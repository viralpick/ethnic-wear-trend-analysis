"""GeminiVisionLLMClient — Phase 2 실 구현.

Gemini 2.5 Flash 를 google-genai SDK 로 호출해 GarmentAnalysis 반환.

Phase 0 실측:
- 파일럿 20 post: 19/20 JSON 형식 OK (1건 outfit 닫는 `}` 누락). BBOX 정확도 gpt-5-mini
  대비 tighter. ~$0.0009 / call, ~12s / call.

JSON 무결성 전략 (spec §4 timeout 가드 허용 범위 내):
1. 1차: `response_mime_type="application/json"` + `json.loads` → Pydantic validate.
2. 1차 json.loads 실패 시: `json_repair_util.parse_json_with_repair` 로 코드 기반 수리
   (LLM 호출 없음). 결과를 동일하게 Pydantic validate.
3. validate 실패 시 1회 Gemini retry (프롬프트에 "CRITICAL: STRICT JSON" 추가 문구).
   retry 결과에도 repair 적용. 그래도 실패면 raise — 상위에서 해당 post drop.
4. validate 통과 후 traditional_filter.apply_to_analysis 로 whitelist 기반 post-filter
   적용 (True → False flip 안전망).

Retry 분기:
- JSONDecodeError / ValueError (json_repair non-dict) → retry
- ValidationError → **즉시 raise**. 스키마 위반은 재시도해도 같은 버그가 재발할 가능성이
  높고, 형식 버그보다 명시 실패로 노출하는 게 낫다 (spec §4 failure-not-hidden).

top-level import 로 google.genai / python-dotenv 사용. `[vision]` extras 미설치 환경에서는
이 모듈 import 자체가 ImportError — core 는 절대 이 파일 top-level import 하지 말 것.
"""
from __future__ import annotations

import json
import logging
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import ValidationError

from contracts.vision import GarmentAnalysis
from vision.json_repair_util import parse_json_with_repair
from vision.llm_cache import VisionLLMCacheBackend, compute_cache_key
from vision.prompts import PROMPT_VERSION, SYSTEM_PROMPT, build_user_payload
from vision.traditional_filter import apply_to_analysis

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "gemini-2.5-flash"

# llm-safety §3 — timeout 가드 필수. Gemini 2.5 Flash 평균 ~12s/call (Phase 0 실측),
# 60s 는 image+prompt 안정 + hang 차단. SDK 가 ms 단위.
_GEMINI_TIMEOUT_MS = 60_000


class GeminiVisionLLMClient:
    """google-genai SDK 기반 vision LLM 클라이언트. VisionLLMClient Protocol 구현."""

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        prompt_version: str = PROMPT_VERSION,
        cache: VisionLLMCacheBackend | None = None,
        api_key: str | None = None,
    ) -> None:
        if api_key is None:
            load_dotenv()
            api_key = os.environ["GEMINI_API_KEY"]
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=_GEMINI_TIMEOUT_MS),
        )
        self._model_id = model_id
        self._prompt_version = prompt_version
        self._cache = cache

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def prompt_version(self) -> str:
        return self._prompt_version

    def extract_garment(
        self, image_bytes: bytes, *, preset: list[dict[str, str]]
    ) -> GarmentAnalysis:
        cache_key = compute_cache_key(
            image_bytes,
            prompt_version=self._prompt_version,
            model_id=self._model_id,
        )
        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                logger.debug("gemini_cache_hit cache_key=%s", cache_key[:12])
                return cached

        analysis = self._call_with_retry(image_bytes, preset)
        if self._cache is not None:
            self._cache.put(cache_key, analysis)
        return analysis

    def _call_with_retry(
        self, image_bytes: bytes, preset: list[dict[str, str]]
    ) -> GarmentAnalysis:
        """JSON 형식 버그는 1회 retry, 스키마 위반 (ValidationError) 은 즉시 raise."""
        try:
            return self._call_once(image_bytes, preset, strict_hint=False)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning(
                "gemini_format_error retry=1 reason=%s",
                f"{type(exc).__name__}: {exc}",
            )
            return self._call_once(image_bytes, preset, strict_hint=True)

    def _call_once(
        self,
        image_bytes: bytes,
        preset: list[dict[str, str]],
        *,
        strict_hint: bool,
    ) -> GarmentAnalysis:
        system_text = SYSTEM_PROMPT
        if strict_hint:
            system_text += (
                "\n\nCRITICAL: Return STRICT JSON only. All brackets balanced. "
                "No trailing commas. No markdown fences."
            )
        contents = [
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            system_text + "\n\n" + build_user_payload(preset),
        ]
        resp = self._client.models.generate_content(
            model=self._model_id,
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0,
            ),
        )
        # Cost baseline (2026-05-01): token rate 측정 — frame resize 안 하면 384×384 이하
        # (~258 tok) vs 1024+ (~1,290 tok, 5×). 1주 운영 후 평균 산출 → resize/quality
        # tuning 의사결정 데이터. usage_metadata 누락 시 silent skip (정책상 안전).
        usage = getattr(resp, "usage_metadata", None)
        if usage is not None:
            logger.info(
                "gemini_usage model=%s image_bytes=%d prompt_tokens=%s output_tokens=%s",
                self._model_id,
                len(image_bytes),
                getattr(usage, "prompt_token_count", None),
                getattr(usage, "candidates_token_count", None),
            )
        raw = resp.text or ""
        if not raw.strip():
            raise ValueError("gemini returned empty text")
        payload = parse_json_with_repair(raw)
        try:
            analysis = GarmentAnalysis.model_validate(payload)
        except ValidationError:
            # 스키마 위반은 retry 해도 LLM 이 같은 실수 반복할 가능성 높아 즉시 raise
            logger.warning(
                "gemini_schema_violation payload_head=%r",
                json.dumps(payload)[:200] if isinstance(payload, dict) else str(payload)[:200],
            )
            raise
        return apply_to_analysis(analysis)
