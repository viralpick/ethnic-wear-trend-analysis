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

from contracts.vision import GarmentAnalysis, KMeansAnchoredPickResponse
from vision.json_repair_util import parse_json_with_repair
from vision.llm_cache import (
    KMeansAnchoredPickCacheBackend,
    VisionLLMCacheBackend,
    compute_cache_key,
    compute_v010_cache_key,
)
from vision.prompts import (
    COLOR_PICK_V010_PROMPT_VERSION,
    COLOR_PICK_V010_SYSTEM_PROMPT,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    build_color_pick_v010_user_payload,
    build_user_payload,
)
from vision.traditional_filter import apply_to_analysis

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "gemini-2.5-flash"

# llm-safety §3 — timeout 가드 필수. Gemini 2.5 Flash 평균 ~12s/call (Phase 0 실측),
# 60s 는 image+prompt 안정 + hang 차단. SDK 가 ms 단위.
_GEMINI_TIMEOUT_MS = 60_000

# 절대 규칙 #5 (재현성) — `temperature=0.0` 만으론 완전 결정론 보장 X. seed 도 명시.
_GEMINI_SEED = 42


class GeminiVisionLLMClient:
    """google-genai SDK 기반 vision LLM 클라이언트. VisionLLMClient Protocol 구현."""

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        prompt_version: str = PROMPT_VERSION,
        cache: VisionLLMCacheBackend | None = None,
        v010_cache: KMeansAnchoredPickCacheBackend | None = None,
        api_key: str | None = None,
    ) -> None:
        if api_key is None:
            load_dotenv()
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError(
                    "GEMINI_API_KEY missing — set in .env or pass api_key= explicitly. "
                    "see .env.example"
                )
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=_GEMINI_TIMEOUT_MS),
        )
        self._model_id = model_id
        self._prompt_version = prompt_version
        self._cache = cache
        # color.B v0.10 — Pass 2 cache (cluster_top_n_hex 포함 key). 미주입 시 매 호출.
        self._v010_cache = v010_cache

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
                seed=_GEMINI_SEED,
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

    def pick_colors_from_kmeans(
        self,
        image_bytes: bytes,
        *,
        garment_classification: dict[str, object],
        kmeans_clusters: list[dict[str, object]],
    ) -> KMeansAnchoredPickResponse:
        """color.B v0.10 Pass 2 — Gemini 가 KMeans cluster top-N 안에서 색 pick.

        실패 시 raise — adapter 가 catch 해 Pass 1 picks 로 fallback (spec v0.10 결정 1-a).
        v010_cache 가 주입돼 있으면 cache hit/miss 분기 (cache key 는 image + Pass 2
        prompt_version + model_id + cluster_top_n hex 시퀀스).
        """
        if not kmeans_clusters:
            raise ValueError("kmeans_clusters must not be empty for v0.10 pick")
        # cache lookup — v0.10 prompt_version 은 caller 의 self._prompt_version 와 별개
        # (그건 Pass 1 v0.9). Pass 2 prompt 는 COLOR_PICK_V010_PROMPT_VERSION 상수.
        cache_key: str | None = None
        if self._v010_cache is not None:
            cluster_hexes = tuple(
                str(c.get("hex", "")) for c in kmeans_clusters
            )
            cache_key = compute_v010_cache_key(
                image_bytes,
                prompt_version=COLOR_PICK_V010_PROMPT_VERSION,
                model_id=self._model_id,
                cluster_hexes=cluster_hexes,
            )
            cached = self._v010_cache.get(cache_key)
            if cached is not None:
                logger.debug("gemini_v010_cache_hit cache_key=%s", cache_key[:12])
                return cached
        contents = [
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            COLOR_PICK_V010_SYSTEM_PROMPT
            + "\n\n"
            + build_color_pick_v010_user_payload(
                garment_classification=garment_classification,
                kmeans_clusters=kmeans_clusters,
            ),
        ]
        resp = self._client.models.generate_content(
            model=self._model_id,
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0,
                seed=_GEMINI_SEED,
            ),
        )
        usage = getattr(resp, "usage_metadata", None)
        if usage is not None:
            logger.info(
                "gemini_v010_usage model=%s image_bytes=%d prompt_tokens=%s output_tokens=%s",
                self._model_id,
                len(image_bytes),
                getattr(usage, "prompt_token_count", None),
                getattr(usage, "candidates_token_count", None),
            )
        raw = resp.text or ""
        if not raw.strip():
            raise ValueError("gemini v0.10 returned empty text")
        payload = parse_json_with_repair(raw)
        try:
            response = KMeansAnchoredPickResponse.model_validate(payload)
        except ValidationError:
            logger.warning(
                "gemini_v010_schema_violation payload_head=%r",
                json.dumps(payload)[:200]
                if isinstance(payload, dict)
                else str(payload)[:200],
            )
            raise
        if self._v010_cache is not None and cache_key is not None:
            self._v010_cache.put(cache_key, response)
        return response
