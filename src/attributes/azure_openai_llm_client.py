"""Azure OpenAI 기반 LLM 속성 추출 클라이언트 (spec §6.3).

연결 크리덴셜 (.env):
  AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT,
  AZURE_OPENAI_DEPLOYMENT, AZURE_OPENAI_API_VERSION

설계 원칙:
- temperature=0, seed=42 — 결정론 최대화
- 배치 단위 요청 (batch_size=10 기본) — 비용/속도 균형
- max_workers > 1 이면 배치들 thread pool 로 병렬 호출 (rate limit 감안 8 권장)
- LLM 출력 enum 외 값 → Pydantic ValidationError → 해당 post DROP (retry 없음)
- rate limit / timeout → raise (상위에서 처리)

top-level import 로 openai / python-dotenv 사용. `[llm]` extras 미설치 환경에서는
이 모듈 import 자체가 ImportError — core 는 절대 top-level import 하지 말 것.
"""
from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor

import openai
from dotenv import load_dotenv
from pydantic import ValidationError

from attributes.extract_text_attributes_llm import LLMExtractionResult
from contracts.common import (
    EmbellishmentIntensity,
    Fabric,
    GarmentType,
    Occasion,
    StylingCombo,
    Technique,
)
from contracts.normalized import NormalizedContentItem

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = f"""You are a fashion attribute extractor for Indian ethnic wear content.
Extract structured attributes from Instagram captions and hashtags (or YouTube titles/descriptions).

Return a JSON object with key "results" — an array with one entry per input post (same order).
Each entry:
{{
  "post_id": "<copy from input>",
  "garment_type": one of {[e.value for e in GarmentType]} or null,
  "technique": one of {[e.value for e in Technique]} or null,
  "fabric": one of {[e.value for e in Fabric]} or null,
  "occasion": one of {[e.value for e in Occasion]} or null,
  "styling_combo": one of {[e.value for e in StylingCombo]} or null,
  "embellishment_intensity": one of {[e.value for e in EmbellishmentIntensity]} or null,
  "brand_mentioned": "<brand name string>" or null
}}

Rules:
- Use ONLY the exact enum values listed above. null if not determinable.
- occasion guidance: festive_lite = weddings/pujas/festivals/functions, office = work/corporate,
  travel = trips/vacations, weekend = parties/evenings/brunches, campus = college/university,
  casual = everyday/relaxed/indo-western.
- embellishment_intensity: minimal = solid/simple print, moderate = some work, heavy = zardozi/mirror/heavy embroidery.
- brand_mentioned: extract brand/designer handle or name only if explicitly mentioned.
- Do NOT guess — null is correct when evidence is absent.
"""


class AzureOpenAILLMClient:
    """Azure OpenAI 기반 LLM 속성 추출. LLMClient Protocol 구현."""

    def __init__(
        self, batch_size: int = 10, seed: int = 42, max_workers: int = 1,
    ) -> None:
        load_dotenv()
        self._client = openai.AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        )
        self._deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]
        self._batch_size = batch_size
        self._seed = seed
        self._max_workers = max(1, max_workers)

    @classmethod
    def from_env(
        cls, batch_size: int = 10, max_workers: int = 1,
    ) -> "AzureOpenAILLMClient":
        return cls(batch_size=batch_size, max_workers=max_workers)

    def extract_attributes(
        self, posts: list[NormalizedContentItem]
    ) -> list[LLMExtractionResult]:
        batches = [
            posts[i : i + self._batch_size]
            for i in range(0, len(posts), self._batch_size)
        ]
        if not batches:
            return []
        if self._max_workers <= 1 or len(batches) <= 1:
            return [r for batch in batches for r in self._extract_batch(batch)]
        # openai SDK + httpx 클라이언트는 thread-safe — concurrent batch 안전.
        # rate limit 은 호출자가 max_workers 로 통제 (Azure deployment TPM 한도 고려).
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            nested = list(executor.map(self._extract_batch, batches))
        return [r for batch_results in nested for r in batch_results]

    def _extract_batch(
        self, posts: list[NormalizedContentItem]
    ) -> list[LLMExtractionResult]:
        user_content = json.dumps(
            [{"post_id": p.source_post_id, "text": p.text_blob[:800]} for p in posts],
            ensure_ascii=False,
        )
        try:
            response = self._client.chat.completions.create(
                model=self._deployment,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                response_format={"type": "json_object"},
                seed=self._seed,
            )
        except openai.OpenAIError as exc:
            logger.warning("llm_batch_error reason=%s", exc)
            return []

        raw = response.choices[0].message.content or "{}"
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("llm_json_parse_error reason=%s raw=%s", exc, raw[:200])
            return []

        results: list[LLMExtractionResult] = []
        for item in payload.get("results", []):
            try:
                results.append(LLMExtractionResult(**item))
            except (ValidationError, TypeError) as exc:
                logger.info("llm_result_skip post_id=%s reason=%s", item.get("post_id"), exc)
        return results
