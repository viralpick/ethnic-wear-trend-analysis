"""Stage 2b: LLM 배치 속성 추출 (spec §6.1 + §6.3).

안전 기본값:
- temperature=0, seed 노출 (기본 42), structured JSON 출력
- LLM 이 enum 밖 값을 내면 LLMExtractionResult 생성 시 Pydantic 이 ValidationError — 해당 post
  결과를 DROP 하고 로그만 남긴다 (retry 없음, coercion 없음). 정확도 퇴행이 숨지 않도록.
- FakeLLMClient 가 기본 (실제 HTTP 호출은 이 스켈레톤에 없다).
"""
from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from attributes.extract_text_attributes import AttributeExtractionState
from contracts.common import (
    ClassificationMethod,
    EmbellishmentIntensity,
    Fabric,
    GarmentType,
    Occasion,
    StylingCombo,
    Technique,
)
from contracts.enriched import BrandInfo
from contracts.normalized import NormalizedContentItem
from utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_LLM_SEED = 42


class LLMExtractionResult(BaseModel):
    """spec §6.3 LLM 출력 JSON 의 타입드 모양.

    LLM 이 채울 수 있는 필드만 포함 (color/silhouette 는 VLM 영역).
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    post_id: str
    garment_type: GarmentType | None = None
    technique: Technique | None = None
    fabric: Fabric | None = None
    embellishment_intensity: EmbellishmentIntensity | None = None
    occasion: Occasion | None = None
    styling_combo: StylingCombo | None = None
    brand_mentioned: str | None = None  # 자유 텍스트 원본. tier 매핑은 별도 단계.


@runtime_checkable
class LLMClient(Protocol):
    def extract_attributes(
        self, posts: list[NormalizedContentItem]
    ) -> list[LLMExtractionResult]: ...


class FakeLLMClient:
    """테스트용 결정론적 LLM. 입력 source_post_id 해시로 고정 응답 생성.

    설계 목표: 같은 입력 → 같은 출력, 운영 비용 0. 실제 HTTP 호출 없음.
    """

    def __init__(self, seed: int = DEFAULT_LLM_SEED) -> None:
        self._seed = seed

    def extract_attributes(
        self, posts: list[NormalizedContentItem]
    ) -> list[LLMExtractionResult]:
        return [self._synthesize(post) for post in posts]

    def _synthesize(self, post: NormalizedContentItem) -> LLMExtractionResult:
        # 해시 기반 결정론. seed 가 바뀌면 응답 분포도 재현 가능하게 달라진다.
        digest = hashlib.sha256(f"{self._seed}:{post.source_post_id}".encode()).digest()
        pick_garment = _pick(digest, 0, list(GarmentType))
        pick_technique = _pick(digest, 1, list(Technique))
        pick_fabric = _pick(digest, 2, list(Fabric))
        return LLMExtractionResult(
            post_id=post.source_post_id,
            garment_type=pick_garment,
            technique=pick_technique,
            fabric=pick_fabric,
        )


def _pick(digest: bytes, offset: int, choices: list) -> object:
    return choices[digest[offset] % len(choices)]


def apply_llm_extraction(
    states: list[AttributeExtractionState],
    llm_client: LLMClient,
) -> None:
    """LLM 이 채울 수 있는 속성 중 하나라도 None 인 state 를 LLM 에 태운다.

    LLM 대상: garment_type, technique, fabric, occasion, styling_combo, embellishment_intensity.
    In-place 로 state 를 업데이트한다. stage 2a 에서 잡힌 값은 절대 덮어쓰지 않는다.
    retry 없음, validation 실패 시 해당 post 결과는 버리고 계속 진행.
    """
    candidates = [
        s for s in states
        if s.garment_type is None
        or s.technique is None
        or s.occasion is None
        or s.styling_combo is None
        or s.embellishment_intensity is None
    ]
    if not candidates:
        return

    # TODO(§6.3): 동일 post_id 를 재처리하지 않게 캐시 레이어 추가 (성능·비용). 현재는 매번 호출.
    results = llm_client.extract_attributes([s.normalized for s in candidates])
    by_post_id = {r.post_id: r for r in results}

    for state in candidates:
        result = by_post_id.get(state.normalized.source_post_id)
        if result is None:
            logger.info("llm_skip post_id=%s reason=no_result", state.normalized.source_post_id)
            continue
        _merge_llm_result(state, result)


def _merge_llm_result(state: AttributeExtractionState, result: LLMExtractionResult) -> None:
    """LLM 결과로 state 의 None 필드만 채운다. rule 단계 값은 보존."""
    if state.garment_type is None and result.garment_type is not None:
        state.garment_type = result.garment_type
        state.method_per_attribute["garment_type"] = ClassificationMethod.LLM
    if state.technique is None and result.technique is not None:
        state.technique = result.technique
        state.method_per_attribute["technique"] = ClassificationMethod.LLM
    if state.fabric is None and result.fabric is not None:
        state.fabric = result.fabric
        state.method_per_attribute["fabric"] = ClassificationMethod.LLM
    if state.embellishment_intensity is None and result.embellishment_intensity is not None:
        state.embellishment_intensity = result.embellishment_intensity
        state.method_per_attribute["embellishment_intensity"] = ClassificationMethod.LLM
    if state.occasion is None and result.occasion is not None:
        state.occasion = result.occasion
        state.method_per_attribute["occasion"] = ClassificationMethod.LLM
    if state.styling_combo is None and result.styling_combo is not None:
        state.styling_combo = result.styling_combo
        state.method_per_attribute["styling_combo"] = ClassificationMethod.LLM
    # LLM 의 brand_mentioned (free text) 는 rule 단계 brands 가 비어있을 때만 1건 fallback.
    # tier 미매핑 — 향후 registry 매칭으로 보강 가능.
    if not state.brands and result.brand_mentioned:
        state.brands = [BrandInfo(name=result.brand_mentioned)]
        state.method_per_attribute["brand"] = ClassificationMethod.LLM
