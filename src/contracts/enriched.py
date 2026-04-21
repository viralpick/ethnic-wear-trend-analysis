"""EnrichedContentItem — Normalized + 8 속성 + per-attribute method map + 클러스터 키.

속성값 enum 검증은 Pydantic이 자동 처리. LLM/VLM 추출기가 enum 에 없는 값을 내면
ValidationError → 호출자가 해당 필드를 None 으로 떨어뜨린다. fuzzy matching 금지.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from contracts.common import (
    BrandTier,
    ClassificationMethod,
    ColorFamily,
    EmbellishmentIntensity,
    Fabric,
    GarmentType,
    Occasion,
    Silhouette,
    StylingCombo,
    Technique,
)
from contracts.normalized import NormalizedContentItem


class ColorInfo(BaseModel):
    """
    purpose: 포스트 1건에서 VLM 이 추출한 대표 색상 (spec §4.1 ④)
    stage: enriched
    ownership: analysis-owned
    stability: negotiable (VLM 프롬프트 확정 — 4/24)
    """
    model_config = ConfigDict(frozen=True)

    r: int
    g: int
    b: int
    r_pct: float | None = None
    g_pct: float | None = None
    b_pct: float | None = None
    name: str | None = None
    family: ColorFamily | None = None


class BrandInfo(BaseModel):
    """
    purpose: 브랜드 언급 자유 텍스트 + 티어 매핑 (spec §4.1 ⑧)
    stage: enriched
    ownership: analysis-owned
    stability: negotiable (tier 매핑 리스트 확정 전)
    """
    model_config = ConfigDict(frozen=True)

    name: str
    tier: BrandTier | None = None


class EnrichedContentItem(BaseModel):
    """
    purpose: 정규화된 콘텐츠 1건 + 추출된 속성 + 클러스터 배정 결과
    stage: enriched
    ownership: analysis-owned
    stability: locked (4/24 기준 속성 스키마 동결 예정)
    """
    model_config = ConfigDict(frozen=True)

    normalized: NormalizedContentItem

    # spec §4.1 의 8개 속성 + embellishment_intensity 보조 플래그. 전부 Optional.
    garment_type: GarmentType | None = None
    fabric: Fabric | None = None
    technique: Technique | None = None
    embellishment_intensity: EmbellishmentIntensity | None = None
    color: ColorInfo | None = None
    silhouette: Silhouette | None = None
    occasion: Occasion | None = None
    styling_combo: StylingCombo | None = None
    brand: BrandInfo | None = None

    # spec §5 — garment_type × technique × fabric 조합. 부분 매칭 시 "unknown" placeholder.
    # 전부 null 이면 "unclassified".
    trend_cluster_key: str | None = None

    # 속성명 → 추출 방법. 값이 None 인 속성은 키가 없음 (partial map).
    # 키 예시: "garment_type", "technique", "color", "silhouette", "occasion",
    #        "styling_combo", "brand", "embellishment_intensity", "fabric"
    classification_method_per_attribute: dict[str, ClassificationMethod] = Field(
        default_factory=dict
    )
