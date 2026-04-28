"""EnrichedContentItem — Normalized + 8 속성 + per-attribute method map + 클러스터 키.

속성값 enum 검증은 Pydantic이 자동 처리. LLM/VLM 추출기가 enum 에 없는 값을 내면
ValidationError → 호출자가 해당 필드를 None 으로 떨어뜨린다. fuzzy matching 금지.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from contracts.common import (
    BrandTier,
    ClassificationMethod,
    EmbellishmentIntensity,
    Fabric,
    GarmentType,
    Occasion,
    PaletteCluster,
    StylingCombo,
    Technique,
)
from contracts.normalized import NormalizedContentItem
from contracts.vision import CanonicalOutfit


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

    # spec §4.1 의 속성 중 text 기반으로 채워지는 단일값만 post-level 유지.
    # Color 3층 재설계 (2026-04-24) 로 color 단일값은 post_palette 로 이동.
    # B3d (2026-04-24) 로 silhouette 단일값 제거 — vision 기반이므로
    # canonicals[*].representative.silhouette 로만 접근 (feedback_post_level_single_value).
    garment_type: GarmentType | None = None
    fabric: Fabric | None = None
    technique: Technique | None = None
    embellishment_intensity: EmbellishmentIntensity | None = None
    occasion: Occasion | None = None
    styling_combo: StylingCombo | None = None

    # M3.F brand registry — 한 post 에 여러 brand 가능 (haul / collab / styled-by).
    # account_handle + caption @mention 모두 dedup 수집. 매칭 0건 시 빈 list.
    brands: list[BrandInfo] = Field(default_factory=list)

    # post-level palette (2026-04-24 신설) — canonicals[*].palette 들을 ΔE76 greedy merge.
    # multi-outfit 전제, max 3 cluster. B3 build_palette 에서 채움.
    post_palette: list[PaletteCluster] = Field(default_factory=list, max_length=3)

    # post 안의 outfit dedup 결과. B3 adapter 가 canonical_extractor 결과를 그대로 적재.
    # canonical 단위 palette / silhouette / garment type 은 여기 CanonicalOutfit 내부에서만
    # 유지 (post-level 단일값 금지 — multi-outfit 가정, feedback_post_level_single_value).
    canonicals: list[CanonicalOutfit] = Field(default_factory=list)

    # spec §5 — garment_type × technique × fabric 조합. 부분 매칭 시 "unknown" placeholder.
    # 전부 null 이면 "unclassified".
    trend_cluster_key: str | None = None

    # 속성명 → 추출 방법. 값이 None 인 속성은 키가 없음 (partial map).
    # 키 예시: "garment_type", "technique", "occasion", "styling_combo", "brand",
    #        "embellishment_intensity", "fabric". silhouette 은 B3d 에서 제거.
    classification_method_per_attribute: dict[str, ClassificationMethod] = Field(
        default_factory=dict
    )
