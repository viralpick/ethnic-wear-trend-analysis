"""EnrichedContentItem — Normalized + 8 속성 + per-attribute method map + 클러스터 키.

속성값 enum 검증은 Pydantic이 자동 처리. LLM/VLM 추출기가 enum 에 없는 값을 내면
ValidationError → 호출자가 해당 필드를 None 으로 떨어뜨린다. fuzzy matching 금지.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

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
    # ζ (2026-04-28): trend_cluster_shares 의 max-share derived 대표값. shares 가
    # canonical source — winner 단독 read 는 fallback 용도.
    trend_cluster_key: str | None = None

    # ζ (2026-04-28): G/T/F cross-product fan-out share dict. winner-only collapse 해소.
    # N=3 representative 승격 case 는 item_cluster_shares 가 다중 entry 채움 (β2).
    # N<3 partial 은 {winner_key: 1.0} single-entry. read-cast: 기존 enriched JSON 의
    # trend_cluster_key 만 있는 경우 _backfill_shares_from_key 가 자동 채움.
    trend_cluster_shares: dict[str, float] = Field(default_factory=dict)

    # 속성명 → 추출 방법. 값이 None 인 속성은 키가 없음 (partial map).
    # 키 예시: "garment_type", "technique", "occasion", "styling_combo", "brand",
    #        "embellishment_intensity", "fabric". silhouette 은 B3d 에서 제거.
    classification_method_per_attribute: dict[str, ClassificationMethod] = Field(
        default_factory=dict
    )

    @model_validator(mode="before")
    @classmethod
    def _backfill_shares_from_key(cls, values: Any) -> Any:
        """ζ read-cast: 기존 enriched JSON 의 trend_cluster_key → {key: 1.0} 1-entry dict.

        shares 가 명시돼 있으면 그대로 둔다 (write 측 권한 우선). shares 가 없거나
        비어있고 key 만 있으면 single-entry 로 채워서 read 측 (.items() 순회) 호환.
        """
        if not isinstance(values, dict):
            return values
        shares = values.get("trend_cluster_shares")
        key = values.get("trend_cluster_key")
        if not shares and key:
            values["trend_cluster_shares"] = {key: 1.0}
        return values
