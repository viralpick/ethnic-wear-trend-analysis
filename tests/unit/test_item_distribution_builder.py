"""enriched_to_item_distribution pinning — pipeline_spec_v1.0 §2.4 입력 변환.

검증:
- text 단일값 (rule/LLM) 만 있는 post → distribution = {text_value: 1.0}, source 정합.
- vision canonical 만 있는 post → group_to_item_contrib 비례 분배 (build_distribution 위임).
- text + vision 합산 — text 가중치 (rule=6, llm=3) + vision share 합 (G=log2(Σn+1)) 정규화.
- canonical 의 representative.upper_garment_type 이 garment value, lower_garment_type 무시.
- text/vision 모두 비면 빈 dict, 그래도 ItemDistribution 자체는 생성됨 (item_id/source).
- item_id = `{source}__{source_post_id}` 포맷 고정.
"""
from __future__ import annotations

from datetime import datetime

from aggregation.item_distribution_builder import enriched_to_item_distribution
from contracts.common import (
    ClassificationMethod,
    ContentSource,
    Fabric,
    GarmentType,
    Technique,
)
from contracts.enriched import EnrichedContentItem
from contracts.normalized import NormalizedContentItem
from contracts.vision import CanonicalOutfit, EthnicOutfit, OutfitMember


def _normalized(post_id: str, source: ContentSource = ContentSource.INSTAGRAM) -> NormalizedContentItem:
    return NormalizedContentItem(
        source=source,
        source_post_id=post_id,
        text_blob="",
        hashtags=[],
        image_urls=[],
        post_date=datetime(2026, 4, 27),
        engagement_raw=0,
    )


def _outfit(
    *,
    upper: str | None = "kurta",
    lower: str | None = "palazzo",
    fabric: str | None = "cotton",
    technique: str | None = "block_print",
    bbox: tuple[float, float, float, float] = (0.1, 0.1, 0.5, 0.7),
    area_ratio: float = 0.35,
    upper_is_ethnic: bool | None = True,
    lower_is_ethnic: bool | None = True,
    dress_as_single: bool = False,
) -> EthnicOutfit:
    return EthnicOutfit(
        person_bbox=bbox,
        person_bbox_area_ratio=area_ratio,
        upper_garment_type=upper,
        upper_is_ethnic=upper_is_ethnic,
        lower_garment_type=lower,
        lower_is_ethnic=lower_is_ethnic,
        dress_as_single=dress_as_single,
        fabric=fabric,
        technique=technique,
        color_preset_picks_top3=[],
    )


def _canonical(
    index: int,
    *,
    outfit: EthnicOutfit,
    member_bboxes: list[tuple[float, float, float, float]] | None = None,
) -> CanonicalOutfit:
    bboxes = member_bboxes or [outfit.person_bbox]
    members = [
        OutfitMember(image_id=f"img_{i}", outfit_index=0, person_bbox=bb)
        for i, bb in enumerate(bboxes)
    ]
    return CanonicalOutfit(
        canonical_index=index,
        representative=outfit,
        members=members,
    )


def test_text_only_rule_distribution_is_text_value() -> None:
    item = EnrichedContentItem(
        normalized=_normalized("p_text"),
        garment_type=GarmentType.KURTA_SET,
        fabric=Fabric.COTTON,
        technique=Technique.BLOCK_PRINT,
        canonicals=[],
        classification_method_per_attribute={
            "garment_type": ClassificationMethod.RULE,
            "fabric": ClassificationMethod.RULE,
            "technique": ClassificationMethod.LLM,
        },
    )
    dist = enriched_to_item_distribution(item)

    assert dist.item_id == "instagram__p_text"
    assert dist.source == ContentSource.INSTAGRAM
    assert dist.garment_type == {"kurta_set": 1.0}
    assert dist.fabric == {"cotton": 1.0}
    assert dist.technique == {"block_print": 1.0}


def test_text_with_no_method_drops_to_vision_only() -> None:
    # text_method 가 없으면 가중치 0 → text 무가중치 → vision 만 남음.
    canonical = _canonical(
        0,
        outfit=_outfit(upper="kurta", fabric="cotton", technique="block_print"),
    )
    item = EnrichedContentItem(
        normalized=_normalized("p_no_method"),
        garment_type=GarmentType.KURTA_SET,
        fabric=Fabric.COTTON,
        technique=Technique.BLOCK_PRINT,
        canonicals=[canonical],
        classification_method_per_attribute={},  # method 없음
    )
    dist = enriched_to_item_distribution(item)

    # text 가중치 0 + vision group 1개 → 100% vision value (vision_normalize 매핑됨).
    # raw "kurta" → STRAIGHT_KURTA, raw "block_print" → BLOCK_PRINT (직접).
    assert dist.garment_type == {"straight_kurta": 1.0}
    assert dist.fabric == {"cotton": 1.0}
    assert dist.technique == {"block_print": 1.0}


def test_vision_only_uses_upper_garment_type_not_lower() -> None:
    # text 미설정 + canonical representative 의 upper="kurta", lower="palazzo".
    # garment_type distribution 은 upper 만 반영해야 함 (lower 는 styling_combo 입력).
    canonical = _canonical(
        0,
        outfit=_outfit(upper="kurta", lower="palazzo"),
    )
    item = EnrichedContentItem(
        normalized=_normalized("p_vision"),
        canonicals=[canonical],
    )
    dist = enriched_to_item_distribution(item)

    # raw "kurta" → STRAIGHT_KURTA, raw "palazzo" 는 lower 라 garment 에 미반영.
    assert "straight_kurta" in dist.garment_type
    assert "palazzo" not in dist.garment_type
    assert dist.garment_type == {"straight_kurta": 1.0}


def test_text_and_vision_combine_with_weighted_sum() -> None:
    # text=KURTA_SET (rule weight=6) + vision canonical (upper="saree").
    # build_distribution: total = 6 + share_saree, normalized.
    canonical = _canonical(
        0,
        outfit=_outfit(upper="saree", fabric="silk", technique="zardosi"),
    )
    item = EnrichedContentItem(
        normalized=_normalized("p_mix"),
        garment_type=GarmentType.KURTA_SET,
        canonicals=[canonical],
        classification_method_per_attribute={
            "garment_type": ClassificationMethod.RULE,
        },
    )
    dist = enriched_to_item_distribution(item)

    # 두 키 모두 등장 + 합 = 1.0 + text(rule=6) > vision share.
    # raw "saree" → CASUAL_SAREE 매핑.
    assert set(dist.garment_type.keys()) == {"kurta_set", "casual_saree"}
    assert abs(sum(dist.garment_type.values()) - 1.0) < 1e-9
    assert dist.garment_type["kurta_set"] > dist.garment_type["casual_saree"]


def test_no_text_no_canonicals_returns_empty_distributions() -> None:
    item = EnrichedContentItem(
        normalized=_normalized("p_empty"),
        canonicals=[],
    )
    dist = enriched_to_item_distribution(item)

    assert dist.item_id == "instagram__p_empty"
    assert dist.garment_type == {}
    assert dist.fabric == {}
    assert dist.technique == {}


def test_canonical_with_none_value_drops_silently() -> None:
    # representative.upper_garment_type=None → garment_type group 자연 drop.
    # upper=None 이면 upper_is_ethnic 도 False/None 이라 case D 아닌 단순 drop.
    canonical = _canonical(
        0,
        outfit=_outfit(
            upper=None, lower=None, fabric="cotton", technique=None,
            upper_is_ethnic=None, lower_is_ethnic=None,
        ),
    )
    item = EnrichedContentItem(
        normalized=_normalized("p_none"),
        canonicals=[canonical],
    )
    dist = enriched_to_item_distribution(item)

    # canonical 자체가 비-ethnic (둘 다 None) 이라 is_canonical_ethnic=False → 전체 drop.
    assert dist.garment_type == {}
    assert dist.fabric == {}
    assert dist.technique == {}


def test_youtube_source_propagates_to_item_id_and_source() -> None:
    item = EnrichedContentItem(
        normalized=_normalized("yt123", source=ContentSource.YOUTUBE),
        garment_type=GarmentType.KURTA_SET,
        canonicals=[],
        classification_method_per_attribute={
            "garment_type": ClassificationMethod.RULE,
        },
    )
    dist = enriched_to_item_distribution(item)

    assert dist.item_id == "youtube__yt123"
    assert dist.source == ContentSource.YOUTUBE


def test_two_canonicals_share_by_group_to_item_contrib() -> None:
    # 두 canonical 모두 같은 upper="kurta", 다른 fabric/technique.
    # garment_type 은 단일 키로 합쳐져 1.0, fabric/technique 은 2-way split.
    big = _canonical(
        0,
        outfit=_outfit(upper="kurta", fabric="silk", technique="zardosi", area_ratio=0.5),
        member_bboxes=[(0.0, 0.0, 0.7, 0.7)],
    )
    small = _canonical(
        1,
        outfit=_outfit(upper="kurta", fabric="cotton", technique="plain", area_ratio=0.1),
        member_bboxes=[(0.6, 0.6, 0.3, 0.3)],
    )
    item = EnrichedContentItem(
        normalized=_normalized("p_two_canon"),
        canonicals=[big, small],
    )
    dist = enriched_to_item_distribution(item)

    # 둘 다 raw "kurta" → STRAIGHT_KURTA 로 합쳐짐.
    assert dist.garment_type == {"straight_kurta": 1.0}
    # fabric 은 2 키 모두 등장, 큰 canonical 이 더 큰 share.
    # silk 은 enum 신규 추가됨.
    assert set(dist.fabric.keys()) == {"silk", "cotton"}
    assert dist.fabric["silk"] > dist.fabric["cotton"]
    assert abs(sum(dist.fabric.values()) - 1.0) < 1e-9


# --------------------------------------------------------------------------- #
# 비-ethnic canonical 차단 가드 pinning (2026-04-28)

def test_non_ethnic_canonical_excluded_from_distribution() -> None:
    """canonical_extractor 의 라벨 보존으로 enriched.canonicals 에 비-ethnic 이 살아남아도
    representative_weekly contribution 단계에서 차단돼야 한다."""
    eth = _canonical(0, outfit=_outfit(upper="kurta", fabric="cotton"))
    non_eth = _canonical(
        1,
        outfit=_outfit(
            upper="t_shirt",
            lower="jeans",
            fabric="denim",
            upper_is_ethnic=False,
            lower_is_ethnic=False,
        ),
    )
    item = EnrichedContentItem(
        normalized=_normalized("p_mixed"),
        canonicals=[eth, non_eth],
    )
    dist = enriched_to_item_distribution(item)

    # ethnic canonical 1개만 contribution. raw "kurta" → STRAIGHT_KURTA.
    assert dist.garment_type == {"straight_kurta": 1.0}
    assert dist.fabric == {"cotton": 1.0}
    # 비-ethnic canonical 의 t_shirt / denim 누락 확인 (denim 은 enum 외라 어차피 drop)
    assert "t_shirt" not in dist.garment_type
    assert "denim" not in dist.fabric


def test_dress_as_single_non_ethnic_excluded() -> None:
    """dress_as_single=True + upper_is_ethnic=False (e.g. evening dress) 도 차단."""
    eth_dress = _canonical(
        0,
        outfit=_outfit(
            upper="anarkali",
            lower=None,
            fabric="silk",
            upper_is_ethnic=True,
            lower_is_ethnic=None,
            dress_as_single=True,
        ),
    )
    non_eth_dress = _canonical(
        1,
        outfit=_outfit(
            upper="cocktail_dress",
            lower=None,
            fabric="polyester",
            upper_is_ethnic=False,
            lower_is_ethnic=None,
            dress_as_single=True,
        ),
    )
    item = EnrichedContentItem(
        normalized=_normalized("p_dress"),
        canonicals=[eth_dress, non_eth_dress],
    )
    dist = enriched_to_item_distribution(item)
    # raw "anarkali" → ANARKALI, "silk" → SILK (신규 enum).
    assert dist.garment_type == {"anarkali": 1.0}
    assert dist.fabric == {"silk": 1.0}


def test_all_non_ethnic_canonicals_yield_empty_distribution() -> None:
    """post 안 모든 canonical 이 비-ethnic 이면 distribution 비어야 한다 (text 미주입)."""
    non_eth = _canonical(
        0,
        outfit=_outfit(
            upper="t_shirt",
            lower="jeans",
            upper_is_ethnic=False,
            lower_is_ethnic=False,
        ),
    )
    item = EnrichedContentItem(
        normalized=_normalized("p_all_non"),
        canonicals=[non_eth],
    )
    dist = enriched_to_item_distribution(item)
    assert dist.garment_type == {}
    assert dist.fabric == {}
    assert dist.technique == {}
