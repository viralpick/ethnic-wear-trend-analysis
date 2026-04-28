"""_vision_reassign_cluster_key pinning — 갭 #3 (B).

text 단계 partial cluster_key 가 vision 채움 후에도 representative_key 와 어긋나는
문제 해소. ItemDistribution top-1 G/T/F 로 키를 재합성해서 representative_builder
가 만드는 representative_key 와 mechanical 하게 일치시킨다.

검증 axes:
- canonicals 비어있으면 기존 키 유지 (text 결정 보존).
- vision dist 의 G/T/F 한 axis 라도 비면 기존 키 유지 (어차피 N=3 적재 안됨).
- text-level partial 가 vision 결과로 exact 키 승격 (canonical EXACT 정합).
- vision raw 값이 enum 멤버에 없으면 기존 키 유지 (free-form 방어).
- 동률 시 (-pct, value asc) 결정론적 tiebreak 으로 키 안정.
"""
from __future__ import annotations

from datetime import datetime

from clustering.assign_trend_cluster import build_exact_key
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
from pipelines.run_daily_pipeline import _vision_reassign_cluster_key


def _normalized(post_id: str = "p1") -> NormalizedContentItem:
    return NormalizedContentItem(
        source=ContentSource.INSTAGRAM,
        source_post_id=post_id,
        text_blob="",
        hashtags=[],
        image_urls=[],
        post_date=datetime(2026, 4, 27),
        engagement_raw=0,
    )


def _outfit(
    *,
    upper: str | None,
    fabric: str | None,
    technique: str | None,
    area_ratio: float = 0.35,
    bbox: tuple[float, float, float, float] = (0.1, 0.1, 0.5, 0.7),
) -> EthnicOutfit:
    return EthnicOutfit(
        person_bbox=bbox,
        person_bbox_area_ratio=area_ratio,
        upper_garment_type=upper,
        upper_is_ethnic=True,
        lower_garment_type=None,
        lower_is_ethnic=None,
        dress_as_single=True,
        fabric=fabric,
        technique=technique,
        color_preset_picks_top3=[],
    )


def _canonical(outfit: EthnicOutfit) -> CanonicalOutfit:
    return CanonicalOutfit(
        canonical_index=0,
        representative=outfit,
        members=[
            OutfitMember(
                image_id="img_0",
                outfit_index=0,
                person_bbox=outfit.person_bbox,
            )
        ],
    )


def test_no_canonicals_keeps_existing_key() -> None:
    item = EnrichedContentItem(
        normalized=_normalized(),
        garment_type=GarmentType.KURTA_SET,
        canonicals=[],
        trend_cluster_key="kurta_set__unknown__unknown",
    )
    assert _vision_reassign_cluster_key(item) == "kurta_set__unknown__unknown"


def test_partial_text_promoted_to_exact_after_vision() -> None:
    # text: garment_type=KURTA_SET (rule), 나머지 None → partial 키.
    # vision: canonical 1 개로 G/T/F 모두 enum 매핑 가능한 값.
    canonical = _canonical(
        _outfit(upper="kurta_set", fabric="cotton", technique="block_print"),
    )
    item = EnrichedContentItem(
        normalized=_normalized(),
        garment_type=GarmentType.KURTA_SET,
        canonicals=[canonical],
        classification_method_per_attribute={
            "garment_type": ClassificationMethod.RULE,
        },
        trend_cluster_key="kurta_set__unknown__unknown",
    )
    new_key = _vision_reassign_cluster_key(item)
    assert new_key == build_exact_key(
        GarmentType.KURTA_SET, Technique.BLOCK_PRINT, Fabric.COTTON
    )


def test_vision_axis_missing_keeps_existing_key() -> None:
    # canonical 의 fabric=None → fabric distribution 비어 N<3 → 기존 키 유지.
    canonical = _canonical(
        _outfit(upper="kurta_set", fabric=None, technique="block_print"),
    )
    item = EnrichedContentItem(
        normalized=_normalized(),
        garment_type=GarmentType.KURTA_SET,
        canonicals=[canonical],
        classification_method_per_attribute={
            "garment_type": ClassificationMethod.RULE,
        },
        trend_cluster_key="kurta_set__unknown__unknown",
    )
    assert _vision_reassign_cluster_key(item) == "kurta_set__unknown__unknown"


def test_vision_freeform_value_keeps_existing_key() -> None:
    # vision 이 enum 멤버에 없는 free-form 값을 채움 → ValueError → 기존 키 유지.
    canonical = _canonical(
        _outfit(upper="not_a_real_garment", fabric="cotton", technique="block_print"),
    )
    item = EnrichedContentItem(
        normalized=_normalized(),
        canonicals=[canonical],
        trend_cluster_key="unknown__unknown__unknown",
    )
    assert _vision_reassign_cluster_key(item) == "unknown__unknown__unknown"


def test_text_dominates_when_rule_weight_outweighs_vision() -> None:
    # text=KURTA_SET (rule weight=6) + vision saree 1 canonical (작은 area).
    # rule weight 6 > vision share, 따라서 garment top-1 = kurta_set.
    canonical = _canonical(
        _outfit(
            upper="straight_kurta",
            fabric="cotton",
            technique="block_print",
            area_ratio=0.05,
        ),
    )
    item = EnrichedContentItem(
        normalized=_normalized(),
        garment_type=GarmentType.KURTA_SET,
        fabric=Fabric.COTTON,
        technique=Technique.BLOCK_PRINT,
        canonicals=[canonical],
        classification_method_per_attribute={
            "garment_type": ClassificationMethod.RULE,
            "fabric": ClassificationMethod.RULE,
            "technique": ClassificationMethod.RULE,
        },
        trend_cluster_key=build_exact_key(
            GarmentType.KURTA_SET, Technique.BLOCK_PRINT, Fabric.COTTON
        ),
    )
    new_key = _vision_reassign_cluster_key(item)
    # text rule 가 vision 보다 큼 → kurta_set 유지.
    assert new_key == build_exact_key(
        GarmentType.KURTA_SET, Technique.BLOCK_PRINT, Fabric.COTTON
    )


def test_vision_overrides_when_text_method_missing() -> None:
    # text_method 없으면 weight=0 → vision top-1 만 반영.
    canonical = _canonical(
        _outfit(upper="anarkali", fabric="georgette", technique="thread_embroidery"),
    )
    item = EnrichedContentItem(
        normalized=_normalized(),
        garment_type=GarmentType.KURTA_SET,
        fabric=Fabric.COTTON,
        technique=Technique.SOLID,
        canonicals=[canonical],
        classification_method_per_attribute={},  # method 없음 → text weight=0.
        trend_cluster_key="kurta_set__solid__cotton",
    )
    new_key = _vision_reassign_cluster_key(item)
    assert new_key == build_exact_key(
        GarmentType.ANARKALI, Technique.THREAD_EMBROIDERY, Fabric.GEORGETTE
    )


def test_tiebreak_is_deterministic_by_value_asc() -> None:
    # 두 canonical 동률 (같은 area, 같은 n_objects) → garment_type 두 키 share 동일.
    # tiebreak: (-pct, value asc) → "anarkali" < "kurta_set" 로 anarkali 선택.
    canonical_a = _canonical(
        _outfit(
            upper="kurta_set",
            fabric="cotton",
            technique="block_print",
            area_ratio=0.3,
            bbox=(0.0, 0.0, 0.5, 0.5),
        ),
    )
    canonical_b = _canonical(
        _outfit(
            upper="anarkali",
            fabric="cotton",
            technique="block_print",
            area_ratio=0.3,
            bbox=(0.5, 0.5, 0.5, 0.5),
        ),
    )
    item = EnrichedContentItem(
        normalized=_normalized(),
        canonicals=[canonical_a, canonical_b],
    )
    new_key = _vision_reassign_cluster_key(item)
    # garment_type tiebreak → anarkali. fabric/technique 모두 단일 cotton/block_print.
    assert new_key == build_exact_key(
        GarmentType.ANARKALI, Technique.BLOCK_PRINT, Fabric.COTTON
    )


def test_returns_none_when_existing_is_none_and_canonicals_empty() -> None:
    item = EnrichedContentItem(
        normalized=_normalized(),
        canonicals=[],
        trend_cluster_key=None,
    )
    assert _vision_reassign_cluster_key(item) is None
