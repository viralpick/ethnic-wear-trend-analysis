"""_vision_reassign_cluster_shares + _winner_key_from_shares pinning — ζ + 갭 #3 (B).

ζ (2026-04-28): 기존 winner-only `_vision_reassign_cluster_key` 함수가 share dict 기반
2-step (shares 재계산 → winner derive) 으로 분리됐다. shares 는 score path / summary path
의 fan-out 입력 (β2/β4 와 정합), trend_cluster_key 는 max-share derived 대표값.

검증 axes:
- canonicals 비어있으면 shares 보존 (text-level 결정).
- vision dist 의 G/T/F 한 axis 라도 비면 (N<3) shares 보존 (picking 손실 방지 — assign_shares
  의 multiplier_ratio 0.5/0.2 가 case2_picking_min_share=0.10 cutoff 에 걸리는 문제).
- text-level partial → vision N=3 으로 cross-product fan-out (winner-only collapse 해소 ★).
- shares 의 max-share key 가 winner_key. 동률 시 (-pct, value asc) 결정론적 tiebreak.
- 빈 shares dict → winner_key = None.
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
from pipelines.run_daily_pipeline import (
    _vision_reassign_cluster_shares,
    _winner_key_from_shares,
)


def _normalized(post_id: str = "p1") -> NormalizedContentItem:
    return NormalizedContentItem(
        source=ContentSource.INSTAGRAM,
        source_post_id=post_id,
        text_blob="",
        hashtags=[],
        image_urls=[],
        post_date=datetime(2026, 4, 27),
        engagement_raw_count=0,
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


def _reassigned_key(item: EnrichedContentItem) -> str | None:
    """test 편의 — 2-step 합성. 기존 _vision_reassign_cluster_key 와 동일 시그니처."""
    return _winner_key_from_shares(_vision_reassign_cluster_shares(item))


def test_no_canonicals_keeps_existing_shares() -> None:
    """canonicals 비어있음 → shares 보존 (text-level 결정 그대로)."""
    item = EnrichedContentItem(
        normalized=_normalized(),
        garment_type=GarmentType.KURTA_SET,
        canonicals=[],
        trend_cluster_key="kurta_set__unknown",
        trend_cluster_shares={"kurta_set__unknown": 1.0},
    )
    new_shares = _vision_reassign_cluster_shares(item)
    assert new_shares == {"kurta_set__unknown": 1.0}
    assert _winner_key_from_shares(new_shares) == "kurta_set__unknown"


def test_partial_text_promoted_to_exact_after_vision() -> None:
    # text: garment_type=KURTA_SET (rule), 나머지 None → partial.
    # vision: canonical 1 개로 G/T/F 모두 enum 매핑 가능한 값 → N=3 cross-product fan-out.
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
        trend_cluster_key="kurta_set__unknown",
        trend_cluster_shares={"kurta_set__unknown": 1.0},
    )
    new_shares = _vision_reassign_cluster_shares(item)
    expected_key = build_exact_key(GarmentType.KURTA_SET, Fabric.COTTON)
    # N=3 cross-product 결과: G/T/F 모두 enum 매핑 → 단일 cluster_key (분포가 collapsed)
    # → shares = {expected_key: 1.0}. multiplier_ratio = 1.0 (N=3).
    assert new_shares == {expected_key: 1.0}
    assert _winner_key_from_shares(new_shares) == expected_key


def test_vision_axis_missing_keeps_shares() -> None:
    """canonical 의 fabric=None → fabric distribution 비어 N<3 → shares 보존."""
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
        trend_cluster_key="kurta_set__unknown",
        trend_cluster_shares={"kurta_set__unknown": 1.0},
    )
    new_shares = _vision_reassign_cluster_shares(item)
    # N<3 → shares 보존 (picking 손실 방지 — multiplier_ratio 0.5/0.2 곱하면 0.10 cutoff
    # 에 걸려 picking 후보 0개 됨).
    assert new_shares == {"kurta_set__unknown": 1.0}
    assert _winner_key_from_shares(new_shares) == "kurta_set__unknown"


def test_vision_only_text_rule_ignored_for_garment_fabric() -> None:
    """Phase 3.2 (2026-04-30): G/F/T 의 text rule 폐지 — vision 만. text=KURTA_SET 라도
    vision (canonical.upper=straight_kurta) 가 winner. 옛 동작 (text rule weight 우세)
    회귀 방지."""
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
        garment_type=GarmentType.KURTA_SET,  # text 가 KURTA_SET 매핑
        fabric=Fabric.COTTON,
        technique=Technique.BLOCK_PRINT,
        canonicals=[canonical],              # vision = straight_kurta
        classification_method_per_attribute={
            "garment_type": ClassificationMethod.RULE,
            "fabric": ClassificationMethod.RULE,
            "technique": ClassificationMethod.RULE,
        },
        trend_cluster_key=build_exact_key(GarmentType.KURTA_SET, Fabric.COTTON),
        trend_cluster_shares={
            build_exact_key(GarmentType.KURTA_SET, Fabric.COTTON): 1.0
        },
    )
    # vision (straight_kurta) 가 정확한 결과. text KURTA_SET 무시.
    expected_winner = build_exact_key(GarmentType.STRAIGHT_KURTA, Fabric.COTTON)
    assert _reassigned_key(item) == expected_winner


def test_vision_overrides_when_text_method_missing() -> None:
    # text_method 없으면 weight=0 → vision top-1 만 winner.
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
        trend_cluster_key="kurta_set__cotton",
        trend_cluster_shares={"kurta_set__cotton": 1.0},
    )
    expected_winner = build_exact_key(GarmentType.ANARKALI, Fabric.GEORGETTE)
    assert _reassigned_key(item) == expected_winner


def test_tiebreak_is_deterministic_by_value_asc() -> None:
    # 두 canonical 동률 → garment_type 두 키 share 동일.
    # ζ winner tiebreak: (-share, key_str asc) → "anarkali__..." < "kurta_set__..." 로
    # anarkali key 가 winner.
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
    new_shares = _vision_reassign_cluster_shares(item)
    # cross-product: 2 garment × 1 fabric = 2 cluster, share=0.5 each.
    anarkali_key = build_exact_key(GarmentType.ANARKALI, Fabric.COTTON)
    kurta_key = build_exact_key(GarmentType.KURTA_SET, Fabric.COTTON)
    assert set(new_shares.keys()) == {anarkali_key, kurta_key}
    # winner tiebreak: alpha asc → anarkali (a < k).
    assert _winner_key_from_shares(new_shares) == anarkali_key


def test_winner_key_returns_none_when_shares_empty() -> None:
    """canonicals 빈데 trend_cluster_shares 도 빈 dict (text 단계도 unclassified) → winner=None."""
    item = EnrichedContentItem(
        normalized=_normalized(),
        canonicals=[],
        trend_cluster_key=None,
        trend_cluster_shares={},
    )
    new_shares = _vision_reassign_cluster_shares(item)
    assert new_shares == {}
    assert _winner_key_from_shares(new_shares) is None
