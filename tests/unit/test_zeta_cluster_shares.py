"""ζ (2026-04-28) pinning — trend_cluster_shares contract + Case2 share-based picking.

ζ scope:
- contract `trend_cluster_shares: dict[str, float]` 추가 + read-cast validator (기존
  enriched JSON 의 trend_cluster_key 만 → {key: 1.0} 1-entry dict 자동 채움).
- `_case2_targets` 가 trend_cluster_shares.items() 순회 — share≥min_share 인 모든
  cluster 에 picking 후보 등록 (winner-only collapse 해소 ★).

검증:
1. read-cast: legacy enriched JSON 호환.
2. multi-cluster picking: 한 item 이 X+Y 둘 다 후보 등장.
3. min_share threshold drop: 작은 share cluster 는 picking 후보 제외.
4. UNCLASSIFIED skip + IG-only filter 유지.
"""
from __future__ import annotations

from datetime import datetime

from clustering.assign_trend_cluster import UNCLASSIFIED
from contracts.common import ContentSource, GarmentType
from contracts.enriched import EnrichedContentItem
from contracts.normalized import NormalizedContentItem
from pipelines.run_daily_pipeline import _case2_targets


def _normalized(post_id: str, *, source: ContentSource = ContentSource.INSTAGRAM,
                engagement: int = 0) -> NormalizedContentItem:
    return NormalizedContentItem(
        source=source,
        source_post_id=post_id,
        text_blob="",
        hashtags=[],
        image_urls=[],
        post_date=datetime(2026, 4, 28),
        engagement_raw=engagement,
    )


# --------------------------------------------------------------------------- #
# read-cast (contract level)
# --------------------------------------------------------------------------- #

def test_read_cast_legacy_key_only_backfills_single_entry_shares() -> None:
    """기존 enriched JSON 의 trend_cluster_key 만 있는 raw dict → shares = {key: 1.0}."""
    raw = {
        "normalized": _normalized("p_legacy").model_dump(),
        "trend_cluster_key": "kurta_set__solid__cotton",
    }
    item = EnrichedContentItem.model_validate(raw)
    assert item.trend_cluster_key == "kurta_set__solid__cotton"
    assert item.trend_cluster_shares == {"kurta_set__solid__cotton": 1.0}


def test_read_cast_explicit_shares_take_precedence_over_key() -> None:
    """write 측이 명시한 multi-entry shares 는 read-cast 가 덮지 않음."""
    raw = {
        "normalized": _normalized("p_explicit").model_dump(),
        "trend_cluster_key": "kurta_set__solid__cotton",
        "trend_cluster_shares": {
            "kurta_set__solid__cotton": 0.6,
            "anarkali__solid__cotton": 0.4,
        },
    }
    item = EnrichedContentItem.model_validate(raw)
    assert item.trend_cluster_shares == {
        "kurta_set__solid__cotton": 0.6,
        "anarkali__solid__cotton": 0.4,
    }


def test_read_cast_no_key_no_shares_keeps_empty() -> None:
    """key/shares 둘 다 None → shares = 빈 dict (read-cast 작동 안 함)."""
    raw = {
        "normalized": _normalized("p_unclass").model_dump(),
    }
    item = EnrichedContentItem.model_validate(raw)
    assert item.trend_cluster_key is None
    assert item.trend_cluster_shares == {}


# --------------------------------------------------------------------------- #
# _case2_targets — share-based picking
# --------------------------------------------------------------------------- #

def _ig_item(post_id: str, shares: dict[str, float], engagement: int = 100) -> EnrichedContentItem:
    """IG item with multi-cluster shares."""
    winner = max(shares.items(), key=lambda kv: (-kv[1], kv[0]))[0] if shares else None
    return EnrichedContentItem(
        normalized=_normalized(post_id, engagement=engagement),
        trend_cluster_key=winner,
        trend_cluster_shares=shares,
    )


def test_multi_cluster_picking_item_appears_in_all_above_threshold() -> None:
    """한 item 이 X(0.6)+Y(0.3)+Z(0.05) fan-out → min_share=0.10 일 때 X+Y 두 cluster 후보 등장.

    Z(0.05) 는 threshold drop. X+Y 는 picking 후보 → 각 cluster 에서 상위 cap 안에 들어감.
    이게 ζ 의 본 목적: winner-only 가 아니라 fan-out cluster 모두 picking 후보.
    """
    item_a = _ig_item("p_a", {"X": 0.6, "Y": 0.3, "Z": 0.05}, engagement=100)
    item_b_x = _ig_item("p_b_x", {"X": 1.0}, engagement=50)
    item_b_y = _ig_item("p_b_y", {"Y": 1.0}, engagement=40)

    picks = _case2_targets(
        [item_a, item_b_x, item_b_y], cap_per_cluster=10, min_share=0.10
    )
    pick_ids = [p.normalized.source_post_id for p in picks]
    # item_a 가 X cluster + Y cluster 양쪽 picking 후보 → 결과에 2번 등장.
    assert pick_ids.count("p_a") == 2
    # Z cluster 후보는 0 (drop).
    # X cluster: [p_a, p_b_x], Y cluster: [p_a, p_b_y] = 총 4 picks.
    assert sorted(pick_ids) == ["p_a", "p_a", "p_b_x", "p_b_y"]


def test_min_share_threshold_drops_long_tail() -> None:
    """share < min_share cluster 는 picking 후보 제외 (노이즈 차단)."""
    item = _ig_item("p_long_tail", {"X": 0.6, "Y": 0.05, "Z": 0.04}, engagement=100)
    picks = _case2_targets([item], cap_per_cluster=10, min_share=0.10)
    # X(0.6) 만 통과, Y/Z drop → 1 pick.
    assert len(picks) == 1
    assert picks[0].normalized.source_post_id == "p_long_tail"


def test_min_share_zero_keeps_all_share_above_zero() -> None:
    """min_share=0.0 → share>0 인 모든 cluster 살림 (cap 이 자연 cutoff)."""
    item = _ig_item("p_zero_thr", {"X": 0.5, "Y": 0.05, "Z": 0.001}, engagement=100)
    picks = _case2_targets([item], cap_per_cluster=10, min_share=0.0)
    # 3 cluster 모두 picking 후보 → 3 picks.
    assert len(picks) == 3


def test_unclassified_cluster_is_skipped() -> None:
    """UNCLASSIFIED cluster_key 는 picking 후보에서 제외 (기존 동작 유지)."""
    item = _ig_item(
        "p_unclass", {UNCLASSIFIED: 1.0, "X": 0.3}, engagement=100
    )
    picks = _case2_targets([item], cap_per_cluster=10, min_share=0.10)
    # UNCLASSIFIED skip, X(0.3) 만 picking → 1 pick.
    assert len(picks) == 1
    pick_clusters = [p for p in picks]
    # picking 후보로 등록된 cluster 는 X.
    assert pick_clusters[0].normalized.source_post_id == "p_unclass"


def test_youtube_items_excluded_from_case2() -> None:
    """Case2 = IG only. YouTube item 은 multi-cluster shares 가 있어도 picking 안 됨."""
    yt_item = EnrichedContentItem(
        normalized=_normalized("p_yt", source=ContentSource.YOUTUBE, engagement=999),
        garment_type=GarmentType.KURTA_SET,
        trend_cluster_key="X",
        trend_cluster_shares={"X": 0.6, "Y": 0.4},
    )
    picks = _case2_targets([yt_item], cap_per_cluster=10, min_share=0.10)
    assert picks == []


def test_cap_per_cluster_caps_engagement_top_k() -> None:
    """cap_per_cluster=2 → 각 cluster 의 top engagement 2 개만 picking."""
    items = [
        _ig_item(f"p_{i}", {"X": 1.0}, engagement=i * 10) for i in range(5)
    ]
    picks = _case2_targets(items, cap_per_cluster=2, min_share=0.10)
    # cap=2 → engagement 상위 2 = p_4, p_3.
    assert len(picks) == 2
    assert {p.normalized.source_post_id for p in picks} == {"p_4", "p_3"}
