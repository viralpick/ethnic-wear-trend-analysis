from __future__ import annotations

from clustering.assign_trend_cluster import UNCLASSIFIED, assign_cluster, assign_shares
from contracts.common import Fabric, GarmentType, Technique


def test_exact_match_when_all_three_resolved() -> None:
    key = assign_cluster(
        GarmentType.KURTA_SET, Technique.CHIKANKARI, Fabric.COTTON,
        cluster_totals={},
    )

    assert key == "kurta_set__chikankari__cotton"


def test_partial_match_picks_highest_post_count_candidate() -> None:
    # garment + fabric 고정, technique 이 null. 기존 정확 키들 중 매칭되는 것만 후보.
    key = assign_cluster(
        GarmentType.KURTA_SET, None, Fabric.COTTON,
        cluster_totals={
            "kurta_set__chikankari__cotton": 5,
            "kurta_set__block_print__cotton": 12,
            "co_ord__block_print__linen": 30,  # 다른 garment_type — 후보 아님
        },
    )

    assert key == "kurta_set__block_print__cotton"


def test_all_null_routes_to_unclassified() -> None:
    key = assign_cluster(None, None, None, cluster_totals={})

    assert key == UNCLASSIFIED


def test_tie_break_picks_lexicographically_smallest_on_equal_count() -> None:
    # 두 후보 모두 count=5. "block_print" < "chikankari" 이므로 block_print 가 승자.
    key = assign_cluster(
        GarmentType.KURTA_SET, None, Fabric.COTTON,
        cluster_totals={
            "kurta_set__chikankari__cotton": 5,
            "kurta_set__block_print__cotton": 5,
        },
    )

    assert key == "kurta_set__block_print__cotton"


def test_partial_with_no_history_builds_unknown_placeholder_key() -> None:
    # 첫 런 (cluster_totals 비어 있음) + 부분 매칭 케이스.
    key = assign_cluster(
        GarmentType.KURTA_SET, None, None,
        cluster_totals={},
    )

    assert key == "kurta_set__unknown__unknown"


# --------------------------------------------------------------------------- #
# Phase α (2026-04-28) — share-weighted assign (N:N path)
# --------------------------------------------------------------------------- #


def test_assign_shares_cross_product_matches_spec_example() -> None:
    # pipeline_spec §2.4 예시 — 0.42 / 0.28 / 0.18 / 0.12 4 keys.
    shares = assign_shares(
        garment_dist={"kurta": 0.7, "saree": 0.3},
        technique_dist={"block_print": 1.0},
        fabric_dist={"cotton": 0.6, "silk": 0.4},
    )

    assert shares == {
        "kurta__block_print__cotton": 0.7 * 1.0 * 0.6,
        "kurta__block_print__silk":   0.7 * 1.0 * 0.4,
        "saree__block_print__cotton": 0.3 * 1.0 * 0.6,
        "saree__block_print__silk":   0.3 * 1.0 * 0.4,
    }
    # input 분포 합 = 1.0 → 결과 share 합 = 1.0 invariant.
    assert abs(sum(shares.values()) - 1.0) < 1e-9


def test_assign_shares_empty_when_any_distribution_missing() -> None:
    # G/T/F 한 축이라도 비면 N<3 → 빈 dict (현 phase 정책).
    assert assign_shares({}, {"a": 1.0}, {"b": 1.0}) == {}
    assert assign_shares({"a": 1.0}, {}, {"b": 1.0}) == {}
    assert assign_shares({"a": 1.0}, {"b": 1.0}, {}) == {}


def test_assign_shares_drops_zero_share_entries() -> None:
    # share=0 인 cross-product 항은 dict 에 안 들어감.
    shares = assign_shares(
        garment_dist={"kurta": 1.0, "saree": 0.0},
        technique_dist={"block_print": 1.0},
        fabric_dist={"cotton": 1.0},
    )

    assert shares == {"kurta__block_print__cotton": 1.0}


def test_assign_shares_min_share_threshold_drops_below() -> None:
    # min_share=0.2 → 0.18 / 0.12 케이스 drop.
    shares = assign_shares(
        garment_dist={"kurta": 0.7, "saree": 0.3},
        technique_dist={"block_print": 1.0},
        fabric_dist={"cotton": 0.6, "silk": 0.4},
        min_share=0.2,
    )

    assert set(shares.keys()) == {
        "kurta__block_print__cotton",
        "kurta__block_print__silk",
    }


def test_assign_shares_single_entry_per_axis_returns_share_1() -> None:
    # exact-key path 와 동치: 각 축 1 value → 1 cluster_key, share = 1.0.
    shares = assign_shares(
        garment_dist={"kurta": 1.0},
        technique_dist={"chikankari": 1.0},
        fabric_dist={"cotton": 1.0},
    )

    assert shares == {"kurta__chikankari__cotton": 1.0}
