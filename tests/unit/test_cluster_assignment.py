from __future__ import annotations

from clustering.assign_trend_cluster import UNCLASSIFIED, assign_cluster
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
