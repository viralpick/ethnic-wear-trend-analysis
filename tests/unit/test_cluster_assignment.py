from __future__ import annotations

import pytest

from clustering.assign_trend_cluster import UNCLASSIFIED, assign_cluster, assign_shares
from contracts.common import Fabric, GarmentType


def test_exact_match_when_both_axes_resolved() -> None:
    key = assign_cluster(GarmentType.KURTA_SET, Fabric.COTTON, {})

    assert key == "kurta_set__cotton"


def test_partial_match_picks_highest_post_count_candidate() -> None:
    key = assign_cluster(
        GarmentType.KURTA_SET,
        None,
        {
            "kurta_set__cotton": 5,
            "kurta_set__linen": 12,
            "co_ord__linen": 30,
        },
    )

    assert key == "kurta_set__linen"


def test_all_null_routes_to_unclassified() -> None:
    key = assign_cluster(None, None, {})

    assert key == UNCLASSIFIED


def test_tie_break_picks_lexicographically_smallest_on_equal_count() -> None:
    key = assign_cluster(
        GarmentType.KURTA_SET,
        None,
        {
            "kurta_set__linen": 5,
            "kurta_set__cotton": 5,
        },
    )

    assert key == "kurta_set__cotton"


def test_partial_with_no_history_builds_unknown_placeholder_key() -> None:
    key = assign_cluster(GarmentType.KURTA_SET, None, {})

    assert key == "kurta_set__unknown"


def test_assign_shares_cross_product_matches_spec_example() -> None:
    shares = assign_shares(
        {"kurta": 0.7, "saree": 0.3},
        {"cotton": 0.6, "silk": 0.4},
    )

    assert shares == {
        "kurta__cotton": 0.7 * 0.6,
        "kurta__silk": 0.7 * 0.4,
        "saree__cotton": 0.3 * 0.6,
        "saree__silk": 0.3 * 0.4,
    }
    assert abs(sum(shares.values()) - 1.0) < 1e-9


def test_assign_shares_partial_emits_with_unknown_axis() -> None:
    assert assign_shares({}, {"b": 1.0}) == {"unknown__b": pytest.approx(0.5)}
    assert assign_shares({"a": 1.0}, {}) == {"a__unknown": pytest.approx(0.5)}
    assert assign_shares({}, {}) == {}


def test_assign_shares_drops_zero_share_entries() -> None:
    shares = assign_shares({"kurta": 1.0, "saree": 0.0}, {"cotton": 1.0})

    assert shares == {"kurta__cotton": 1.0}


def test_assign_shares_min_share_threshold_drops_below() -> None:
    shares = assign_shares(
        {"kurta": 0.7, "saree": 0.3},
        {"cotton": 0.6, "silk": 0.4},
        min_share=0.2,
    )

    assert set(shares.keys()) == {
        "kurta__cotton",
        "kurta__silk",
    }


def test_assign_shares_single_entry_per_axis_returns_share_1() -> None:
    shares = assign_shares({"kurta": 1.0}, {"cotton": 1.0})

    assert shares == {"kurta__cotton": 1.0}
