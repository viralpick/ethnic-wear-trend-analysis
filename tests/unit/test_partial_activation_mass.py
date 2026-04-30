"""partial(G/F) 활성화 (2026-04-30) — N=1/2 emit + mass invariant pinning."""
from __future__ import annotations

import pytest

from aggregation.representative_builder import (
    ItemDistribution,
    build_contributions,
    effective_item_count,
    item_cluster_shares,
)
from clustering.assign_trend_cluster import assign_shares
from contracts.common import ContentSource


def test_assign_shares_n1_garment_only() -> None:
    shares = assign_shares({"kurta": 1.0}, {})
    assert shares == {"kurta__unknown": pytest.approx(0.5)}


def test_assign_shares_n1_fabric_only() -> None:
    shares = assign_shares({}, {"cotton": 1.0})
    assert shares == {"unknown__cotton": pytest.approx(0.5)}


def test_assign_shares_n2_garment_fabric() -> None:
    shares = assign_shares({"kurta": 1.0}, {"cotton": 1.0})
    assert shares == {"kurta__cotton": pytest.approx(1.0)}


def test_assign_shares_n2_with_distribution_split() -> None:
    shares = assign_shares({"kurta": 0.7, "saree": 0.3}, {"cotton": 1.0})
    assert shares == {
        "kurta__cotton": pytest.approx(0.7),
        "saree__cotton": pytest.approx(0.3),
    }
    assert sum(shares.values()) == pytest.approx(1.0)


def test_assign_shares_n0_returns_empty() -> None:
    assert assign_shares({}, {}) == {}


def test_assign_shares_mass_n2_equals_one() -> None:
    shares = assign_shares(
        {"kurta": 0.7, "saree": 0.3},
        {"cotton": 0.6, "silk": 0.4},
    )
    assert sum(shares.values()) == pytest.approx(1.0)


def test_assign_shares_mass_n1_equals_half() -> None:
    shares = assign_shares({"kurta": 0.7, "saree": 0.3}, {})
    assert sum(shares.values()) == pytest.approx(0.5)


def test_build_contributions_n1_multiplier_1() -> None:
    item = ItemDistribution(
        item_id="p1",
        source=ContentSource.INSTAGRAM,
        garment_type={"kurta": 1.0},
        fabric={},
    )
    [c] = build_contributions([item])
    assert c.representative_key == "kurta__unknown"
    assert c.match_share == pytest.approx(1.0)
    assert c.multiplier == pytest.approx(1.0)
    assert c.contribution == pytest.approx(1.0)


def test_build_contributions_n2_multiplier_2_5() -> None:
    item = ItemDistribution(
        item_id="p1",
        source=ContentSource.INSTAGRAM,
        garment_type={"kurta": 1.0},
        fabric={"cotton": 1.0},
    )
    [c] = build_contributions([item])
    assert c.representative_key == "kurta__cotton"
    assert c.match_share == pytest.approx(1.0)
    assert c.multiplier == pytest.approx(2.5)
    assert c.contribution == pytest.approx(2.5)


def test_effective_item_count_uses_normalized_n2_denominator() -> None:
    items = [
        ItemDistribution(
            item_id="p1",
            source=ContentSource.INSTAGRAM,
            garment_type={"kurta": 1.0},
            fabric={"cotton": 1.0},
        ),
        ItemDistribution(
            item_id="p2",
            source=ContentSource.INSTAGRAM,
            garment_type={"kurta": 1.0},
            fabric={},
        ),
        ItemDistribution(
            item_id="p3",
            source=ContentSource.INSTAGRAM,
            garment_type={},
            fabric={},
        ),
    ]
    eic = effective_item_count(items)
    mass_via_shares = sum(sum(item_cluster_shares(it).values()) for it in items)
    assert eic == pytest.approx(1.4)
    assert mass_via_shares == pytest.approx(1.5)


def test_assign_shares_and_build_contributions_use_same_cluster_key() -> None:
    item = ItemDistribution(
        item_id="p1",
        source=ContentSource.INSTAGRAM,
        garment_type={"kurta": 1.0},
        fabric={"cotton": 1.0},
    )
    shares_keys = set(assign_shares(item.garment_type, item.fabric).keys())
    contrib_keys = {c.representative_key for c in build_contributions([item])}
    assert shares_keys == contrib_keys
