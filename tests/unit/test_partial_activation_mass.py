"""partial(g) 활성화 (2026-04-28) — N=1/2 emit + multiplier_ratio mass invariant pinning.

검증 포인트:
- assign_shares N=1 → 1 partial cluster (g__unknown__unknown), share=0.2
- assign_shares N=2 → cross-product unknown placeholder, share × 0.5
- _item_contributions N=1 → multiplier=1.0, contribution = share × 1.0
- _item_contributions N=2 → multiplier=2.5, contribution = share × 2.5
- per-item mass invariant: Σ assign_shares = multiplier_ratio (N=3=1.0 / N=2=0.5 / N=1=0.2 / N=0=0)
- assign_shares 와 _item_contributions 의 cluster_key 포맷 정합 (`unknown` placeholder 동일)
- effective_item_count 와 Σ assign_shares 단위 정합 (β1 ↔ β2/β3 align, γ 마이그 전 raw scale)
"""
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

# --------------------------------------------------------------------------- #
# assign_shares — N=1 / N=2 / N=3 mass
# --------------------------------------------------------------------------- #


def test_assign_shares_n1_garment_only() -> None:
    """N=1 (G만) → 1 partial cluster (kurta__unknown__unknown), share=0.2."""
    shares = assign_shares({"kurta": 1.0}, {}, {})
    assert shares == {"kurta__unknown__unknown": pytest.approx(0.2)}


def test_assign_shares_n1_technique_only() -> None:
    """N=1 (T만) → unknown__chikankari__unknown, share=0.2."""
    shares = assign_shares({}, {"chikankari": 1.0}, {})
    assert shares == {"unknown__chikankari__unknown": pytest.approx(0.2)}


def test_assign_shares_n2_garment_fabric_unknown_technique() -> None:
    """N=2 (G/F만) → kurta__unknown__cotton, share=0.5."""
    shares = assign_shares({"kurta": 1.0}, {}, {"cotton": 1.0})
    assert shares == {"kurta__unknown__cotton": pytest.approx(0.5)}


def test_assign_shares_n2_with_distribution_split() -> None:
    """N=2 + 분포 split → 두 cluster, 각 share × multiplier_ratio."""
    shares = assign_shares({"kurta": 0.7, "saree": 0.3}, {}, {"cotton": 1.0})
    assert shares == {
        "kurta__unknown__cotton": pytest.approx(0.7 * 0.5),
        "saree__unknown__cotton": pytest.approx(0.3 * 0.5),
    }
    assert sum(shares.values()) == pytest.approx(0.5)  # multiplier_ratio


def test_assign_shares_n3_full_mass_unchanged() -> None:
    """N=3 → multiplier_ratio=1.0 → 기존 raw share 동작 그대로."""
    shares = assign_shares({"kurta": 1.0}, {"chikankari": 1.0}, {"cotton": 1.0})
    assert shares == {"kurta__chikankari__cotton": pytest.approx(1.0)}


def test_assign_shares_n0_returns_empty() -> None:
    """N=0 → 빈 dict (representative 후보 아님)."""
    assert assign_shares({}, {}, {}) == {}


# --------------------------------------------------------------------------- #
# Per-item mass invariant
# --------------------------------------------------------------------------- #


def test_assign_shares_mass_n3_equals_one() -> None:
    """per-item mass invariant: N=3 → Σ shares = 1.0."""
    shares = assign_shares(
        {"kurta": 0.7, "saree": 0.3},
        {"block_print": 1.0},
        {"cotton": 0.6, "silk": 0.4},
    )
    assert sum(shares.values()) == pytest.approx(1.0)


def test_assign_shares_mass_n2_equals_half() -> None:
    """per-item mass invariant: N=2 → Σ shares = 0.5."""
    shares = assign_shares({"kurta": 0.7, "saree": 0.3}, {}, {"cotton": 1.0})
    assert sum(shares.values()) == pytest.approx(0.5)


def test_assign_shares_mass_n1_equals_one_fifth() -> None:
    """per-item mass invariant: N=1 → Σ shares = 0.2."""
    shares = assign_shares({"kurta": 0.7, "saree": 0.3}, {}, {})
    assert sum(shares.values()) == pytest.approx(0.2)


# --------------------------------------------------------------------------- #
# _item_contributions (build_contributions) — multiplier 비례
# --------------------------------------------------------------------------- #


def test_build_contributions_n1_multiplier_1() -> None:
    """N=1 → multiplier=1.0. contribution = share × 1.0."""
    item = ItemDistribution(
        item_id="p1",
        source=ContentSource.INSTAGRAM,
        garment_type={"kurta": 1.0},
        technique={},
        fabric={},
    )
    [c] = build_contributions([item])
    assert c.representative_key == "kurta__unknown__unknown"
    assert c.match_share == pytest.approx(1.0)
    assert c.multiplier == pytest.approx(1.0)
    assert c.contribution == pytest.approx(1.0)


def test_build_contributions_n2_multiplier_2_5() -> None:
    """N=2 → multiplier=2.5. contribution = share × 2.5."""
    item = ItemDistribution(
        item_id="p1",
        source=ContentSource.INSTAGRAM,
        garment_type={"kurta": 1.0},
        technique={},
        fabric={"cotton": 1.0},
    )
    [c] = build_contributions([item])
    assert c.representative_key == "kurta__unknown__cotton"
    assert c.match_share == pytest.approx(1.0)
    assert c.multiplier == pytest.approx(2.5)
    assert c.contribution == pytest.approx(2.5)


def test_build_contributions_n3_multiplier_5() -> None:
    """N=3 → multiplier=5.0 (기존 동작 그대로)."""
    item = ItemDistribution(
        item_id="p1",
        source=ContentSource.INSTAGRAM,
        garment_type={"kurta": 1.0},
        technique={"chikankari": 1.0},
        fabric={"cotton": 1.0},
    )
    [c] = build_contributions([item])
    assert c.multiplier == pytest.approx(5.0)
    assert c.contribution == pytest.approx(5.0)


# --------------------------------------------------------------------------- #
# β1 effective_item_count ↔ β2/β3 assign_shares mass align
# --------------------------------------------------------------------------- #


def test_effective_item_count_aligns_with_assign_shares_mass() -> None:
    """β1 effective_item_count(items) = Σ multiplier_ratio per item.
    Σ assign_shares(item) per item = multiplier_ratio (per-item mass).
    → 두 값이 정확히 같아야 단위 정합 (γ 에서 minmax view 분자/분모 align 가능)."""
    items = [
        # N=3 → 1.0
        ItemDistribution(
            item_id="p1", source=ContentSource.INSTAGRAM,
            garment_type={"kurta": 1.0}, technique={"chikankari": 1.0}, fabric={"cotton": 1.0},
        ),
        # N=2 → 0.5
        ItemDistribution(
            item_id="p2", source=ContentSource.INSTAGRAM,
            garment_type={"kurta": 1.0}, technique={}, fabric={"cotton": 1.0},
        ),
        # N=1 → 0.2
        ItemDistribution(
            item_id="p3", source=ContentSource.INSTAGRAM,
            garment_type={"kurta": 1.0}, technique={}, fabric={},
        ),
        # N=0 → 0.0
        ItemDistribution(
            item_id="p4", source=ContentSource.INSTAGRAM,
            garment_type={}, technique={}, fabric={},
        ),
    ]
    eic = effective_item_count(items)
    mass_via_shares = sum(sum(item_cluster_shares(it).values()) for it in items)
    assert eic == pytest.approx(1.7)  # 1.0 + 0.5 + 0.2 + 0
    assert mass_via_shares == pytest.approx(eic)


# --------------------------------------------------------------------------- #
# Cluster key format consistency
# --------------------------------------------------------------------------- #


def test_assign_shares_and_build_contributions_use_same_cluster_key() -> None:
    """assign_shares cluster_key 와 _item_contributions representative_key 가 같은 포맷.
    같은 input 으로 동일 cluster_key 가 양쪽에서 emit되어야 fan-out 정합 (β2 ↔ β3)."""
    item = ItemDistribution(
        item_id="p1",
        source=ContentSource.INSTAGRAM,
        garment_type={"kurta": 1.0},
        technique={},
        fabric={"cotton": 1.0},
    )
    shares_keys = set(assign_shares(item.garment_type, item.technique, item.fabric).keys())
    contrib_keys = {c.representative_key for c in build_contributions([item])}
    assert shares_keys == contrib_keys
