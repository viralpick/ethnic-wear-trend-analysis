"""representative_builder pinning — pipeline_spec_v1.0 §2.4.

검증 대상:
- multiplier 테이블: N=1 → 1.0 / N=2 → 2.5 / N=3 → 5.0 / 그 외 → 0.0.
- representative_key 포맷 = "g__t__f".
- cross-product: 한 attr 이라도 비면 contribution 없음 (spec §C.2).
- contribution = share × multiplier × item_base_unit.
- sparse filter: total_item_contribution > 0 만 emit.
- factor_contribution: source 별 비율, 합=1.0, 모든 등록 source 키 존재 (없으면 0.0).
"""
from __future__ import annotations

import pytest

from aggregation.representative_builder import (
    ItemDistribution,
    RepresentativeContribution,
    aggregate_representatives,
    build_contributions,
    effective_item_count,
    item_cluster_shares,
    multiplier_for_n,
    representative_key,
    top_evidence_per_source,
)
from contracts.common import ContentSource


def test_multiplier_table() -> None:
    assert multiplier_for_n(1) == 1.0
    assert multiplier_for_n(2) == 2.5
    assert multiplier_for_n(3) == 5.0
    assert multiplier_for_n(0) == 0.0
    assert multiplier_for_n(4) == 0.0


def test_representative_key_format() -> None:
    assert representative_key("kurta", "block_print", "cotton") == "kurta__block_print__cotton"


def test_build_contributions_cross_product() -> None:
    item = ItemDistribution(
        item_id="ig_001",
        source=ContentSource.INSTAGRAM,
        garment_type={"kurta": 0.7, "saree": 0.3},
        technique={"block_print": 1.0},
        fabric={"cotton": 0.6, "silk": 0.4},
    )
    contribs = build_contributions([item])
    assert len(contribs) == 4  # 2 × 1 × 2
    keys = sorted(c.representative_key for c in contribs)
    assert keys == [
        "kurta__block_print__cotton",
        "kurta__block_print__silk",
        "saree__block_print__cotton",
        "saree__block_print__silk",
    ]
    # share 검증: kurta + block_print + cotton = 0.7 × 1.0 × 0.6 = 0.42
    by_key = {c.representative_key: c for c in contribs}
    kbc = by_key["kurta__block_print__cotton"]
    assert kbc.match_share == pytest.approx(0.42)
    # multiplier=5.0 (N=3, all decided) → contribution = 0.42 × 5.0 = 2.1
    assert kbc.contribution == pytest.approx(2.1)


def test_partial_distribution_drops_item() -> None:
    # technique 비어있으면 N<3, contribution 0개.
    item = ItemDistribution(
        item_id="ig_002",
        source=ContentSource.INSTAGRAM,
        garment_type={"kurta": 1.0},
        technique={},
        fabric={"cotton": 1.0},
    )
    assert build_contributions([item]) == []


def test_aggregate_factor_contribution_single_source() -> None:
    item = ItemDistribution(
        item_id="ig_001",
        source=ContentSource.INSTAGRAM,
        garment_type={"kurta": 1.0},
        technique={"block_print": 1.0},
        fabric={"cotton": 1.0},
    )
    contribs = build_contributions([item])
    aggs = aggregate_representatives(contribs)
    assert len(aggs) == 1
    a = aggs[0]
    assert a.representative_key == "kurta__block_print__cotton"
    assert a.total_item_contribution == pytest.approx(5.0)
    assert a.factor_contribution[ContentSource.INSTAGRAM] == pytest.approx(1.0)
    assert a.factor_contribution[ContentSource.YOUTUBE] == pytest.approx(0.0)
    assert a.member_count == 1


def test_aggregate_factor_contribution_mixed_sources() -> None:
    items = [
        ItemDistribution(
            item_id="ig_001",
            source=ContentSource.INSTAGRAM,
            garment_type={"kurta": 1.0},
            technique={"block_print": 1.0},
            fabric={"cotton": 1.0},
        ),
        ItemDistribution(
            item_id="yt_001",
            source=ContentSource.YOUTUBE,
            garment_type={"kurta": 1.0},
            technique={"block_print": 1.0},
            fabric={"cotton": 1.0},
        ),
    ]
    aggs = aggregate_representatives(build_contributions(items))
    assert len(aggs) == 1
    a = aggs[0]
    # 두 item 모두 contribution=5.0, total=10.0, 각각 0.5 비중.
    assert a.total_item_contribution == pytest.approx(10.0)
    assert a.factor_contribution[ContentSource.INSTAGRAM] == pytest.approx(0.5)
    assert a.factor_contribution[ContentSource.YOUTUBE] == pytest.approx(0.5)
    assert sum(a.factor_contribution.values()) == pytest.approx(1.0)
    assert a.member_count == 2


def test_aggregate_sparse_filter() -> None:
    # 빈 contributions → 빈 결과 (분모 0 발산 방지).
    aggs = aggregate_representatives([])
    assert aggs == []


def _mk_contrib(item_id: str, source: ContentSource, contribution: float) -> RepresentativeContribution:
    return RepresentativeContribution(
        representative_key="kurta__block_print__cotton",
        item_id=item_id,
        source=source,
        match_share=contribution / 5.0,
        multiplier=5.0,
        contribution=contribution,
    )


def test_top_evidence_per_source_sorts_desc_and_caps_k() -> None:
    contribs = [
        _mk_contrib("ig_a", ContentSource.INSTAGRAM, 1.0),
        _mk_contrib("ig_b", ContentSource.INSTAGRAM, 3.0),
        _mk_contrib("ig_c", ContentSource.INSTAGRAM, 2.0),
        _mk_contrib("ig_d", ContentSource.INSTAGRAM, 4.0),
        _mk_contrib("ig_e", ContentSource.INSTAGRAM, 0.5),
    ]
    out = top_evidence_per_source(contribs, k=3)
    assert [r.item_id for r in out[ContentSource.INSTAGRAM]] == ["ig_d", "ig_b", "ig_c"]


def test_top_evidence_tie_break_by_item_id_asc() -> None:
    # 동일 contribution 일 때 item_id asc 가 tie-break.
    contribs = [
        _mk_contrib("ig_z", ContentSource.INSTAGRAM, 1.0),
        _mk_contrib("ig_a", ContentSource.INSTAGRAM, 1.0),
        _mk_contrib("ig_m", ContentSource.INSTAGRAM, 1.0),
    ]
    out = top_evidence_per_source(contribs, k=2)
    assert [r.item_id for r in out[ContentSource.INSTAGRAM]] == ["ig_a", "ig_m"]


def test_top_evidence_separates_sources() -> None:
    contribs = [
        _mk_contrib("ig_1", ContentSource.INSTAGRAM, 5.0),
        _mk_contrib("yt_1", ContentSource.YOUTUBE, 4.0),
        _mk_contrib("yt_2", ContentSource.YOUTUBE, 6.0),
    ]
    out = top_evidence_per_source(contribs, k=4)
    assert [r.item_id for r in out[ContentSource.INSTAGRAM]] == ["ig_1"]
    assert [r.item_id for r in out[ContentSource.YOUTUBE]] == ["yt_2", "yt_1"]


def test_aggregate_deterministic_sort() -> None:
    items = [
        ItemDistribution(
            item_id="i1",
            source=ContentSource.INSTAGRAM,
            garment_type={"saree": 1.0, "kurta": 1.0},
            technique={"plain": 1.0},
            fabric={"cotton": 1.0},
        ),
    ]
    aggs = aggregate_representatives(build_contributions(items))
    keys = [a.representative_key for a in aggs]
    assert keys == sorted(keys)  # deterministic ascending


# --------------------------------------------------------------------------- #
# Phase α (2026-04-28) — item_cluster_shares (share-weighted summary 입력)
# --------------------------------------------------------------------------- #


def test_item_cluster_shares_matches_spec_cross_product() -> None:
    item = ItemDistribution(
        item_id="i1",
        source=ContentSource.INSTAGRAM,
        garment_type={"kurta": 0.7, "saree": 0.3},
        technique={"block_print": 1.0},
        fabric={"cotton": 0.6, "silk": 0.4},
    )

    shares = item_cluster_shares(item)

    assert shares == {
        "kurta__block_print__cotton": 0.7 * 1.0 * 0.6,
        "kurta__block_print__silk":   0.7 * 1.0 * 0.4,
        "saree__block_print__cotton": 0.3 * 1.0 * 0.6,
        "saree__block_print__silk":   0.3 * 1.0 * 0.4,
    }
    assert abs(sum(shares.values()) - 1.0) < 1e-9


def test_item_cluster_shares_empty_when_n_lt_3() -> None:
    # build_contributions 와 동일 정책 — N<3 빈 dict.
    item = ItemDistribution(
        item_id="i1",
        source=ContentSource.INSTAGRAM,
        garment_type={"kurta": 1.0},
        technique={},
        fabric={"cotton": 1.0},
    )

    assert item_cluster_shares(item) == {}


def test_item_cluster_shares_no_multiplier_applied() -> None:
    # build_contributions 의 contribution = share × multiplier(=5.0). raw share 만 반환.
    item = ItemDistribution(
        item_id="i1",
        source=ContentSource.INSTAGRAM,
        garment_type={"kurta": 1.0},
        technique={"chikankari": 1.0},
        fabric={"cotton": 1.0},
    )

    assert item_cluster_shares(item) == {"kurta__chikankari__cotton": 1.0}
    # build_contributions 는 동일 input 으로 contribution=5.0 (× multiplier).
    [contrib] = build_contributions([item])
    assert contrib.contribution == 5.0
    assert contrib.match_share == 1.0


# --------------------------------------------------------------------------- #
# Phase β1 (2026-04-28) — effective_item_count (multiplier-scaled denominator)
# --------------------------------------------------------------------------- #


def _mk_item(item_id: str, *, g: bool, t: bool, f: bool) -> ItemDistribution:
    """Helper — N 축 결정 여부만 토글하는 fixture."""
    return ItemDistribution(
        item_id=item_id,
        source=ContentSource.INSTAGRAM,
        garment_type={"kurta": 1.0} if g else {},
        technique={"block_print": 1.0} if t else {},
        fabric={"cotton": 1.0} if f else {},
    )


def test_effective_item_count_n_eq_3() -> None:
    items = [_mk_item("i1", g=True, t=True, f=True)]
    assert effective_item_count(items) == pytest.approx(1.0)


def test_effective_item_count_n_eq_2() -> None:
    # N=2 → multiplier 2.5 / 5.0 = 0.5
    items = [_mk_item("i1", g=True, t=True, f=False)]
    assert effective_item_count(items) == pytest.approx(0.5)


def test_effective_item_count_n_eq_1() -> None:
    # N=1 → multiplier 1.0 / 5.0 = 0.2
    items = [_mk_item("i1", g=True, t=False, f=False)]
    assert effective_item_count(items) == pytest.approx(0.2)


def test_effective_item_count_n_eq_0() -> None:
    items = [_mk_item("i1", g=False, t=False, f=False)]
    assert effective_item_count(items) == 0.0


def test_effective_item_count_mixed_batch_sums_correctly() -> None:
    items = [
        _mk_item("a", g=True, t=True, f=True),    # 1.0
        _mk_item("b", g=True, t=True, f=False),   # 0.5
        _mk_item("c", g=True, t=False, f=False),  # 0.2
        _mk_item("d", g=False, t=False, f=False), # 0.0
    ]
    assert effective_item_count(items) == pytest.approx(1.7)


def test_effective_item_count_empty_returns_zero() -> None:
    assert effective_item_count([]) == 0.0
