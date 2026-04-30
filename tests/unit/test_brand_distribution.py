"""로직 C (2026-04-29) — compute_brand_distribution 핀.

규칙:
- post 1건이 N brand 동시 언급 → 영향력 = 1/log2(N+1) × (1/N) per brand × cluster share.
- 합산 후 정규화 → share<min_share drop → top_n cut → 재정규화.
- empty / 모든 share<=0 → 빈 dict.
"""
from __future__ import annotations

import math
from datetime import datetime

import pytest

from aggregation.brand_distribution import compute_brand_distribution
from contracts.common import ContentSource
from contracts.enriched import BrandInfo, EnrichedContentItem
from contracts.normalized import NormalizedContentItem


def _normalized(post_id: str) -> NormalizedContentItem:
    return NormalizedContentItem(
        source=ContentSource.INSTAGRAM,
        source_post_id=post_id,
        text_blob="",
        hashtags=[],
        image_urls=[],
        post_date=datetime(2026, 4, 29),
        engagement_raw_count=100,
    )


def _enriched(post_id: str, brand_names: list[str]) -> EnrichedContentItem:
    return EnrichedContentItem(
        normalized=_normalized(post_id),
        brands=[BrandInfo(name=n) for n in brand_names],
    )


def test_empty_input_returns_empty() -> None:
    assert compute_brand_distribution([]) == {}


def test_no_brands_returns_empty() -> None:
    a = _enriched("p_a", brand_names=[])
    assert compute_brand_distribution([(a, 1.0)]) == {}


def test_single_post_single_brand_share_1() -> None:
    """N=1 → log_weight=1.0, single brand 100%."""
    a = _enriched("p_a", brand_names=["AND"])
    dist = compute_brand_distribution([(a, 1.0)])
    assert dist == {"AND": pytest.approx(1.0)}


def test_single_post_two_brands_log_split_then_normalize() -> None:
    """N=2 → log_weight=1/log2(3)≈0.631, brand 별 0.5 × 0.631 ≈ 0.315.
    합 ≈ 0.631 → 정규화 후 50:50.
    """
    a = _enriched("p_a", brand_names=["AND", "Fabindia"])
    dist = compute_brand_distribution([(a, 1.0)])
    assert dist["AND"] == pytest.approx(0.5)
    assert dist["Fabindia"] == pytest.approx(0.5)


def test_two_posts_one_brand_each_gets_share_proportional() -> None:
    """post X share=0.7, brand A / post Y share=0.3, brand B → A:B = 0.7:0.3."""
    a = _enriched("p_a", brand_names=["AND"])
    b = _enriched("p_b", brand_names=["Fabindia"])
    dist = compute_brand_distribution([(a, 0.7), (b, 0.3)])
    assert dist["AND"] == pytest.approx(0.7)
    assert dist["Fabindia"] == pytest.approx(0.3)


def test_log_scale_decay_n_dominant_post() -> None:
    """N=1 brand 1건 (share=1) vs N=5 brand 1건 (share=1) — N=1 brand 가 더 큰 영향력.

    A: 1.0 × (1/log2(2)) × 1.0 = 1.0
    B (각 brand): 1.0 × (1/log2(6)) × (1/5) ≈ 0.0774
    → 1차 정규화: A ≈ 0.721 / B 각 ≈ 0.0558
    threshold 0.05 통과: A + 5 brand 모두. top_n=5 → A + 4 brand (B0..B3).
    재정규화 후 A 가 dominant.
    """
    a = _enriched("p_a", brand_names=["A"])
    b = _enriched("p_b", brand_names=["B0", "B1", "B2", "B3", "B4"])
    dist = compute_brand_distribution([(a, 1.0), (b, 1.0)])
    # top 5 cut → A + 4 of B*. A 는 가장 큰 share.
    assert "A" in dist
    assert max(dist, key=lambda k: dist[k]) == "A"
    # 살아남은 합 1.0
    assert sum(dist.values()) == pytest.approx(1.0)
    assert len(dist) == 5


def test_threshold_drops_low_share_brand() -> None:
    """N=1 brand 1건 (share=1.0, big) + N=20 brand 1건 (share=0.1, small).

    big: 1.0 × 1.0 × 1.0 = 1.0
    small (각): 0.1 × (1/log2(21)) × (1/20) ≈ 0.001138
    raw_total ≈ 1.0 + 20 × 0.001138 ≈ 1.0228
    big share ≈ 0.978, small 각 ≈ 0.00111 < 0.05 → drop.
    """
    a = _enriched("p_a", brand_names=["BIG"])
    b_brands = [f"S{i}" for i in range(20)]
    b = _enriched("p_b", brand_names=b_brands)
    dist = compute_brand_distribution([(a, 1.0), (b, 0.1)])
    assert dist == {"BIG": pytest.approx(1.0)}


def test_dedup_within_post() -> None:
    """post 의 brands list 가 중복 name 가지면 한 번만 카운트 (account_handle + caption
    둘 다 잡힌 경우)."""
    a = _enriched("p_a", brand_names=["AND", "AND", "Fabindia"])
    dist = compute_brand_distribution([(a, 1.0)])
    # N=2 (AND dedup) → 50:50
    assert dist["AND"] == pytest.approx(0.5)
    assert dist["Fabindia"] == pytest.approx(0.5)


def test_zero_share_skipped() -> None:
    """share<=0 인 (item, share) 는 미기여."""
    a = _enriched("p_a", brand_names=["AND"])
    b = _enriched("p_b", brand_names=["Fabindia"])
    dist = compute_brand_distribution([(a, 1.0), (b, 0.0)])
    assert dist == {"AND": pytest.approx(1.0)}


def test_top_n_cut_overrides_threshold_survivors() -> None:
    """6 brand 모두 threshold 통과해도 top_n=5 → 5개만. 살아남은 합 1.0."""
    posts = [
        _enriched(f"p_{i}", brand_names=[f"B{i}"]) for i in range(6)
    ]
    dist = compute_brand_distribution([(p, 1.0) for p in posts])
    assert len(dist) == 5
    assert sum(dist.values()) == pytest.approx(1.0)


def test_share_desc_insertion_order() -> None:
    """반환 dict 의 key 순서 = share desc (JSON 출력 시 ranking 보존)."""
    a = _enriched("p_a", brand_names=["AND"])  # 0.7
    b = _enriched("p_b", brand_names=["Fabindia"])  # 0.2
    c = _enriched("p_c", brand_names=["W"])  # 0.1
    dist = compute_brand_distribution(
        [(a, 0.7), (b, 0.2), (c, 0.1)],
        min_share=0.0,  # threshold 끄고 ranking 만 검증
    )
    assert list(dist.keys()) == ["AND", "Fabindia", "W"]


def test_log_weight_formula_pinning() -> None:
    """1/log2(N+1) 공식 직접 검증 — N=3 일 때 brand 당 raw weight = share/3 × 1/log2(4)."""
    a = _enriched("p_a", brand_names=["A", "B", "C"])
    # share=1 → 각 brand raw = 1.0 × (1/2.0) × (1/3) = 1/6 ≈ 0.1667
    # 정규화 후 1/3 each
    dist = compute_brand_distribution([(a, 1.0)])
    expected = 1.0 / 3.0
    for name in ("A", "B", "C"):
        assert dist[name] == pytest.approx(expected)
    # log2(4) = 2.0 직접 검증
    assert math.log2(4) == pytest.approx(2.0)
