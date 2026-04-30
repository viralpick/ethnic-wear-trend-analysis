"""compute_growth_rate / growth_rate_factor_map pinning tests.

Phase 3 (2026-04-30): collected_at 기반 Δ days 검증. post_date 는 게시일 불변이라
부적합 (multi-snapshot 동일 → Δ days = 0 → 모든 entry 미수록 버그) — 그 회귀 방지.
"""
from __future__ import annotations

from datetime import datetime, timezone

from contracts.common import ContentSource
from contracts.enriched import EnrichedContentItem
from contracts.normalized import NormalizedContentItem
from pipelines.load_enriched import compute_growth_rate, growth_rate_factor_map


def _make_enriched(
    *,
    short_tag: str,
    source: ContentSource,
    growth_metric: int,
    collected_at: datetime | None,
    post_date: datetime = datetime(2026, 4, 1, tzinfo=timezone.utc),
    source_post_id: str = "stub",
) -> EnrichedContentItem:
    normalized = NormalizedContentItem(
        source=source,
        source_post_id=source_post_id,
        url_short_tag=short_tag,
        text_blob="",
        hashtags=[],
        image_urls=[],
        post_date=post_date,
        collected_at=collected_at,
        growth_metric=growth_metric,
        engagement_score=0.0,
        engagement_raw_count=0,
        account_followers=0,
        account_handle=None,
    )
    return EnrichedContentItem(
        normalized=normalized,
        canonicals=[],
        post_palette=[],
        is_india_ethnic_wear=True,
        occasion=None,
        brands=[],
    )


def test_growth_rate_uses_collected_at_not_post_date() -> None:
    """같은 url_short_tag 의 multi-snapshot 은 post_date 동일이라 Δ days = 0 위험.
    collected_at 기준이면 정상 산출."""
    same_post_date = datetime(2026, 4, 1, tzinfo=timezone.utc)
    items = [
        _make_enriched(
            short_tag="abc", source=ContentSource.INSTAGRAM,
            growth_metric=100, post_date=same_post_date,
            collected_at=datetime(2026, 4, 10, tzinfo=timezone.utc),
        ),
        _make_enriched(
            short_tag="abc", source=ContentSource.INSTAGRAM,
            growth_metric=300, post_date=same_post_date,
            collected_at=datetime(2026, 4, 20, tzinfo=timezone.utc),
        ),
    ]
    out = compute_growth_rate(items)
    assert "abc" in out
    source, rate = out["abc"]
    assert source == "instagram"
    # Δ metric 200 / Δ days 10 = 20.0 likes/day
    assert rate == 20.0


def test_growth_rate_skips_when_collected_at_missing() -> None:
    """collected_at None 이면 시계열 비교 불가 → 미수록."""
    items = [
        _make_enriched(
            short_tag="x", source=ContentSource.INSTAGRAM,
            growth_metric=10, collected_at=None,
        ),
        _make_enriched(
            short_tag="x", source=ContentSource.INSTAGRAM,
            growth_metric=20, collected_at=None,
        ),
    ]
    assert compute_growth_rate(items) == {}


def test_growth_rate_skips_single_snapshot() -> None:
    items = [
        _make_enriched(
            short_tag="solo", source=ContentSource.INSTAGRAM,
            growth_metric=100,
            collected_at=datetime(2026, 4, 10, tzinfo=timezone.utc),
        ),
    ]
    assert compute_growth_rate(items) == {}


def test_growth_rate_factor_map_per_source_normalization() -> None:
    """IG max likes/day 와 YT max views/day 가 서로 다른 단위라 source 별 분리 정규화."""
    growth = {
        "ig_a": ("instagram", 100.0),  # IG max
        "ig_b": ("instagram", 50.0),
        "yt_a": ("youtube", 5000.0),  # YT max
        "yt_b": ("youtube", 1000.0),
    }
    factors = growth_rate_factor_map(growth)
    # IG max → factor 2.0 (1 + 100/100)
    assert factors["ig_a"] == 2.0
    # IG half max → 1.5 (1 + 50/100)
    assert factors["ig_b"] == 1.5
    # YT max → 2.0 (1 + 5000/5000)
    assert factors["yt_a"] == 2.0
    # YT 1/5 max → 1.2 (1 + 1000/5000)
    assert factors["yt_b"] == 1.2


def test_growth_rate_factor_map_negative_growth_floor_to_one() -> None:
    """음수 성장률 (likes 감소) → factor 1.0 (감소 무시)."""
    growth = {
        "down": ("instagram", -50.0),
        "up": ("instagram", 100.0),
    }
    factors = growth_rate_factor_map(growth)
    assert factors["down"] == 1.0
    assert factors["up"] == 2.0


def test_growth_rate_zero_delta_days_skipped() -> None:
    """동일 collected_at 두 snapshot → Δ days = 0 → skip."""
    same_collected = datetime(2026, 4, 10, tzinfo=timezone.utc)
    items = [
        _make_enriched(
            short_tag="zero", source=ContentSource.INSTAGRAM,
            growth_metric=10, collected_at=same_collected,
        ),
        _make_enriched(
            short_tag="zero", source=ContentSource.INSTAGRAM,
            growth_metric=20, collected_at=same_collected,
        ),
    ]
    assert compute_growth_rate(items) == {}
