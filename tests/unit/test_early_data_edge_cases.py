"""Early-data edge cases (non-negotiable, user mandate).

- day 1 / day 3 / day 7 에 대해 weekly_direction / data_maturity / lifecycle 기대값
- momentum 은 7일 baseline 이 0 일 때 crash 없이 0 반환
- post_count_total < threshold 이면 lifecycle 은 EARLY
"""
from __future__ import annotations

from datetime import date

import pytest

from contracts.common import DataMaturity, Direction, LifecycleStage
from scoring.direction import (
    classify_data_maturity,
    classify_lifecycle,
    classify_weekly_direction,
)
from scoring.score_momentum import safe_growth
from settings import DataMaturityConfig, LifecycleConfig


@pytest.fixture
def maturity_cfg() -> DataMaturityConfig:
    return DataMaturityConfig(bootstrap_below_days=3, full_from_days=7)


@pytest.fixture
def lifecycle_cfg() -> LifecycleConfig:
    return LifecycleConfig(
        early_below=30.0, growth_until=65.0, early_post_count_threshold=10
    )


def _days_since(start: date, target: date) -> int:
    return (target - start).days + 1


# --------------------------------------------------------------------------- #
# day 1, day 3, day 7 시나리오
# --------------------------------------------------------------------------- #

def test_day_one_weekly_flat_baseline_missing(maturity_cfg: DataMaturityConfig) -> None:
    # day 1 은 7일 전 entry 가 없어 weekly baseline 부재 → FLAT 강제.
    start = date(2026, 4, 21)
    today = date(2026, 4, 21)
    days = _days_since(start, today)
    assert days == 1
    assert classify_weekly_direction(
        50.0, 5.0, weekly_baseline_exists=False
    ) == Direction.FLAT
    assert classify_data_maturity(days, maturity_cfg) == DataMaturity.BOOTSTRAP


def test_day_three_partial_maturity_baseline_still_missing(
    maturity_cfg: DataMaturityConfig,
) -> None:
    # day 3 부터 partial maturity. weekly baseline 은 7일 누적 필요라 여전히 부재.
    start = date(2026, 4, 21)
    today = date(2026, 4, 23)
    days = _days_since(start, today)
    assert days == 3
    assert classify_data_maturity(days, maturity_cfg) == DataMaturity.PARTIAL
    assert classify_weekly_direction(
        12.0, 5.0, weekly_baseline_exists=False
    ) == Direction.FLAT


def test_day_seven_full_maturity_baseline_present(
    maturity_cfg: DataMaturityConfig,
) -> None:
    # day 7+ 는 7일 전 entry 가 있을 수 있음 → baseline_exists=True 시 정상 classify.
    start = date(2026, 4, 21)
    today = date(2026, 4, 27)
    days = _days_since(start, today)
    assert days == 7
    assert classify_data_maturity(days, maturity_cfg) == DataMaturity.FULL
    assert classify_weekly_direction(
        12.0, 5.0, weekly_baseline_exists=True
    ) == Direction.UP


# --------------------------------------------------------------------------- #
# momentum denominator 0 → growth 0 (raise 없음)
# --------------------------------------------------------------------------- #

def test_momentum_zero_baseline_does_not_raise() -> None:
    assert safe_growth(current=10.0, baseline=0.0) == 0.0


def test_momentum_negative_baseline_does_not_raise() -> None:
    # 음수 baseline 도 0 으로 처리 (정의상 의미 없음 — post_count 가 음수일 수 없음).
    assert safe_growth(current=10.0, baseline=-1.0) == 0.0


# --------------------------------------------------------------------------- #
# lifecycle 은 post_count_total < threshold 이면 EARLY 강제
# --------------------------------------------------------------------------- #

def test_lifecycle_forced_early_regardless_of_score(
    lifecycle_cfg: LifecycleConfig,
) -> None:
    # 높은 점수도 post_count 적으면 early (초창기 노이즈 억제).
    result = classify_lifecycle(
        score=95.0, post_count_total=9, score_trend="rising", cfg=lifecycle_cfg
    )
    assert result == LifecycleStage.EARLY


def test_lifecycle_becomes_growth_above_threshold(
    lifecycle_cfg: LifecycleConfig,
) -> None:
    result = classify_lifecycle(
        score=50.0, post_count_total=10, score_trend="rising", cfg=lifecycle_cfg
    )
    assert result == LifecycleStage.GROWTH


# --------------------------------------------------------------------------- #
# Sub-score 계산이 빈/0 입력에서 crash 하지 않는다
# --------------------------------------------------------------------------- #

def test_score_computation_with_all_zero_inputs_does_not_raise() -> None:
    from scoring import score_cultural, score_momentum, score_social, score_youtube
    from scoring.cluster_context import ClusterScoringContext
    from tests.unit.test_scoring_formulas import _cfg

    ctx = ClusterScoringContext(
        cluster_key="zero",
        social_weighted_engagement=0.0,
        youtube_video_count=0,
        youtube_views_total=0.0,
        youtube_view_growth=0.0,
        cultural_festival_match=0.0,
        cultural_bollywood_presence=0.0,
        momentum_post_growth=0.0,
        momentum_hashtag_velocity=0.0,
        momentum_new_ig_account_ratio=0.0,
        momentum_new_yt_channel_ratio=0.0,
        post_count_total=0,
        post_count_today=0,
        avg_engagement_rate=0.0,
    )
    cfg = _cfg()
    # 모든 sub-score raw 가 0 에서 crash 없이 0.0 을 내야 한다.
    assert score_social.compute(ctx, cfg) == 0.0
    assert score_youtube.compute(ctx, cfg) == 0.0
    assert score_cultural.compute(ctx, cfg) == 0.0
    assert score_momentum.compute(ctx, cfg) == 0.0
