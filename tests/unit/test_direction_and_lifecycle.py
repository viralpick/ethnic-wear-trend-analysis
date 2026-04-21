"""Direction / lifecycle / data_maturity 단위 테스트 (spec §9.3, §9.4)."""
from __future__ import annotations

import pytest

from contracts.common import DataMaturity, Direction, LifecycleStage
from scoring.direction import (
    change_pct,
    classify_data_maturity,
    classify_direction,
    classify_lifecycle,
    classify_weekly_direction,
)
from settings import DataMaturityConfig, LifecycleConfig


@pytest.fixture
def lifecycle_cfg() -> LifecycleConfig:
    return LifecycleConfig(
        early_below=30.0, growth_until=65.0, early_post_count_threshold=10
    )


@pytest.fixture
def maturity_cfg() -> DataMaturityConfig:
    return DataMaturityConfig(bootstrap_below_days=3, full_from_days=7)


# --------------------------------------------------------------------------- #
# classify_direction: ±5% 경계
# --------------------------------------------------------------------------- #

def test_direction_above_threshold_is_up() -> None:
    assert classify_direction(5.01, 5.0) == Direction.UP


def test_direction_at_positive_boundary_is_up() -> None:
    # ">= threshold" 경계. +5% 정확히도 up (포함 경계).
    assert classify_direction(5.0, 5.0) == Direction.UP


def test_direction_within_band_is_flat() -> None:
    assert classify_direction(4.9, 5.0) == Direction.FLAT
    assert classify_direction(-4.9, 5.0) == Direction.FLAT


def test_direction_below_threshold_is_down() -> None:
    assert classify_direction(-5.0, 5.0) == Direction.DOWN
    assert classify_direction(-7.5, 5.0) == Direction.DOWN


# --------------------------------------------------------------------------- #
# weekly direction: <3 일 → FLAT 강제
# --------------------------------------------------------------------------- #

def test_weekly_direction_forced_flat_when_days_less_than_three() -> None:
    # days_collected < 3 이면 변화율과 무관하게 flat.
    assert classify_weekly_direction(50.0, 5.0, days_collected=1) == Direction.FLAT
    assert classify_weekly_direction(-50.0, 5.0, days_collected=2) == Direction.FLAT


def test_weekly_direction_uses_threshold_when_enough_days() -> None:
    assert classify_weekly_direction(12.0, 5.0, days_collected=7) == Direction.UP
    assert classify_weekly_direction(-12.0, 5.0, days_collected=7) == Direction.DOWN


# --------------------------------------------------------------------------- #
# lifecycle: Early/Growth/Maturity/Decline
# --------------------------------------------------------------------------- #

def test_lifecycle_forced_early_when_post_count_below_threshold(
    lifecycle_cfg: LifecycleConfig,
) -> None:
    # 점수가 아무리 높아도 post_count 10 미만이면 EARLY.
    result = classify_lifecycle(
        score=90.0, post_count_total=5, score_trend="rising", cfg=lifecycle_cfg
    )
    assert result == LifecycleStage.EARLY


def test_lifecycle_early_below_score_threshold(lifecycle_cfg: LifecycleConfig) -> None:
    result = classify_lifecycle(
        score=25.0, post_count_total=50, score_trend="rising", cfg=lifecycle_cfg
    )
    assert result == LifecycleStage.EARLY


def test_lifecycle_growth_mid_score(lifecycle_cfg: LifecycleConfig) -> None:
    result = classify_lifecycle(
        score=45.0, post_count_total=50, score_trend="rising", cfg=lifecycle_cfg
    )
    assert result == LifecycleStage.GROWTH


def test_lifecycle_maturity_at_high_score(lifecycle_cfg: LifecycleConfig) -> None:
    result = classify_lifecycle(
        score=80.0, post_count_total=200, score_trend="flat", cfg=lifecycle_cfg
    )
    assert result == LifecycleStage.MATURITY


def test_lifecycle_decline_when_trend_falling(lifecycle_cfg: LifecycleConfig) -> None:
    # 점수가 성숙 구간이어도 3일 연속 하락 시그널이면 Decline.
    result = classify_lifecycle(
        score=70.0, post_count_total=200, score_trend="falling", cfg=lifecycle_cfg
    )
    assert result == LifecycleStage.DECLINE


# --------------------------------------------------------------------------- #
# data_maturity 판정
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "days,expected",
    [
        (0, DataMaturity.BOOTSTRAP),
        (1, DataMaturity.BOOTSTRAP),
        (2, DataMaturity.BOOTSTRAP),
        (3, DataMaturity.PARTIAL),
        (6, DataMaturity.PARTIAL),
        (7, DataMaturity.FULL),
        (30, DataMaturity.FULL),
    ],
)
def test_data_maturity_bands(
    days: int, expected: DataMaturity, maturity_cfg: DataMaturityConfig
) -> None:
    assert classify_data_maturity(days, maturity_cfg) == expected


# --------------------------------------------------------------------------- #
# change_pct: baseline 0 safe 처리
# --------------------------------------------------------------------------- #

def test_change_pct_zero_baseline_returns_zero() -> None:
    assert change_pct(10.0, 0.0) == 0.0
    assert change_pct(10.0, 0) == 0.0


def test_change_pct_normal_case() -> None:
    assert change_pct(110.0, 100.0) == pytest.approx(10.0)
    assert change_pct(90.0, 100.0) == pytest.approx(-10.0)
