"""trajectory pinning — pipeline_spec_v1.0 §3.3 / §3.4.

검증 대상:
- consecutive_direction_streak: 끝에서부터 연속 동일 방향 카운트, padding 0 끊음.
- is_within_band: 최근 N 주 ±band_pct 이내 변동 여부.
- monthly_score_avg: 단순 평균 (sparse 0 그대로).
- weighted_distribution_avg: contribution-weighted, 합=1.0, weight=0 무시.
"""
from __future__ import annotations

import pytest

from aggregation.trajectory import (
    consecutive_direction_streak,
    is_within_band,
    monthly_score_avg,
    weighted_distribution_avg,
)
from contracts.common import Direction


def test_consecutive_up_streak() -> None:
    # 0,0,...,30,40,50,60 → up 3연속
    traj = [0.0] * 8 + [30.0, 40.0, 50.0, 60.0]
    assert consecutive_direction_streak(traj, direction=Direction.UP) == 3


def test_padding_zero_breaks_streak() -> None:
    # 끝에서 직전이 0 이면 streak 끊김 (방향 정의 불가).
    traj = [0.0] * 11 + [50.0]
    assert consecutive_direction_streak(traj, direction=Direction.UP) == 0


def test_consecutive_down_streak() -> None:
    traj = [0.0] * 8 + [60.0, 50.0, 40.0, 30.0]
    assert consecutive_direction_streak(traj, direction=Direction.DOWN) == 3


def test_streak_below_threshold_counts_flat() -> None:
    # 1% 변동만이면 up 임계값 5% 못 넘어서 flat → up streak = 0.
    traj = [0.0] * 9 + [100.0, 101.0, 102.0]
    assert consecutive_direction_streak(traj, direction=Direction.UP) == 0


def test_is_within_band_true() -> None:
    # 4주 모두 ±5% 이내.
    traj = [0.0] * 7 + [60.0, 61.0, 62.0, 63.0, 64.0]
    assert is_within_band(traj, weeks=4, band_pct=5.0) is True


def test_is_within_band_false_when_padding() -> None:
    # 검사 범위 안에 0 padding 들어가면 False.
    traj = [0.0] * 11 + [60.0]
    assert is_within_band(traj, weeks=4, band_pct=5.0) is False


def test_monthly_score_avg_simple() -> None:
    assert monthly_score_avg([10.0, 20.0, 30.0, 40.0]) == 25.0


def test_monthly_score_avg_with_padding() -> None:
    # sparse 영향 그대로 — 0 padding 도 평균에 포함.
    assert monthly_score_avg([0.0, 0.0, 30.0, 40.0]) == pytest.approx(17.5)


def test_monthly_score_avg_empty() -> None:
    assert monthly_score_avg([]) == 0.0


def test_weighted_distribution_avg() -> None:
    # 두 주 distribution, weight 비례.
    dists = [
        ({"kurta": 0.6, "saree": 0.4}, 10.0),
        ({"kurta": 0.2, "saree": 0.8}, 30.0),
    ]
    avg = weighted_distribution_avg(dists)
    # kurta = (0.6×10 + 0.2×30) / 40 = 12 / 40 = 0.3
    # saree = (0.4×10 + 0.8×30) / 40 = 28 / 40 = 0.7
    assert avg["kurta"] == pytest.approx(0.3)
    assert avg["saree"] == pytest.approx(0.7)
    assert sum(avg.values()) == pytest.approx(1.0)


def test_weighted_distribution_avg_zero_weight_ignored() -> None:
    dists = [
        ({"kurta": 1.0}, 0.0),
        ({"saree": 1.0}, 5.0),
    ]
    avg = weighted_distribution_avg(dists)
    assert avg == {"saree": pytest.approx(1.0)}


def test_weighted_distribution_avg_all_zero_returns_empty() -> None:
    dists = [({"kurta": 1.0}, 0.0)]
    assert weighted_distribution_avg(dists) == {}
