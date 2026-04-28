"""Trajectory + monthly 4-week rolling — pipeline_spec_v1.0 §3.3 / §3.4.

L9: 12주 trajectory 는 `WeeklyScoreHistory.get_trajectory_12w` 가 0 padding 까지 처리.
이 모듈은 (a) trajectory 위에 derived 시그널 (consecutive up/down), (b) monthly 4주 rolling
read-time 합성 헬퍼만 책임진다.

monthly 합성 = 최근 4주의 score 평균 + (필요 시) distribution / palette contribution-weighted
평균. distribution 합성은 read-time 이라 representative_weekly row 4개를 받아서 weight 로
누적. 이 phase 에선 score 평균 + direction 만 우선 (palette 합성은 read-side adapter 로
이연 가능).

spec §3.4 lifecycle 환산 ("3주 연속 상승" / "3주 연속 하락") 헬퍼도 여기 둔다.
"""
from __future__ import annotations

from collections.abc import Iterable

from contracts.common import Direction


def _direction(prev: float, curr: float, threshold_pct: float) -> Direction:
    """spec §3.4 — +threshold% 이상 up, -threshold% 이하 down, 그 외 flat. prev=0 → flat."""
    if prev <= 0.0:
        return Direction.FLAT
    delta_pct = (curr - prev) / prev * 100.0
    if delta_pct >= threshold_pct:
        return Direction.UP
    if delta_pct <= -threshold_pct:
        return Direction.DOWN
    return Direction.FLAT


def consecutive_direction_streak(
    trajectory: list[float],
    *,
    direction: Direction,
    threshold_pct: float = 5.0,
) -> int:
    """trajectory 끝에서 시작해서 같은 방향이 몇 주 연속됐는지 반환 (oldest → newest 입력).

    spec §9.4 환산: growth 조건 "3주 연속 상승" / decline "3주 연속 하락" 판정용.
    0 padding 슬롯 (prev=0) 은 streak 끊는다 (방향 정의 불가).
    """
    if len(trajectory) < 2:
        return 0
    streak = 0
    for prev, curr in zip(reversed(trajectory[:-1]), reversed(trajectory[1:])):
        if _direction(prev, curr, threshold_pct) is direction:
            streak += 1
        else:
            break
    return streak


def is_within_band(
    trajectory: list[float],
    *,
    weeks: int,
    band_pct: float,
) -> bool:
    """maturity 조건 — 최근 `weeks` 주가 모두 ±band_pct 이내 변동인지.

    padding 0 이 포함된 슬롯은 비교 불가 → False (= maturity 아님).
    """
    if len(trajectory) < weeks + 1:
        return False
    recent = trajectory[-(weeks + 1):]
    for prev, curr in zip(recent[:-1], recent[1:], strict=True):
        if prev <= 0.0:
            return False
        delta_pct = abs(curr - prev) / prev * 100.0
        if delta_pct > band_pct:
            return False
    return True


def monthly_score_avg(weekly_scores: Iterable[float]) -> float:
    """spec §3.3 — 최근 4주 weekly score 평균. 0 padding 도 포함 (sparse 영향 그대로 노출).

    호출자는 4개 길이 리스트를 전달 (부족분은 0 으로 채워서). 빈 리스트 → 0.0.
    """
    scores = list(weekly_scores)
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def weighted_distribution_avg(
    distributions: list[tuple[dict[str, float], float]],
) -> dict[str, float]:
    """spec §3.3 — 최근 4주 distribution 의 contribution-weighted 평균.

    입력: [(distribution, weight), ...]. weight = total_item_contribution 권장.
    출력: 합 = 1.0 (정규화). 모든 weight 0 → {}.
    """
    accum: dict[str, float] = {}
    total = 0.0
    for dist, weight in distributions:
        if weight <= 0.0:
            continue
        total += weight
        for value, share in dist.items():
            accum[value] = accum.get(value, 0.0) + share * weight
    if total <= 0.0:
        return {}
    return {k: v / total for k, v in accum.items()}
