"""Direction / lifecycle / data_maturity 판정 (spec §9.3, §9.4).

Early-data 규칙 (non-negotiable):
- weekly_direction 은 days_collected < 3 일 때 FLAT 로 강제.
- momentum 은 denominator 0 일 때 growth factors = 0 (score_momentum.compute 참고).
- lifecycle 은 post_count_total < early_post_count_threshold 이면 점수와 관계없이 EARLY.
- data_maturity 는 days_collected 만으로 결정.

이 모듈은 stub 가 아닌 real code. 외부 의존 없이 pure 함수로 구성한다.
"""
from __future__ import annotations

from contracts.common import DataMaturity, Direction, LifecycleStage
from settings import DataMaturityConfig, LifecycleConfig


def classify_direction(change_pct: float, threshold_pct: float) -> Direction:
    """spec §9.3 — ±threshold 바깥이면 up/down, 아니면 flat."""
    if change_pct >= threshold_pct:
        return Direction.UP
    if change_pct <= -threshold_pct:
        return Direction.DOWN
    return Direction.FLAT


def classify_weekly_direction(
    change_pct: float,
    threshold_pct: float,
    days_collected: int,
) -> Direction:
    """days_collected < 3 이면 FLAT 강제 (baseline 부재). 그 외는 classify_direction 과 동일."""
    if days_collected < 3:
        return Direction.FLAT
    return classify_direction(change_pct, threshold_pct)


def classify_lifecycle(
    score: float,
    post_count_total: float,
    score_trend: str,
    cfg: LifecycleConfig,
) -> LifecycleStage:
    """spec §9.4 — Early/Growth/Maturity/Decline.

    score_trend: "rising" | "falling" | "flat" — 직전 3일 흐름 요약.
    post_count_total < early_post_count_threshold 이면 점수와 관계없이 EARLY (non-negotiable).

    옵션 C (2026-04-29): post_count_total int → float (share-weighted fan-out 의 fractional
    mass). early threshold 비교는 부동소수 비교로 자연 동작.
    """
    if post_count_total < cfg.early_post_count_threshold:
        return LifecycleStage.EARLY
    if score < cfg.early_below:
        return LifecycleStage.EARLY
    if score_trend == "falling":
        return LifecycleStage.DECLINE
    if score >= cfg.growth_until:
        return LifecycleStage.MATURITY
    return LifecycleStage.GROWTH


def classify_data_maturity(
    days_collected: int, cfg: DataMaturityConfig
) -> DataMaturity:
    """spec — bootstrap(<3) / partial(3-6) / full(≥7)."""
    if days_collected < cfg.bootstrap_below_days:
        return DataMaturity.BOOTSTRAP
    if days_collected >= cfg.full_from_days:
        return DataMaturity.FULL
    return DataMaturity.PARTIAL


def change_pct(current: float, baseline: float) -> float:
    """safe % 변화율. baseline 이 0/None 이면 0.0 (divide-by-zero 방어)."""
    if not baseline:
        return 0.0
    return (current - baseline) / baseline * 100.0
