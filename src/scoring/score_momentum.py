"""Momentum 팩터 raw 계산 (spec §9.2 Momentum).

Early-data non-negotiable:
- 7일 rolling denominator 가 0 이거나 missing 이면 growth factors = 0 처리 (divide-by-zero 방어).
- ClusterScoringContext 의 momentum_* 필드는 orchestrator 가 채우기 전에 이미 safe 값으로
  들어와야 한다 — 여기서는 그 약속을 믿고 weighted sum 만 낸다.
"""
from __future__ import annotations

from scoring.cluster_context import ClusterScoringContext
from settings import ScoringConfig


def compute(cluster_ctx: ClusterScoringContext, cfg: ScoringConfig) -> float:
    w = cfg.momentum_factor_weights
    return (
        cluster_ctx.momentum_post_growth * w.post_growth
        + cluster_ctx.momentum_hashtag_velocity * w.hashtag_velocity
        + cluster_ctx.momentum_new_account_ratio * w.new_account_ratio
    )


def safe_growth(current: float, baseline: float) -> float:
    """post_growth = (current - baseline) / baseline. baseline <= 0 이면 0.0 반환."""
    if baseline <= 0:
        return 0.0
    return (current - baseline) / baseline
