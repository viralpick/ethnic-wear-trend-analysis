"""Cultural Fit 팩터 raw 계산 (spec §9.2 Cultural + §9.5).

festival_match × 0.6 + bollywood_presence × 0.4 — pre-aggregated 값 사용.
festival_boost / bollywood_bonus 의 적용은 build_cluster_summary 단계에서 미리 수행되어
ClusterScoringContext 에 들어온다 (pure 유지).
"""
from __future__ import annotations

from scoring.cluster_context import ClusterScoringContext
from settings import ScoringConfig


def compute(cluster_ctx: ClusterScoringContext, cfg: ScoringConfig) -> float:
    w = cfg.cultural_factor_weights
    return (
        cluster_ctx.cultural_festival_match * w.festival
        + cluster_ctx.cultural_bollywood_presence * w.bollywood
    )
