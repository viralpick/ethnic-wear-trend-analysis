"""Social 팩터 raw 계산 (spec §9.2 Social).

compute() 는 pure. 정규화 / 가중치 scaling 은 compute_scores orchestrator 담당.
"""
from __future__ import annotations

from scoring.cluster_context import ClusterScoringContext
from settings import ScoringConfig


def compute(cluster_ctx: ClusterScoringContext, cfg: ScoringConfig) -> float:  # noqa: ARG001
    """인플루언서 가중 engagement 합을 그대로 반환 (pre-aggregated).

    cfg 는 signature 통일 위해 받지만 Social 은 pre-aggregate 결과를 그대로 사용한다.
    influencer_weight 는 ClusterScoringContext 구성 단계에서 이미 적용된다 (build_cluster_summary).
    """
    return cluster_ctx.social_weighted_engagement
