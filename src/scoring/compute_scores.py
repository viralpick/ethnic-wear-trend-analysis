"""Sub-score 합성 orchestrator (spec §9).

흐름: contexts → 각 sub-score raw → same-run minmax → weight 로 scale → ScoreBreakdown.
learned / 휴리스틱 re-ranker 일절 없음 (user mandate: explainable from raw counts).
"""
from __future__ import annotations

from contracts.output import MomentumComponents, ScoreBreakdown
from scoring import score_cultural, score_momentum, score_social, score_youtube
from scoring.cluster_context import ClusterScoringContext
from scoring.normalize import apply_normalization
from settings import ScoringConfig


def _scale(normalized: list[float], weight_cap: float) -> list[float]:
    return [n * weight_cap for n in normalized]


def score_clusters(
    contexts: list[ClusterScoringContext], cfg: ScoringConfig
) -> dict[str, ScoreBreakdown]:
    """클러스터 배치 → cluster_key: ScoreBreakdown.

    B-2 (M3.G/H 후): MomentumComponents 로 sub-signal raw 가시화. score_breakdown
    의 momentum 합산은 그대로, IG/YT 신규 entity 시그널을 breakdown 에서 분리 추적.
    """
    if not contexts:
        return {}

    social_raws = [score_social.compute(c, cfg) for c in contexts]
    youtube_raws = [score_youtube.compute(c, cfg) for c in contexts]
    cultural_raws = [score_cultural.compute(c, cfg) for c in contexts]
    momentum_raws = [score_momentum.compute(c, cfg) for c in contexts]

    method = cfg.normalization_method
    social_scaled = _scale(apply_normalization(social_raws, method), cfg.weights.social)
    youtube_scaled = _scale(apply_normalization(youtube_raws, method), cfg.weights.youtube)
    cultural_scaled = _scale(apply_normalization(cultural_raws, method), cfg.weights.cultural)
    momentum_scaled = _scale(apply_normalization(momentum_raws, method), cfg.weights.momentum)

    return {
        ctx.cluster_key: ScoreBreakdown(
            social=social_scaled[i],
            youtube=youtube_scaled[i],
            cultural=cultural_scaled[i],
            momentum=momentum_scaled[i],
            momentum_components=MomentumComponents(
                post_growth=ctx.momentum_post_growth,
                hashtag_velocity=ctx.momentum_hashtag_velocity,
                new_ig_account_ratio=ctx.momentum_new_ig_account_ratio,
                new_yt_channel_ratio=ctx.momentum_new_yt_channel_ratio,
            ),
        )
        for i, ctx in enumerate(contexts)
    }


def total_score(breakdown: ScoreBreakdown) -> float:
    """ScoreBreakdown 의 4 필드 합 (0~100). momentum_components 는 raw 시그널 가시화용."""
    return breakdown.social + breakdown.youtube + breakdown.cultural + breakdown.momentum
