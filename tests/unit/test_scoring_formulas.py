"""Sub-score raw 계산 + compute_scores orchestrator (spec §9.2).

숫자는 전부 손으로 계산 가능한 수준만 사용한다 (랜덤 드리프트 방지).
"""
from __future__ import annotations

import pytest

from scoring import score_cultural, score_momentum, score_social, score_youtube
from scoring.cluster_context import ClusterScoringContext
from scoring.compute_scores import score_clusters, total_score
from settings import (
    CulturalFactorWeights,
    DataMaturityConfig,
    InfluencerTierThresholds,
    InfluencerWeights,
    LifecycleConfig,
    MomentumFactorWeights,
    ScoringConfig,
    ScoringWeights,
    YouTubeFactorWeights,
)


def _ctx(cluster_key: str, **overrides: float | int) -> ClusterScoringContext:
    base = {
        "cluster_key": cluster_key,
        "social_weighted_engagement": 0.0,
        "youtube_video_count": 0,
        "youtube_views_total": 0.0,
        "youtube_view_growth": 0.0,
        "cultural_festival_match": 0.0,
        "cultural_bollywood_presence": 0.0,
        "momentum_post_growth": 0.0,
        "momentum_hashtag_velocity": 0.0,
        "momentum_new_account_ratio": 0.0,
        "post_count_total": 0,
        "post_count_today": 0,
        "avg_engagement_rate": 0.0,
    }
    base.update(overrides)
    return ClusterScoringContext(**base)


def _cfg() -> ScoringConfig:
    return ScoringConfig(
        weights=ScoringWeights(social=40.0, youtube=25.0, cultural=15.0, momentum=20.0),
        normalization_method="minmax_same_run",
        direction_threshold_pct=5.0,
        lifecycle=LifecycleConfig(
            early_below=30.0, growth_until=65.0, early_post_count_threshold=10
        ),
        influencer_weights=InfluencerWeights(mega=3.0, macro=2.0, mid=1.5, micro=1.0),
        influencer_tier_thresholds=InfluencerTierThresholds(
            mega=1_000_000, macro=100_000, mid=10_000
        ),
        youtube_factor_weights=YouTubeFactorWeights(
            video_count=0.3, views=0.4, view_growth=0.3
        ),
        cultural_factor_weights=CulturalFactorWeights(festival=0.6, bollywood=0.4),
        cultural_festival_boost=1.5,
        cultural_bollywood_bonus=0.3,
        cultural_festivals=[],
        momentum_factor_weights=MomentumFactorWeights(
            post_growth=0.4, hashtag_velocity=0.3, new_account_ratio=0.3
        ),
        momentum_window_days=7,
        data_maturity=DataMaturityConfig(bootstrap_below_days=3, full_from_days=7),
    )


def test_score_social_returns_pre_aggregated_engagement() -> None:
    ctx = _ctx("c1", social_weighted_engagement=1234.5)
    assert score_social.compute(ctx, _cfg()) == pytest.approx(1234.5)


def test_score_youtube_weighted_sum() -> None:
    # 10 × 0.3 + 100 × 0.4 + 2 × 0.3 = 3 + 40 + 0.6 = 43.6
    ctx = _ctx("c1", youtube_video_count=10, youtube_views_total=100.0,
               youtube_view_growth=2.0)
    assert score_youtube.compute(ctx, _cfg()) == pytest.approx(43.6)


def test_score_cultural_weighted_sum() -> None:
    # 2.0 × 0.6 + 1.0 × 0.4 = 1.2 + 0.4 = 1.6
    ctx = _ctx("c1", cultural_festival_match=2.0, cultural_bollywood_presence=1.0)
    assert score_cultural.compute(ctx, _cfg()) == pytest.approx(1.6)


def test_score_momentum_weighted_sum() -> None:
    # 0.5 × 0.4 + 0.2 × 0.3 + 0.1 × 0.3 = 0.2 + 0.06 + 0.03 = 0.29
    ctx = _ctx("c1", momentum_post_growth=0.5, momentum_hashtag_velocity=0.2,
               momentum_new_account_ratio=0.1)
    assert score_momentum.compute(ctx, _cfg()) == pytest.approx(0.29)


def test_score_momentum_safe_growth_zero_denominator() -> None:
    # baseline=0 → divide-by-zero 대신 0.0.
    assert score_momentum.safe_growth(current=100.0, baseline=0.0) == 0.0
    assert score_momentum.safe_growth(current=100.0, baseline=-5.0) == 0.0


# --------------------------------------------------------------------------- #
# Orchestrator: minmax normalization + weight scaling 검증
# --------------------------------------------------------------------------- #

def test_score_clusters_empty_returns_empty() -> None:
    assert score_clusters([], _cfg()) == {}


def test_score_clusters_minmax_scales_to_full_weight_at_max() -> None:
    # Social raw 10 vs 100 → normalized [0, 1] → scaled [0, 40].
    contexts = [
        _ctx("A", social_weighted_engagement=10.0),
        _ctx("B", social_weighted_engagement=100.0),
    ]
    result = score_clusters(contexts, _cfg())
    assert result["A"].social == pytest.approx(0.0)
    assert result["B"].social == pytest.approx(40.0)


def test_score_clusters_all_equal_collapses_to_zero() -> None:
    # 동률이면 same-run minmax 전부 0 → 스케일 후 0.
    contexts = [
        _ctx("A", social_weighted_engagement=7.0),
        _ctx("B", social_weighted_engagement=7.0),
    ]
    result = score_clusters(contexts, _cfg())
    assert result["A"].social == 0.0
    assert result["B"].social == 0.0


def test_total_score_sums_breakdown() -> None:
    contexts = [
        _ctx("A", social_weighted_engagement=10.0, youtube_views_total=100.0),
        _ctx("B", social_weighted_engagement=100.0, youtube_views_total=10.0),
    ]
    result = score_clusters(contexts, _cfg())
    # B 의 social = 40, youtube = 0; A 의 social = 0, youtube = 25 (view diff dominates).
    assert 0.0 <= total_score(result["A"]) <= 100.0
    assert 0.0 <= total_score(result["B"]) <= 100.0
