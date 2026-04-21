"""YouTube 팩터 raw 계산 (spec §9.2 YouTube).

spec 의 내부 `normalize(views)` / `normalize(view_growth)` 는 orchestrator 의 same-run minmax
한 단계로 합쳐진다 (user mandate: "no hidden transforms"). 여기서는 weighted sum 만 낸다.
"""
from __future__ import annotations

from scoring.cluster_context import ClusterScoringContext
from settings import ScoringConfig


def compute(cluster_ctx: ClusterScoringContext, cfg: ScoringConfig) -> float:
    """V × w_v + views × w_views + view_growth × w_growth. 음수 growth 도 그대로 반영."""
    w = cfg.youtube_factor_weights
    return (
        cluster_ctx.youtube_video_count * w.video_count
        + cluster_ctx.youtube_views_total * w.views
        + cluster_ctx.youtube_view_growth * w.view_growth
    )
