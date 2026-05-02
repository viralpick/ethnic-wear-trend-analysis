"""YouTube 팩터 raw 계산 (spec §9.2 YouTube).

spec 의 내부 `normalize(views)` / `normalize(view_growth)` 는 orchestrator 의 same-run minmax
한 단계로 합쳐진다 (user mandate: "no hidden transforms"). 여기서는 weighted sum 만 낸다.
"""
from __future__ import annotations

from scoring.cluster_context import ClusterScoringContext
from settings import ScoringConfig


def compute(cluster_ctx: ClusterScoringContext, cfg: ScoringConfig) -> float:
    """V × w_v + views × w_views + view_growth × w_growth.

    view_growth 음수는 0 으로 clamp (2026-05-02) — 음수 raw 를 그대로 두면 minmax 정규화
    에서 raw 가 0 인 cluster (= YT 데이터 부재) 가 max=1 로 잡혀 weight 25 가 그대로
    곱해지는 의도 외 동작 발생. raw youtube ≥ 0 보장으로 차단. 추가 가드: video_count
    + views_total 모두 0 인 cluster 는 short-circuit (가독성 + history view_growth 의
    edge case 안전망).
    """
    w = cfg.youtube_factor_weights
    if cluster_ctx.youtube_video_count == 0 and cluster_ctx.youtube_views_total == 0:
        return 0.0
    growth = max(0.0, cluster_ctx.youtube_view_growth)
    return (
        cluster_ctx.youtube_video_count * w.video_count
        + cluster_ctx.youtube_views_total * w.views
        + growth * w.view_growth
    )
