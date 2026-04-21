"""TrendClusterSummary 빌더 (spec §5.4, §8.2).

흐름 (pure):
- enriched items 를 cluster_key 로 group
- 각 클러스터에서 DrilldownPayload 구성 (palette, distributions, top_posts, top_videos)
- 외부에서 넘겨준 ScoreBreakdown / direction / lifecycle / data_maturity 와 결합
- TrendClusterSummary 리스트 반환

이 모듈은 스코어링 / 팔레트 계산을 직접 하지 않는다. 역할은 "재료 조립".
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date

from aggregation.color_palette import build_palette
from contracts.common import (
    ContentSource,
    DataMaturity,
    Direction,
    DistributionMap,
    LifecycleStage,
)
from contracts.enriched import EnrichedContentItem
from contracts.output import (
    DrilldownPayload,
    ScoreBreakdown,
    TrendClusterSummary,
)
from settings import PaletteConfig


@dataclass(frozen=True)
class ClusterDecision:
    """한 클러스터에 대한 스코어링/라이프사이클 결정 묶음."""
    score_breakdown: ScoreBreakdown
    daily_direction: Direction
    weekly_direction: Direction
    daily_change_pct: float
    weekly_change_pct: float
    lifecycle_stage: LifecycleStage
    data_maturity: DataMaturity
    display_name: str
    post_count_total: int
    post_count_today: int
    avg_engagement_rate: float
    total_video_views: int
    top_video_ids: list[str]


def group_by_cluster(
    items: list[EnrichedContentItem],
) -> dict[str, list[EnrichedContentItem]]:
    """None/unclassified 는 제외. 모두 exact 또는 partial 키 단위로 묶는다."""
    grouped: dict[str, list[EnrichedContentItem]] = {}
    for item in items:
        key = item.trend_cluster_key
        if not key:
            continue
        grouped.setdefault(key, []).append(item)
    return grouped


def _distribution(values: list[str | None]) -> DistributionMap:
    valid = [v for v in values if v]
    if not valid:
        return {}
    counter: Counter[str] = Counter(valid)
    total = sum(counter.values())
    return {k: v / total for k, v in counter.items()}


def _top_posts(
    items: list[EnrichedContentItem], limit: int
) -> list[str]:
    """engagement_raw 내림차순 → source_post_id 상위 N."""
    ig = [i for i in items if i.normalized.source == ContentSource.INSTAGRAM]
    ig.sort(key=lambda i: -i.normalized.engagement_raw)
    return [i.normalized.source_post_id for i in ig[:limit]]


def _top_influencers(items: list[EnrichedContentItem], limit: int) -> list[str]:
    """TODO(§8.2): 현재 normalized 뷰에 account_handle 이 없다.

    raw IG 포스트로부터 주입되도록 post→account 매핑을 build_cluster_summary 진입점에
    넘기는 확장을 M3 에서 추가 예정. 현재는 빈 리스트.
    """
    _ = (items, limit)
    return []


def make_drilldown(
    items: list[EnrichedContentItem],
    palette_cfg: PaletteConfig,
    top_post_limit: int,
    top_video_ids: list[str],
) -> DrilldownPayload:
    """한 클러스터의 enriched items → DrilldownPayload."""
    colors = [i.color for i in items if i.color is not None]
    palette = build_palette(colors, palette_cfg)

    silhouettes = [i.silhouette.value if i.silhouette else None for i in items]
    occasions = [i.occasion.value if i.occasion else None for i in items]
    stylings = [i.styling_combo.value if i.styling_combo else None for i in items]

    return DrilldownPayload(
        color_palette=palette,
        silhouette_distribution=_distribution(silhouettes),
        occasion_distribution=_distribution(occasions),
        styling_distribution=_distribution(stylings),
        top_posts=_top_posts(items, top_post_limit),
        top_videos=list(top_video_ids),
        top_influencers=_top_influencers(items, top_post_limit),
    )


def build_summary(
    cluster_key: str,
    items: list[EnrichedContentItem],
    decision: ClusterDecision,
    target_date: date,
    palette_cfg: PaletteConfig,
    top_post_limit: int,
) -> TrendClusterSummary:
    """DrilldownPayload 를 포함한 최종 TrendClusterSummary 합성."""
    drilldown = make_drilldown(
        items=items,
        palette_cfg=palette_cfg,
        top_post_limit=top_post_limit,
        top_video_ids=decision.top_video_ids,
    )
    from scoring.compute_scores import total_score  # 순환 방지용 지연 import

    return TrendClusterSummary(
        cluster_key=cluster_key,
        display_name=decision.display_name,
        date=target_date,
        score=total_score(decision.score_breakdown),
        score_breakdown=decision.score_breakdown,
        daily_direction=decision.daily_direction,
        weekly_direction=decision.weekly_direction,
        daily_change_pct=decision.daily_change_pct,
        weekly_change_pct=decision.weekly_change_pct,
        lifecycle_stage=decision.lifecycle_stage,
        data_maturity=decision.data_maturity,
        drilldown=drilldown,
        post_count_total=decision.post_count_total,
        post_count_today=decision.post_count_today,
        avg_engagement_rate=decision.avg_engagement_rate,
        total_video_views=decision.total_video_views,
    )
