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

from aggregation.cluster_palette import build_cluster_palette
from aggregation.item_distribution_builder import enriched_to_item_distribution
from aggregation.representative_builder import item_cluster_shares
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
    """G×T×F cross-product fan-out grouping (Phase β3, spec §2.4).

    한 item 의 garment_type / technique / fabric distribution 의 cross-product 결과로
    share > 0 인 모든 cluster_key 에 그 item 등록. β2 의 _accumulate_share_weighted 와
    같은 cluster space → score path ↔ summary path align (sparse rep_with_summary 해소).

    N<3 (G/T/F 한 축이라도 비어있는) item 은 item_cluster_shares 가 빈 dict 반환 →
    어떤 cluster 에도 등장 X (mass preservation 정합). partial(g) 활성화 후엔 N<3 도
    multiplier 가중 share 로 등장.

    contract `trend_cluster_key` (winner 단일) 는 read 안 함 — ζ 에서 deprecate 예정.
    """
    grouped: dict[str, list[EnrichedContentItem]] = {}
    for item in items:
        shares = item_cluster_shares(enriched_to_item_distribution(item))
        if not shares:
            continue
        for cluster_key in shares:
            grouped.setdefault(cluster_key, []).append(item)
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
    """engagement 상위 IG 계정 핸들 목록 (account_handle 있는 것만)."""
    from contracts.common import ContentSource  # noqa: PLC0415
    ig = [
        i for i in items
        if i.normalized.source == ContentSource.INSTAGRAM
        and i.normalized.account_handle
    ]
    ig.sort(key=lambda i: -i.normalized.engagement_raw)
    seen: set[str] = set()
    result: list[str] = []
    for item in ig:
        handle = item.normalized.account_handle
        if handle not in seen:
            seen.add(handle)
            result.append(handle)
        if len(result) >= limit:
            break
    return result


def _canonical_silhouette_vote(item: EnrichedContentItem) -> str | None:
    """B3d: post 당 1표. canonicals[0] 이 rep.area_ratio desc 0 번째이므로 post 내
    대표 outfit. rep.silhouette 이 None 이거나 canonicals 가 비어 있으면 미기여.
    """
    if not item.canonicals:
        return None
    rep_silhouette = item.canonicals[0].representative.silhouette
    return rep_silhouette.value if rep_silhouette else None


def make_drilldown(
    items: list[EnrichedContentItem],
    palette_cfg: PaletteConfig,
    top_post_limit: int,
    top_video_ids: list[str],
) -> DrilldownPayload:
    """한 클러스터의 enriched items → DrilldownPayload.

    Color 3층 재설계 (2026-04-24) B3c: cluster palette 는 각 item 의 post_palette 를
    one-post-one-vote 로 flatten → ΔE76 merge → top 5 cap. post_palette 가 비어 있는
    item (vision 비활성 smoke 등) 은 자연 빈 기여.

    B3d (2026-04-24): silhouette_distribution 은 canonicals[0].representative.silhouette
    에서 one-post-one-vote 로 집계 (post-level item.silhouette 제거). multi-outfit 이어도
    대표 canonical 1 개만 1표. occasion / styling 은 text 기반 속성이라 post-level 유지.
    """
    _ = palette_cfg  # 3층 설계에서 magic 은 aggregation.cluster_palette 상수로 이동
    silhouettes = [_canonical_silhouette_vote(i) for i in items]
    occasions = [i.occasion.value if i.occasion else None for i in items]
    stylings = [i.styling_combo.value if i.styling_combo else None for i in items]

    return DrilldownPayload(
        color_palette=build_cluster_palette([item.post_palette for item in items]),
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
