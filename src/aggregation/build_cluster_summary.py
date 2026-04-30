"""TrendClusterSummary 빌더 (spec §5.4, §8.2).

흐름 (pure):
- enriched items 를 cluster_key 로 group — Phase β4 (2026-04-28) 부터 (item, share)
  tuple 로 등록. share = item_cluster_shares(item)[cluster_key] (= G×T×F cross-product
  × multiplier_ratio). 같은 item 이 multi-fan-out 되어도 각 cluster 의 list 에는 그
  cluster 안에서의 share 가 같이 실린다.
- 각 클러스터에서 DrilldownPayload 구성 (palette, distributions, top_posts, top_videos)
  — Phase β4 이후 share-weighted vote (silhouette/occasion/styling/cluster_palette/
  top_posts/top_influencers 모두 cluster 안 item share 가중).
- 외부에서 넘겨준 ScoreBreakdown / direction / lifecycle / data_maturity 와 결합
- TrendClusterSummary 리스트 반환

이 모듈은 스코어링 / 팔레트 계산을 직접 하지 않는다. 역할은 "재료 조립".
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date

from aggregation.brand_distribution import compute_brand_distribution
from aggregation.cluster_palette import build_cluster_palette
from aggregation.item_distribution_builder import (
    build_styling_combo_distribution,
    enriched_to_item_distribution,
)
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

# 한 cluster 안에서 (item, share) 페어. share 합 = cluster 내 mass (= effective_item_count).
ItemWithShare = tuple[EnrichedContentItem, float]


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
    # 옵션 C (2026-04-29): share-weighted fan-out (β2/β3/β4) 으로 fractional.
    post_count_total: float
    post_count_today: float
    avg_engagement_rate: float
    total_video_views: int
    top_video_ids: list[str]


def group_by_cluster(
    items: list[EnrichedContentItem],
) -> dict[str, list[ItemWithShare]]:
    """G×T×F cross-product fan-out grouping (Phase β3 + β4, spec §2.4).

    한 item 의 garment_type / technique / fabric distribution 의 cross-product 결과로
    share > 0 인 모든 cluster_key 에 그 item 을 (item, share) 페어로 등록 (β4).
    share 는 그 cluster 안에서의 item 기여도 (= item_cluster_shares 결과 = G×T×F ×
    multiplier_ratio). β2 의 score path 와 같은 cluster space → score path ↔ summary
    path align (sparse rep_with_summary 해소).

    N<3 (G/T/F 한 축이라도 비어있는) item 은 item_cluster_shares 가 빈 dict 반환 →
    어떤 cluster 에도 등장 X (mass preservation 정합). partial(g) 활성화 후엔 N<3 도
    multiplier 가중 share 로 등장.

    contract `trend_cluster_key` (winner 단일) 는 read 안 함 — ζ 에서 deprecate 예정.
    """
    grouped: dict[str, list[ItemWithShare]] = {}
    for item in items:
        shares = item_cluster_shares(enriched_to_item_distribution(item))
        if not shares:
            continue
        for cluster_key, share in shares.items():
            grouped.setdefault(cluster_key, []).append((item, share))
    return grouped


def _share_weighted_distribution(
    values_with_share: list[tuple[str | None, float]],
) -> DistributionMap:
    """Phase β4 — share-weighted distribution {value: pct}, 합=1.0 (정규화).

    None / share<=0 entry 는 미기여. 모든 weight 합이 0 이면 빈 dict.
    """
    totals: defaultdict[str, float] = defaultdict(float)
    for value, share in values_with_share:
        if value is None or share <= 0.0:
            continue
        totals[value] += share
    total_sum = sum(totals.values())
    if total_sum <= 0.0:
        return {}
    return {k: v / total_sum for k, v in totals.items()}


def _share_weighted_dict_aggregate(
    dists_with_share: list[tuple[dict[str, float], float]],
) -> DistributionMap:
    """로직 B (2026-04-29) — per-item distribution 을 cluster share 가중 합산 + 정규화.

    `dist[value] * share` 누적 → 합 0 이면 빈 dict, 아니면 정규화.
    각 per-item distribution 은 이미 mass=1 (또는 빈 dict) 로 정규화돼 있다고 가정 —
    `build_styling_combo_distribution` 등 build_distribution 출력. share<=0 / 빈 dist
    는 자연 미기여.
    """
    totals: defaultdict[str, float] = defaultdict(float)
    for dist, share in dists_with_share:
        if share <= 0.0 or not dist:
            continue
        for value, pct in dist.items():
            if pct <= 0.0:
                continue
            totals[value] += pct * share
    total_sum = sum(totals.values())
    if total_sum <= 0.0:
        return {}
    return {k: v / total_sum for k, v in totals.items()}


def _top_posts(
    items_with_share: list[ItemWithShare], limit: int
) -> list[str]:
    """engagement_raw_count × share 내림차순 → IG source_post_id 상위 N (β4 share-weighted)."""
    ig = [
        (item, share) for item, share in items_with_share
        if item.normalized.source == ContentSource.INSTAGRAM
    ]
    ig.sort(key=lambda pair: -pair[0].normalized.engagement_raw_count * pair[1])
    return [item.normalized.source_post_id for item, _ in ig[:limit]]


def _top_influencers(items_with_share: list[ItemWithShare], limit: int) -> list[str]:
    """engagement × share 상위 IG 계정 핸들 (account_handle 있는 것만, dedup, β4 가중)."""
    ig = [
        (item, share) for item, share in items_with_share
        if item.normalized.source == ContentSource.INSTAGRAM
        and item.normalized.account_handle
    ]
    ig.sort(key=lambda pair: -pair[0].normalized.engagement_raw_count * pair[1])
    seen: set[str] = set()
    result: list[str] = []
    for item, _ in ig:
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
    β4 부터는 호출자에서 cluster 안 item share 와 페어로 묶어 share-weighted vote 구성.
    """
    if not item.canonicals:
        return None
    rep_silhouette = item.canonicals[0].representative.silhouette
    return rep_silhouette.value if rep_silhouette else None


def make_drilldown(
    items_with_share: list[ItemWithShare],
    palette_cfg: PaletteConfig,
    top_post_limit: int,
    top_video_ids: list[str],
) -> DrilldownPayload:
    """한 클러스터의 (item, share) 페어 → DrilldownPayload (Phase β4 share-weighted).

    cluster 안 한 item 의 share = item_cluster_shares(item)[cluster_key]. 모든 distribution
    / palette / top_* ranking 이 그 share 로 가중. multi-fan-out item 은 자연스럽게 자기
    가 가장 많이 기여하는 cluster 에서 큰 vote, 적게 기여하는 cluster 에서 작은 vote.

    Color 3층 재설계 (2026-04-24) B3c: cluster palette 는 각 item 의 post_palette 를
    one-post-one-vote 로 flatten → ΔE76 merge → top 5 cap (β4 부터 vote 자체에 item
    share 곱셈, post_palette 내부 share 와 multiplier).

    B3d (2026-04-24): silhouette_distribution 은 canonicals[0].representative.silhouette
    에서 1 표 (post 당), β4 부터 그 표가 cluster 안 item share 로 가중. occasion / styling
    은 text 기반 속성이라 post-level item 1 표 (역시 share 가중).
    """
    _ = palette_cfg  # 3층 설계에서 magic 은 aggregation.cluster_palette 상수로 이동
    silhouettes_with_share = [
        (_canonical_silhouette_vote(item), share)
        for item, share in items_with_share
    ]
    occasions_with_share = [
        (item.occasion.value if item.occasion else None, share)
        for item, share in items_with_share
    ]
    # styling_combo 는 로직 B (2026-04-29) — vision-side derive_styling_from_outfit 결과를
    # canonical 단위 vote 로 합산한 per-item distribution × cluster share. text-only post
    # 는 build_styling_combo_distribution 안에서 text 채널로 자연 fallback.
    stylings_with_share = [
        (build_styling_combo_distribution(item), share)
        for item, share in items_with_share
    ]

    return DrilldownPayload(
        color_palette=build_cluster_palette([
            (item.post_palette, share) for item, share in items_with_share
        ]),
        silhouette_distribution=_share_weighted_distribution(silhouettes_with_share),
        occasion_distribution=_share_weighted_distribution(occasions_with_share),
        styling_distribution=_share_weighted_dict_aggregate(stylings_with_share),
        # 로직 C: log-scale 균등 분배 → threshold/top-N → 정규화. 빈 결과 시 빈 dict.
        brand_distribution=compute_brand_distribution(items_with_share),
        top_posts=_top_posts(items_with_share, top_post_limit),
        top_videos=list(top_video_ids),
        top_influencers=_top_influencers(items_with_share, top_post_limit),
    )


def build_summary(
    cluster_key: str,
    items_with_share: list[ItemWithShare],
    decision: ClusterDecision,
    target_date: date,
    palette_cfg: PaletteConfig,
    top_post_limit: int,
) -> TrendClusterSummary:
    """DrilldownPayload 를 포함한 최종 TrendClusterSummary 합성 (β4 share-weighted)."""
    drilldown = make_drilldown(
        items_with_share=items_with_share,
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
