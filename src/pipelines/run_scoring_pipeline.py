"""Scoring / aggregation / export 파이프라인 (spec §10.1 Step 3 + Step 5 일부).

입력: outputs/{date}/enriched.json (이미 속성 추출 + 클러스터 배정 + VLM 보강까지 된 상태).
출력: outputs/{date}/summaries.json, outputs/{date}/payload.json, outputs/score_history.json.

이 파이프라인은 run_daily_pipeline 에서도 in-memory 로 재사용되는 함수를 export 한다
(score_and_export). entry point 는 CLI 용 main().
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from pydantic import TypeAdapter

from aggregation.build_cluster_summary import (
    ClusterDecision,
    build_summary,
    group_by_cluster,
)
from aggregation.item_distribution_builder import enriched_to_item_distribution
from aggregation.representative_builder import item_cluster_shares
from contracts.common import ContentSource, InstagramSourceType
from contracts.enriched import EnrichedContentItem
from contracts.output import TrendClusterSummary
from exporters.write_json_output import write_payload, write_summaries
from scoring.cluster_context import ClusterScoringContext
from scoring.compute_scores import score_clusters, total_score
from scoring.direction import (
    change_pct,
    classify_data_maturity,
    classify_direction,
    classify_lifecycle,
    classify_weekly_direction,
)
from scoring.score_history import ScoreHistory
from scoring.score_history_weekly import WeeklyScoreHistory
from settings import ScoringConfig, Settings, load_settings
from utils.logging import get_logger

logger = get_logger(__name__)

_TOP_POST_LIMIT = 10
_BOLLYWOOD_SOURCE = "bollywood_decode"
_enriched_adapter = TypeAdapter(list[EnrichedContentItem])


def _load_enriched(path: Path) -> list[EnrichedContentItem]:
    return _enriched_adapter.validate_json(path.read_bytes())


def _days_collected(start: date, target: date) -> int:
    delta = (target - start).days + 1
    return max(delta, 0)


def _display_name(cluster_key: str) -> str:
    parts = cluster_key.split("__")
    return " / ".join(p.replace("_", " ") for p in parts)


def _influencer_weight(followers: int, cfg: ScoringConfig) -> float:
    """spec §9.2 — 팔로워 수 → 인플루언서 티어 가중치."""
    t = cfg.influencer_tier_thresholds
    w = cfg.influencer_weights
    if followers >= t.mega:
        return w.mega
    if followers >= t.macro:
        return w.macro
    if followers >= t.mid:
        return w.mid
    return w.micro


def _source_type_weight(source_type: str | None, cfg: ScoringConfig) -> float:
    """M3.E — ig_source_type 분류별 Social multiplier.

    normalization 단계에서 파생된 `NormalizedContentItem.ig_source_type` 값을 받아
    `cfg.source_type_weights` 의 해당 multiplier 반환. 알 수 없는 값이면 1.0 (no-op).
    """
    w = cfg.source_type_weights
    mapping = {
        InstagramSourceType.INFLUENCER_FIXED.value: w.influencer_fixed,
        InstagramSourceType.HASHTAG_TRACKING.value: w.hashtag_tracking,
        InstagramSourceType.HASHTAG_HAUL.value: w.hashtag_haul,
        InstagramSourceType.BOLLYWOOD_DECODE.value: w.bollywood_decode,
    }
    return mapping.get(source_type or "", 1.0)


def _post_growth(today_count: float, window_counts: list[int]) -> float:
    """spec §9.2 Momentum — (오늘 - 7일평균) / 7일평균. 평균 0 이면 0.

    Phase β2: share-weighted today_count 가 float (history int 와 mixed precision OK).
    """
    if not window_counts:
        return 0.0
    avg = sum(window_counts) / len(window_counts)
    return (today_count - avg) / avg if avg > 0 else 0.0


def _hashtag_counts(items: list[EnrichedContentItem]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        for tag in item.normalized.hashtags:
            counts[tag] = counts.get(tag, 0) + 1
    return counts


@dataclass
class _ClusterAggregate:
    """Phase β2 share-weighted accumulator — share-fanned 누적값 모음."""
    social_weighted_engagement: float = 0.0
    youtube_video_count: float = 0.0
    youtube_views_total: float = 0.0
    festival_match: float = 0.0  # boost 곱하기 전 raw match 합 (× share)
    bollywood_count: float = 0.0
    post_count_today: float = 0.0  # share 합 = N=3 item 의 mass


def _active_festival_tags(target_date: date, cfg: ScoringConfig) -> set[str]:
    """target_date 가 들어있는 festival 의 tag set (lowercase). 없으면 빈 set."""
    for festival in cfg.cultural_festivals:
        if festival.window_start <= target_date <= festival.window_end:
            return {t.lower() for t in festival.tags}
    return set()


def _per_item_signals(
    item: EnrichedContentItem,
    cfg: ScoringConfig,
    festival_tags: set[str],
) -> tuple[float, float, float, float, float]:
    """1 item → (social_e, yt_count, yt_views, bolly, festival_match) — share fan-out 전.

    festival_tags 가 비면 festival_match = 0. boost 는 호출 측에서 한 번에 곱함.
    """
    n = item.normalized
    is_ig = n.source == ContentSource.INSTAGRAM
    is_yt = n.source == ContentSource.YOUTUBE
    social_e = (
        n.engagement_raw
        * _influencer_weight(n.account_followers, cfg)
        * _source_type_weight(n.ig_source_type, cfg)
    ) if is_ig else 0.0
    yt_count = 1.0 if is_yt else 0.0
    yt_views = float(n.engagement_raw) if is_yt else 0.0
    bolly = 1.0 if (is_ig and n.ig_source_type == _BOLLYWOOD_SOURCE) else 0.0
    festival_match = 1.0 if (
        festival_tags and any(h.lower() in festival_tags for h in n.hashtags)
    ) else 0.0
    return social_e, yt_count, yt_views, bolly, festival_match


def _accumulate_share_weighted(
    grouped: dict[str, list[EnrichedContentItem]],
    target_date: date,
    cfg: ScoringConfig,
) -> dict[str, _ClusterAggregate]:
    """spec §2.4 — G×T×F cross-product fan-out 으로 cluster 별 share-weighted 누적.

    N<3 (G/T/F 한 축이라도 없는) item 은 assign_shares 가 빈 dict 반환 → 어떤 cluster
    에도 기여 0 (mass preservation 결과: 분모는 N=3 item 만).
    """
    festival_tags = _active_festival_tags(target_date, cfg)
    festival_boost = cfg.cultural_festival_boost if festival_tags else 0.0
    acc: dict[str, _ClusterAggregate] = defaultdict(_ClusterAggregate)
    for items in grouped.values():
        for item in items:
            shares = item_cluster_shares(enriched_to_item_distribution(item))
            if not shares:
                continue
            social_e, yt_c, yt_v, bolly, fest = _per_item_signals(item, cfg, festival_tags)
            for cluster_key, share in shares.items():
                a = acc[cluster_key]
                a.social_weighted_engagement += social_e * share
                a.youtube_video_count += yt_c * share
                a.youtube_views_total += yt_v * share
                a.festival_match += fest * share
                a.bollywood_count += bolly * share
                a.post_count_today += share
    if festival_boost:
        for a in acc.values():
            a.festival_match *= festival_boost
    return dict(acc)


def _build_contexts(
    grouped: dict[str, list[EnrichedContentItem]],
    target_date: date,
    cfg: ScoringConfig,
    history: ScoreHistory,
) -> list[ClusterScoringContext]:
    """share-weighted enriched fan-out → ClusterScoringContext (Phase β2, spec §2.4).

    accounts 는 winner-keyed 로 유지 (account 식별 의미라 fan-out 시 중복으로 부풀려짐).
    post_count_total 은 history int + round(share-sum) — γ 에서 history schema 마이그 후 float.

    grouped 의 모든 winner key 는 zero-aggregate 라도 context 를 받음 — partial cluster
    (e.g. `kurta_set__unknown__cotton`, `unclassified`) 가 score_and_export 의
    `decisions[key]` 룩업에서 KeyError 가 나지 않도록. N<3 의미는 그대로 보존
    (모든 share-weighted 필드 = 0).
    """
    accumulators = _accumulate_share_weighted(grouped, target_date, cfg)
    for key in grouped:
        accumulators.setdefault(key, _ClusterAggregate())
    contexts: list[ClusterScoringContext] = []
    for key, a in accumulators.items():
        winner_ig = [
            i for i in grouped.get(key, [])
            if i.normalized.source == ContentSource.INSTAGRAM
        ]
        accounts = [
            i.normalized.account_handle for i in winner_ig
            if i.normalized.account_handle
        ]
        window_counts = history.get_post_count_history(
            key, target_date, cfg.momentum_window_days
        )
        avg_denom = a.post_count_today if a.post_count_today > 0 else 1.0
        contexts.append(
            ClusterScoringContext(
                cluster_key=key,
                social_weighted_engagement=a.social_weighted_engagement,
                youtube_video_count=a.youtube_video_count,
                youtube_views_total=a.youtube_views_total,
                youtube_view_growth=history.get_youtube_view_growth(key, target_date),
                cultural_festival_match=a.festival_match,
                cultural_bollywood_presence=a.bollywood_count,
                momentum_post_growth=_post_growth(a.post_count_today, window_counts),
                momentum_hashtag_velocity=history.get_hashtag_velocity(key, target_date),
                momentum_new_account_ratio=history.get_new_account_ratio(
                    key, target_date, cfg.new_account_window_days, accounts
                ),
                post_count_total=history.get_total_post_count(key) + round(a.post_count_today),
                post_count_today=a.post_count_today,
                avg_engagement_rate=(
                    a.social_weighted_engagement + a.youtube_views_total
                ) / avg_denom,
            )
        )
    return contexts


def _decide_clusters(
    contexts: list[ClusterScoringContext],
    settings: Settings,
    days_collected: int,
    grouped: dict[str, list[EnrichedContentItem]],
    target_date: date,
    history: ScoreHistory,
) -> dict[str, ClusterDecision]:
    breakdowns = score_clusters(contexts, settings.scoring)
    maturity = classify_data_maturity(days_collected, settings.scoring.data_maturity)
    decisions: dict[str, ClusterDecision] = {}

    for ctx in contexts:
        bd = breakdowns[ctx.cluster_key]
        total = total_score(bd)

        daily_base = history.get_daily_baseline(ctx.cluster_key, target_date)
        weekly_base = history.get_weekly_baseline(ctx.cluster_key, target_date)

        daily_change = change_pct(total, daily_base) if daily_base is not None else 0.0
        weekly_change = change_pct(total, weekly_base) if weekly_base is not None else 0.0

        decisions[ctx.cluster_key] = ClusterDecision(
            score_breakdown=bd,
            daily_direction=classify_direction(
                daily_change, settings.scoring.direction_threshold_pct
            ),
            weekly_direction=classify_weekly_direction(
                weekly_change, settings.scoring.direction_threshold_pct, days_collected
            ),
            daily_change_pct=daily_change,
            weekly_change_pct=weekly_change,
            lifecycle_stage=classify_lifecycle(
                total, ctx.post_count_total, "flat", settings.scoring.lifecycle
            ),
            data_maturity=maturity,
            display_name=_display_name(ctx.cluster_key),
            post_count_total=ctx.post_count_total,
            # Phase β2: ctx.post_count_today float → ClusterDecision int round.
            # summary path 형변경은 β3 (drilldown share-vote) 에서 float 노출.
            post_count_today=round(ctx.post_count_today),
            avg_engagement_rate=ctx.avg_engagement_rate,
            total_video_views=int(ctx.youtube_views_total),
            top_video_ids=[
                i.normalized.source_post_id
                for i in grouped[ctx.cluster_key]
                if i.normalized.source == ContentSource.YOUTUBE
            ][:_TOP_POST_LIMIT],
        )
    return decisions


def score_and_export(
    enriched: list[EnrichedContentItem],
    settings: Settings,
    target_date: date,
    output_root: Path,
) -> tuple[Path, list[TrendClusterSummary]]:
    """공용 진입점 — daily pipeline 에서도 호출."""
    history_path = output_root / "score_history.json"
    history = ScoreHistory(history_path)
    weekly_history_path = output_root / "score_history_weekly.json"
    weekly_history = WeeklyScoreHistory(weekly_history_path)

    grouped = group_by_cluster(enriched)
    contexts = _build_contexts(grouped, target_date, settings.scoring, history)
    days = _days_collected(settings.pipeline.collection_start_date, target_date)
    decisions = _decide_clusters(contexts, settings, days, grouped, target_date, history)

    summaries = [
        build_summary(
            cluster_key=key,
            items=grouped[key],
            decision=decisions[key],
            target_date=target_date,
            palette_cfg=settings.palette,
            top_post_limit=_TOP_POST_LIMIT,
        )
        for key in grouped
    ]
    summaries.sort(key=lambda s: -s.score)

    # score_history 갱신 후 저장 (다음 실행의 direction + momentum 계산 기반)
    # weekly_history 도 함께 갱신 — sink 활성화 여부와 무관하게 trajectory 데이터 누적
    # (Step 7.7 결정: sink_runner 가 자가 update 시 sink 분기에 따라 trajectory 가
    # 비결정론적이 됨. score_and_export 에서 한 번 update 하면 sink=none 도 누적).
    ctx_by_key = {ctx.cluster_key: ctx for ctx in contexts}
    for summary in summaries:
        ctx = ctx_by_key.get(summary.cluster_key)
        cluster_items = grouped.get(summary.cluster_key, [])
        ig_items = [i for i in cluster_items if i.normalized.source == ContentSource.INSTAGRAM]
        hashtag_counts = _hashtag_counts(cluster_items)
        accounts = [
            i.normalized.account_handle for i in ig_items
            if i.normalized.account_handle
        ]
        # Phase β2: ctx.post_count_today 가 float (share-weighted) 이라 history int
        # schema 와 단위 mismatch — round 로 보존. γ 에서 history schema float 마이그.
        post_count_int = round(ctx.post_count_today) if ctx else 0
        history.update(
            summary.cluster_key, target_date, summary.score,
            post_count=post_count_int,
            youtube_views_total=ctx.youtube_views_total if ctx else 0.0,
            hashtag_counts=hashtag_counts,
            accounts=accounts,
        )
        weekly_history.update_weekly(
            summary.cluster_key, target_date, summary.score,
            post_count=post_count_int,
            youtube_views_total=ctx.youtube_views_total if ctx else 0.0,
            hashtag_counts=hashtag_counts,
            accounts=accounts,
        )
    history.save()
    weekly_history.save()

    path = write_summaries(
        output_root, target_date, summaries, filename=settings.export.summaries_filename
    )
    write_payload(output_root, target_date, summaries)
    logger.info(
        "wrote summaries count=%d path=%s days=%d", len(summaries), path, days
    )
    return path, summaries


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scoring pipeline (Step 3-B)")
    parser.add_argument("--enriched", type=Path, required=False,
                        help="enriched.json 경로. 미지정 시 outputs/{target_date}/enriched.json.")
    parser.add_argument("--date", type=str, required=False,
                        help="ISO 날짜. 미지정 시 settings.pipeline.target_date 또는 today.")
    return parser.parse_args()


def _resolve_target_date(cli: str | None, settings_target: date | None) -> date:
    if cli:
        return datetime.strptime(cli, "%Y-%m-%d").date()
    return settings_target or date.today()


def main() -> None:
    args = _parse_args()
    settings = load_settings()
    target = _resolve_target_date(args.date, settings.pipeline.target_date)
    enriched_path = args.enriched or (
        settings.paths.outputs / target.isoformat() / settings.export.enriched_filename
    )
    enriched = _load_enriched(enriched_path)
    score_and_export(enriched, settings, target, settings.paths.outputs)


if __name__ == "__main__":
    main()
