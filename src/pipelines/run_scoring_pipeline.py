"""Scoring / aggregation / export 파이프라인 (spec §10.1 Step 3 + Step 5 일부).

입력: outputs/{date}/enriched.json (이미 속성 추출 + 클러스터 배정 + VLM 보강까지 된 상태).
출력: outputs/{date}/summaries.json, outputs/{date}/payload.json, outputs/score_history.json.

이 파이프라인은 run_daily_pipeline 에서도 in-memory 로 재사용되는 함수를 export 한다
(score_and_export). entry point 는 CLI 용 main().
"""
from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path

from pydantic import TypeAdapter

from aggregation.build_cluster_summary import (
    ClusterDecision,
    build_summary,
    group_by_cluster,
)
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


def _festival_match_score(
    items: list[EnrichedContentItem],
    target_date: date,
    cfg: ScoringConfig,
) -> float:
    """spec §9.2 Cultural — target_date 가 festival 윈도우 안이면 태그 매치 포스트 수 × boost."""
    for festival in cfg.cultural_festivals:
        if festival.window_start <= target_date <= festival.window_end:
            tag_set = {t.lower() for t in festival.tags}
            matched = sum(
                1 for item in items
                if any(h.lower() in tag_set for h in item.normalized.hashtags)
            )
            return float(matched) * cfg.cultural_festival_boost
    return 0.0


def _post_growth(today_count: int, window_counts: list[int]) -> float:
    """spec §9.2 Momentum — (오늘 - 7일평균) / 7일평균. 평균 0 이면 0."""
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


def _build_contexts(
    grouped: dict[str, list[EnrichedContentItem]],
    target_date: date,
    cfg: ScoringConfig,
    history: ScoreHistory,
) -> list[ClusterScoringContext]:
    """enriched group → ClusterScoringContext (spec §9.2 Social + Cultural + Momentum 실 계산)."""
    contexts: list[ClusterScoringContext] = []
    for key, items in grouped.items():
        ig_items = [i for i in items if i.normalized.source == ContentSource.INSTAGRAM]
        yt_items = [i for i in items if i.normalized.source == ContentSource.YOUTUBE]

        # Social: 인플루언서 티어 × source_type 가중 engagement 합산 (spec §9.2, M3.E)
        ig_engagement = sum(
            i.normalized.engagement_raw
            * _influencer_weight(i.normalized.account_followers, cfg)
            * _source_type_weight(i.normalized.ig_source_type, cfg)
            for i in ig_items
        )
        yt_engagement = float(sum(i.normalized.engagement_raw for i in yt_items))

        # Cultural: festival 태그 매칭 + bollywood presence (spec §9.2, §9.5)
        festival_match = _festival_match_score(items, target_date, cfg)
        bollywood_count = float(sum(
            1 for i in ig_items
            if i.normalized.ig_source_type == _BOLLYWOOD_SOURCE
        ))

        # post_count: 오늘 배치 + 히스토리 누적 합산
        post_count_today = len(items)
        post_count_total = history.get_total_post_count(key) + post_count_today

        # Momentum: post_growth + hashtag_velocity + new_account_ratio (spec §9.2)
        window_counts = history.get_post_count_history(key, target_date, cfg.momentum_window_days)
        growth = _post_growth(post_count_today, window_counts)
        hashtag_vel = history.get_hashtag_velocity(key, target_date)
        accounts = [
            i.normalized.account_handle for i in ig_items
            if i.normalized.account_handle
        ]
        new_acct_ratio = history.get_new_account_ratio(
            key, target_date, cfg.new_account_window_days, accounts
        )

        # YouTube view growth (spec §9.2)
        view_growth = history.get_youtube_view_growth(key, target_date)

        contexts.append(
            ClusterScoringContext(
                cluster_key=key,
                social_weighted_engagement=ig_engagement,
                youtube_video_count=len(yt_items),
                youtube_views_total=yt_engagement,
                youtube_view_growth=view_growth,
                cultural_festival_match=festival_match,
                cultural_bollywood_presence=bollywood_count,
                momentum_post_growth=growth,
                momentum_hashtag_velocity=hashtag_vel,
                momentum_new_account_ratio=new_acct_ratio,
                post_count_total=post_count_total,
                post_count_today=post_count_today,
                avg_engagement_rate=(ig_engagement + yt_engagement) / max(len(items), 1),
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
            post_count_today=ctx.post_count_today,
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
        history.update(
            summary.cluster_key, target_date, summary.score,
            post_count=ctx.post_count_today if ctx else 0,
            youtube_views_total=ctx.youtube_views_total if ctx else 0.0,
            hashtag_counts=hashtag_counts,
            accounts=accounts,
        )
        weekly_history.update_weekly(
            summary.cluster_key, target_date, summary.score,
            post_count=ctx.post_count_today if ctx else 0,
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
