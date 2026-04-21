"""Scoring / aggregation / export 파이프라인 (spec §10.1 Step 3 + Step 5 일부).

입력: outputs/{date}/enriched.json (이미 속성 추출 + 클러스터 배정 + VLM 보강까지 된 상태).
출력: outputs/{date}/summaries.json.

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
from contracts.common import ContentSource
from contracts.enriched import EnrichedContentItem
from contracts.output import TrendClusterSummary
from exporters.write_json_output import write_summaries
from scoring.cluster_context import ClusterScoringContext
from scoring.compute_scores import score_clusters, total_score
from scoring.direction import (
    change_pct,
    classify_data_maturity,
    classify_direction,
    classify_lifecycle,
    classify_weekly_direction,
)
from settings import Settings, load_settings
from utils.logging import get_logger

logger = get_logger(__name__)

_TOP_POST_LIMIT = 10
_enriched_adapter = TypeAdapter(list[EnrichedContentItem])


def _load_enriched(path: Path) -> list[EnrichedContentItem]:
    return _enriched_adapter.validate_json(path.read_bytes())


def _days_collected(start: date, target: date) -> int:
    delta = (target - start).days + 1
    return max(delta, 0)


def _display_name(cluster_key: str) -> str:
    parts = cluster_key.split("__")
    return " / ".join(p.replace("_", " ") for p in parts)


def _build_contexts(
    grouped: dict[str, list[EnrichedContentItem]],
) -> list[ClusterScoringContext]:
    """enriched group → ClusterScoringContext. 역사적 비교는 이 skeleton 에서 0 으로 seed.

    TODO(§9.2 Social): influencer_tier 가중치 적용 — raw IG post 의 account_followers 가 필요.
    현재는 engagement_raw (weight=1) 합산만. M3 에서 raw 주입 통로가 생기면 가중치 반영.
    """
    contexts: list[ClusterScoringContext] = []
    for key, items in grouped.items():
        ig_items = [i for i in items if i.normalized.source == ContentSource.INSTAGRAM]
        yt_items = [i for i in items if i.normalized.source == ContentSource.YOUTUBE]
        ig_engagement = float(sum(i.normalized.engagement_raw for i in ig_items))
        yt_engagement = float(sum(i.normalized.engagement_raw for i in yt_items))
        contexts.append(
            ClusterScoringContext(
                cluster_key=key,
                social_weighted_engagement=ig_engagement,
                youtube_video_count=len(yt_items),
                youtube_views_total=yt_engagement,
                youtube_view_growth=0.0,              # TODO(§9.2): 역사 주입 (M3)
                cultural_festival_match=0.0,          # TODO(§9.2): festival 윈도우 매치 (M3)
                cultural_bollywood_presence=0.0,      # TODO(§9.2): IG source_type 주입 (M3)
                momentum_post_growth=0.0,             # TODO(§9.2): 7일 baseline (M3)
                momentum_hashtag_velocity=0.0,
                momentum_new_account_ratio=0.0,
                post_count_total=len(items),
                post_count_today=len(items),
                avg_engagement_rate=(ig_engagement + yt_engagement) / max(len(items), 1),
            )
        )
    return contexts


def _decide_clusters(
    contexts: list[ClusterScoringContext],
    settings: Settings,
    days_collected: int,
    grouped: dict[str, list[EnrichedContentItem]],
) -> dict[str, ClusterDecision]:
    breakdowns = score_clusters(contexts, settings.scoring)
    maturity = classify_data_maturity(days_collected, settings.scoring.data_maturity)
    decisions: dict[str, ClusterDecision] = {}
    for ctx in contexts:
        bd = breakdowns[ctx.cluster_key]
        total = total_score(bd)
        # TODO(§9.3/§9.4): daily/weekly baseline 주입 — skeleton 에서는 0 (flat).
        daily_change = change_pct(total, baseline=total)
        weekly_change = change_pct(total, baseline=total)
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
    grouped = group_by_cluster(enriched)
    contexts = _build_contexts(grouped)
    days = _days_collected(settings.pipeline.collection_start_date, target_date)
    decisions = _decide_clusters(contexts, settings, days, grouped)

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
    path = write_summaries(
        output_root, target_date, summaries, filename=settings.export.summaries_filename
    )
    logger.info("wrote summaries count=%d path=%s days=%d", len(summaries), path, days)
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
