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
from contracts.common import ContentSource, InstagramSourceType
from contracts.enriched import EnrichedContentItem
from contracts.output import TrendClusterSummary
from exporters.write_json_output import write_payload, write_summaries
from scoring.cluster_context import ClusterScoringContext
from scoring.compute_scores import score_clusters, total_score
from scoring.direction import (
    change_pct,
    classify_data_maturity,
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
    # Social score (rate-based, 2026-04-30): engagement_score (likes/F + comments*2/F)
    # × influencer_weight × source_type_weight. F = max(followers, 100).
    # rate 가 follower 자동 반영하지만 mega/macro tier weight 는 narrative 신뢰도 시그널
    # 로 유지 (user 결정).
    social_e = (
        n.engagement_score
        * _influencer_weight(n.account_followers, cfg)
        * _source_type_weight(n.ig_source_type, cfg)
    ) if is_ig else 0.0
    yt_count = 1.0 if is_yt else 0.0
    # YT raw 절대 view_count + like + comment*2 는 engagement_raw_count 에 보존됨.
    # spec §9.2 의 yt_views 는 절대값 노출이라 engagement_raw_count 사용.
    yt_views = float(n.engagement_raw_count) if is_yt else 0.0
    bolly = 1.0 if (is_ig and n.ig_source_type == _BOLLYWOOD_SOURCE) else 0.0
    festival_match = 1.0 if (
        festival_tags and any(h.lower() in festival_tags for h in n.hashtags)
    ) else 0.0
    return social_e, yt_count, yt_views, bolly, festival_match


def _accumulate_share_weighted(
    grouped: dict[str, list[tuple[EnrichedContentItem, float]]],
    target_date: date,
    cfg: ScoringConfig,
) -> dict[str, _ClusterAggregate]:
    """spec §2.4 — G×T×F cross-product fan-out 으로 cluster 별 share-weighted 누적.

    Phase β4 (2026-04-28): grouped entry 가 (item, share) tuple 이라 cluster_key 별로
    그 cluster 안에서의 item share 가 직접 주어진다. outer 가 cluster_key 단위라 multi-fan-out
    item 도 cluster 마다 자기 share 만 한 번씩 기여 — over-count 자연 차단 (PR #16 의
    id() dedup 은 signature 변경으로 자연 unused, 함께 제거).

    N<3 (G/T/F 한 축이라도 없는) item 은 group_by_cluster 단계에서 partial cluster 에
    multiplier_ratio 가중 share 로 등장 (partial(g) 활성화).
    """
    festival_tags = _active_festival_tags(target_date, cfg)
    festival_boost = cfg.cultural_festival_boost if festival_tags else 0.0
    acc: dict[str, _ClusterAggregate] = defaultdict(_ClusterAggregate)
    for cluster_key, items_with_share in grouped.items():
        for item, share in items_with_share:
            if share <= 0.0:
                continue
            social_e, yt_c, yt_v, bolly, fest = _per_item_signals(item, cfg, festival_tags)
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
    grouped: dict[str, list[tuple[EnrichedContentItem, float]]],
    target_date: date,
    cfg: ScoringConfig,
    history: ScoreHistory,
) -> list[ClusterScoringContext]:
    """share-weighted enriched fan-out → ClusterScoringContext (Phase β2 + β4 + γ, spec §2.4).

    accounts 는 fan-out cluster 마다 등장 (β4: cluster 안 item share>0 인 IG account).
    같은 account 가 multi-cluster 에 등장하는 건 정상 (cluster 별 trend 시그널 분리).
    post_count_total = history float + a.post_count_today 직접 합 (Phase γ: round 제거 →
    분자/분모 단위 정합. β1 effective_item_count_today ↔ history.get_total_post_count
    모두 float 단위).

    grouped 의 모든 cluster key 는 zero-aggregate 라도 context 를 받음 — partial cluster
    (e.g. `kurta_set__unknown__cotton`, `unclassified`) 가 score_and_export 의
    `decisions[key]` 룩업에서 KeyError 가 나지 않도록. N=0 의미는 그대로 보존
    (group_by_cluster 단계에서 자연 제외).
    """
    accumulators = _accumulate_share_weighted(grouped, target_date, cfg)
    for key in grouped:
        accumulators.setdefault(key, _ClusterAggregate())
    contexts: list[ClusterScoringContext] = []
    for key, a in accumulators.items():
        cluster_ig_items = [
            item for item, _ in grouped.get(key, [])
            if item.normalized.source == ContentSource.INSTAGRAM
        ]
        cluster_yt_items = [
            item for item, _ in grouped.get(key, [])
            if item.normalized.source == ContentSource.YOUTUBE
        ]
        ig_accounts = [
            item.normalized.account_handle for item in cluster_ig_items
            if item.normalized.account_handle
        ]
        # B-2 (M3.G/H 후): YT channel 을 별도 sub-signal 로 분리. normalize_content 가
        # video.channel 을 account_handle 에 매핑하므로 동일 필드에서 source 분기로 추출.
        yt_channels = [
            item.normalized.account_handle for item in cluster_yt_items
            if item.normalized.account_handle
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
                momentum_new_ig_account_ratio=history.get_new_ig_account_ratio(
                    key, target_date, cfg.new_account_window_days, ig_accounts
                ),
                momentum_new_yt_channel_ratio=history.get_new_yt_channel_ratio(
                    key, target_date, cfg.new_account_window_days, yt_channels
                ),
                post_count_total=history.get_total_post_count(key) + a.post_count_today,
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
    grouped: dict[str, list[tuple[EnrichedContentItem, float]]],
    target_date: date,
    history: ScoreHistory,
) -> dict[str, ClusterDecision]:
    breakdowns = score_clusters(contexts, settings.scoring)
    maturity = classify_data_maturity(days_collected, settings.scoring.data_maturity)
    decisions: dict[str, ClusterDecision] = {}

    for ctx in contexts:
        bd = breakdowns[ctx.cluster_key]
        total = total_score(bd)

        weekly_base = history.get_weekly_baseline(ctx.cluster_key, target_date)
        weekly_change = change_pct(total, weekly_base) if weekly_base is not None else 0.0

        decisions[ctx.cluster_key] = ClusterDecision(
            score_breakdown=bd,
            weekly_direction=classify_weekly_direction(
                weekly_change,
                settings.scoring.direction_threshold_pct,
                weekly_baseline_exists=weekly_base is not None,
            ),
            weekly_change_pct=weekly_change,
            lifecycle_stage=classify_lifecycle(
                total, ctx.post_count_total, "flat", settings.scoring.lifecycle
            ),
            data_maturity=maturity,
            display_name=_display_name(ctx.cluster_key),
            # 옵션 C (2026-04-29): ClusterDecision / TrendClusterSummary 가 float 화 →
            # round() 제거. score path / summary path / output contract 모두 같은
            # share-weighted fan-out 의 fractional mass 를 그대로 보존.
            post_count_total=ctx.post_count_total,
            post_count_today=ctx.post_count_today,
            avg_engagement_rate=ctx.avg_engagement_rate,
            total_video_views=int(ctx.youtube_views_total),
            top_video_ids=[
                item.normalized.source_post_id
                for item, _ in grouped[ctx.cluster_key]
                if item.normalized.source == ContentSource.YOUTUBE
            ][:_TOP_POST_LIMIT],
        )
    return decisions


def score_and_export(
    enriched: list[EnrichedContentItem],
    settings: Settings,
    target_date: date,
    output_root: Path,
    *,
    growth_factor_by_tag: dict[str, float] | None = None,
) -> tuple[Path, list[TrendClusterSummary]]:
    """공용 진입점 — daily pipeline 에서도 호출.

    Phase 3 (2026-04-30): growth_factor_by_tag — {url_short_tag: factor} dict.
    rep phase 가 시계열 growth rate 계산 후 source 별 정규화한 factor (1.0~2.0).
    canonical 단위 mass 분배 시 item_base_unit 으로 곱해져 cluster 가중. None →
    가중 없음 (모든 item factor=1.0).
    """
    history_path = output_root / "score_history.json"
    history = ScoreHistory(history_path)
    weekly_history_path = output_root / "score_history_weekly.json"
    weekly_history = WeeklyScoreHistory(weekly_history_path)

    grouped = group_by_cluster(enriched, item_base_units=growth_factor_by_tag)
    contexts = _build_contexts(grouped, target_date, settings.scoring, history)
    days = _days_collected(settings.pipeline.collection_start_date, target_date)
    decisions = _decide_clusters(contexts, settings, days, grouped, target_date, history)

    summaries = [
        build_summary(
            cluster_key=key,
            items_with_share=grouped[key],
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
        cluster_items = [item for item, _ in grouped.get(summary.cluster_key, [])]
        ig_items = [i for i in cluster_items if i.normalized.source == ContentSource.INSTAGRAM]
        yt_items = [i for i in cluster_items if i.normalized.source == ContentSource.YOUTUBE]
        hashtag_counts = _hashtag_counts(cluster_items)
        accounts = [
            i.normalized.account_handle for i in ig_items
            if i.normalized.account_handle
        ]
        # B-2 (M3.G/H 후): YT channel 도 history 에 누적 (new_yt_channel_ratio 추적용).
        channels = [
            i.normalized.account_handle for i in yt_items
            if i.normalized.account_handle
        ]
        # Phase γ: history schema float 마이그 후 share-weighted post_count 직접 적재.
        # ctx 가 None 인 경우 (sink-only 등) 0.0 fallback.
        post_count_float = ctx.post_count_today if ctx else 0.0
        history.update(
            summary.cluster_key, target_date, summary.score,
            post_count=post_count_float,
            youtube_views_total=ctx.youtube_views_total if ctx else 0.0,
            hashtag_counts=hashtag_counts,
            accounts=accounts,
            channels=channels,
        )
        weekly_history.update_weekly(
            summary.cluster_key, target_date, summary.score,
            post_count=post_count_float,
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


from pipelines._cli_common import resolve_target_date as _resolve_target_date  # noqa: E402, F401


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
