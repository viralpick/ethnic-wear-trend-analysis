"""StarRocks 4-tier 적재 orchestrator — Step 7.7 진입점.

`run_daily_pipeline.py --sink starrocks` 가 호출. enriched + summaries 를 받아
row_builder 4 함수로 변환 후 `StarRocksWriter` (Stream Load 또는 Fake) 에 순차 적재.

적재 순서 (의미상 의존, FK 없음):
  1. item                — post 단위, item_distribution 가 group/object 의 가중치 기여원
  2. canonical_group     — outfit (canonical) 단위
  3. canonical_object    — member (segformer instance) 단위
  4. representative_weekly — 주간 G/T/F 트렌드 결과

설계 결정 (Step 7.7, 2026-04-27 advisor 권고):
- WeeklyScoreHistory update 는 `score_and_export` 에서 이미 끝난 가정. sink_runner 는
  read-only 로 trajectory 만 조회. sink 활성화 여부에 trajectory 가 의존하면 안 됨.
- item_id ↔ source_post_id 매핑은 sink_runner 내부 dict 로 명시 유지. item_id 의
  string split 의존 회피 (item_id format 가 future 변경되어도 깨지지 않게).
- distributions 6 키 중 silhouette/occasion/styling_combo 는 summary.drilldown 에서,
  garment/fabric/technique 는 NULL (representative 단위 단일값이라 redundant — advisor #4).
- cluster_key ↔ representative_key 직접 string equality 매칭 (둘 다 `g__t__f` 포맷,
  N=3 only emit 이라 unknown 미포함 cluster 와 1:1).
- color_palette: cluster-level palette = summary.drilldown.color_palette 재사용
  (spec §2.3 — cluster_palette 가 곧 representative palette).
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from aggregation.item_distribution_builder import enriched_to_item_distribution
from aggregation.representative_builder import (
    ItemDistribution,
    RepresentativeContribution,
    aggregate_representatives,
    build_contributions,
    effective_item_count,
    top_evidence_per_source,
)
from contracts.common import ContentSource, PaletteCluster
from contracts.enriched import EnrichedContentItem
from contracts.output import TrendClusterSummary
from exporters.starrocks.row_builder import (
    build_group_rows,
    build_item_row,
    build_object_rows,
    build_representative_row,
)
from exporters.starrocks.writer import StarRocksWriter
from scoring.score_history_weekly import WeeklyScoreHistory, week_start_monday
from utils.logging import get_logger

logger = get_logger(__name__)

ITEM_TABLE = "item"
GROUP_TABLE = "canonical_group"
OBJECT_TABLE = "canonical_object"
REPRESENTATIVE_TABLE = "representative_weekly"

_DATETIME_FMT = "%Y-%m-%d %H:%M:%S"
_EVIDENCE_TOP_K = 4


def _format_datetime(dt: datetime) -> str:
    return dt.strftime(_DATETIME_FMT)


def _now_utc_str() -> str:
    return _format_datetime(datetime.now(timezone.utc).replace(tzinfo=None))


def _build_item_distributions(
    enriched: list[EnrichedContentItem],
) -> tuple[list[ItemDistribution], dict[str, str]]:
    """ItemDistribution list + (item_id → source_post_id) map.

    item_id 의 string split 안전성보다 명시적 map 유지 (item_id format 결합도 차단).
    """
    distributions: list[ItemDistribution] = []
    id_map: dict[str, str] = {}
    for item in enriched:
        dist = enriched_to_item_distribution(item)
        distributions.append(dist)
        id_map[dist.item_id] = item.normalized.source_post_id
    return distributions, id_map


def _group_contributions_by_key(
    contributions: list[RepresentativeContribution],
) -> dict[str, list[RepresentativeContribution]]:
    by_key: dict[str, list[RepresentativeContribution]] = defaultdict(list)
    for c in contributions:
        by_key[c.representative_key].append(c)
    return by_key


def _build_evidence(
    contributions: list[RepresentativeContribution],
    item_id_to_source_post_id: dict[str, str],
) -> tuple[list[str], list[str]]:
    """top_evidence_per_source 결과를 (IG post_ids, YT video_ids) 로 변환."""
    top = top_evidence_per_source(contributions, k=_EVIDENCE_TOP_K)
    ig_ids = [
        item_id_to_source_post_id[c.item_id]
        for c in top.get(ContentSource.INSTAGRAM, [])
        if c.item_id in item_id_to_source_post_id
    ]
    yt_ids = [
        item_id_to_source_post_id[c.item_id]
        for c in top.get(ContentSource.YOUTUBE, [])
        if c.item_id in item_id_to_source_post_id
    ]
    return ig_ids, yt_ids


def _summary_to_distributions(
    summary: TrendClusterSummary | None,
) -> dict[str, dict[str, float] | None]:
    """7 키 적재. garment_type/fabric 은 cluster_key 자체에 들어가 representative
    단일값으로 redundant 라 None. technique 은 Phase 2 (2026-04-30) 부터 cluster_key
    에서 빠져 drilldown distribution 으로만 노출 — 적재.
    """
    if summary is None:
        return {
            "silhouette": None,
            "occasion": None,
            "styling_combo": None,
            "garment_type": None,
            "fabric": None,
            "technique": None,
            "brand": None,
        }
    drill = summary.drilldown
    return {
        "silhouette": dict(drill.silhouette_distribution) or None,
        "occasion": dict(drill.occasion_distribution) or None,
        "styling_combo": dict(drill.styling_distribution) or None,
        "garment_type": None,
        "fabric": None,
        "technique": dict(drill.technique_distribution) or None,
        "brand": dict(drill.brand_distribution) or None,
    }


def _summary_color_palette(
    summary: TrendClusterSummary | None,
) -> list[PaletteCluster]:
    if summary is None:
        return []
    return list(summary.drilldown.color_palette)


def _summary_score_breakdown(
    summary: TrendClusterSummary | None,
) -> dict[str, float] | None:
    if summary is None:
        return None
    bd = summary.score_breakdown
    return {
        "social": bd.social,
        "youtube": bd.youtube,
        "cultural": bd.cultural,
        "momentum": bd.momentum,
    }


def _build_representative_rows(
    contributions: list[RepresentativeContribution],
    summaries_by_key: dict[str, TrendClusterSummary],
    item_id_to_source_post_id: dict[str, str],
    weekly_history: WeeklyScoreHistory,
    *,
    week_start_iso: str,
    target_date: date,
    computed_at: str,
    batch_effective_item_count: float,
) -> list[dict[str, Any]]:
    aggregates = aggregate_representatives(contributions)
    if not aggregates:
        return []

    grouped = _group_contributions_by_key(contributions)
    rows: list[dict[str, Any]] = []
    for agg in aggregates:
        summary = summaries_by_key.get(agg.representative_key)
        ig_ids, yt_ids = _build_evidence(
            grouped[agg.representative_key], item_id_to_source_post_id
        )
        trajectory = weekly_history.get_trajectory_12w(
            agg.representative_key, target_date
        )
        rows.append(
            build_representative_row(
                agg,
                week_start_date=week_start_iso,
                computed_at=computed_at,
                score_total=summary.score if summary is not None else None,
                score_breakdown=_summary_score_breakdown(summary),
                lifecycle_stage=(
                    summary.lifecycle_stage.value if summary is not None else None
                ),
                weekly_change_pct=(
                    summary.weekly_change_pct if summary is not None else None
                ),
                weekly_direction=(
                    summary.weekly_direction.value if summary is not None else None
                ),
                color_palette=_summary_color_palette(summary),
                distributions=_summary_to_distributions(summary),
                evidence_ig_post_ids=ig_ids,
                evidence_yt_video_ids=yt_ids,
                trajectory=trajectory,
                effective_item_count=batch_effective_item_count,
                display_name=summary.display_name if summary is not None else None,
            )
        )
    return rows


def emit_items_only(
    enriched: list[EnrichedContentItem],
    writer: StarRocksWriter,
    *,
    computed_at: str | None = None,
) -> dict[str, int]:
    """item / canonical_group / canonical_object 만 적재 — phase item 진입점.

    representative_weekly 는 별도 emit_representatives_only 가 처리.
    """
    if not enriched:
        logger.info("emit_items_only skip — empty enriched")
        return {ITEM_TABLE: 0, GROUP_TABLE: 0, OBJECT_TABLE: 0}

    computed = computed_at or _now_utc_str()
    item_distributions, _ = _build_item_distributions(enriched)

    item_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    object_rows: list[dict[str, Any]] = []
    for item, item_dist in zip(enriched, item_distributions, strict=True):
        item_rows.append(
            build_item_row(
                item,
                computed_at=computed,
                posted_at=_format_datetime(item.normalized.post_date),
                item_distribution=item_dist,
            )
        )
        group_rows.extend(build_group_rows(item, computed_at=computed))
        object_rows.extend(build_object_rows(item, computed_at=computed))

    counts: dict[str, int] = {
        ITEM_TABLE: writer.write_batch(ITEM_TABLE, item_rows),
        GROUP_TABLE: writer.write_batch(GROUP_TABLE, group_rows),
        OBJECT_TABLE: writer.write_batch(OBJECT_TABLE, object_rows),
    }
    logger.info(
        "emit_items_only done item=%d group=%d object=%d",
        counts[ITEM_TABLE], counts[GROUP_TABLE], counts[OBJECT_TABLE],
    )
    return counts


def emit_representatives_only(
    enriched: list[EnrichedContentItem],
    summaries: Iterable[TrendClusterSummary],
    target_date: date,
    writer: StarRocksWriter,
    *,
    weekly_history_path: Path,
    computed_at: str | None = None,
) -> dict[str, int]:
    """representative_weekly 만 적재 — phase representative 진입점.

    enriched 는 evidence (item_id → source_post_id) 매핑 + contribution 계산용.
    item/group/object 는 별도 emit_items_only 가 처리 (이미 적재된 상태 가정).
    """
    if not enriched:
        logger.info("emit_representatives_only skip — empty enriched")
        return {REPRESENTATIVE_TABLE: 0}

    computed = computed_at or _now_utc_str()
    week_start_iso = week_start_monday(target_date).isoformat()
    item_distributions, id_map = _build_item_distributions(enriched)
    summaries_by_key: dict[str, TrendClusterSummary] = {
        s.cluster_key: s for s in summaries
    }
    contributions = build_contributions(item_distributions)
    batch_eff_count = effective_item_count(item_distributions)
    weekly_history = WeeklyScoreHistory(weekly_history_path)
    representative_rows = _build_representative_rows(
        contributions,
        summaries_by_key,
        id_map,
        weekly_history,
        week_start_iso=week_start_iso,
        target_date=target_date,
        computed_at=computed,
        batch_effective_item_count=batch_eff_count,
    )

    aggregate_keys = {row["representative_key"] for row in representative_rows}
    summary_keys = set(summaries_by_key.keys())
    matched_keys = aggregate_keys & summary_keys
    coverage = len(matched_keys) / len(summary_keys) if summary_keys else 1.0
    rep_with_summary_rate = (
        len(matched_keys) / len(aggregate_keys) if aggregate_keys else 1.0
    )
    logger.info(
        "emit_representatives cluster_match agg=%d summary=%d matched=%d "
        "summary_coverage=%.3f rep_with_summary=%.3f week_start=%s",
        len(aggregate_keys), len(summary_keys),
        len(matched_keys), coverage, rep_with_summary_rate, week_start_iso,
    )

    counts = {
        REPRESENTATIVE_TABLE: writer.write_batch(
            REPRESENTATIVE_TABLE, representative_rows
        )
    }
    logger.info(
        "emit_representatives_only done representative=%d", counts[REPRESENTATIVE_TABLE]
    )
    return counts


def emit_to_starrocks(
    enriched: list[EnrichedContentItem],
    summaries: Iterable[TrendClusterSummary],
    target_date: date,
    writer: StarRocksWriter,
    *,
    weekly_history_path: Path,
    computed_at: str | None = None,
) -> dict[str, int]:
    """4 base 테이블 모두 적재 (phase=all backwards-compat 진입점).

    내부적으로 emit_items_only + emit_representatives_only 를 동일 computed_at 으로
    순차 호출. 새 코드는 phase 분리 함수 직접 호출 권장.
    """
    if not enriched:
        logger.info("emit_to_starrocks skip — empty enriched")
        return {ITEM_TABLE: 0, GROUP_TABLE: 0, OBJECT_TABLE: 0, REPRESENTATIVE_TABLE: 0}

    computed = computed_at or _now_utc_str()
    item_counts = emit_items_only(enriched, writer, computed_at=computed)
    rep_counts = emit_representatives_only(
        enriched, summaries, target_date, writer,
        weekly_history_path=weekly_history_path,
        computed_at=computed,
    )
    counts = {**item_counts, **rep_counts}
    logger.info(
        "starrocks_emit done item=%d group=%d object=%d representative=%d",
        counts[ITEM_TABLE], counts[GROUP_TABLE],
        counts[OBJECT_TABLE], counts[REPRESENTATIVE_TABLE],
    )
    return counts
