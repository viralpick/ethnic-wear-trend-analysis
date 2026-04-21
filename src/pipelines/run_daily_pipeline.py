"""Daily pipeline (spec §10.1 Step 1~5 통합).

흐름: load raw → normalize → rule extract → LLM fill → cluster assign →
      color extraction Case1 (unclassified IG) → Case2 (top-engagement IG per cluster) →
      enriched persist → scoring + aggregation + summaries persist.

ColorExtractor 선택 (Step D):
- 기본 `fake`: FakeColorExtractor (해시 결정론, vision extras 불필요)
- `--color-extractor pipeline_b`: PipelineBColorExtractor (YOLO+segformer+LAB KMeans,
  vision extras + 모델 다운 필요). `--image-root` 로 로컬 이미지 디렉토리 지정 가능.

이 스켈레톤은 DB/백엔드 POST/스케줄러 없음. 로컬 sample_data 만으로 끝까지 흐르는 baseline.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path

from attributes.extract_text_attributes import (
    AttributeExtractionState,
    extract_rule_based,
)
from attributes.extract_text_attributes_llm import (
    DEFAULT_LLM_SEED,
    FakeLLMClient,
    LLMClient,
    apply_llm_extraction,
)
from attributes.unknown_signal_tracker import run_tracker
from clustering.assign_trend_cluster import UNCLASSIFIED, assign_cluster
from contracts.common import ContentSource
from contracts.enriched import ColorInfo, EnrichedContentItem
from exporters.write_json_output import write_enriched
from loaders.raw_loader import LocalSampleLoader, RawDailyBatch
from normalization.normalize_content import normalize_batch
from pipelines.run_scoring_pipeline import score_and_export
from settings import Settings, load_settings
from utils.logging import get_logger
from vision.color_extractor import (
    ColorExtractionResult,
    ColorExtractor,
    FakeColorExtractor,
    run_color_extraction,
)

logger = get_logger(__name__)


def _load_raw(input_dir: Path, target_date: date) -> RawDailyBatch:
    return LocalSampleLoader(input_dir).load_batch(target_date)


def _assign_clusters(
    states: list[AttributeExtractionState],
) -> list[EnrichedContentItem]:
    cluster_totals: dict[str, int] = {}
    enriched: list[EnrichedContentItem] = []
    for state in states:
        key = assign_cluster(
            state.garment_type, state.technique, state.fabric, cluster_totals
        )
        enriched.append(state.to_enriched(cluster_key=key))
        if key != UNCLASSIFIED and "unknown" not in key:
            cluster_totals[key] = cluster_totals.get(key, 0) + 1
    return enriched


def _case1_targets(
    enriched: list[EnrichedContentItem], cap: int
) -> list[EnrichedContentItem]:
    """Case1: IG, garment_type 가 여전히 None 인 아이템 (spec §7.2)."""
    candidates = [
        e for e in enriched
        if e.normalized.source == ContentSource.INSTAGRAM and e.garment_type is None
    ]
    return candidates[:cap]


def _case2_targets(
    enriched: list[EnrichedContentItem], cap_per_cluster: int
) -> list[EnrichedContentItem]:
    """Case2: cluster 당 IG top-engagement 중 color 가 아직 없는 포스트 (spec §7.2)."""
    by_cluster: dict[str, list[EnrichedContentItem]] = {}
    for item in enriched:
        if item.normalized.source != ContentSource.INSTAGRAM:
            continue
        if item.color is not None:
            continue
        if not item.trend_cluster_key or item.trend_cluster_key == UNCLASSIFIED:
            continue
        by_cluster.setdefault(item.trend_cluster_key, []).append(item)

    picks: list[EnrichedContentItem] = []
    for cluster_items in by_cluster.values():
        cluster_items.sort(key=lambda i: -i.normalized.engagement_raw)
        picks.extend(cluster_items[:cap_per_cluster])
    return picks


def _apply_extraction_result(
    enriched: list[EnrichedContentItem],
    results: list[ColorExtractionResult],
) -> list[EnrichedContentItem]:
    """extraction 결과로 enriched 를 동결 상태 그대로 re-build (frozen Pydantic)."""
    by_id = {r.source_post_id: r for r in results}
    updated: list[EnrichedContentItem] = []
    for item in enriched:
        result = by_id.get(item.normalized.source_post_id)
        if result is None:
            updated.append(item)
            continue
        color = item.color
        has_rgb = result.r is not None and result.g is not None and result.b is not None
        if color is None and has_rgb:
            color = ColorInfo(
                r=result.r, g=result.g, b=result.b,
                name=result.name, family=result.family,
            )
        silhouette = item.silhouette or result.silhouette
        updated.append(
            item.model_copy(update={"color": color, "silhouette": silhouette})
        )
    return updated


def _run_color_extraction(
    enriched: list[EnrichedContentItem],
    extractor: ColorExtractor,
    settings: Settings,
) -> list[EnrichedContentItem]:
    case1 = _case1_targets(enriched, settings.vlm.case1_daily_cap)
    results1 = run_color_extraction(
        [e.normalized for e in case1], extractor, cap=settings.vlm.case1_daily_cap
    )
    enriched = _apply_extraction_result(enriched, results1)

    case2 = _case2_targets(enriched, settings.vlm.case2_per_cluster_cap)
    results2 = run_color_extraction([e.normalized for e in case2], extractor)
    enriched = _apply_extraction_result(enriched, results2)
    logger.info("color_extraction case1=%d case2=%d", len(case1), len(case2))
    return enriched


def run_pipeline(
    settings: Settings,
    target_date: date,
    llm_client: LLMClient,
    color_extractor: ColorExtractor,
) -> None:
    batch = _load_raw(settings.paths.sample_data, target_date)
    logger.info("loaded ig=%d yt=%d", len(batch.instagram), len(batch.youtube))

    normalized = normalize_batch(batch.instagram, batch.youtube)
    states = [extract_rule_based(item) for item in normalized]
    apply_llm_extraction(states, llm_client)
    enriched = _assign_clusters(states)

    enriched = _run_color_extraction(enriched, color_extractor, settings)

    write_enriched(
        settings.paths.outputs, target_date, enriched,
        filename=settings.export.enriched_filename,
    )
    run_tracker(normalized, settings.paths.outputs / "unknown_signals.json", target_date)

    score_and_export(enriched, settings, target_date, settings.paths.outputs)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Daily pipeline (Step 1~5 로컬)")
    parser.add_argument("--date", type=str, required=False,
                        help="ISO 날짜. 미지정 시 settings.pipeline.target_date or today.")
    parser.add_argument(
        "--color-extractor", choices=["fake", "pipeline_b"], default="fake",
        help="color/silhouette 추출기 선택. pipeline_b 는 `uv sync --extra vision` 필요.",
    )
    parser.add_argument(
        "--image-root", type=Path, default=None,
        help="pipeline_b 모드에서 URL basename 매핑용 이미지 디렉토리 (예: sample_data/image).",
    )
    return parser.parse_args()


def _resolve_target_date(cli: str | None, settings_target: date | None) -> date:
    if cli:
        return datetime.strptime(cli, "%Y-%m-%d").date()
    return settings_target or date.today()


def _select_color_extractor(
    choice: str, settings: Settings, image_root: Path | None
) -> ColorExtractor:
    """CLI flag 기반 ColorExtractor DI. pipeline_b 는 lazy import (vision extras 격리)."""
    if choice == "pipeline_b":
        # vision extras 미설치 환경에서는 이 import 자체가 ImportError — 즉시 실패가 맞음.
        from vision.pipeline_b_adapter import PipelineBColorExtractor  # noqa: I001
        from vision.pipeline_b_extractor import load_models
        bundle = load_models()
        return PipelineBColorExtractor(bundle, settings.vision, image_root=image_root)
    return FakeColorExtractor(cfg=settings.vlm)


def main() -> None:
    args = _parse_args()
    settings = load_settings()
    target = _resolve_target_date(args.date, settings.pipeline.target_date)
    llm_client = FakeLLMClient(seed=DEFAULT_LLM_SEED)
    color_extractor = _select_color_extractor(args.color_extractor, settings, args.image_root)
    run_pipeline(settings, target, llm_client, color_extractor)


if __name__ == "__main__":
    main()
