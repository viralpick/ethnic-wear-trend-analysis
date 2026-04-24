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
from contracts.enriched import EnrichedContentItem
from exporters.write_json_output import write_enriched
from loaders.raw_loader import LocalSampleLoader, RawDailyBatch, RawLoader
from loaders.tsv_raw_loader import TsvRawLoader
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


def _load_raw(loader: RawLoader, target_date: date) -> RawDailyBatch:
    return loader.load_batch(target_date)


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
    """Case2: cluster 당 IG top-engagement 포스트 (spec §7.2).

    Color 3층 재설계 (2026-04-24) 로 post-level ColorInfo 제거, "color 아직 없는" 필터
    탈락. B3 에서 post_palette 채우기 루틴으로 재배선 예정.
    """
    by_cluster: dict[str, list[EnrichedContentItem]] = {}
    for item in enriched:
        if item.normalized.source != ContentSource.INSTAGRAM:
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
    """extraction 결과로 enriched 를 동결 상태 그대로 re-build (frozen Pydantic).

    Color 3층 재설계 B3a/B3b/B3d (2026-04-24): canonicals + post_palette 반영. post-level
    silhouette 단일값은 제거됨 — canonical silhouette 은 result.canonicals 복사로 자연 전달.
    """
    by_id = {r.source_post_id: r for r in results}
    updated: list[EnrichedContentItem] = []
    for item in enriched:
        result = by_id.get(item.normalized.source_post_id)
        if result is None or not result.canonicals:
            # 빈 결과는 기존 item (이전 Case 결과 포함) 를 덮지 않음.
            # Case1 / Case2 2-pass 간 의도치 않은 wipe 방어.
            updated.append(item)
            continue
        updated.append(item.model_copy(update={
            "canonicals": list(result.canonicals),
            "post_palette": list(result.post_palette),
        }))
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
    raw_loader: RawLoader | None = None,
) -> None:
    loader = raw_loader or LocalSampleLoader(settings.paths.sample_data)
    batch = _load_raw(loader, target_date)
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
    parser.add_argument(
        "--blob-cache", type=Path, default=None,
        help="pipeline_b + Azure Blob 모드: 다운로드 이미지 캐시 디렉토리. "
             "지정 시 AZURE_STORAGE_* 환경변수 필요.",
    )
    parser.add_argument(
        "--source", choices=["local", "tsv", "starrocks"], default="local",
        help="raw source loader 선택. local=sample JSON, tsv=TSV 파일, starrocks=StarRocks 직접.",
    )
    parser.add_argument(
        "--tsv-dir", type=Path, default=None,
        help="--source tsv 일 때 TSV 디렉토리 (기본: settings.paths.sample_data).",
    )
    # StarRocks 로드 모드 옵션
    parser.add_argument(
        "--window-mode", choices=["count", "date"], default=None,
        help="StarRocks 로드 모드. count=갯수 기준(LIMIT/OFFSET), date=기간 기준. "
             "미지정 시 settings.pipeline.window_mode 사용.",
    )
    parser.add_argument(
        "--page-size", type=int, default=None,
        help="count 모드: 배치당 포스트 수 (기본: settings.pipeline.page_size).",
    )
    parser.add_argument(
        "--page-index", type=int, default=None,
        help="count 모드: 몇 번째 배치인지 직접 지정. "
             "미지정 시 (target_date - collection_start).days.",
    )
    parser.add_argument(
        "--window-days", type=int, default=None,
        help="date 모드: target_date 기준 N일 이내 포스트 로드 "
             "(기본: settings.pipeline.window_days).",
    )
    parser.add_argument(
        "--llm", choices=["fake", "azure-openai"], default="fake",
        help="LLM 클라이언트 선택. azure-openai 는 AZURE_OPENAI_* 환경변수 필요.",
    )
    parser.add_argument(
        "--vision-llm", choices=["fake", "gemini"], default="fake",
        help="Vision LLM (BBOX+attributes) 클라이언트 선택. gemini 는 GEMINI_API_KEY 필요. "
             "--color-extractor pipeline_b 일 때만 사용됨.",
    )
    return parser.parse_args()


def _resolve_target_date(cli: str | None, settings_target: date | None) -> date:
    if cli:
        return datetime.strptime(cli, "%Y-%m-%d").date()
    return settings_target or date.today()


def _build_vision_llm(choice: str, settings: Settings):
    """CLI --vision-llm 플래그 기반 VisionLLMClient DI. lazy import 로 extras 격리."""
    from vision.llm_client import FakeVisionLLMClient  # noqa: I001
    if choice == "gemini":
        from vision.gemini_client import GeminiVisionLLMClient  # noqa: I001
        from vision.llm_cache import LocalJSONCache
        cfg = settings.vision_llm
        cache = LocalJSONCache(
            cfg.cache_dir,
            model_id=cfg.model_id,
            prompt_version=cfg.prompt_version,
        )
        return GeminiVisionLLMClient(
            model_id=cfg.model_id,
            prompt_version=cfg.prompt_version,
            cache=cache,
        )
    return FakeVisionLLMClient()


def _select_color_extractor(
    choice: str,
    settings: Settings,
    image_root: Path | None,
    *,
    vision_llm_choice: str = "fake",
    blob_cache: Path | None = None,
) -> ColorExtractor:
    """CLI flag 기반 ColorExtractor DI. pipeline_b 는 lazy import (vision extras 격리)."""
    if choice != "pipeline_b":
        return FakeColorExtractor(cfg=settings.vlm)

    from vision.color_family_preset import load_preset_views  # noqa: I001
    from vision.pipeline_b_adapter import PipelineBColorExtractor
    from vision.pipeline_b_extractor import load_models

    bundle = load_models()
    views = load_preset_views(settings.outfit_dedup.preset_path)
    vision_llm = _build_vision_llm(vision_llm_choice, settings)

    blob_downloader = None
    if blob_cache is not None:
        from loaders.blob_downloader import BlobDownloader  # noqa: I001
        blob_downloader = BlobDownloader.from_env()

    return PipelineBColorExtractor(
        bundle=bundle,
        cfg=settings.vision,
        vision_llm=vision_llm,
        llm_preset=views.llm_preset,
        matcher_entries=views.matcher_entries,
        family_map=views.family_map,
        dedup_cfg=settings.outfit_dedup,
        image_root=image_root,
        blob_downloader=blob_downloader,
        blob_cache_dir=blob_cache,
    )


def _select_llm_client(choice: str, settings: Settings) -> LLMClient:
    """CLI flag 기반 LLMClient DI. azure-openai 는 lazy import (openai 패키지 필요)."""
    if choice == "azure-openai":
        from attributes.azure_openai_llm_client import AzureOpenAILLMClient  # noqa: I001
        return AzureOpenAILLMClient(seed=DEFAULT_LLM_SEED)
    return FakeLLMClient(seed=DEFAULT_LLM_SEED)


def _select_raw_loader(
    choice: str,
    settings: Settings,
    tsv_dir: Path | None,
    window_mode: str | None = None,
    page_size: int | None = None,
    page_index: int | None = None,
    window_days: int | None = None,
    target_date: date | None = None,
) -> RawLoader:
    """CLI flag 기반 RawLoader DI."""
    if choice == "starrocks":
        from loaders.starrocks_raw_loader import StarRocksRawLoader  # noqa: I001 — lazy (db extras)
        mode = window_mode or settings.pipeline.window_mode
        p_size = page_size or settings.pipeline.page_size
        w_days = window_days or settings.pipeline.window_days
        # page_index 직접 지정 시 collection_start 를 역산해서 loader 에 주입.
        if page_index is not None and target_date is not None:
            from datetime import timedelta
            collection_start = target_date - timedelta(days=page_index)
        else:
            collection_start = settings.pipeline.collection_start_date
        return StarRocksRawLoader.from_env(
            window_mode=mode,
            page_size=p_size,
            window_days=w_days,
            collection_start=collection_start,
        )
    if choice == "tsv":
        return TsvRawLoader(tsv_dir or settings.paths.sample_data)
    return LocalSampleLoader(settings.paths.sample_data)


def main() -> None:
    args = _parse_args()
    settings = load_settings()
    target = _resolve_target_date(args.date, settings.pipeline.target_date)
    llm_client = _select_llm_client(args.llm, settings)
    color_extractor = _select_color_extractor(
        args.color_extractor, settings, args.image_root,
        vision_llm_choice=args.vision_llm,
        blob_cache=args.blob_cache,
    )
    raw_loader = _select_raw_loader(
        args.source, settings, args.tsv_dir,
        window_mode=args.window_mode,
        page_size=args.page_size,
        page_index=args.page_index,
        window_days=args.window_days,
        target_date=target,
    )
    run_pipeline(settings, target, llm_client, color_extractor, raw_loader=raw_loader)


if __name__ == "__main__":
    main()
