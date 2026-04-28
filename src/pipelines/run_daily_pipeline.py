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

from aggregation.item_distribution_builder import enriched_to_item_distribution
from aggregation.representative_builder import item_cluster_shares
from attributes.brand_registry import BrandRegistry, load_brand_registry
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


def _load_brand_registry_or_warn(settings: Settings) -> BrandRegistry | None:
    """settings.paths.brand_registry 가 있으면 로드. 없거나 파일 부재면 None + warn."""
    path = settings.paths.brand_registry
    if path is None:
        logger.info("brand_registry path not configured — brand 추출 skip")
        return None
    if not path.exists():
        logger.warning("brand_registry file not found path=%s — brand 추출 skip", path)
        return None
    return load_brand_registry(path)


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
    enriched: list[EnrichedContentItem],
    cap_per_cluster: int,
    min_share: float,
) -> list[EnrichedContentItem]:
    """Case2: cluster 당 IG top-engagement 포스트 (spec §7.2).

    Color 3층 재설계 (2026-04-24) 로 post-level ColorInfo 제거, "color 아직 없는" 필터
    탈락. B3 에서 post_palette 채우기 루틴으로 재배선 예정.

    ζ (2026-04-28): trend_cluster_shares.items() 순회 — share≥min_share 인 모든
    cluster 에 picking 후보 등록 (winner-only collapse 해소). N<3 partial 은 write 측에서
    {winner_key: 1.0} single-entry 로 채워져 자연스럽게 winner 단일 picking 유지.
    """
    by_cluster: dict[str, list[EnrichedContentItem]] = {}
    for item in enriched:
        if item.normalized.source != ContentSource.INSTAGRAM:
            continue
        for cluster_key, share in item.trend_cluster_shares.items():
            if not cluster_key or cluster_key == UNCLASSIFIED:
                continue
            if share < min_share:
                continue
            by_cluster.setdefault(cluster_key, []).append(item)

    picks: list[EnrichedContentItem] = []
    for cluster_items in by_cluster.values():
        cluster_items.sort(key=lambda i: -i.normalized.engagement_raw)
        picks.extend(cluster_items[:cap_per_cluster])
    return picks


def _vision_reassign_cluster_shares(item: EnrichedContentItem) -> dict[str, float]:
    """canonicals 채워진 후 G/T/F cross-product fan-out share dict 재계산 (ζ + 갭 #3 B).

    text-level partial 의 single-entry shares (= {winner_key: 1.0}) 를 vision-aware
    multi-cluster fan-out 으로 확장. representative_builder.item_cluster_shares 와 동일
    cross-product space — score path (β2) ↔ summary path (β4) ↔ picking path (ζ) 정합.

    N (resolved axis 수) 별 동작:
    - N=3 (G/T/F 모두 채워짐) → item_cluster_shares cross-product dict (winner-only
      collapse 해소 ★ ζ 본 목적). multiplier_ratio=1.0.
    - N<3 (partial) → 기존 shares 유지. assign_shares 가 N<3 일 때 multiplier_ratio 0.5/0.2
      를 곱해 share 가 작아져 picking_min_share=0.10 threshold 에 걸려 picking 손실
      방지. text-level winner single-entry ({winner_key: 1.0}) 가 그대로 picking 됨.
    - canonicals 비어있음 → 기존 shares 유지 (text-level 결정 보존).
    """
    if not item.canonicals:
        return item.trend_cluster_shares
    dist = enriched_to_item_distribution(item)
    if not dist.garment_type or not dist.technique or not dist.fabric:
        return item.trend_cluster_shares
    return item_cluster_shares(dist)


def _winner_key_from_shares(shares: dict[str, float]) -> str | None:
    """ζ (2026-04-28): shares dict → winner cluster_key (max share, alpha asc tiebreak).

    trend_cluster_key 는 shares 의 derived 대표값 — score 측은 shares 모두 fan-out 으로
    가중하지만 summary 매칭 / 단일 cluster reference 가 필요한 경우 (e.g. log message)
    위해 winner key 를 따로 들고 있다.
    """
    if not shares:
        return None
    return min(shares.items(), key=lambda kv: (-kv[1], kv[0]))[0]


def _apply_extraction_result(
    enriched: list[EnrichedContentItem],
    results: list[ColorExtractionResult],
) -> list[EnrichedContentItem]:
    """extraction 결과로 enriched 를 동결 상태 그대로 re-build (frozen Pydantic).

    Color 3층 재설계 B3a/B3b/B3d (2026-04-24): canonicals + post_palette 반영. post-level
    silhouette 단일값은 제거됨 — canonical silhouette 은 result.canonicals 복사로 자연 전달.

    ζ (2026-04-28): canonicals 가 채워지면 trend_cluster_shares + trend_cluster_key 둘
    다 vision-aware 재계산. shares 가 canonical (β2/β3/β4 의 fan-out 입력),
    trend_cluster_key 는 shares 의 max-share derived 대표값.
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
        new_item = item.model_copy(update={
            "canonicals": list(result.canonicals),
            "post_palette": list(result.post_palette),
        })
        new_shares = _vision_reassign_cluster_shares(new_item)
        new_key = _winner_key_from_shares(new_shares) or item.trend_cluster_key
        new_item = new_item.model_copy(update={
            "trend_cluster_shares": new_shares,
            "trend_cluster_key": new_key,
        })
        updated.append(new_item)
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

    # Defensive read — case2_picking_min_share 가 인스턴스에 없는 환경 (Pydantic
    # frozen + yaml partial override 의 미해소 이슈) 대비. 기본값은 VLMConfig 정의와 동일.
    case2 = _case2_targets(
        enriched,
        cap_per_cluster=settings.vlm.case2_per_cluster_cap,
        min_share=getattr(settings.vlm, "case2_picking_min_share", 0.10),
    )
    results2 = run_color_extraction([e.normalized for e in case2], extractor)
    enriched = _apply_extraction_result(enriched, results2)
    logger.info("color_extraction case1=%d case2=%d", len(case1), len(case2))
    return enriched


def _build_writer(sink: str):
    """sink 종류별 StarRocksWriter 인스턴스 생성. emit_*_only 가 공유."""
    if sink == "starrocks":
        from exporters.starrocks.stream_load_writer import (  # noqa: I001
            StarRocksStreamLoadWriter,
        )
        return StarRocksStreamLoadWriter.from_env()
    if sink == "starrocks_insert":
        from exporters.starrocks.insert_writer import (  # noqa: I001
            StarRocksInsertWriter,
        )
        return StarRocksInsertWriter.from_env()
    if sink == "dry_run":
        from exporters.starrocks.fake_writer import FakeStarRocksWriter  # noqa: I001
        return FakeStarRocksWriter()
    raise ValueError(f"unknown sink: {sink!r}")


def run_pipeline(
    settings: Settings,
    target_date: date,
    llm_client: LLMClient,
    color_extractor: ColorExtractor,
    raw_loader: RawLoader | None = None,
    *,
    sink: str = "none",
    phase: str = "all",
) -> None:
    """phase=all (기본, 전 단계) / phase=item (raw → enriched → items 적재).

    phase=representative 는 별도 진입점 (run_representative_phase) — raw 안 읽음.
    """
    if phase not in ("all", "item"):
        raise ValueError(f"run_pipeline phase must be all|item, got {phase!r}")

    loader = raw_loader or LocalSampleLoader(settings.paths.sample_data)
    batch = _load_raw(loader, target_date)
    logger.info("loaded ig=%d yt=%d", len(batch.instagram), len(batch.youtube))

    haul_tags = frozenset(settings.normalization.haul_tags)
    normalized = normalize_batch(batch.instagram, batch.youtube, haul_tags)
    brand_registry = _load_brand_registry_or_warn(settings)
    states = [extract_rule_based(item, brand_registry) for item in normalized]
    apply_llm_extraction(states, llm_client)
    enriched = _assign_clusters(states)

    enriched = _run_color_extraction(enriched, color_extractor, settings)

    write_enriched(
        settings.paths.outputs, target_date, enriched,
        filename=settings.export.enriched_filename,
    )
    run_tracker(normalized, settings.paths.outputs / "unknown_signals.json", target_date)

    summaries: list = []
    if phase == "all":
        _, summaries = score_and_export(
            enriched, settings, target_date, settings.paths.outputs
        )

    if sink not in ("starrocks", "starrocks_insert", "dry_run"):
        return

    writer = _build_writer(sink)
    if phase == "item":
        from exporters.starrocks.sink_runner import emit_items_only  # noqa: I001
        counts = emit_items_only(enriched, writer)
        if sink == "dry_run":
            from exporters.starrocks.dry_run import log_dry_run_summary  # noqa: I001
            log_dry_run_summary(writer, counts)
        return

    # phase == "all"
    from exporters.starrocks.sink_runner import emit_to_starrocks  # noqa: I001
    weekly_history_path = settings.paths.outputs / "score_history_weekly.json"
    counts = emit_to_starrocks(
        enriched, summaries, target_date, writer,
        weekly_history_path=weekly_history_path,
    )
    if sink == "dry_run":
        from exporters.starrocks.dry_run import log_dry_run_summary  # noqa: I001
        log_dry_run_summary(writer, counts)


def run_representative_phase(
    settings: Settings,
    *,
    enriched_glob: str,
    start_date: date,
    end_date: date,
    sink: str,
) -> None:
    """phase=representative — enriched JSON 글롭 로드 → 날짜 필터 → score → representative 적재.

    target_date 는 end_date 사용 (week_start_date / trajectory 조회 기준).
    """
    from pipelines.load_enriched import load_enriched_files, filter_by_date_range
    from exporters.starrocks.sink_runner import emit_representatives_only

    pool = load_enriched_files(enriched_glob)
    enriched = filter_by_date_range(pool, start_date=start_date, end_date=end_date)
    if not enriched:
        logger.warning(
            "run_representative_phase empty after filter glob=%s start=%s end=%s",
            enriched_glob, start_date, end_date,
        )
        return

    target_date = end_date  # week_start_date 계산 + score_history_weekly 갱신 기준
    # rep phase 는 raw 를 안 읽으므로 enriched.json 도 따로 써둬야 HTML 빌더가
    # 동일 윈도우 데이터를 읽을 수 있음 (filtered 314 개 등).
    write_enriched(
        settings.paths.outputs, target_date, enriched,
        filename=settings.export.enriched_filename,
    )
    _, summaries = score_and_export(
        enriched, settings, target_date, settings.paths.outputs
    )

    if sink not in ("starrocks", "starrocks_insert", "dry_run"):
        return

    writer = _build_writer(sink)
    weekly_history_path = settings.paths.outputs / "score_history_weekly.json"
    counts = emit_representatives_only(
        enriched, summaries, target_date, writer,
        weekly_history_path=weekly_history_path,
    )
    if sink == "dry_run":
        from exporters.starrocks.dry_run import log_dry_run_summary  # noqa: I001
        log_dry_run_summary(writer, counts)
    logger.info(
        "run_representative_phase done window=[%s,%s] enriched=%d representatives=%d",
        start_date, end_date, len(enriched),
        counts.get("representative_weekly", 0),
    )


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
    parser.add_argument(
        "--sink", choices=["none", "starrocks", "starrocks_insert", "dry_run"],
        default="none",
        help="결과 적재 sink. starrocks 는 STARROCKS_HOST/USER/PASSWORD/RESULT_DATABASE "
             "+ optional STARROCKS_STREAM_LOAD_PORT(8030) 환경변수 필요. "
             "starrocks_insert 는 9030 query 포트 INSERT VALUES fallback (8030 차단 환경용). "
             "dry_run 은 FakeStarRocksWriter 로 in-memory 적재 후 row 갯수 + sample 출력 (HTTP 안 보냄).",
    )
    parser.add_argument(
        "--text-workers", type=int, default=1,
        help="text LLM (azure-openai) batch parallel worker 수. "
             "기본 1 (sequential). Azure deployment TPM 한도 고려해 8 권장.",
    )
    parser.add_argument(
        "--vision-workers", type=int, default=1,
        help="vision pipeline (pipeline_b) per-post parallel worker 수. "
             "기본 1 (sequential). IO-bound (blob download + Gemini) 라 8~16 권장.",
    )
    parser.add_argument(
        "--phase", choices=["all", "item", "representative"], default="all",
        help="pipeline phase. all=raw→enriched→item+rep 모두, "
             "item=raw→enriched→item/group/object (rep 스킵), "
             "representative=enriched JSON 로드 → rep 만 적재 (raw 안 읽음).",
    )
    parser.add_argument(
        "--start-date", type=str, default=None,
        help="--phase representative: posted_at IST 기준 시작 (YYYY-MM-DD, inclusive).",
    )
    parser.add_argument(
        "--end-date", type=str, default=None,
        help="--phase representative: posted_at IST 기준 끝 (YYYY-MM-DD, inclusive). "
             "week_start_date / trajectory 조회 기준 target_date 로도 사용.",
    )
    parser.add_argument(
        "--enriched-glob", type=str, default=None,
        help="--phase representative: enriched.json 글롭 패턴. "
             "예: outputs/backfill/page_*_enriched.json",
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
    max_workers: int = 1,
) -> ColorExtractor:
    """CLI flag 기반 ColorExtractor DI. pipeline_b 는 lazy import (vision extras 격리)."""
    if choice != "pipeline_b":
        return FakeColorExtractor(cfg=settings.vlm)

    from vision.color_family_preset import load_preset_views  # noqa: I001
    from vision.pipeline_b_adapter import PipelineBColorExtractor
    from vision.pipeline_b_extractor import load_models

    # SceneFilterConfig 를 load_models 에 명시 전달 — yaml `enabled: true` 가 무시되던
    # leak 방어 (Phase 4, 2026-04-25). canonical path 는 별도 인자로 같은 SceneFilter
    # 객체 재사용 (CLIP ~600MB 중복 로드 회피).
    bundle = load_models(scene_filter_cfg=settings.vision.scene_filter)
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
        scene_filter=bundle.scene_filter,
        max_workers=max_workers,
    )


def _select_llm_client(
    choice: str, settings: Settings, max_workers: int = 1,
) -> LLMClient:
    """CLI flag 기반 LLMClient DI. azure-openai 는 lazy import (openai 패키지 필요)."""
    if choice == "azure-openai":
        from attributes.azure_openai_llm_client import AzureOpenAILLMClient  # noqa: I001
        return AzureOpenAILLMClient(seed=DEFAULT_LLM_SEED, max_workers=max_workers)
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


_LIVE_SINKS: frozenset[str] = frozenset({"starrocks", "starrocks_insert"})


def _validate_sink_extractor(
    sink: str, color_extractor: str, vision_llm: str, phase: str = "all",
) -> None:
    """`--sink starrocks{,_insert}` + fake 조합 reject (color_extractor / vision_llm 양쪽).

    silent stale 시나리오 (2026-04-27 SL smoke 발견):

    1. `--color-extractor fake` 면 FakeColorExtractor 가 의도적으로 빈 canonicals 반환
       (color_extractor.py:54-73, B3a 이후) → group/object 0 적재.
    2. `--vision-llm fake` 면 FakeVisionLLMClient 가 ethnic decision 채우지 않아
       canonical_extractor 의 pools=[] → `_attach_palette` skip → canonical/member palette
       빈 list → DB color_palette NULL.

    DB view 모델이 DUPLICATE KEY append-only + MAX(computed_at) 라 빈 적재가 latest 가
    되면 직전 정상 row 를 가리고 운영 query 는 stale data 노출. main argparse 직후 reject —
    `--sink none` (snapshot/test path) 은 그대로 허용.
    """
    if sink not in _LIVE_SINKS:
        return
    if phase == "representative":
        # phase=representative 는 raw 안 읽고 enriched JSON 만 쓰므로 extractor/llm 무관.
        return
    missing: list[str] = []
    if color_extractor != "pipeline_b":
        missing.append("--color-extractor pipeline_b")
    if vision_llm != "gemini":
        missing.append("--vision-llm gemini")
    if missing:
        raise SystemExit(
            f"--sink {sink} requires {' + '.join(missing)} "
            "(fake variants produce empty canonicals/palette — silent stale on DB view). "
            "Use --sink none for fake-based local runs."
        )


def main() -> None:
    args = _parse_args()
    _validate_sink_extractor(args.sink, args.color_extractor, args.vision_llm, args.phase)
    if (args.sink in ("starrocks", "starrocks_insert")
            or args.source == "starrocks" or args.blob_cache):
        # raw_loader / blob_downloader / stream_load_writer 모두 from_env() 패턴 —
        # CLI main 에서 한 번만 .env 로드 (편의). 이미 process env 에 있으면 no-op.
        from dotenv import load_dotenv  # noqa: I001 — optional dep
        load_dotenv()

    settings = load_settings()

    if args.phase == "representative":
        if not args.enriched_glob or not args.start_date or not args.end_date:
            raise SystemExit(
                "--phase representative 는 --enriched-glob / --start-date / --end-date 필수"
            )
        start = datetime.strptime(args.start_date, "%Y-%m-%d").date()
        end = datetime.strptime(args.end_date, "%Y-%m-%d").date()
        run_representative_phase(
            settings,
            enriched_glob=args.enriched_glob,
            start_date=start,
            end_date=end,
            sink=args.sink,
        )
        return

    target = _resolve_target_date(args.date, settings.pipeline.target_date)
    llm_client = _select_llm_client(args.llm, settings, max_workers=args.text_workers)
    color_extractor = _select_color_extractor(
        args.color_extractor, settings, args.image_root,
        vision_llm_choice=args.vision_llm,
        blob_cache=args.blob_cache,
        max_workers=args.vision_workers,
    )
    raw_loader = _select_raw_loader(
        args.source, settings, args.tsv_dir,
        window_mode=args.window_mode,
        page_size=args.page_size,
        page_index=args.page_index,
        window_days=args.window_days,
        target_date=target,
    )
    run_pipeline(
        settings, target, llm_client, color_extractor,
        raw_loader=raw_loader, sink=args.sink, phase=args.phase,
    )


if __name__ == "__main__":
    main()
