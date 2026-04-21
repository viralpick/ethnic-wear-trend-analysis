"""M1 baseline 파이프라인 (spec §10.1 Step 1~3 + unknown 추적).

흐름:
  load → normalize → rule extract (2a) → LLM fill (2b, FakeLLMClient) → cluster assign
       → unknown signal tracking → persist enriched per-day

이 파이프라인은 아직 scoring/aggregation/export 를 포함하지 않는다 (Step 3-B 이후).
"""
from __future__ import annotations

import argparse
from datetime import date
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
from contracts.enriched import EnrichedContentItem
from loaders.raw_loader import LocalSampleLoader, RawDailyBatch
from normalization.normalize_content import normalize_batch
from settings import load_settings
from utils.io import write_json_atomic
from utils.logging import get_logger

logger = get_logger(__name__)


def _load_raw(input_dir: Path, target_date: date) -> RawDailyBatch:
    return LocalSampleLoader(input_dir).load_batch(target_date)


def _is_exact_key(cluster_key: str) -> bool:
    if cluster_key == UNCLASSIFIED:
        return False
    parts = cluster_key.split("__")
    return len(parts) == 3 and "unknown" not in parts


def _assign_and_build_enriched(
    states: list[AttributeExtractionState],
) -> list[EnrichedContentItem]:
    """states 순서대로 클러스터 배정 + EnrichedContentItem 구성.

    TODO(§10.1): cluster_totals 는 현재 빈 dict 에서 시작. v1 프로덕션에서는 전날
    summaries.json 로부터 누적된 cluster totals 를 시드한다.
    """
    cluster_totals: dict[str, int] = {}
    enriched: list[EnrichedContentItem] = []
    for state in states:
        key = assign_cluster(
            state.garment_type, state.technique, state.fabric, cluster_totals,
        )
        enriched.append(state.to_enriched(cluster_key=key))
        if _is_exact_key(key):
            cluster_totals[key] = cluster_totals.get(key, 0) + 1
    return enriched


def run_pipeline(
    input_dir: Path,
    output_dir: Path,
    target_date: date,
    llm_client: LLMClient,
) -> None:
    batch = _load_raw(input_dir, target_date)
    logger.info("loaded ig=%d yt=%d", len(batch.instagram), len(batch.youtube))

    normalized = normalize_batch(batch.instagram, batch.youtube)
    logger.info("normalized items=%d", len(normalized))

    states = [extract_rule_based(item) for item in normalized]
    resolved_after_rule = sum(1 for s in states if s.garment_type and s.technique and s.fabric)
    logger.info("rule_stage exact_resolved=%d partial_or_none=%d",
                resolved_after_rule, len(states) - resolved_after_rule)

    apply_llm_extraction(states, llm_client)

    enriched = _assign_and_build_enriched(states)
    by_key: dict[str, int] = {}
    for item in enriched:
        key = item.trend_cluster_key or UNCLASSIFIED
        by_key[key] = by_key.get(key, 0) + 1
    for key, count in sorted(by_key.items(), key=lambda kv: -kv[1]):
        logger.info("cluster=%s posts=%d", key, count)

    enriched_path = output_dir / target_date.isoformat() / "enriched.json"
    write_json_atomic(enriched_path, [e.model_dump(mode="json") for e in enriched])
    logger.info("wrote enriched=%d path=%s", len(enriched), enriched_path)

    signals = run_tracker(normalized, output_dir / "unknown_signals.json", target_date)
    logger.info("unknown_signals surfaced=%d (threshold=10)", len(signals))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attribute pipeline (M1 baseline)")
    parser.add_argument("--config", type=Path, default=None,
                        help="configs/local.yaml 경로. 미지정 시 CWD 상향 탐색.")
    parser.add_argument("--input", type=Path, default=None,
                        help="sample_data 디렉토리. 미지정 시 settings.paths.sample_data.")
    parser.add_argument("--output", type=Path, default=None,
                        help="outputs 디렉토리. 미지정 시 settings.paths.outputs.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    # NOTE(§settings): --config override 는 아직 구현 전. v1 에서 필요 시 load_settings(path=...).
    settings = load_settings()
    input_dir = args.input or settings.paths.sample_data
    output_dir = args.output or settings.paths.outputs
    llm_client = FakeLLMClient(seed=DEFAULT_LLM_SEED)
    run_pipeline(input_dir, output_dir, date.today(), llm_client)


if __name__ == "__main__":
    main()
