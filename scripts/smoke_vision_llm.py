"""Phase 2 — GeminiVisionLLMClient + LocalJSONCache 실 smoke (3 post).

검증 목표:
1. `response_mime_type="application/json"` + `GarmentAnalysis.model_validate` 로 Phase 0
   관측 JSON 형식 버그 (1/20) 가 잡히는지 (retry 또는 1회 성공).
2. 2회차 호출 시 cache hit — API call 0, latency << 1s.

비용: 3 post × $0.0009 ≈ $0.003. 재실행은 캐시 덕에 0원.

사용:
    uv run python scripts/smoke_vision_llm.py \
        --sample-dir outputs/blob_cache_preset \
        --preset outputs/color_preset/color_preset.json \
        --cache-dir outputs/llm_cache --n 3 --seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

from vision.gemini_client import GeminiVisionLLMClient
from vision.llm_cache import LocalJSONCache, compute_cache_key

logger = logging.getLogger("smoke_vision_llm")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sample-dir", type=Path, default=Path("outputs/blob_cache_preset"))
    p.add_argument("--preset", type=Path, default=Path("outputs/color_preset/color_preset.json"))
    p.add_argument("--cache-dir", type=Path, default=Path("outputs/llm_cache"))
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model-id", type=str, default="gemini-2.5-flash")
    return p.parse_args()


def _load_preset(path: Path) -> list[dict[str, str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [{"name": e["name"], "hex": e["hex"]} for e in raw]


def _pick_sample(sample_dir: Path, n: int, seed: int) -> list[Path]:
    files = sorted(sample_dir.glob("*.jpg"))
    if not files:
        raise FileNotFoundError(f"no .jpg in {sample_dir}")
    rng = random.Random(seed)
    return rng.sample(files, min(n, len(files)))


def _run_pass(
    label: str,
    client: GeminiVisionLLMClient,
    cache: LocalJSONCache,
    samples: list[Path],
    preset: list[dict[str, str]],
) -> tuple[int, int, float]:
    """한 pass 실행. (parse_ok, cache_hit_count, total_latency) 반환."""
    parse_ok = 0
    cache_hits = 0
    total_latency = 0.0
    for idx, path in enumerate(samples, 1):
        image_bytes = path.read_bytes()
        cache_key = compute_cache_key(
            image_bytes,
            prompt_version=client.prompt_version,
            model_id=client.model_id,
        )
        had_cache = cache.get(cache_key) is not None
        start = time.perf_counter()
        try:
            analysis = client.extract_garment(image_bytes, preset=preset)
        except Exception as exc:
            logger.error(
                "[%s] %d/%d %s PARSE FAIL %s: %s",
                label, idx, len(samples), path.stem,
                type(exc).__name__, exc,
            )
            continue
        latency = time.perf_counter() - start
        total_latency += latency
        parse_ok += 1
        if had_cache:
            cache_hits += 1
        n_outfits = len(analysis.outfits)
        binary = analysis.is_india_ethnic_wear
        logger.info(
            "[%s] %d/%d %s ethnic=%s outfits=%d latency=%.3fs %s",
            label, idx, len(samples), path.stem[:20],
            binary, n_outfits, latency,
            "(cache hit)" if had_cache else "(API call)",
        )
    return parse_ok, cache_hits, total_latency


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()

    preset = _load_preset(args.preset)
    samples = _pick_sample(args.sample_dir, args.n, args.seed)
    logger.info("samples=%d preset=%d model=%s", len(samples), len(preset), args.model_id)

    cache = LocalJSONCache(
        args.cache_dir, model_id=args.model_id, prompt_version="v0.1"
    )
    client = GeminiVisionLLMClient(model_id=args.model_id, cache=cache)

    # Pass 1 — cache miss 시나리오 (처음 돌리거나 기존 캐시 무시)
    logger.info("=" * 50)
    logger.info("PASS 1 (expect cache miss = API calls)")
    logger.info("=" * 50)
    ok1, hits1, lat1 = _run_pass("P1", client, cache, samples, preset)

    # Pass 2 — 동일 input 재호출, 전부 cache hit 이어야
    logger.info("=" * 50)
    logger.info("PASS 2 (expect cache hit = 0 API calls)")
    logger.info("=" * 50)
    ok2, hits2, lat2 = _run_pass("P2", client, cache, samples, preset)

    logger.info("=" * 50)
    logger.info("SUMMARY")
    logger.info("  pass1: parse_ok=%d/%d cache_hits=%d latency=%.2fs", ok1, len(samples), hits1, lat1)
    logger.info("  pass2: parse_ok=%d/%d cache_hits=%d latency=%.2fs", ok2, len(samples), hits2, lat2)
    logger.info("=" * 50)

    # 종합 판정
    if ok1 < len(samples):
        logger.warning("pass1 parse failures — Gemini JSON integrity 검토 필요")
    if ok2 != len(samples) or hits2 != len(samples):
        logger.error("pass2 cache hit mismatch — LocalJSONCache 동작 이상")
        return 1
    if lat2 >= lat1 * 0.3:
        logger.warning("pass2 latency 가 pass1 의 30%% 이상 — cache 단축 효과 불분명")
    logger.info("SMOKE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
