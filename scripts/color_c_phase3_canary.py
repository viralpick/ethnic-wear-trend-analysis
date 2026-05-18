"""color.C Phase 3 — sample 100 frame mask-based verify (운영 흐름 일부).

spec: docs/color_c_illumination_spec.md (Phase 3 단계).

목적: Phase 2 의 frame-level surrogate 와 달리 **mask-based ΔE76** (garment 영역만) 측정.
운영 흐름 (pipeline_b_adapter._analyze_images) 의 verify guard 와 동일한 segformer wear
mask 사용 → verify reject rate 의 정확한 측정 + garment 단위 보정 효과 정량.

데이터: outputs/weekly_review/_video_thumbs/ 중 random sample 100 frame (Phase 2 와
동일 seed=42 → 같은 sample → frame-level vs garment-level 직접 비교).

측정 metric (mask-based):
- mask_pixel_count: garment mask 픽셀 수 (0 이면 garment 없음 — skip)
- triggered_rate: Phase 2 와 동일 detection rule
- verify_reject_rate (mask-based): apply_correction(verify on, segment_fn) 의 실제 reject 율
- garment_skew_a/b_reduction (median): mask 내 |a_mean| 의 감소량
- garment_deltae76 (median, p90): mask 내 LAB median 의 보정 magnitude

Phase 2 (frame-level surrogate) 와 비교 → mask 효과 정량.

family disagreement / cliff / cluster matching 은 별도 full canary 세션에서 (Gemini 호출).

출력: docs/color_c_phase3_sample100_results.md.
"""
from __future__ import annotations

import json
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from settings import (  # noqa: E402
    IlluminationCorrectionConfig,
    IlluminationCorrectionDetectionConfig,
    IlluminationCorrectionVerifyConfig,
)
from vision.illumination_correction import (  # noqa: E402
    _rgb_to_lab_float,
    apply_correction,
    compute_lab_stats,
    needs_correction,
)
from vision.pipeline_b_adapter import _compute_wear_mask  # noqa: E402
from vision.pipeline_b_extractor import SegBundle, load_models  # noqa: E402

THUMB_DIR = ROOT / "outputs" / "weekly_review" / "_video_thumbs"
def _out_md_path(p: int) -> Path:
    suffix = "" if p == 6 else f"_p{p}"
    return ROOT / "docs" / f"color_c_phase3_sample100_results{suffix}.md"


def _out_json_path(p: int) -> Path:
    suffix = "" if p == 6 else f"_p{p}"
    return ROOT / "outputs" / f"color_c_phase3_sample100{suffix}.json"

SAMPLE_SIZE = 100
SEED = 42
P_VALUE = int(sys.argv[1]) if len(sys.argv) > 1 else 6  # CLI: p=6 default
VERIFY_THRESHOLD = 30.0


@dataclass
class FrameResult:
    path: str
    has_mask: bool
    mask_pixel_count: int
    L_orig: float
    a_orig: float
    b_orig: float
    triggered: bool
    verify_accept: bool | None
    # mask-based metric
    garment_a_orig: float | None
    garment_b_orig: float | None
    garment_a_corr: float | None
    garment_b_corr: float | None
    garment_deltae76: float | None


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.array(img.convert("RGB"), dtype=np.uint8)


def _build_cfg() -> IlluminationCorrectionConfig:
    return IlluminationCorrectionConfig(
        enabled=True,
        minkowski_p=P_VALUE,
        detection=IlluminationCorrectionDetectionConfig(
            a_skew_threshold=8.0,
            b_skew_threshold=8.0,
            l_low_threshold=30.0,
            l_high_threshold=80.0,
        ),
        verify=IlluminationCorrectionVerifyConfig(
            enabled=True, deltae76_threshold=VERIFY_THRESHOLD
        ),
    )


def _garment_lab_median(rgb: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    lab = _rgb_to_lab_float(rgb)
    masked = lab[mask]
    return (
        float(np.median(masked[..., 0])),
        float(np.median(masked[..., 1])),
        float(np.median(masked[..., 2])),
    )


def _sample_frames() -> list[Path]:
    all_jpgs = sorted(THUMB_DIR.glob("*.jpg"))
    if len(all_jpgs) < SAMPLE_SIZE:
        raise SystemExit(f"insufficient frames: {len(all_jpgs)} < {SAMPLE_SIZE}")
    rng = random.Random(SEED)
    return rng.sample(all_jpgs, SAMPLE_SIZE)


def _process_frame(
    path: Path, bundle: SegBundle, cfg: IlluminationCorrectionConfig
) -> FrameResult:
    rgb = _load_rgb(path)
    L_o, a_o, b_o = compute_lab_stats(rgb)

    # segformer wear mask (원본 frame)
    mask = _compute_wear_mask(rgb, bundle)
    has_mask = mask is not None and bool(mask.any())
    mask_pixel_count = int(mask.sum()) if has_mask else 0

    triggered = needs_correction(L_o, a_o, b_o, cfg)
    if not triggered:
        return FrameResult(
            str(path.name), has_mask, mask_pixel_count,
            L_o, a_o, b_o, False, None,
            None, None, None, None, None,
        )

    # mask-based verify segment_fn — Phase 1 v2 와 동일 흐름
    segment_fn = lambda rgb_uint8: _compute_wear_mask(rgb_uint8, bundle)  # noqa: E731
    corrected, info = apply_correction(rgb, cfg, segment_fn=segment_fn)
    verify_accept = bool(info.get("verify_accept", True))

    if has_mask:
        gL_o, ga_o, gb_o = _garment_lab_median(rgb, mask)
        gL_c, ga_c, gb_c = _garment_lab_median(corrected, mask)
        de = float(
            np.sqrt(
                (gL_c - gL_o) ** 2 + (ga_c - ga_o) ** 2 + (gb_c - gb_o) ** 2
            )
        )
    else:
        ga_o = gb_o = ga_c = gb_c = de = None

    return FrameResult(
        str(path.name), has_mask, mask_pixel_count,
        L_o, a_o, b_o, True, verify_accept,
        ga_o, gb_o, ga_c, gb_c, de,
    )


def _summarize(results: list[FrameResult]) -> dict:
    n_total = len(results)
    n_with_mask = sum(1 for r in results if r.has_mask)
    n_triggered = sum(1 for r in results if r.triggered)
    n_triggered_with_mask = sum(1 for r in results if r.triggered and r.has_mask)
    n_verify_reject = sum(
        1 for r in results if r.triggered and r.verify_accept is False
    )

    # mask-based metric (triggered + has_mask only)
    skew_a_red: list[float] = []
    skew_b_red: list[float] = []
    deltae: list[float] = []
    for r in results:
        if not (r.triggered and r.has_mask and r.garment_deltae76 is not None):
            continue
        skew_a_red.append(abs(r.garment_a_orig) - abs(r.garment_a_corr))
        skew_b_red.append(abs(r.garment_b_orig) - abs(r.garment_b_corr))
        deltae.append(r.garment_deltae76)

    if deltae:
        deltae_sorted = sorted(deltae)
        de_med = statistics.median(deltae)
        de_p90 = deltae_sorted[int(0.9 * (len(deltae_sorted) - 1))]
        sa_med = statistics.median(skew_a_red)
        sb_med = statistics.median(skew_b_red)
    else:
        de_med = de_p90 = sa_med = sb_med = 0.0

    return {
        "sample_size": n_total,
        "p_value": P_VALUE,
        "verify_threshold": VERIFY_THRESHOLD,
        "mask_coverage": {
            "n_with_mask": n_with_mask,
            "rate": n_with_mask / n_total,
        },
        "trigger": {
            "n_triggered": n_triggered,
            "n_triggered_with_mask": n_triggered_with_mask,
            "rate": n_triggered / n_total,
        },
        "verify": {
            "n_reject": n_verify_reject,
            "rate_over_triggered": (
                n_verify_reject / n_triggered if n_triggered else 0.0
            ),
        },
        "garment_skew_a_reduction_median": sa_med,
        "garment_skew_b_reduction_median": sb_med,
        "garment_deltae76_median": de_med,
        "garment_deltae76_p90": de_p90,
    }


def _write_markdown(summary: dict, runtime_sec: float) -> None:
    s = summary
    lines: list[str] = []
    lines.append("# color.C Phase 3 — sample 100 frame mask-based canary")
    lines.append("")
    lines.append("spec: `docs/color_c_illumination_spec.md` Phase 3 단계 (sample 100 일부).")
    lines.append("")
    lines.append("## 데이터 / 셋업")
    lines.append("")
    lines.append(f"- frame pool: `outputs/weekly_review/_video_thumbs/` 8057 JPG")
    lines.append(f"- sample: {SAMPLE_SIZE} (random, seed={SEED}) — Phase 2 와 동일")
    lines.append(f"- p: {P_VALUE} (Phase 2 권장값 / spec 권장)")
    lines.append(f"- verify: enabled=True, deltae76_threshold={VERIFY_THRESHOLD:.0f}")
    lines.append("- segformer: segformer_b2_clothes (MPS device), UPPER∪LOWER∪DRESS wear mask")
    lines.append(f"- runtime: {runtime_sec:.1f}s")
    lines.append("")
    lines.append("## 결과 요약")
    lines.append("")
    lines.append(
        f"- mask 보유 frame: {s['mask_coverage']['n_with_mask']} / {s['sample_size']} "
        f"({s['mask_coverage']['rate'] * 100:.1f}%)"
    )
    lines.append(
        f"- detection trigger: {s['trigger']['n_triggered']} / {s['sample_size']} "
        f"({s['trigger']['rate'] * 100:.1f}%) — "
        f"이 중 mask 보유 {s['trigger']['n_triggered_with_mask']}"
    )
    lines.append(
        f"- **verify reject (mask-based)**: {s['verify']['n_reject']} / "
        f"{s['trigger']['n_triggered']} = "
        f"**{s['verify']['rate_over_triggered'] * 100:.1f}%**"
    )
    lines.append("")
    lines.append("## garment 단위 보정 효과 (triggered + has_mask)")
    lines.append("")
    lines.append("| metric | median |")
    lines.append("|---|---|")
    lines.append(f"| skew_a reduction (garment mask 내) | {s['garment_skew_a_reduction_median']:+.2f} |")
    lines.append(f"| skew_b reduction (garment mask 내) | {s['garment_skew_b_reduction_median']:+.2f} |")
    lines.append(f"| ΔE76 (garment LAB median, 보정 magnitude) | {s['garment_deltae76_median']:.2f} |")
    lines.append(f"| ΔE76 p90 | {s['garment_deltae76_p90']:.2f} |")
    lines.append("")
    lines.append("## Phase 2 vs Phase 3 비교 가이드")
    lines.append("")
    lines.append("- Phase 2 (frame-level surrogate, p=6): ΔE76 median 6.79 / reject 0%")
    lines.append(
        f"- Phase 3 (mask-based, p=6): ΔE76 median {s['garment_deltae76_median']:.2f} / "
        f"reject {s['verify']['rate_over_triggered'] * 100:.1f}%"
    )
    lines.append("- mask-based 가 frame-level 보다 크면: garment 가 보정 영향을 더 받음 (의류 색이 frame 평균에서 멀리)")
    lines.append("- mask-based 가 frame-level 보다 작으면: 보정이 배경 위주, garment 는 안정")
    lines.append("- verify reject 가 0 이상이면 mask 가드의 실 effectiveness 검증됨 (Phase 1 sanity 의 0건 보강)")
    lines.append("")
    lines.append("## Phase 3 enable 결정 5조건 vs 현재 측정")
    lines.append("")
    lines.append("| metric | baseline | 목표 | 현재 (sample 100) |")
    lines.append("|---|---|---|---|")
    lines.append("| family disagreement rate | 50.9% | < 30% | (full canary 필요 — Gemini 호출) |")
    lines.append(
        f"| ΔE76 분산 median | 17.7 | < 10 | "
        f"보정 magnitude {s['garment_deltae76_median']:.2f} (다른 metric — 분산은 multi-member 비교)"
    )
    lines.append(
        f"| verify 거부율 | — | < 20% | "
        f"**{s['verify']['rate_over_triggered'] * 100:.1f}%** {'✅' if s['verify']['rate_over_triggered'] < 0.2 else '⚠️'}"
    )
    lines.append(
        f"| 회귀 cliff 신규 발생 | 0 | 0 (절대) | (full canary 필요 — cluster matching)"
    )
    lines.append(
        f"| 운영 시간 증가 | 1× | < 30% | (full canary 필요 — Gemini round-trip 측정)"
    )
    lines.append("")
    lines.append("> 본 Phase 3 sample 100 은 verify 가드 + garment-level 보정 강도 정량까지. family/cliff/runtime 은 별도 full canary 세션 (Gemini Pass 1/2 + Stream Load) 필요.")
    _out_md_path(P_VALUE).parent.mkdir(parents=True, exist_ok=True)
    _out_md_path(P_VALUE).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    print(f"[phase3] sample {SAMPLE_SIZE} frames from {THUMB_DIR}")
    sample_paths = _sample_frames()

    print("[phase3] loading models (YOLO + segformer)")
    t0 = time.perf_counter()
    bundle = load_models()
    print(f"[phase3] models loaded in {time.perf_counter() - t0:.1f}s on device={bundle.device}")

    cfg = _build_cfg()
    results: list[FrameResult] = []
    t1 = time.perf_counter()
    for i, path in enumerate(sample_paths, 1):
        r = _process_frame(path, bundle, cfg)
        results.append(r)
        if i % 20 == 0:
            elapsed = time.perf_counter() - t1
            print(f"[phase3] {i}/{SAMPLE_SIZE} processed ({elapsed:.1f}s)")
    runtime = time.perf_counter() - t1
    print(f"[phase3] done {SAMPLE_SIZE} frames in {runtime:.1f}s")

    summary = _summarize(results)
    _out_json_path(P_VALUE).parent.mkdir(parents=True, exist_ok=True)
    _out_json_path(P_VALUE).write_text(
        json.dumps(
            {
                "summary": summary,
                "seed": SEED,
                "runtime_sec": runtime,
                "frames": [asdict(r) for r in results],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_markdown(summary, runtime)
    print(f"[phase3] wrote {_out_md_path(P_VALUE)}")
    print(f"[phase3] wrote {_out_json_path(P_VALUE)}")
    print()
    print(
        f"==> verify reject (mask-based, p={P_VALUE}): "
        f"{summary['verify']['n_reject']} / "
        f"{summary['trigger']['n_triggered']} "
        f"({summary['verify']['rate_over_triggered'] * 100:.1f}%)"
    )
    print(
        f"==> garment ΔE76 median: {summary['garment_deltae76_median']:.2f} "
        f"(Phase 2 frame-level: 6.79)"
    )


if __name__ == "__main__":
    main()
