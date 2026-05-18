"""color.C Phase 2 — Minkowski p ablation (shades-of-gray).

spec: docs/color_c_illumination_spec.md (결정 2 Phase 2 단계).

목적: p ∈ {1, 4, 6, 8, 16} 중 16w 백필 frame 의 illumination bias 감소 효과 비교 →
최적 p 재선정 또는 6 권장값 검증.

데이터: outputs/weekly_review/_video_thumbs/ (16w 백필 video frame 8057장) 중 random sample
100 frame (seed=42).

측정 metric (frame-level surrogate — mask 없는 빠른 비교):
- trigger_rate: detection rule (a/b skew |·|>8 또는 L<30/L>80) 발동 비율. p 와 무관 동일.
- skew_reduction_a/b: |a_mean|_before - |a_mean|_after, b 동일. 보정 효과의 직접 측정 (median).
- L_shift_abs: |L_mean_before - L_mean_after| (median). 과도한 L 변화 = 의류 색 왜곡 risk.
- frame_deltae76: √((L_c-L_o)² + (a_c-a_o)² + (b_c-b_o)²). 보정 강도 magnitude.
- verify_reject_proxy: frame ΔE76 > 30 비율 (mask-based verify 의 proxy).

mask-based verify (segformer 동원) 는 Phase 3 full canary 에서 수행 — Phase 2 는 p 비교가
목적이라 frame-level surrogate 만 사용.

출력: docs/color_c_phase2_p_ablation_results.md (markdown table).
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
    compute_lab_stats,
    needs_correction,
    shades_of_gray,
)

THUMB_DIR = ROOT / "outputs" / "weekly_review" / "_video_thumbs"
OUT_MD = ROOT / "docs" / "color_c_phase2_p_ablation_results.md"
OUT_JSON = ROOT / "outputs" / "color_c_phase2_p_ablation.json"

P_VALUES = [1, 4, 6, 8, 16]
SAMPLE_SIZE = 100
SEED = 42
VERIFY_THRESHOLD = 30.0  # spec 결정 4 — garment LAB median ΔE76 임계


@dataclass
class FrameMetric:
    path: str
    triggered: bool
    L_mean: float
    a_mean: float
    b_mean: float
    # p 별 결과는 dict로 별도 저장


@dataclass
class PMetric:
    p: int
    n_triggered: int
    n_total: int
    skew_a_reduction_median: float
    skew_b_reduction_median: float
    L_shift_abs_median: float
    deltae76_median: float
    deltae76_p90: float
    verify_reject_rate: float
    runtime_sec: float


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.array(img.convert("RGB"), dtype=np.uint8)


def _build_cfg(p: int) -> IlluminationCorrectionConfig:
    return IlluminationCorrectionConfig(
        enabled=True,
        minkowski_p=p,
        detection=IlluminationCorrectionDetectionConfig(
            a_skew_threshold=8.0,
            b_skew_threshold=8.0,
            l_low_threshold=30.0,
            l_high_threshold=80.0,
        ),
        verify=IlluminationCorrectionVerifyConfig(
            enabled=False, deltae76_threshold=VERIFY_THRESHOLD
        ),
    )


def _sample_frames() -> list[Path]:
    all_jpgs = sorted(THUMB_DIR.glob("*.jpg"))
    if len(all_jpgs) < SAMPLE_SIZE:
        raise SystemExit(f"insufficient frames: {len(all_jpgs)} < {SAMPLE_SIZE}")
    rng = random.Random(SEED)
    return rng.sample(all_jpgs, SAMPLE_SIZE)


def _measure_p(
    p: int,
    frames: list[tuple[Path, np.ndarray, tuple[float, float, float], bool]],
) -> PMetric:
    """각 p 별 한 번 돈다. frames 는 (path, rgb, lab_stats_orig, triggered) tuple list."""
    skew_a_red: list[float] = []
    skew_b_red: list[float] = []
    L_shifts: list[float] = []
    delta_e: list[float] = []
    n_triggered = 0
    n_verify_reject = 0

    t0 = time.perf_counter()
    for _path, rgb, (L_o, a_o, b_o), triggered in frames:
        if not triggered:
            continue
        n_triggered += 1
        corrected = shades_of_gray(rgb, p=p)
        L_c, a_c, b_c = compute_lab_stats(corrected)
        skew_a_red.append(abs(a_o) - abs(a_c))
        skew_b_red.append(abs(b_o) - abs(b_c))
        L_shifts.append(abs(L_o - L_c))
        de = float(np.sqrt((L_c - L_o) ** 2 + (a_c - a_o) ** 2 + (b_c - b_o) ** 2))
        delta_e.append(de)
        if de > VERIFY_THRESHOLD:
            n_verify_reject += 1
    runtime = time.perf_counter() - t0

    if not delta_e:
        return PMetric(p, 0, len(frames), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, runtime)

    delta_e_sorted = sorted(delta_e)
    p90_idx = int(0.9 * (len(delta_e_sorted) - 1))
    return PMetric(
        p=p,
        n_triggered=n_triggered,
        n_total=len(frames),
        skew_a_reduction_median=statistics.median(skew_a_red),
        skew_b_reduction_median=statistics.median(skew_b_red),
        L_shift_abs_median=statistics.median(L_shifts),
        deltae76_median=statistics.median(delta_e),
        deltae76_p90=delta_e_sorted[p90_idx],
        verify_reject_rate=n_verify_reject / n_triggered,
        runtime_sec=runtime,
    )


def _write_markdown(metrics: list[PMetric], trigger_rate: float) -> None:
    lines: list[str] = []
    lines.append("# color.C Phase 2 — Minkowski p ablation")
    lines.append("")
    lines.append("spec: `docs/color_c_illumination_spec.md` 결정 2 Phase 2 단계.")
    lines.append("")
    lines.append(f"## 데이터 / 셋업")
    lines.append("")
    lines.append(f"- frame pool: `outputs/weekly_review/_video_thumbs/` 8057 JPG")
    lines.append(f"- sample: {SAMPLE_SIZE} (random, seed={SEED})")
    lines.append(f"- p 후보: {P_VALUES}")
    lines.append(f"- detection rule: |a|>8 또는 |b|>8 또는 L<30 또는 L>80 (Phase 1 동일)")
    lines.append(
        f"- verify threshold (frame-level surrogate): ΔE76 > {VERIFY_THRESHOLD:.0f}"
    )
    lines.append("- mask-based verify (segformer): Phase 3 에서 수행")
    lines.append("")
    lines.append(f"## detection trigger rate")
    lines.append("")
    lines.append(
        f"- {trigger_rate * 100:.1f}% (p 와 무관 — detection rule 동일). "
        f"spec 예상 50~60% 와 비교."
    )
    lines.append("")
    lines.append("## p ablation metric")
    lines.append("")
    lines.append(
        "| p | trigger N | skew_a↓ median | skew_b↓ median | L_shift median | ΔE76 median | ΔE76 p90 | verify reject % | runtime |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for m in metrics:
        lines.append(
            f"| {m.p} | {m.n_triggered}/{m.n_total} "
            f"| {m.skew_a_reduction_median:+.2f} "
            f"| {m.skew_b_reduction_median:+.2f} "
            f"| {m.L_shift_abs_median:.2f} "
            f"| {m.deltae76_median:.2f} "
            f"| {m.deltae76_p90:.2f} "
            f"| {m.verify_reject_rate * 100:.1f}% "
            f"| {m.runtime_sec:.2f}s |"
        )
    lines.append("")
    lines.append("## 해석 가이드")
    lines.append("")
    lines.append("- **skew_a / skew_b reduction (+)**: 보정 후 a/b skew 절대값이 감소한 양.")
    lines.append("  양수 = 보정 효과 (gray balance 회복). 음수 = 보정이 오히려 skew 증가.")
    lines.append("- **L_shift**: 보정 전후 L mean 의 absolute 차. 클수록 명도 손상 risk.")
    lines.append("- **ΔE76**: frame LAB mean 의 보정 magnitude. p 작을수록 (gray-world 에 가까울수록) ↑.")
    lines.append("- **verify reject (surrogate)**: ΔE76 > 30 = 과도한 보정 후보. mask-based 와 다를 수 있음.")
    lines.append("")
    lines.append("## 권장값 판단 기준")
    lines.append("")
    lines.append("1. skew_a/skew_b reduction 이 충분히 큰 p (effect size)")
    lines.append("2. verify reject 가 낮은 p (안전 margin)")
    lines.append("3. L_shift 가 합리적 (e.g. < 5 ~ 10) 한 p")
    lines.append("4. tie 시 spec 권장값 p=6 유지")
    lines.append("")
    lines.append("> Phase 3 full canary 에서 mask-based verify + family disagreement / 운영 metric 재측정 후 enable 결정.")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    print(f"[phase2] sample {SAMPLE_SIZE} frames from {THUMB_DIR}")
    sample_paths = _sample_frames()

    cfg = _build_cfg(p=6)  # detection rule 만 공유

    # 1차: frame 별 원본 LAB stats + triggered 측정 (p 와 무관)
    print(f"[phase2] loading {SAMPLE_SIZE} frames + computing LAB stats")
    t0 = time.perf_counter()
    frames: list[tuple[Path, np.ndarray, tuple[float, float, float], bool]] = []
    n_trig = 0
    for path in sample_paths:
        rgb = _load_rgb(path)
        L_o, a_o, b_o = compute_lab_stats(rgb)
        triggered = needs_correction(L_o, a_o, b_o, cfg)
        if triggered:
            n_trig += 1
        frames.append((path, rgb, (L_o, a_o, b_o), triggered))
    print(
        f"[phase2] frame load+stats {time.perf_counter() - t0:.1f}s — "
        f"triggered {n_trig}/{SAMPLE_SIZE} ({n_trig / SAMPLE_SIZE * 100:.1f}%)"
    )

    # 2차: p 별 보정 + metric 측정
    metrics: list[PMetric] = []
    for p in P_VALUES:
        print(f"[phase2] p={p} ablation")
        m = _measure_p(p, frames)
        print(
            f"  -> ΔE76 median {m.deltae76_median:.2f} / "
            f"reject {m.verify_reject_rate * 100:.1f}% / runtime {m.runtime_sec:.2f}s"
        )
        metrics.append(m)

    trigger_rate = n_trig / SAMPLE_SIZE
    _write_markdown(metrics, trigger_rate)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(
            {
                "sample_size": SAMPLE_SIZE,
                "seed": SEED,
                "p_values": P_VALUES,
                "trigger_rate": trigger_rate,
                "n_triggered": n_trig,
                "verify_threshold": VERIFY_THRESHOLD,
                "metrics": [asdict(m) for m in metrics],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[phase2] wrote {OUT_MD}")
    print(f"[phase2] wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
