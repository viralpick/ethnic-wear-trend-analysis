"""color.C — frame illumination correction (shades-of-gray + detection + verify).

spec: `docs/color_c_illumination_spec.md` (2026-05-17).

frame 단계 보정 (segformer 입력 전). 적용 흐름:
  1. detection — LAB a/b skew 또는 L extreme 이면 trigger (Phase 1: garment mask
     channel imbalance 제외, segformer 순환 의존 회피).
  2. shades_of_gray — Minkowski p-norm gray balance (Finlayson & Trezzi 2004).
     p=1 gray-world / p=∞ white-patch / p=6 Finlayson 권장 perceptual 최고.
  3. verify — 보정 후 garment LAB median ΔE76 > threshold 면 원본 회수
     (Phase 2 구현 — segment_fn DI 콜백 필요. Phase 1 v1 은 verify skip).

설계 원칙:
- pure function — np.ndarray RGB 입력 / 출력. 결정론.
- segformer 의존 없음 (verify 의 segment_fn 만 외부 callable, optional).
- skip 시 input ndarray 그대로 반환 (copy 없음).

통합 지점:
- `pipeline_b_adapter._analyze_images` 에서 SceneFilter / LLM 호출 후 segformer
  입력용 RGB 만 보정 (SceneFilter / Gemini 는 원본 안전성 유지).

비용:
- detection: cv2.cvtColor (cv2.COLOR_RGB2LAB) + 3 mean — O(N pixel)
- shades_of_gray: O(N pixel × 3 channel × p)
- verify (Phase 2): segformer 호출 1회 추가 (또는 원본 mask 재사용 — IoU>0.95 가정 검증 후)
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

import cv2
import numpy as np

from settings import IlluminationCorrectionConfig

logger = logging.getLogger(__name__)


def _rgb_to_lab_float(rgb_uint8: np.ndarray) -> np.ndarray:
    """sRGB uint8 → CIE LAB float64.

    cv2 의 LAB 는 OpenCV scale: L ∈ [0, 255], a/b ∈ [0, 255] (offset 128).
    표준 CIE 로 변환: L = L_cv * 100 / 255, a = a_cv - 128, b = b_cv - 128.
    """
    lab_cv = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LAB).astype(np.float64)
    lab_cie = np.empty_like(lab_cv)
    lab_cie[..., 0] = lab_cv[..., 0] * (100.0 / 255.0)
    lab_cie[..., 1] = lab_cv[..., 1] - 128.0
    lab_cie[..., 2] = lab_cv[..., 2] - 128.0
    return lab_cie


def compute_lab_stats(rgb_uint8: np.ndarray) -> tuple[float, float, float]:
    """frame 전체 CIE LAB mean (L, a, b). detection trigger 측정용.

    Returns:
        (L_mean, a_mean, b_mean) — L ∈ [0, 100], a/b ∈ [-128, 127].
    """
    lab = _rgb_to_lab_float(rgb_uint8)
    return (
        float(lab[..., 0].mean()),
        float(lab[..., 1].mean()),
        float(lab[..., 2].mean()),
    )


def needs_correction(
    L_mean: float,
    a_mean: float,
    b_mean: float,
    cfg: IlluminationCorrectionConfig,
) -> bool:
    """detection trigger — (a/b skew) OR (L extreme).

    Phase 1: (c) garment mask 내 channel imbalance 제외 (순환 의존). 의류 본연 색 가드는
    verify (Phase 2) 가 담당.
    """
    d = cfg.detection
    return (
        abs(a_mean) > d.a_skew_threshold
        or abs(b_mean) > d.b_skew_threshold
        or L_mean < d.l_low_threshold
        or L_mean > d.l_high_threshold
    )


def shades_of_gray(rgb_uint8: np.ndarray, p: int = 6) -> np.ndarray:
    """Minkowski p-norm gray balance (Finlayson & Trezzi 2004 "Shades of Gray").

    각 채널 c ∈ {R, G, B} 의 Minkowski p-norm 을 gray balance 목표값으로 normalize:
        k_c = ((1/N) Σ |I_c(x)|^p)^(1/p)
        gain_c = k_gray / k_c   (k_gray = mean(k_R, k_G, k_B))
        I_c' = I_c × gain_c   (clip [0, 1])

    Args:
        rgb_uint8: (H, W, 3) uint8 RGB.
        p: Minkowski exponent. 1=gray-world (각 채널 mean), 6=Finlayson 권장 절충,
           수치 ∞ 시 white-patch (각 채널 max). 본 구현은 finite p (>=1) 만 — white-patch
           는 별도 함수 권장.

    Returns:
        (H, W, 3) uint8 보정된 RGB.
    """
    if p < 1:
        raise ValueError(f"Minkowski p must be >= 1, got {p}")
    img = rgb_uint8.astype(np.float64) / 255.0
    powered = np.power(img, p)
    # (H*W, 3) average → (3,) per-channel norm
    k = powered.mean(axis=(0, 1)) ** (1.0 / p)
    k_gray = float(k.mean())
    gain = k_gray / np.clip(k, 1e-6, None)
    corrected = img * gain
    corrected = np.clip(corrected, 0.0, 1.0)
    return (corrected * 255.0).astype(np.uint8)


SegmentFn = Callable[[np.ndarray], Optional[np.ndarray]]


def _verify_correction(
    rgb_original: np.ndarray,
    rgb_corrected: np.ndarray,
    segment_fn: SegmentFn,
    deltae76_threshold: float,
) -> bool:
    """verify 가드 — 보정 후 garment LAB median ΔE76 ≤ threshold 면 accept.

    Phase 2 구현 예정 — 본 함수는 인터페이스만 정의. segment_fn 은 rgb → bool mask
    (H, W) callable. mask 없으면 (None 또는 빈 mask) 안전하게 accept (보정 적용).

    원본 frame mask 1회 호출 (Phase 1 v2 단순 — Phase 2 에서 IoU>0.95 가정 검증 후
    mask_o 만 사용해서 mask_c 호출 절약 검토).
    """
    mask = segment_fn(rgb_original)
    if mask is None or not mask.any():
        return True
    lab_o = _rgb_to_lab_float(rgb_original)[mask]
    lab_c = _rgb_to_lab_float(rgb_corrected)[mask]
    median_o = np.median(lab_o, axis=0)
    median_c = np.median(lab_c, axis=0)
    delta_e = float(np.sqrt(((median_c - median_o) ** 2).sum()))
    return delta_e <= deltae76_threshold


def apply_correction(
    rgb_uint8: np.ndarray,
    cfg: IlluminationCorrectionConfig,
    *,
    segment_fn: SegmentFn | None = None,
) -> tuple[np.ndarray, dict]:
    """color.C entry — detection → (optional) shades_of_gray → (optional) verify.

    Args:
        rgb_uint8: (H, W, 3) uint8 RGB frame.
        cfg: IlluminationCorrectionConfig — enabled / minkowski_p / detection / verify.
        segment_fn: optional rgb→mask callable for verify guard (Phase 2). None 이면
            verify skip (cfg.verify.enabled=True 라도 동작 안 함 — Phase 1 v1 단순화).

    Returns:
        (output_rgb, info). output_rgb 는 보정된 RGB 또는 원본 (skip / verify reject).
        info dict: {enabled, triggered, L_mean, a_mean, b_mean, verify_accept}.
        canary 측정에서 trigger 비율 / verify 거부율 추적용.
    """
    info: dict = {"enabled": cfg.enabled}
    if not cfg.enabled:
        info["triggered"] = False
        return rgb_uint8, info

    L_mean, a_mean, b_mean = compute_lab_stats(rgb_uint8)
    info["L_mean"] = L_mean
    info["a_mean"] = a_mean
    info["b_mean"] = b_mean
    if not needs_correction(L_mean, a_mean, b_mean, cfg):
        info["triggered"] = False
        return rgb_uint8, info
    info["triggered"] = True

    corrected = shades_of_gray(rgb_uint8, p=cfg.minkowski_p)

    # verify (Phase 2) — segment_fn 콜백 제공 시만 동작.
    if cfg.verify.enabled and segment_fn is not None:
        accept = _verify_correction(
            rgb_uint8,
            corrected,
            segment_fn,
            cfg.verify.deltae76_threshold,
        )
        info["verify_accept"] = accept
        if not accept:
            logger.info(
                "illumination_correction_rejected L=%.1f a=%.1f b=%.1f ΔE76>%.1f",
                L_mean, a_mean, b_mean, cfg.verify.deltae76_threshold,
            )
            return rgb_uint8, info
    else:
        info["verify_accept"] = True  # verify skip 또는 미주입

    return corrected, info


__all__ = [
    "apply_correction",
    "compute_lab_stats",
    "needs_correction",
    "shades_of_gray",
]
