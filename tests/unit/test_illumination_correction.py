"""color.C illumination_correction 단위 test — detection / shades_of_gray / apply_correction."""
from __future__ import annotations

import numpy as np
import pytest

from settings import (
    IlluminationCorrectionConfig,
    IlluminationCorrectionDetectionConfig,
    IlluminationCorrectionVerifyConfig,
)
from vision.illumination_correction import (
    apply_correction,
    compute_lab_stats,
    needs_correction,
    shades_of_gray,
)


def _cfg(enabled: bool = True, p: int = 6, verify_enabled: bool = True) -> IlluminationCorrectionConfig:
    return IlluminationCorrectionConfig(
        enabled=enabled,
        minkowski_p=p,
        detection=IlluminationCorrectionDetectionConfig(),
        verify=IlluminationCorrectionVerifyConfig(enabled=verify_enabled),
    )


def _solid_frame(rgb: tuple[int, int, int], shape: tuple[int, int] = (64, 64)) -> np.ndarray:
    h, w = shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[..., 0] = rgb[0]
    img[..., 1] = rgb[1]
    img[..., 2] = rgb[2]
    return img


# ---- compute_lab_stats ----


def test_compute_lab_stats_neutral_gray_zero_skew() -> None:
    # 중간 회색 — LAB a/b ≈ 0, L ≈ 53.4 (sRGB 128 의 CIE LAB)
    gray = _solid_frame((128, 128, 128))
    L, a, b = compute_lab_stats(gray)
    assert abs(a) < 1.0
    assert abs(b) < 1.0
    assert 50.0 < L < 60.0


def test_compute_lab_stats_red_positive_a() -> None:
    red = _solid_frame((255, 0, 0))
    L, a, b = compute_lab_stats(red)
    assert a > 50.0  # 강한 red shift


def test_compute_lab_stats_dark_low_L() -> None:
    dark = _solid_frame((20, 20, 20))
    L, _, _ = compute_lab_stats(dark)
    assert L < 20.0


def test_compute_lab_stats_bright_high_L() -> None:
    bright = _solid_frame((240, 240, 240))
    L, _, _ = compute_lab_stats(bright)
    assert L > 90.0


# ---- needs_correction ----


def test_needs_correction_neutral_frame_no_trigger() -> None:
    cfg = _cfg()
    assert needs_correction(L_mean=50.0, a_mean=0.0, b_mean=0.0, cfg=cfg) is False


def test_needs_correction_a_skew_triggers() -> None:
    cfg = _cfg()
    assert needs_correction(L_mean=50.0, a_mean=10.0, b_mean=0.0, cfg=cfg) is True
    assert needs_correction(L_mean=50.0, a_mean=-10.0, b_mean=0.0, cfg=cfg) is True


def test_needs_correction_b_skew_triggers() -> None:
    cfg = _cfg()
    assert needs_correction(L_mean=50.0, a_mean=0.0, b_mean=12.0, cfg=cfg) is True
    assert needs_correction(L_mean=50.0, a_mean=0.0, b_mean=-12.0, cfg=cfg) is True


def test_needs_correction_l_low_triggers() -> None:
    cfg = _cfg()
    assert needs_correction(L_mean=25.0, a_mean=0.0, b_mean=0.0, cfg=cfg) is True


def test_needs_correction_l_high_triggers() -> None:
    cfg = _cfg()
    assert needs_correction(L_mean=85.0, a_mean=0.0, b_mean=0.0, cfg=cfg) is True


def test_needs_correction_threshold_boundary() -> None:
    # 정확히 threshold 면 trigger 안 됨 (> 사용)
    cfg = _cfg()
    assert needs_correction(L_mean=30.0, a_mean=0.0, b_mean=0.0, cfg=cfg) is False
    assert needs_correction(L_mean=80.0, a_mean=0.0, b_mean=0.0, cfg=cfg) is False
    assert needs_correction(L_mean=50.0, a_mean=8.0, b_mean=0.0, cfg=cfg) is False


# ---- shades_of_gray ----


def test_shades_of_gray_neutral_frame_unchanged() -> None:
    # 회색 frame — 모든 채널 같은 norm → gain=1 → 변화 없음
    gray = _solid_frame((128, 128, 128))
    corrected = shades_of_gray(gray, p=6)
    assert np.array_equal(corrected, gray)


def test_shades_of_gray_warm_shift_rebalanced() -> None:
    # warm shift (R 더 강한 cast) → 보정 후 채널 mean 균등화 방향
    # gradient 로 frame 채우면 KMeans norm 의미 있음. 단순 solid R 우세는 white-patch 와 동일
    rng = np.random.default_rng(42)
    base = rng.integers(60, 200, size=(64, 64, 3), dtype=np.uint8)
    # R 채널 의도적으로 +20 shift (warm)
    base[..., 0] = np.clip(base[..., 0].astype(int) + 20, 0, 255).astype(np.uint8)
    corrected = shades_of_gray(base, p=6)
    # 보정 후 채널별 mean 이 원본 보다 균등 (max-min 차이 감소)
    orig_range = base.mean(axis=(0, 1)).max() - base.mean(axis=(0, 1)).min()
    new_range = corrected.mean(axis=(0, 1)).max() - corrected.mean(axis=(0, 1)).min()
    assert new_range < orig_range


def test_shades_of_gray_deterministic() -> None:
    rng = np.random.default_rng(42)
    frame = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    a = shades_of_gray(frame, p=6)
    b = shades_of_gray(frame, p=6)
    assert np.array_equal(a, b)


def test_shades_of_gray_p_below_one_raises() -> None:
    frame = _solid_frame((100, 100, 100))
    with pytest.raises(ValueError, match="p must be >= 1"):
        shades_of_gray(frame, p=0)


def test_shades_of_gray_output_uint8_in_range() -> None:
    rng = np.random.default_rng(42)
    frame = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    corrected = shades_of_gray(frame, p=6)
    assert corrected.dtype == np.uint8
    assert corrected.shape == frame.shape
    assert corrected.min() >= 0
    assert corrected.max() <= 255


# ---- apply_correction ----


def test_apply_correction_disabled_returns_original() -> None:
    frame = _solid_frame((100, 100, 100))
    cfg = _cfg(enabled=False)
    out, info = apply_correction(frame, cfg)
    assert np.array_equal(out, frame)
    assert info["enabled"] is False
    assert info["triggered"] is False


def test_apply_correction_enabled_neutral_no_trigger() -> None:
    # 회색 frame — detection 무필요 → 원본 그대로
    gray = _solid_frame((128, 128, 128))
    cfg = _cfg(enabled=True)
    out, info = apply_correction(gray, cfg)
    assert np.array_equal(out, gray)
    assert info["enabled"] is True
    assert info["triggered"] is False
    assert "L_mean" in info
    assert "a_mean" in info
    assert "b_mean" in info


def test_apply_correction_dark_frame_triggers_and_corrects() -> None:
    # L extreme (어두운) → detection trigger → shades_of_gray 적용
    dark = _solid_frame((20, 20, 20))
    cfg = _cfg(enabled=True)
    out, info = apply_correction(dark, cfg)
    assert info["triggered"] is True
    # 회색 frame 이라 norm 균등 → shades_of_gray 결과 거의 동일하지만 dtype/shape 보존 확인
    assert out.shape == dark.shape
    assert out.dtype == np.uint8


def test_apply_correction_warm_random_frame_triggers() -> None:
    rng = np.random.default_rng(42)
    base = rng.integers(60, 200, size=(64, 64, 3), dtype=np.uint8)
    # warm: R + b 강하게 shift
    base[..., 0] = np.clip(base[..., 0].astype(int) + 40, 0, 255).astype(np.uint8)
    cfg = _cfg(enabled=True)
    out, info = apply_correction(base, cfg)
    assert info["triggered"] is True
    # 보정 후 base 와 다름
    assert not np.array_equal(out, base)


def test_apply_correction_verify_skip_when_segment_fn_none() -> None:
    # verify.enabled=True 라도 segment_fn=None 이면 verify skip (Phase 1 v1)
    dark = _solid_frame((20, 20, 20))
    cfg = _cfg(enabled=True, verify_enabled=True)
    out, info = apply_correction(dark, cfg, segment_fn=None)
    assert info["triggered"] is True
    assert info["verify_accept"] is True  # skip 으로 간주


def test_apply_correction_verify_accept_when_low_delta() -> None:
    # segment_fn mock — 항상 같은 mask 반환. 회색 frame 이라 보정 후 ΔE76 ≈ 0
    dark = _solid_frame((20, 20, 20))
    cfg = _cfg(enabled=True, verify_enabled=True)
    mask = np.ones(dark.shape[:2], dtype=bool)
    out, info = apply_correction(dark, cfg, segment_fn=lambda _rgb: mask)
    assert info["verify_accept"] is True


def test_apply_correction_verify_reject_when_high_delta() -> None:
    # segment_fn mock + verify threshold 매우 낮음 (1) → 항상 reject → 원본 회수
    rng = np.random.default_rng(42)
    base = rng.integers(60, 200, size=(64, 64, 3), dtype=np.uint8)
    base[..., 0] = np.clip(base[..., 0].astype(int) + 60, 0, 255).astype(np.uint8)
    cfg = IlluminationCorrectionConfig(
        enabled=True,
        minkowski_p=6,
        detection=IlluminationCorrectionDetectionConfig(),
        verify=IlluminationCorrectionVerifyConfig(
            enabled=True, deltae76_threshold=1.0,
        ),
    )
    mask = np.ones(base.shape[:2], dtype=bool)
    out, info = apply_correction(base, cfg, segment_fn=lambda _rgb: mask)
    assert info["triggered"] is True
    assert info["verify_accept"] is False
    assert np.array_equal(out, base)  # 원본 회수


def test_apply_correction_verify_disabled_skips_verify() -> None:
    rng = np.random.default_rng(42)
    base = rng.integers(60, 200, size=(64, 64, 3), dtype=np.uint8)
    base[..., 0] = np.clip(base[..., 0].astype(int) + 60, 0, 255).astype(np.uint8)
    # verify.enabled=False 면 segment_fn 주입돼도 verify skip
    cfg = IlluminationCorrectionConfig(
        enabled=True,
        minkowski_p=6,
        detection=IlluminationCorrectionDetectionConfig(),
        verify=IlluminationCorrectionVerifyConfig(
            enabled=False, deltae76_threshold=1.0,
        ),
    )
    mask = np.ones(base.shape[:2], dtype=bool)
    out, info = apply_correction(base, cfg, segment_fn=lambda _rgb: mask)
    assert info["triggered"] is True
    assert info["verify_accept"] is True
    # 보정 적용 됨 (원본과 다름)
    assert not np.array_equal(out, base)


def test_apply_correction_info_keys() -> None:
    dark = _solid_frame((20, 20, 20))
    cfg = _cfg(enabled=True)
    _, info = apply_correction(dark, cfg)
    # canary 측정 / debug 용 info 필수 키
    assert "enabled" in info
    assert "triggered" in info
    assert "L_mean" in info
    assert "verify_accept" in info
