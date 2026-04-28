"""frame_quality pure function pinning — Laplacian / brightness / HSV histogram corr.

cv2 의존 — `[vision]` extras 없으면 collect-time skip.
"""
from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from vision.frame_quality import (
    compute_blur_score,
    compute_brightness,
    compute_quality_score,
    histogram_correlation,
)


def _solid_color(h: int, w: int, rgb: tuple[int, int, int]) -> np.ndarray:
    return np.tile(np.array(rgb, dtype=np.uint8), (h, w, 1))


def _checkerboard(h: int, w: int, square: int = 8) -> np.ndarray:
    """sharp edge 가득한 패턴 — Laplacian variance 매우 큼."""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(0, h, square):
        for x in range(0, w, square):
            if ((y // square) + (x // square)) % 2 == 0:
                arr[y:y + square, x:x + square] = 255
    return arr


def _gaussian_blur(rgb: np.ndarray, ksize: int = 21) -> np.ndarray:
    return cv2.GaussianBlur(rgb, (ksize, ksize), 0)


# --------------------------------------------------------------------------- #
# blur score

def test_blur_score_solid_is_near_zero() -> None:
    """단색 frame 은 라플라시안 응답 0 → variance 0 근처."""
    img = _solid_color(64, 64, (128, 128, 128))
    assert compute_blur_score(img) < 1e-6


def test_blur_score_sharp_pattern_is_large() -> None:
    """체커보드는 edge 가 가득 → variance 매우 큼."""
    img = _checkerboard(64, 64)
    assert compute_blur_score(img) > 1000.0


def test_blur_score_blurred_pattern_is_smaller_than_sharp() -> None:
    """동일 패턴 + GaussianBlur → variance 감소 (blur 검출의 핵심 동작)."""
    sharp = _checkerboard(64, 64)
    blurred = _gaussian_blur(sharp)
    assert compute_blur_score(blurred) < compute_blur_score(sharp)


# --------------------------------------------------------------------------- #
# brightness

def test_brightness_black_frame_is_zero() -> None:
    img = _solid_color(32, 32, (0, 0, 0))
    assert compute_brightness(img) == 0.0


def test_brightness_white_frame_is_255() -> None:
    img = _solid_color(32, 32, (255, 255, 255))
    assert compute_brightness(img) == 255.0


def test_brightness_mid_gray_is_around_128() -> None:
    img = _solid_color(32, 32, (128, 128, 128))
    assert 127.0 <= compute_brightness(img) <= 129.0


# --------------------------------------------------------------------------- #
# composite quality_score

def test_quality_score_zero_when_too_dark() -> None:
    img = _solid_color(32, 32, (10, 10, 10))   # mean=10 < 30
    assert compute_quality_score(img, blur_min=100.0, brightness_range=(30.0, 225.0)) == 0.0


def test_quality_score_zero_when_too_bright() -> None:
    img = _solid_color(32, 32, (240, 240, 240))   # mean=240 > 225
    assert compute_quality_score(img, blur_min=100.0, brightness_range=(30.0, 225.0)) == 0.0


def test_quality_score_zero_when_blurred_below_min() -> None:
    """blur min 기준치 아래는 0 (모션 블러 컷)."""
    img = _solid_color(64, 64, (128, 128, 128))   # variance ≈ 0
    assert compute_quality_score(img, blur_min=100.0, brightness_range=(30.0, 225.0)) == 0.0


def test_quality_score_returns_blur_when_pass() -> None:
    """exposure + blur 둘 다 통과 시 score = blur_score 그대로."""
    img = _checkerboard(64, 64)   # bright pattern, mean ≈ 127, variance huge
    score = compute_quality_score(img, blur_min=100.0, brightness_range=(30.0, 225.0))
    assert score == compute_blur_score(img)
    assert score > 1000.0


# --------------------------------------------------------------------------- #
# histogram correlation

def test_histogram_corr_identical_is_one() -> None:
    img = _checkerboard(64, 64)
    assert histogram_correlation(img, img) == pytest.approx(1.0, abs=1e-6)


def test_histogram_corr_opposite_color_is_low() -> None:
    """순수 빨강 vs 순수 파랑 → HSV H 채널 분포 다름 → correlation 낮음."""
    red = _solid_color(64, 64, (255, 0, 0))
    blue = _solid_color(64, 64, (0, 0, 255))
    corr = histogram_correlation(red, blue)
    assert corr < 0.5


def test_histogram_corr_v_invariance() -> None:
    """같은 색조 (H+S) 의 밝기만 다른 frame → V 제외라 correlation 높게 유지."""
    bright_red = _solid_color(64, 64, (255, 100, 100))
    dim_red = _solid_color(64, 64, (128, 50, 50))
    corr = histogram_correlation(bright_red, dim_red)
    # 밝기 차이가 있어도 H+S 분포는 유사 → corr 높음 (V invariant 검증).
    assert corr > 0.5
