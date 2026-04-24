"""Phase 4 dynamic_palette pinning — 단색/분리/merge/drop/가드/수식 고정.

sklearn (numpy 는 core) 필요 — vision extras 별도 설치 없이 기본 개발 환경에서 동작.
LAB ΔE76 사전 계산 (see scripts 기반 탐색):
  red(220,20,20) ↔ blue(20,20,220) ↔ green(20,180,20) 전부 60+ (merge 안 됨).
  white ↔ ivoryA(240,235,220) 9.48, white ↔ ivoryB(242,238,225) 8.00,
  ivoryA ↔ ivoryB 1.51 — 전부 threshold=10 안 → 1 cluster 로 수렴.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn", reason="sklearn required")

from settings import DynamicPaletteConfig  # noqa: E402
from vision.dynamic_palette import (  # noqa: E402
    _merge_by_deltae76,
    extract_dynamic_palette,
)

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _fill(color: tuple[int, int, int], n: int) -> np.ndarray:
    arr = np.empty((n, 3), dtype=np.uint8)
    arr[:] = color
    return arr


def _default_cfg(**overrides) -> DynamicPaletteConfig:
    return DynamicPaletteConfig(**overrides)


# --------------------------------------------------------------------------- #
# extract_dynamic_palette — e2e
# --------------------------------------------------------------------------- #

def test_single_color_pool_collapses_to_k1() -> None:
    pixels = _fill((220, 20, 20), 1000)
    palette = extract_dynamic_palette(pixels, _default_cfg())
    assert len(palette) == 1
    assert palette[0].share == pytest.approx(1.0, abs=1e-6)


def test_three_distinct_colors_no_merge() -> None:
    # red / blue / green — pair ΔE76 모두 > 60, threshold=10 으로 merge 없음.
    pixels = np.concatenate([
        _fill((220, 20, 20), 500),
        _fill((20, 20, 220), 500),
        _fill((20, 180, 20), 500),
    ], axis=0)
    palette = extract_dynamic_palette(pixels, _default_cfg())
    assert len(palette) == 3
    # share 합 ≈ 1.0 재정규화
    assert sum(p.share for p in palette) == pytest.approx(1.0, abs=1e-5)
    # 각 share 는 ~ 1/3
    assert all(abs(p.share - 1 / 3) < 0.05 for p in palette)


def test_nearby_ivory_colors_merge_to_one() -> None:
    # white / ivoryA / ivoryB — 모든 pair ΔE76 < 10. greedy 로 합쳐 최종 1 cluster.
    pixels = np.concatenate([
        _fill((250, 250, 250), 500),
        _fill((240, 235, 220), 500),
        _fill((242, 238, 225), 500),
    ], axis=0)
    palette = extract_dynamic_palette(pixels, _default_cfg())
    assert len(palette) == 1
    assert palette[0].share == pytest.approx(1.0, abs=1e-6)


def test_small_share_cluster_is_dropped() -> None:
    # red 96% + blue 4% — blue share < 0.05 → drop, red 만 남음.
    pixels = np.concatenate([
        _fill((220, 20, 20), 2400),
        _fill((20, 20, 220), 100),
    ], axis=0)
    palette = extract_dynamic_palette(pixels, _default_cfg())
    assert len(palette) == 1
    # 남은 cluster 는 red 계통
    assert palette[0].rgb[0] > palette[0].rgb[2]  # R > B


def test_below_min_pixels_returns_empty() -> None:
    # 기본 min_pixels=150 미만 → []
    pixels = _fill((220, 20, 20), 100)
    assert extract_dynamic_palette(pixels, _default_cfg()) == []


def test_extract_dynamic_palette_is_deterministic() -> None:
    pixels = np.concatenate([
        _fill((220, 20, 20), 300),
        _fill((20, 20, 220), 300),
        _fill((20, 180, 20), 300),
    ], axis=0)
    first = extract_dynamic_palette(pixels, _default_cfg())
    second = extract_dynamic_palette(pixels, _default_cfg())
    assert [(p.hex, p.share) for p in first] == [(p.hex, p.share) for p in second]


# --------------------------------------------------------------------------- #
# _merge_by_deltae76 — 수식 pinning
# --------------------------------------------------------------------------- #

def test_merge_weighted_centroid_formula() -> None:
    # 두 cluster centroid (50,0,0), (56,0,0), weights 30 / 10 → ΔE=6 (<10) merge.
    # new_center = (30*50 + 10*56)/40 = 51.5, new_weight = 40.
    centers = np.array([[50.0, 0.0, 0.0], [56.0, 0.0, 0.0]], dtype=np.float32)
    weights = np.array([30.0, 10.0], dtype=np.float32)
    merged_centers, merged_weights = _merge_by_deltae76(centers, weights, threshold=10.0)
    assert merged_centers.shape == (1, 3)
    assert merged_weights.tolist() == [40.0]
    assert merged_centers[0].tolist() == pytest.approx([51.5, 0.0, 0.0], abs=1e-5)


def test_merge_respects_threshold() -> None:
    # 두 cluster ΔE=15 (>10) → no merge, 입력 그대로 반환 (deep copy).
    centers = np.array([[50.0, 0.0, 0.0], [65.0, 0.0, 0.0]], dtype=np.float32)
    weights = np.array([20.0, 20.0], dtype=np.float32)
    out_centers, out_weights = _merge_by_deltae76(centers, weights, threshold=10.0)
    assert out_centers.shape == (2, 3)
    assert out_weights.tolist() == [20.0, 20.0]
