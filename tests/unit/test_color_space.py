"""color_space 단위 테스트 — RGB↔LAB, skin 제거, KMeans 팔레트, ΔE, skin_leak.

동료 PoC (`~/dev/clothing-color-extraction-poc`) 에서 이식한 color_utils 의 모든 공용 심볼을
covers. 방법론 비교 목적이라 전 심볼 유지.
"""
from __future__ import annotations

import numpy as np
import pytest

from vision.color_space import (
    SKIN_LAB_MAX,
    SKIN_LAB_MIN,
    delta_e76,
    drop_skin,
    drop_skin_adaptive,
    extract_colors,
    hex_skin_leak,
    hex_to_rgb,
    lab_to_rgb,
    rgb_to_hex,
    rgb_to_lab,
)

# --------------------------------------------------------------------------- #
# HEX ↔ RGB
# --------------------------------------------------------------------------- #

def test_rgb_to_hex_mid_tone() -> None:
    # spec §4.1 ④ 예시.
    assert rgb_to_hex(np.array([184, 212, 195])) == "#B8D4C3"


def test_rgb_to_hex_zero_pads() -> None:
    assert rgb_to_hex(np.array([0, 5, 15])) == "#00050F"


def test_rgb_to_hex_clips_out_of_range() -> None:
    # 범위 밖 값도 0/255 로 clip, 단순 모듈로 하지 않음.
    assert rgb_to_hex(np.array([-10, 300, 128])) == "#00FF80"


def test_hex_to_rgb_roundtrip() -> None:
    rgb = np.array([184, 212, 195], dtype=np.float32)
    parsed = hex_to_rgb(rgb_to_hex(rgb))
    assert np.array_equal(parsed, rgb)


def test_hex_to_rgb_strips_optional_hash() -> None:
    assert np.array_equal(hex_to_rgb("#010203"), hex_to_rgb("010203"))


# --------------------------------------------------------------------------- #
# RGB ↔ LAB round-trip (D65 / sRGB)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("rgb", [
    [128, 128, 128],   # neutral gray
    [184, 212, 195],   # sage
    [255, 165, 0],     # orange
    [20, 30, 200],     # deep blue
    [255, 255, 255],   # pure white
    [0, 0, 0],         # pure black
])
def test_rgb_lab_roundtrip_within_one_unit(rgb: list[int]) -> None:
    # 부동소수 한계. D65 라운드 트립 평균 오차 < 1 을 기대.
    arr = np.array(rgb, dtype=np.float32)
    lab = rgb_to_lab(arr)
    back = lab_to_rgb(lab)
    assert np.max(np.abs(back - arr)) < 1.0


def test_rgb_to_lab_preserves_shape() -> None:
    # (H, W, 3) 입력도 (H, W, 3) LAB 로 변환되어야 한다.
    img = np.full((4, 5, 3), 128, dtype=np.uint8)
    lab = rgb_to_lab(img)
    assert lab.shape == (4, 5, 3)


# --------------------------------------------------------------------------- #
# ΔE76 (방법론 비교용)
# --------------------------------------------------------------------------- #

def test_delta_e76_self_is_zero() -> None:
    lab = np.array([50.0, 10.0, 20.0])
    assert delta_e76(lab, lab) == 0.0


def test_delta_e76_unit_distance() -> None:
    assert delta_e76(np.array([0, 0, 0]), np.array([1, 0, 0])) == pytest.approx(1.0)
    assert delta_e76(np.array([0, 0, 0]), np.array([3, 4, 0])) == pytest.approx(5.0)


# --------------------------------------------------------------------------- #
# drop_skin (default + override box)
# --------------------------------------------------------------------------- #

def _make_pixel_in_box() -> np.ndarray:
    """SKIN_LAB_MIN/MAX 안에 들어가는 RGB pixel 을 역산."""
    # LAB [60, 18, 25] → mid-skin. lab_to_rgb 로 역변환.
    lab = np.array([60.0, 18.0, 25.0], dtype=np.float32)
    rgb = lab_to_rgb(lab).astype(np.uint8)
    # 역변환 결과가 진짜 box 안인지 확인 (float 오차로 경계 밖으로 나갈 수 있어 assert)
    round_lab = rgb_to_lab(rgb.astype(np.float32))
    assert np.all(round_lab >= SKIN_LAB_MIN) and np.all(round_lab <= SKIN_LAB_MAX)
    return rgb


def _make_pixel_outside_box() -> np.ndarray:
    # 선명한 파랑 — LAB 의 a/b 가 음수라 box 밖 보장.
    return np.array([20, 30, 200], dtype=np.uint8)


def test_drop_skin_removes_inside_keeps_outside() -> None:
    skin = _make_pixel_in_box()
    non_skin = _make_pixel_outside_box()
    pixels = np.stack([skin, non_skin, skin, non_skin])
    cleaned = drop_skin(pixels)
    assert cleaned.shape[0] == 2
    # 남은 픽셀은 전부 non-skin (R < G < B 의 파랑 특성).
    assert np.all(cleaned[:, 2] > cleaned[:, 0])


def test_drop_skin_empty_input_safe() -> None:
    empty = np.zeros((0, 3), dtype=np.uint8)
    assert drop_skin(empty).shape == (0, 3)


def test_drop_skin_custom_box_override() -> None:
    # 박스를 L:0~10 으로 좁히면 모든 실 pixel 이 밖이라 아무것도 제거 안 됨.
    pixels = np.stack([_make_pixel_in_box(), _make_pixel_outside_box()])
    narrow_min = np.array([0.0, -50.0, -50.0])
    narrow_max = np.array([10.0, 0.0, 0.0])
    cleaned = drop_skin(pixels, lab_min=narrow_min, lab_max=narrow_max)
    assert cleaned.shape[0] == 2  # 둘 다 유지 (박스 밖)


def test_drop_skin_custom_box_wider_drops_more() -> None:
    # 박스를 전체 LAB 로 넓히면 모든 pixel 이 안쪽 → 전부 drop.
    pixels = np.stack([_make_pixel_in_box(), _make_pixel_outside_box()])
    wide_min = np.array([-200.0, -200.0, -200.0])
    wide_max = np.array([200.0, 200.0, 200.0])
    cleaned = drop_skin(pixels, lab_min=wide_min, lab_max=wide_max)
    assert cleaned.shape[0] == 0


# --------------------------------------------------------------------------- #
# drop_skin_adaptive — skin-tone garment 보호 (Q1 phase 2)
# --------------------------------------------------------------------------- #

def test_adaptive_below_threshold_drops_inside_pixels() -> None:
    # 5 pixel 중 1개만 skin box 안 (20%) → threshold 0.3 미만 → 일반 drop
    skin = _make_pixel_in_box()
    non_skin = _make_pixel_outside_box()
    pixels = np.stack([skin, non_skin, non_skin, non_skin, non_skin])
    cleaned, ratio, kept_whole = drop_skin_adaptive(pixels, keep_threshold_pct=0.3)
    assert cleaned.shape[0] == 4           # skin 1개 제거
    assert ratio == pytest.approx(0.2)
    assert kept_whole is False


def test_adaptive_above_threshold_keeps_whole_garment() -> None:
    # 5 pixel 중 3개가 skin box 안 (60%) → threshold 0.3 초과 → garment-as-skin 판정
    # 원본 전체 유지 (베이지 코트 보호 등)
    skin = _make_pixel_in_box()
    non_skin = _make_pixel_outside_box()
    pixels = np.stack([skin, skin, skin, non_skin, non_skin])
    cleaned, ratio, kept_whole = drop_skin_adaptive(pixels, keep_threshold_pct=0.3)
    assert cleaned.shape[0] == 5           # 전체 유지
    assert ratio == pytest.approx(0.6)
    assert kept_whole is True


def test_adaptive_exactly_at_threshold_drops() -> None:
    # boundary: ratio == threshold 인 경우. 구현은 `>` 이라 drop.
    skin = _make_pixel_in_box()
    non_skin = _make_pixel_outside_box()
    pixels = np.stack([skin, skin, skin, non_skin, non_skin, non_skin, non_skin])
    # 3/7 ≈ 0.428. threshold 0.43 으로 설정 → just below.
    cleaned, ratio, kept_whole = drop_skin_adaptive(pixels, keep_threshold_pct=0.43)
    assert kept_whole is False             # 0.428 < 0.43
    assert cleaned.shape[0] == 4


def test_adaptive_empty_pixels_returns_empty() -> None:
    empty = np.zeros((0, 3), dtype=np.uint8)
    cleaned, ratio, kept_whole = drop_skin_adaptive(empty)
    assert cleaned.shape == (0, 3)
    assert ratio == 0.0
    assert kept_whole is False


# --------------------------------------------------------------------------- #
# hex_skin_leak (drop_skin 사후 QA)
# --------------------------------------------------------------------------- #

def test_hex_skin_leak_true_for_skin_hex() -> None:
    rgb = _make_pixel_in_box()
    assert hex_skin_leak(rgb_to_hex(rgb)) is True


def test_hex_skin_leak_false_for_vivid_blue() -> None:
    assert hex_skin_leak("#1E1EFF") is False


def test_hex_skin_leak_custom_box_override() -> None:
    # 박스를 매우 좁히면 어떤 skin-근사 hex 도 leak 으로 잡히지 않음.
    narrow_min = np.array([0.0, -50.0, -50.0])
    narrow_max = np.array([10.0, 0.0, 0.0])
    rgb = _make_pixel_in_box()
    assert hex_skin_leak(rgb_to_hex(rgb), lab_min=narrow_min, lab_max=narrow_max) is False


# --------------------------------------------------------------------------- #
# extract_colors — KMeans 결정론 / empty / k clamp
# --------------------------------------------------------------------------- #

def _synthetic_bimodal_pixels(n_per_cluster: int = 200) -> np.ndarray:
    """두 뚜렷한 색 cluster (red / blue) 를 각 n_per_cluster 개."""
    rng = np.random.RandomState(0)
    red = rng.randint(200, 256, size=(n_per_cluster, 1))
    red_full = np.concatenate([red, rng.randint(0, 40, size=(n_per_cluster, 2))], axis=1)
    blue = rng.randint(200, 256, size=(n_per_cluster, 1))
    blue_full = np.concatenate([
        rng.randint(0, 40, size=(n_per_cluster, 2)), blue,
    ], axis=1)
    return np.concatenate([red_full, blue_full]).astype(np.uint8)


def test_extract_colors_empty_when_below_min_pixels() -> None:
    # min_pixels=150 미만이면 빈 리스트.
    pixels = np.random.RandomState(0).randint(0, 255, size=(100, 3)).astype(np.uint8)
    assert extract_colors(pixels, k=5, min_pixels=150) == []


def test_extract_colors_determinism() -> None:
    # random_state=0 고정이므로 같은 입력 → 같은 출력.
    pixels = _synthetic_bimodal_pixels()
    a = extract_colors(pixels, k=5)
    b = extract_colors(pixels, k=5)
    assert [c["hex"] for c in a] == [c["hex"] for c in b]
    assert [c["weight"] for c in a] == [c["weight"] for c in b]


def test_extract_colors_weights_sum_to_one() -> None:
    pixels = _synthetic_bimodal_pixels()
    colors = extract_colors(pixels, k=5)
    assert sum(c["weight"] for c in colors) == pytest.approx(1.0)


def test_extract_colors_sorted_desc() -> None:
    pixels = _synthetic_bimodal_pixels()
    colors = extract_colors(pixels, k=5)
    weights = [c["weight"] for c in colors]
    assert weights == sorted(weights, reverse=True)


def test_extract_colors_k_clamped_by_pixel_count() -> None:
    # k_eff = min(k, max(1, N//50)). N=150 → k_eff = min(5, 3) = 3.
    pixels = np.random.RandomState(0).randint(0, 255, size=(150, 3)).astype(np.uint8)
    colors = extract_colors(pixels, k=5, min_pixels=150)
    assert len(colors) == 3


def test_extract_colors_separates_bimodal() -> None:
    # red / blue 합성 입력이면 top-2 cluster 의 hex 가 붉은색과 푸른색이어야 한다.
    pixels = _synthetic_bimodal_pixels()
    colors = extract_colors(pixels, k=2)
    assert len(colors) == 2
    hexes = [c["hex"] for c in colors]
    # 두 cluster 중 하나는 R 지배, 다른 하나는 B 지배.
    rgbs = [hex_to_rgb(h) for h in hexes]
    r_dom = any(rgb[0] > rgb[2] + 100 for rgb in rgbs)
    b_dom = any(rgb[2] > rgb[0] + 100 for rgb in rgbs)
    assert r_dom and b_dom
