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
    drop_skin_2layer,
    drop_skin_adaptive,
    drop_skin_adaptive_spatial,
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


def test_adaptive_upper_ceiling_drops_all_skin_segment() -> None:
    # 10 pixel 전부 skin box 안 (100%) → upper_ceiling (0.97) 초과 → segment 통째 drop.
    # segformer 가 팔/다리를 upper-clothes/pants 로 오분류한 케이스 방어.
    skin = _make_pixel_in_box()
    pixels = np.stack([skin] * 10)
    cleaned, ratio, kept_whole = drop_skin_adaptive(
        pixels, keep_threshold_pct=0.5, upper_ceiling_pct=0.97,
    )
    assert cleaned.shape == (0, 3)
    assert ratio == pytest.approx(1.0)
    assert kept_whole is False


def test_adaptive_just_below_ceiling_keeps_whole() -> None:
    # 20 중 19 skin (95%) — ceiling 0.97 아래이므로 통째 보존 (skin-tone 옷 판정).
    skin = _make_pixel_in_box()
    non_skin = _make_pixel_outside_box()
    pixels = np.stack([skin] * 19 + [non_skin])
    cleaned, ratio, kept_whole = drop_skin_adaptive(
        pixels, keep_threshold_pct=0.5, upper_ceiling_pct=0.97,
    )
    assert cleaned.shape[0] == 20
    assert ratio == pytest.approx(0.95)
    assert kept_whole is True


def test_adaptive_ceiling_takes_precedence_over_threshold() -> None:
    # threshold 와 ceiling 사이 gap 이 없을 때도 (둘 다 0.5) ceiling 분기가 우선 분기.
    # 실 호출에서 threshold > ceiling 설정은 금기지만 분기 순서 검증용 단위 테스트.
    skin = _make_pixel_in_box()
    pixels = np.stack([skin] * 10)
    cleaned, _ratio, kept_whole = drop_skin_adaptive(
        pixels, keep_threshold_pct=0.5, upper_ceiling_pct=0.5,
    )
    assert cleaned.shape == (0, 3)
    assert kept_whole is False


# --------------------------------------------------------------------------- #
# drop_skin_adaptive_spatial — skin class mask + dilate margin 기반 공간 방어
# scipy.ndimage.binary_dilation 필요 — no-extras 환경에선 importorskip 으로 스킵.
# --------------------------------------------------------------------------- #

def _build_spatial_scene(
    inner_skin_color: bool = True,
    edge_skin_color: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """10x10 crop 생성.

    좌측 4 column (x < 4) = skin class mask (ATR face/arm/leg 시뮬).
    우측 6 column (x >= 4) = garment mask.
    garment 내부에 skin-color pixel 을 두 곳 배치:
      - edge: garment 왼쪽 경계 바로 옆 (x=4, y=5) — skin class 인접
      - inner: garment 오른쪽 깊숙이 (x=9, y=5) — skin class 에서 멀리
    나머지는 non-skin 색 (파랑).
    """
    skin_rgb = _make_pixel_in_box()
    non_skin = _make_pixel_outside_box()
    crop = np.tile(non_skin, (10, 10, 1)).astype(np.uint8)

    garment_mask = np.zeros((10, 10), dtype=bool)
    garment_mask[:, 4:] = True

    skin_class_mask = np.zeros((10, 10), dtype=bool)
    skin_class_mask[:, :4] = True

    if edge_skin_color:
        crop[5, 4] = skin_rgb   # garment 경계 바로 옆 → skin-adjacent zone 안
    if inner_skin_color:
        crop[5, 9] = skin_rgb   # garment 내부 먼 곳 → skin-adjacent zone 밖
    return crop, garment_mask, skin_class_mask


def test_spatial_preserves_inner_pattern_drops_edge_leak() -> None:
    """내부 skin-color pattern 은 보존, 경계 leak 만 drop."""
    pytest.importorskip("scipy.ndimage")
    crop, garment_mask, skin_class_mask = _build_spatial_scene()
    cleaned, ratio, kept_whole = drop_skin_adaptive_spatial(
        crop, garment_mask, skin_class_mask,
        keep_threshold_pct=0.5, upper_ceiling_pct=0.97,
        skin_dilate_iterations=2,
    )
    # garment 60 pixel 중 edge 1, inner 1 = 2 개 skin. ratio = 2/60 ≈ 0.033.
    # threshold 0.5 미만 → spatial drop 분기. dilate=2 이면 edge 는 skin_zone 안, inner 는 밖.
    # 따라서 edge 1개만 drop → cleaned = 60 - 1 = 59.
    assert ratio == pytest.approx(2 / 60)
    assert kept_whole is False
    assert cleaned.shape[0] == 59


def test_spatial_dilate_zero_drops_all_inside_same_as_adaptive() -> None:
    """skin_dilate_iterations=0 이면 spatial 방어 비활성 — box-안 pixel 전부 drop."""
    pytest.importorskip("scipy.ndimage")
    crop, garment_mask, skin_class_mask = _build_spatial_scene()
    cleaned, _ratio, _ = drop_skin_adaptive_spatial(
        crop, garment_mask, skin_class_mask,
        keep_threshold_pct=0.5, upper_ceiling_pct=0.97,
        skin_dilate_iterations=0,
    )
    # dilate off → inner skin-color 도 drop → 60 - 2 = 58.
    assert cleaned.shape[0] == 58


def test_spatial_no_skin_class_pixels_means_no_drop() -> None:
    """skin_class_mask 가 전부 False 면 drop 0 (garment 내 skin-color pixel 이어도)."""
    pytest.importorskip("scipy.ndimage")
    crop, garment_mask, _ = _build_spatial_scene()
    empty_skin = np.zeros_like(garment_mask)
    cleaned, _ratio, _ = drop_skin_adaptive_spatial(
        crop, garment_mask, empty_skin,
        keep_threshold_pct=0.5, upper_ceiling_pct=0.97,
        skin_dilate_iterations=4,
    )
    # skin_zone 이 empty → drop_mask 전부 False → garment 60 pixel 전체 보존.
    assert cleaned.shape[0] == 60


def test_spatial_ceiling_takes_precedence() -> None:
    """ratio > ceiling 이면 spatial 무관하게 garment 통째 drop."""
    pytest.importorskip("scipy.ndimage")
    skin_rgb = _make_pixel_in_box()
    # 전체가 skin 색인 garment.
    crop = np.tile(skin_rgb, (10, 10, 1)).astype(np.uint8)
    garment_mask = np.zeros((10, 10), dtype=bool)
    garment_mask[:, 4:] = True
    skin_class_mask = np.zeros((10, 10), dtype=bool)  # skin class 가 없어도 ceiling 우선.
    cleaned, ratio, _ = drop_skin_adaptive_spatial(
        crop, garment_mask, skin_class_mask,
        keep_threshold_pct=0.5, upper_ceiling_pct=0.97,
        skin_dilate_iterations=4,
    )
    assert ratio == pytest.approx(1.0)
    assert cleaned.shape[0] == 0


# --------------------------------------------------------------------------- #
# drop_skin_2layer — Phase 3 재설계 (primary: segformer skin / secondary: LAB box)
# --------------------------------------------------------------------------- #

# LAB [60, 15, 20] = skin box 안. RGB 로 변환한 결정론 값.
SKIN_BEIGE_RGB: tuple[int, int, int] = (182, 134, 110)
# LAB ~[49, 71, 48] — a 채널이 box max(29.6) 훨씬 초과 → box 밖.
VIBRANT_RED_RGB: tuple[int, int, int] = (230, 30, 40)


def _fill_crop(h: int, w: int, rgb: tuple[int, int, int]) -> np.ndarray:
    crop = np.empty((h, w, 3), dtype=np.uint8)
    crop[:, :] = rgb
    return crop


def test_drop_skin_2layer_primary_only_segformer_overlap() -> None:
    # garment_mask 15 pixel 중 5 pixel 이 segformer_skin 과 overlap → primary 5 drop.
    # 색은 전부 VIBRANT_RED (LAB 박스 밖) 라 secondary 0.
    crop = _fill_crop(5, 5, VIBRANT_RED_RGB)
    garment_mask = np.zeros((5, 5), dtype=bool)
    garment_mask[:, 0:3] = True   # 15 pixel
    skin_mask = np.zeros((5, 5), dtype=bool)
    skin_mask[:, 0:1] = True      # 5 pixel, garment 와 overlap
    cleaned, primary, secondary = drop_skin_2layer(crop, garment_mask, skin_mask)
    assert primary == 5
    assert secondary == 0
    assert cleaned.shape == (10, 3)


def test_drop_skin_2layer_secondary_only_lab_edge_leak() -> None:
    # segformer_skin_mask 비어있음 (skin 미검출). garment 중 3/10 픽셀이 skin-beige.
    # ratio 0.3 < threshold 0.5 → edge leak 분기: 3 pixel drop.
    crop = np.empty((1, 10, 3), dtype=np.uint8)
    crop[:, 0:3] = SKIN_BEIGE_RGB
    crop[:, 3:10] = VIBRANT_RED_RGB
    garment_mask = np.ones((1, 10), dtype=bool)
    skin_mask = np.zeros((1, 10), dtype=bool)
    cleaned, primary, secondary = drop_skin_2layer(crop, garment_mask, skin_mask)
    assert primary == 0
    assert secondary == 3
    assert cleaned.shape == (7, 3)


def test_drop_skin_2layer_both_layers_active() -> None:
    # garment 16 pixel. segformer skin overlap 4 (primary drop).
    # 나머지 12 중 4 는 skin-beige (LAB box 안, ratio 4/12=0.33 < 0.5 → edge leak).
    crop = np.empty((4, 8, 3), dtype=np.uint8)
    crop[:, :] = VIBRANT_RED_RGB
    crop[0:2, 2:4] = SKIN_BEIGE_RGB   # 4 pixel skin-beige, 이후 garment 적용 후 LAB-drop 대상
    garment_mask = np.zeros((4, 8), dtype=bool)
    garment_mask[:, 0:4] = True       # 16 pixel
    skin_mask = np.zeros((4, 8), dtype=bool)
    skin_mask[:, 0:1] = True          # 4 pixel (garment 와 overlap 4)
    cleaned, primary, secondary = drop_skin_2layer(crop, garment_mask, skin_mask)
    assert primary == 4
    assert secondary == 4
    assert cleaned.shape == (8, 3)   # 16 - 4 primary - 4 secondary


def test_drop_skin_2layer_upper_ceiling_drops_segment() -> None:
    # garment 전부 skin-beige (ratio 1.0 > 0.97). mis-seg 판정 → empty 반환.
    crop = _fill_crop(3, 3, SKIN_BEIGE_RGB)
    garment_mask = np.ones((3, 3), dtype=bool)
    skin_mask = np.zeros((3, 3), dtype=bool)
    cleaned, primary, secondary = drop_skin_2layer(crop, garment_mask, skin_mask)
    assert cleaned.shape == (0, 3)
    assert primary == 0
    assert secondary == 9


def test_drop_skin_2layer_skin_tone_garment_preserved() -> None:
    # 8/10 = 0.8 > threshold 0.5 ≤ ceiling 0.97 → skin-tone 의류 보존. LAB drop 생략.
    crop = np.empty((1, 10, 3), dtype=np.uint8)
    crop[:, 0:8] = SKIN_BEIGE_RGB
    crop[:, 8:10] = VIBRANT_RED_RGB
    garment_mask = np.ones((1, 10), dtype=bool)
    skin_mask = np.zeros((1, 10), dtype=bool)
    cleaned, primary, secondary = drop_skin_2layer(crop, garment_mask, skin_mask)
    assert primary == 0
    assert secondary == 0
    assert cleaned.shape == (10, 3)


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
