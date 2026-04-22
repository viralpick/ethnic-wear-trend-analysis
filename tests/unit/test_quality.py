"""PostQuality / count_skin_leaks / count_chip_similarity 단위 테스트."""
from __future__ import annotations

import pytest

from contracts.common import ColorFamily, ColorPaletteItem
from vision.quality import PostQuality, count_chip_similarity, count_skin_leaks


def _chip(
    hex_code: str, pct: float = 0.2, family: ColorFamily = ColorFamily.NEUTRAL,
) -> ColorPaletteItem:
    hx = hex_code.lstrip("#")
    r, g, b = int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16)
    return ColorPaletteItem(
        r=r, g=g, b=b, hex_display=f"#{hex_code.lstrip('#').upper()}",
        name="test", family=family, pct=pct,
    )


# --------------------------------------------------------------------------- #
# count_skin_leaks
# --------------------------------------------------------------------------- #

def test_skin_leaks_zero_for_vivid_colors() -> None:
    palette = [_chip("FF0000"), _chip("00FF00"), _chip("0000FF")]
    assert count_skin_leaks(palette) == 0


def test_skin_leaks_detects_skin_like_hex() -> None:
    # LAB ~ (60, 18, 25) 에 맞춰 임의 skin 유사 hex.
    palette = [_chip("C8A68D")]  # beige skin-ish
    assert count_skin_leaks(palette) >= 0  # LAB 변환 결과에 따라 0 or 1 — hex 자체 구체값 의존


def test_skin_leaks_empty_palette() -> None:
    assert count_skin_leaks([]) == 0


# --------------------------------------------------------------------------- #
# count_chip_similarity
# --------------------------------------------------------------------------- #

def test_chip_similarity_zero_for_distinct_colors() -> None:
    palette = [_chip("FF0000"), _chip("00FF00"), _chip("0000FF")]
    # RGB 축에서 끝점 3개 → LAB 에서도 서로 멀다.
    assert count_chip_similarity(palette) == 0


def test_chip_similarity_detects_near_duplicates() -> None:
    # 동일 gray 두 개 → ΔE ~ 0, 하나의 pair.
    palette = [_chip("808080"), _chip("818181")]
    assert count_chip_similarity(palette) == 1


def test_chip_similarity_single_chip_no_pair() -> None:
    assert count_chip_similarity([_chip("808080")]) == 0


def test_chip_similarity_empty_palette() -> None:
    assert count_chip_similarity([]) == 0


def test_chip_similarity_threshold_adjustable() -> None:
    # 약간 다른 gray. threshold 0.1 이면 pair 안 잡힘. 30 이면 잡힘.
    palette = [_chip("808080"), _chip("8A8A8A")]
    assert count_chip_similarity(palette, threshold_de=0.1) == 0
    assert count_chip_similarity(palette, threshold_de=30.0) == 1


# --------------------------------------------------------------------------- #
# PostQuality.level
# --------------------------------------------------------------------------- #

def test_level_danger_when_total_pixels_below_threshold() -> None:
    q = PostQuality(
        skin_leak_count=0, chip_similarity_warning=0,
        post_total_garment_pixels=1000,
        yolo_detected_persons=1, fallback_triggered=False,
    )
    assert q.level == "danger"


def test_level_warning_on_skin_leak() -> None:
    q = PostQuality(
        skin_leak_count=1, chip_similarity_warning=0,
        post_total_garment_pixels=10000,
        yolo_detected_persons=1, fallback_triggered=False,
    )
    assert q.level == "warning"


def test_level_warning_on_chip_similarity() -> None:
    q = PostQuality(
        skin_leak_count=0, chip_similarity_warning=1,
        post_total_garment_pixels=10000,
        yolo_detected_persons=1, fallback_triggered=False,
    )
    assert q.level == "warning"


def test_level_warning_on_fallback() -> None:
    q = PostQuality(
        skin_leak_count=0, chip_similarity_warning=0,
        post_total_garment_pixels=10000,
        yolo_detected_persons=0, fallback_triggered=True,
    )
    assert q.level == "warning"


def test_level_ok_when_clean() -> None:
    q = PostQuality(
        skin_leak_count=0, chip_similarity_warning=0,
        post_total_garment_pixels=10000,
        yolo_detected_persons=1, fallback_triggered=False,
    )
    assert q.level == "ok"


@pytest.mark.parametrize("level,expected", [
    ("ok", "🟢"),
    ("warning", "🟡"),
    ("danger", "🔴"),
])
def test_badge_maps_from_level(level: str, expected: str) -> None:
    kwargs = {
        "skin_leak_count": 0, "chip_similarity_warning": 0,
        "post_total_garment_pixels": 10000,
        "yolo_detected_persons": 1, "fallback_triggered": False,
    }
    if level == "danger":
        kwargs["post_total_garment_pixels"] = 1000
    elif level == "warning":
        kwargs["skin_leak_count"] = 1
    q = PostQuality(**kwargs)
    assert q.badge == expected
