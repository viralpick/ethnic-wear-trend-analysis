"""GarmentInstance + classify_single_color + find_duplicate_groups + aggregate_post_palette 단위."""
from __future__ import annotations

import pytest

from contracts.common import ColorFamily, ColorPaletteItem
from vision.garment_instance import (
    GarmentInstance,
    _weight_for,
    aggregate_post_palette,
    classify_single_color,
    find_duplicate_groups,
)


def _chip(r: int, g: int, b: int, pct: float = 1.0) -> ColorPaletteItem:
    return ColorPaletteItem(
        r=r, g=g, b=b,
        hex_display=f"#{r:02X}{g:02X}{b:02X}",
        name="test", family=ColorFamily.NEUTRAL, pct=pct,
    )


def _inst(
    instance_id: str = "t:p0:upper-clothes",
    garment_class: str = "upper-clothes",
    palette: list[ColorPaletteItem] | None = None,
    pixel_count: int = 1000,
    frame_id: str = "t",
) -> GarmentInstance:
    # palette=[] (빈 리스트) 를 명시적으로 빈 상태로 보존 (falsy 처리 X).
    effective_palette = palette if palette is not None else [_chip(100, 100, 100)]
    return GarmentInstance(
        instance_id=instance_id,
        frame_id=frame_id,
        bbox=(0, 0, 100, 100),
        garment_class=garment_class,
        palette=effective_palette,
        is_single_color=True,
        pixel_count=pixel_count,
        skin_drop_ratio=0.0,
    )


# --------------------------------------------------------------------------- #
# classify_single_color
# --------------------------------------------------------------------------- #

def test_single_color_when_chips_near_identical() -> None:
    # 거의 같은 gray — ΔE 작음.
    palette = [_chip(100, 100, 100, 0.6), _chip(105, 105, 105, 0.4)]
    assert classify_single_color(palette, max_delta_e=10.0) is True


def test_multi_color_when_chips_far_apart() -> None:
    palette = [_chip(200, 30, 30, 0.6), _chip(30, 30, 200, 0.4)]
    assert classify_single_color(palette, max_delta_e=10.0) is False


def test_single_color_on_empty_or_single_chip() -> None:
    assert classify_single_color([], max_delta_e=10.0) is True
    assert classify_single_color([_chip(100, 100, 100)], max_delta_e=10.0) is True


# --------------------------------------------------------------------------- #
# find_duplicate_groups
# --------------------------------------------------------------------------- #

def test_duplicate_groups_same_class_similar_color() -> None:
    # 두 upper-clothes instance 가 같은 gray ≈ duplicate
    a = _inst("a", palette=[_chip(100, 100, 100)])
    b = _inst("b", palette=[_chip(105, 105, 105)])
    c = _inst("c", palette=[_chip(200, 30, 30)])  # 다른 색
    groups = find_duplicate_groups([a, b, c], max_delta_e=10.0)
    assert len(groups) == 2
    assert {i.instance_id for i in groups[0]} == {"a", "b"}
    assert {i.instance_id for i in groups[1]} == {"c"}


def test_duplicate_groups_different_class_not_merged() -> None:
    a = _inst("a", garment_class="upper-clothes", palette=[_chip(100, 100, 100)])
    # 같은 색이지만 garment_class 가 달라 merge 안 되어야 함.
    b = _inst("b", garment_class="pants", palette=[_chip(100, 100, 100)])
    groups = find_duplicate_groups([a, b], max_delta_e=10.0)
    assert len(groups) == 2


def test_duplicate_groups_empty_palette_isolated() -> None:
    a = _inst("a", palette=[])
    b = _inst("b", palette=[_chip(100, 100, 100)])
    groups = find_duplicate_groups([a, b], max_delta_e=10.0)
    # empty palette instance 는 자기만의 group
    assert len(groups) == 2


# --------------------------------------------------------------------------- #
# _weight_for
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("count,formula,expected", [
    (1, "log", 1.0),
    (2, "log", 1.0 + __import__("math").log(2)),
    (1, "linear", 1.0),
    (5, "linear", 5.0),
    (1, "sqrt", 1.0),
    (4, "sqrt", 2.0),
])
def test_weight_for_supported_formulas(count: int, formula: str, expected: float) -> None:
    assert _weight_for(count, formula) == pytest.approx(expected)


def test_weight_for_unknown_raises() -> None:
    with pytest.raises(ValueError):
        _weight_for(5, "unknown")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# aggregate_post_palette
# --------------------------------------------------------------------------- #

def test_aggregate_weights_duplicates_sublinear() -> None:
    # upper 2개 duplicate (gray) + 1 pants (red) — gray 가 weight 1+log(2)≈1.69 로 부각.
    upper_a = _inst("u_a", "upper-clothes", palette=[_chip(100, 100, 100, 1.0)], pixel_count=500)
    upper_b = _inst("u_b", "upper-clothes", palette=[_chip(105, 105, 105, 1.0)], pixel_count=500)
    pants = _inst("p", "pants", palette=[_chip(200, 30, 30, 1.0)], pixel_count=500)
    palette, groups = aggregate_post_palette(
        [upper_a, upper_b, pants], top_k=5, duplicate_max_delta_e=10.0, weight_formula="log",
    )
    assert len(groups) == 2  # gray group + pants
    # gray 가 pants 보다 더 높은 pct (weight 덕분)
    assert len(palette) == 2
    assert palette[0].r < 150  # gray top-1


def test_aggregate_empty_returns_empty() -> None:
    palette, groups = aggregate_post_palette([], top_k=5)
    assert palette == []
    assert groups == []


def test_aggregate_preserves_multi_color_instance_chips() -> None:
    multi = _inst(
        "m", palette=[
            _chip(200, 30, 30, 0.6),
            _chip(30, 200, 30, 0.4),
        ],
        pixel_count=1000,
    )
    palette, _ = aggregate_post_palette([multi], top_k=5)
    assert len(palette) == 2  # 두 chip 모두 보존


def test_aggregate_normalizes_pct() -> None:
    a = _inst("a", palette=[_chip(100, 100, 100, 1.0)], pixel_count=500)
    b = _inst("b", palette=[_chip(200, 30, 30, 1.0)], pixel_count=500)
    # 서로 다른 색 (class 도 같지만 ΔE 커서 separate group)
    palette, groups = aggregate_post_palette(
        [a, b], top_k=5, duplicate_max_delta_e=5.0,  # threshold 작아서 merge 안 됨
    )
    assert len(groups) == 2
    assert sum(c.pct for c in palette) == pytest.approx(1.0)
