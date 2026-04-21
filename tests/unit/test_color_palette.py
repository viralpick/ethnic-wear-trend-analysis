"""Post-level ColorInfo → palette chip (spec §4.1 ④, §5.4).

Step B 이후: LAB KMeans 기반. 결정론 (random_state=0), weight desc, family 최빈값.
"""
from __future__ import annotations

import pytest

from aggregation.color_palette import build_palette
from contracts.common import ColorFamily
from contracts.enriched import ColorInfo
from settings import PaletteConfig


@pytest.fixture
def palette_cfg() -> PaletteConfig:
    return PaletteConfig(top_k=5)


def test_build_palette_empty_returns_empty(palette_cfg: PaletteConfig) -> None:
    assert build_palette([], palette_cfg) == []


def test_build_palette_single_color(palette_cfg: PaletteConfig) -> None:
    colors = [ColorInfo(r=184, g=212, b=195, family=ColorFamily.PASTEL)]
    palette = build_palette(colors, palette_cfg)
    assert len(palette) == 1
    assert palette[0].pct == pytest.approx(1.0)
    assert palette[0].family == ColorFamily.PASTEL


def test_build_palette_weight_sums_to_one(palette_cfg: PaletteConfig) -> None:
    colors = [
        ColorInfo(r=10, g=10, b=10, family=ColorFamily.NEUTRAL),
        ColorInfo(r=12, g=12, b=12, family=ColorFamily.NEUTRAL),
        ColorInfo(r=200, g=10, b=10, family=ColorFamily.BRIGHT),
        ColorInfo(r=10, g=200, b=10, family=ColorFamily.PASTEL),
    ]
    palette = build_palette(colors, palette_cfg)
    assert sum(item.pct for item in palette) == pytest.approx(1.0)


def test_build_palette_sorted_by_pct_desc() -> None:
    # top_k=3 으로 제한. 4 point (gray 2 + red 1 + green 1) → KMeans 가 gray 두 개를 묶어
    # [gray(2), red(1), green(1)] 3 cluster 생성. 첫 pct 0.5 가 top.
    cfg = PaletteConfig(top_k=3)
    colors = [
        ColorInfo(r=10, g=10, b=10, family=ColorFamily.NEUTRAL),
        ColorInfo(r=15, g=15, b=15, family=ColorFamily.NEUTRAL),
        ColorInfo(r=200, g=10, b=10, family=ColorFamily.BRIGHT),
        ColorInfo(r=10, g=200, b=10, family=ColorFamily.PASTEL),
    ]
    palette = build_palette(colors, cfg)
    pcts = [item.pct for item in palette]
    assert pcts == sorted(pcts, reverse=True)
    assert palette[0].pct == pytest.approx(0.5)


def test_build_palette_family_majority_wins() -> None:
    # top_k=2 로 강제. PASTEL 2개는 어두운 gray (근접), NEUTRAL 1개는 밝은 white (멀리)
    # → cluster 0 = {dark PASTEL × 2}, cluster 1 = {light NEUTRAL × 1}. 첫 cluster family=PASTEL.
    cfg = PaletteConfig(top_k=2)
    colors = [
        ColorInfo(r=10, g=10, b=10, family=ColorFamily.PASTEL),
        ColorInfo(r=15, g=15, b=15, family=ColorFamily.PASTEL),
        ColorInfo(r=250, g=250, b=250, family=ColorFamily.NEUTRAL),
    ]
    palette = build_palette(colors, cfg)
    assert len(palette) == 2
    assert palette[0].family == ColorFamily.PASTEL
    assert palette[0].pct == pytest.approx(2 / 3)


def test_build_palette_top_k_clamp_by_color_count(palette_cfg: PaletteConfig) -> None:
    # top_k=5 인데 color 가 3개 → 최대 3 cluster.
    colors = [
        ColorInfo(r=10, g=10, b=10, family=ColorFamily.NEUTRAL),
        ColorInfo(r=200, g=10, b=10, family=ColorFamily.BRIGHT),
        ColorInfo(r=10, g=200, b=10, family=ColorFamily.PASTEL),
    ]
    palette = build_palette(colors, palette_cfg)
    assert len(palette) == 3


def test_build_palette_determinism(palette_cfg: PaletteConfig) -> None:
    # random_state=0 고정 → 같은 입력 같은 출력.
    colors = [
        ColorInfo(r=10, g=10, b=10, family=ColorFamily.NEUTRAL),
        ColorInfo(r=12, g=12, b=12, family=ColorFamily.NEUTRAL),
        ColorInfo(r=200, g=10, b=10, family=ColorFamily.BRIGHT),
        ColorInfo(r=10, g=200, b=10, family=ColorFamily.PASTEL),
    ]
    a = build_palette(colors, palette_cfg)
    b = build_palette(colors, palette_cfg)
    assert [(i.r, i.g, i.b, i.pct) for i in a] == [(i.r, i.g, i.b, i.pct) for i in b]


def test_build_palette_hex_display_uppercase(palette_cfg: PaletteConfig) -> None:
    colors = [ColorInfo(r=184, g=212, b=195, family=ColorFamily.PASTEL)]
    palette = build_palette(colors, palette_cfg)
    # hex_display 는 대문자 + # 포함.
    assert palette[0].hex_display.startswith("#")
    assert palette[0].hex_display == palette[0].hex_display.upper()
