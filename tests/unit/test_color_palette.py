"""Color palette: RGB → HEX + 결정론적 bucketing (spec §4.1 ④, §5.4)."""
from __future__ import annotations

import pytest

from aggregation.color_palette import (
    bucket_rgb,
    build_palette,
    rgb_to_hex,
)
from contracts.common import ColorFamily
from contracts.enriched import ColorInfo
from settings import PaletteConfig


@pytest.fixture
def palette_cfg() -> PaletteConfig:
    return PaletteConfig(bucket_size=32, top_k=5)


def test_rgb_to_hex_happy_path() -> None:
    # spec §4.1 ④ 문서 예시.
    assert rgb_to_hex(184, 212, 195) == "#B8D4C3"


def test_rgb_to_hex_zero_pads_low_values() -> None:
    # 0x00 은 2자리로 패딩되어야 한다.
    assert rgb_to_hex(0, 5, 15) == "#00050F"


def test_bucket_rgb_determinism() -> None:
    # 같은 bucket_size 라면 같은 근접색은 같은 버킷.
    a = bucket_rgb(100, 100, 100, 32)
    b = bucket_rgb(120, 115, 110, 32)
    assert a == b == (96, 96, 96)


def test_build_palette_empty_returns_empty(palette_cfg: PaletteConfig) -> None:
    assert build_palette([], palette_cfg) == []


def test_build_palette_uses_most_common_family(palette_cfg: PaletteConfig) -> None:
    # 같은 버킷 안에서 가장 흔한 family 를 대표값으로 쓴다.
    colors = [
        ColorInfo(r=10, g=10, b=10, family=ColorFamily.PASTEL),
        ColorInfo(r=12, g=12, b=12, family=ColorFamily.PASTEL),
        ColorInfo(r=14, g=14, b=14, family=ColorFamily.NEUTRAL),
    ]
    palette = build_palette(colors, palette_cfg)
    assert len(palette) == 1
    assert palette[0].family == ColorFamily.PASTEL


def test_build_palette_top_k_and_pct_sum(palette_cfg: PaletteConfig) -> None:
    # 세 개 버킷을 기대하도록 명확히 떨어진 색상 생성.
    colors = [
        ColorInfo(r=10, g=10, b=10, family=ColorFamily.NEUTRAL),
        ColorInfo(r=15, g=15, b=15, family=ColorFamily.NEUTRAL),  # 같은 버킷
        ColorInfo(r=200, g=10, b=10, family=ColorFamily.BRIGHT),
        ColorInfo(r=10, g=200, b=10, family=ColorFamily.PASTEL),
    ]
    palette = build_palette(colors, palette_cfg)
    # 3 버킷이 정렬되어 첫 버킷이 가장 큰 비율이어야 한다.
    assert len(palette) == 3
    assert palette[0].pct == pytest.approx(2 / 4)
    assert sum(item.pct for item in palette) == pytest.approx(1.0)


def test_build_palette_hex_display_is_midpoint(palette_cfg: PaletteConfig) -> None:
    # 버킷 0~31 (bucket_size=32) → midpoint 16. 실제 입력이 30 이어도 표시값은 #101010.
    colors = [ColorInfo(r=30, g=30, b=30, family=ColorFamily.NEUTRAL)]
    palette = build_palette(colors, palette_cfg)
    assert palette[0].hex_display == "#101010"
    assert palette[0].r == 16
