"""vision.color_family_preset — LAB → ColorFamily rule + preset mapping."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from contracts.common import ColorFamily
from vision.color_family_preset import lab_to_family, load_preset_family_map


def test_lab_to_family_white_on_white() -> None:
    # L 90 + chroma ~0
    assert lab_to_family(90.0, 0.0, 0.0) is ColorFamily.WHITE_ON_WHITE


def test_lab_to_family_neutral_gray() -> None:
    # chroma < 12, L 중간
    assert lab_to_family(55.0, -2.0, -1.0) is ColorFamily.NEUTRAL


def test_lab_to_family_pastel() -> None:
    # L 78, chroma 20 — soft pink
    assert lab_to_family(78.0, 15.0, 5.0) is ColorFamily.PASTEL


def test_lab_to_family_jewel() -> None:
    # L 35, 강한 chroma — sapphire
    assert lab_to_family(35.0, 16.0, -44.0) is ColorFamily.JEWEL


def test_lab_to_family_bright_saffron() -> None:
    # L 60, chroma 60+ — saffron orange
    assert lab_to_family(60.0, 45.0, 55.0) is ColorFamily.BRIGHT


def test_lab_to_family_earth_rust() -> None:
    # L 45, 중간 chroma, a/b 양수
    assert lab_to_family(45.0, 28.0, 30.0) is ColorFamily.EARTH


def test_lab_to_family_never_returns_dual_or_multi() -> None:
    # single LAB 로는 DUAL_TONE / MULTICOLOR 판정 불가 — 이 rule 은 안 내놓음
    result = lab_to_family(50.0, 10.0, 10.0)
    assert result not in (ColorFamily.DUAL_TONE, ColorFamily.MULTICOLOR)


def test_load_preset_family_map_roundtrip(tmp_path: Path) -> None:
    preset_file = tmp_path / "preset.json"
    preset_file.write_text(
        json.dumps(
            [
                {"name": "ivory", "hex": "#F5F2E8", "lab": [94.0, 0.5, 5.0], "origin": "self"},
                {"name": "black", "hex": "#141314", "lab": [6.3, 0.9, -0.5], "origin": "self"},
                {"name": "saffron", "hex": "#DB633B", "lab": [55.0, 35.0, 50.0], "origin": "self"},
            ]
        ),
        encoding="utf-8",
    )
    m = load_preset_family_map(preset_file)
    assert m["ivory"] is ColorFamily.WHITE_ON_WHITE
    assert m["black"] is ColorFamily.NEUTRAL
    assert m["saffron"] is ColorFamily.BRIGHT


def test_load_preset_family_map_missing_lab_raises(tmp_path: Path) -> None:
    preset_file = tmp_path / "bad.json"
    preset_file.write_text(
        json.dumps([{"name": "bad", "hex": "#000000"}]),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="lab 필드"):
        load_preset_family_map(preset_file)


def test_load_preset_family_map_real_preset_smoke() -> None:
    # 실 50-color preset 이 엣지케이스 없이 로드되는지 smoke
    preset_path = (
        Path(__file__).resolve().parents[2]
        / "outputs"
        / "color_preset"
        / "color_preset.json"
    )
    if not preset_path.exists():
        pytest.skip("color_preset.json 미생성 (빌드 선행 필요)")
    m = load_preset_family_map(preset_path)
    assert len(m) == 50
    # 5 family 중 하나 (DUAL_TONE/MULTICOLOR 는 single hex 에서 안 나와야)
    allowed = {
        ColorFamily.WHITE_ON_WHITE,
        ColorFamily.NEUTRAL,
        ColorFamily.PASTEL,
        ColorFamily.JEWEL,
        ColorFamily.BRIGHT,
        ColorFamily.EARTH,
    }
    assert set(m.values()).issubset(allowed)
