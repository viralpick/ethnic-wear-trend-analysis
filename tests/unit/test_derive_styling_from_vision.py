"""derive_styling_from_outfit pinning — M3.I P0 5 매핑 + P1 3 매핑 + None fallback.

P1 (co_ord_set / dupatta / jacket) 은 EthnicOutfit v0.8 슬롯 (is_co_ord_set / outer_layer)
기반 매핑.
"""
from __future__ import annotations

from attributes.derive_styling_from_vision import derive_styling_from_outfit
from contracts.common import StylingCombo
from contracts.vision import EthnicOutfit


def _outfit(
    *,
    upper: str | None = "kurta",
    lower: str | None = None,
    dress_as_single: bool = False,
    is_co_ord_set: bool | None = None,
    outer_layer: str | None = None,
) -> EthnicOutfit:
    return EthnicOutfit(
        person_bbox=(0.1, 0.1, 0.5, 0.7),
        person_bbox_area_ratio=0.35,
        upper_garment_type=upper,
        lower_garment_type=lower,
        dress_as_single=dress_as_single,
        is_co_ord_set=is_co_ord_set,
        outer_layer=outer_layer,
        color_preset_picks_top3=[],
    )


def test_dress_as_single_returns_standalone() -> None:
    # saree drape / lehenga-as-single / ethnic_dress 케이스.
    outfit = _outfit(upper="saree", lower=None, dress_as_single=True)
    assert derive_styling_from_outfit(outfit) is StylingCombo.STANDALONE


def test_dress_as_single_overrides_lower_garment_type() -> None:
    # dress_as_single=True 면 lower 가 우연히 채워져도 STANDALONE 우선.
    outfit = _outfit(upper="saree", lower="palazzo", dress_as_single=True)
    assert derive_styling_from_outfit(outfit) is StylingCombo.STANDALONE


def test_palazzo_maps_to_with_palazzo() -> None:
    outfit = _outfit(upper="kurta", lower="palazzo")
    assert derive_styling_from_outfit(outfit) is StylingCombo.WITH_PALAZZO


def test_churidar_and_salwar_map_to_with_churidar() -> None:
    assert derive_styling_from_outfit(
        _outfit(upper="kurta", lower="churidar")
    ) is StylingCombo.WITH_CHURIDAR
    assert derive_styling_from_outfit(
        _outfit(upper="kurta", lower="salwar")
    ) is StylingCombo.WITH_CHURIDAR


def test_pants_trousers_pyjama_map_to_with_pants() -> None:
    for value in ("pants", "trousers", "pyjama"):
        assert derive_styling_from_outfit(
            _outfit(upper="kurta", lower=value)
        ) is StylingCombo.WITH_PANTS, value


def test_jeans_and_denim_map_to_with_jeans() -> None:
    assert derive_styling_from_outfit(
        _outfit(upper="kurta", lower="jeans")
    ) is StylingCombo.WITH_JEANS
    assert derive_styling_from_outfit(
        _outfit(upper="kurta", lower="denim")
    ) is StylingCombo.WITH_JEANS


def test_unknown_lower_without_outer_returns_none() -> None:
    # lower 가 매핑 표 외 + outer_layer 도 없으면 None. (dupatta 가 lower 슬롯에 잘못
    # 들어왔을 때의 LLM 실수 — 매핑하지 않음, outer_layer 슬롯에서만 처리).
    outfit = _outfit(upper="kurta", lower="dupatta")
    assert derive_styling_from_outfit(outfit) is None


def test_lower_none_and_not_dress_returns_none() -> None:
    outfit = _outfit(upper="kurta", lower=None, dress_as_single=False)
    assert derive_styling_from_outfit(outfit) is None


def test_uppercase_lower_normalized_to_lowercase_match() -> None:
    # LLM 이 대문자 배합으로 흘려도 .lower() 로 정규화.
    outfit = _outfit(upper="kurta", lower="Palazzo")
    assert derive_styling_from_outfit(outfit) is StylingCombo.WITH_PALAZZO


# ---------------------------------------------------------------------------
# P1 매핑 (M3.I 2026-04-28)
# ---------------------------------------------------------------------------


def test_is_co_ord_set_maps_to_co_ord_set() -> None:
    # upper+lower 가 동일 fabric/print 의 매칭 set → CO_ORD_SET.
    outfit = _outfit(upper="kurta", lower="palazzo", is_co_ord_set=True)
    assert derive_styling_from_outfit(outfit) is StylingCombo.CO_ORD_SET


def test_co_ord_set_overrides_lower_garment_type() -> None:
    # is_co_ord_set=True 면 lower 가 palazzo/churidar 여도 CO_ORD_SET 우선.
    outfit = _outfit(upper="kurta", lower="churidar", is_co_ord_set=True)
    assert derive_styling_from_outfit(outfit) is StylingCombo.CO_ORD_SET


def test_dress_as_single_overrides_co_ord_set() -> None:
    # 단일 piece 가 co_ord 보다 우선 (dress_as_single 은 최상위).
    outfit = _outfit(
        upper="saree", lower=None, dress_as_single=True, is_co_ord_set=True,
    )
    assert derive_styling_from_outfit(outfit) is StylingCombo.STANDALONE


def test_outer_layer_dupatta_maps_to_with_dupatta() -> None:
    # bottom 미상이라 outer_layer 가 dominant signal.
    outfit = _outfit(upper="kurta", lower=None, outer_layer="dupatta")
    assert derive_styling_from_outfit(outfit) is StylingCombo.WITH_DUPATTA


def test_outer_layer_shawl_and_stole_map_to_with_dupatta() -> None:
    for value in ("shawl", "stole"):
        outfit = _outfit(upper="kurta", lower=None, outer_layer=value)
        assert derive_styling_from_outfit(outfit) is StylingCombo.WITH_DUPATTA, value


def test_outer_layer_jacket_family_maps_to_with_jacket() -> None:
    for value in ("jacket", "cardigan", "nehru", "shrug"):
        outfit = _outfit(upper="kurta", lower=None, outer_layer=value)
        assert derive_styling_from_outfit(outfit) is StylingCombo.WITH_JACKET, value


def test_specific_lower_overrides_outer_layer() -> None:
    # 의도적 priority — palazzo + dupatta 면 WITH_PALAZZO (dupatta 는 default 빈도 너무
    # 높아 categorization 무의미해짐 방지).
    outfit = _outfit(upper="kurta", lower="palazzo", outer_layer="dupatta")
    assert derive_styling_from_outfit(outfit) is StylingCombo.WITH_PALAZZO


def test_outer_layer_unknown_returns_none() -> None:
    # outer_layer free-form word 가 매핑 표 외면 None (fuzzy 매칭 없음).
    outfit = _outfit(upper="kurta", lower=None, outer_layer="poncho")
    assert derive_styling_from_outfit(outfit) is None


def test_outer_layer_uppercase_normalized() -> None:
    outfit = _outfit(upper="kurta", lower=None, outer_layer="Dupatta")
    assert derive_styling_from_outfit(outfit) is StylingCombo.WITH_DUPATTA
