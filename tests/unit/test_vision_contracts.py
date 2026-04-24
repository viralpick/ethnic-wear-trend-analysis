"""contracts/vision.py Pydantic 검증 테스트.

enum 외 값 / bbox 범위 위반 / length 초과는 ValidationError — 상위에서 해당 post drop.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from contracts.common import Silhouette
from contracts.vision import EthnicOutfit, GarmentAnalysis


def _valid_outfit(**overrides) -> dict:
    base = dict(
        person_bbox=(0.1, 0.1, 0.5, 0.7),
        person_bbox_area_ratio=0.35,
        upper_garment_type="kurta",
        lower_garment_type="palazzo",
        dress_as_single=False,
        silhouette=Silhouette.A_LINE,
        fabric="cotton",
        technique="chikankari",
        color_preset_picks_top3=["pool_00", "saffron", "pool_12"],
    )
    base.update(overrides)
    return base


def test_ethnic_outfit_valid() -> None:
    o = EthnicOutfit(**_valid_outfit())
    assert o.upper_garment_type == "kurta"
    assert o.silhouette is Silhouette.A_LINE
    assert o.fabric == "cotton"
    assert o.technique == "chikankari"
    assert len(o.color_preset_picks_top3) == 3


def test_ethnic_outfit_fabric_technique_default_none() -> None:
    # 필드 미지정 시 둘 다 None (LLM 이 불확실 판정 → null 낼 케이스)
    payload = _valid_outfit()
    payload.pop("fabric")
    payload.pop("technique")
    o = EthnicOutfit(**payload)
    assert o.fabric is None
    assert o.technique is None


def test_ethnic_outfit_fabric_technique_null_explicit() -> None:
    o = EthnicOutfit(**_valid_outfit(fabric=None, technique=None))
    assert o.fabric is None
    assert o.technique is None


def test_ethnic_outfit_single_piece() -> None:
    o = EthnicOutfit(
        **_valid_outfit(
            upper_garment_type="saree",
            lower_garment_type=None,
            dress_as_single=True,
            silhouette=None,
        )
    )
    assert o.dress_as_single is True
    assert o.lower_garment_type is None
    assert o.silhouette is None


def test_ethnic_outfit_silhouette_out_of_enum_rejected() -> None:
    with pytest.raises(ValidationError):
        EthnicOutfit(**_valid_outfit(silhouette="mermaid"))


def test_ethnic_outfit_bbox_origin_out_of_range_rejected() -> None:
    with pytest.raises(ValidationError):
        EthnicOutfit(**_valid_outfit(person_bbox=(1.2, 0.1, 0.3, 0.3)))


def test_ethnic_outfit_bbox_exceeds_image_rejected() -> None:
    with pytest.raises(ValidationError):
        EthnicOutfit(**_valid_outfit(person_bbox=(0.8, 0.8, 0.5, 0.5)))


def test_ethnic_outfit_bbox_zero_size_rejected() -> None:
    with pytest.raises(ValidationError):
        EthnicOutfit(**_valid_outfit(person_bbox=(0.1, 0.1, 0.0, 0.3)))


def test_ethnic_outfit_area_ratio_out_of_range_rejected() -> None:
    with pytest.raises(ValidationError):
        EthnicOutfit(**_valid_outfit(person_bbox_area_ratio=1.5))


def test_ethnic_outfit_extra_field_rejected() -> None:
    payload = _valid_outfit()
    payload["unexpected_field"] = "leak"
    with pytest.raises(ValidationError):
        EthnicOutfit(**payload)


def test_ethnic_outfit_color_picks_over_three_rejected() -> None:
    with pytest.raises(ValidationError):
        EthnicOutfit(
            **_valid_outfit(
                color_preset_picks_top3=["a", "b", "c", "d"],
            )
        )


def test_ethnic_outfit_color_picks_empty_allowed() -> None:
    # 모호한 경우 LLM 이 빈 array 내도 허용 — 상위에서 drop 여부 결정
    o = EthnicOutfit(**_valid_outfit(color_preset_picks_top3=[]))
    assert o.color_preset_picks_top3 == []


def test_garment_analysis_binary_false_empty_outfits() -> None:
    a = GarmentAnalysis(is_india_ethnic_wear=False, outfits=[])
    assert a.is_india_ethnic_wear is False
    assert a.outfits == []


def test_garment_analysis_max_two_outfits() -> None:
    o = _valid_outfit()
    with pytest.raises(ValidationError):
        GarmentAnalysis(
            is_india_ethnic_wear=True,
            outfits=[EthnicOutfit(**o)] * 3,
        )


def test_garment_analysis_json_roundtrip() -> None:
    a = GarmentAnalysis(
        is_india_ethnic_wear=True,
        outfits=[EthnicOutfit(**_valid_outfit())],
    )
    serialized = a.model_dump(mode="json")
    restored = GarmentAnalysis.model_validate(serialized)
    assert restored == a
