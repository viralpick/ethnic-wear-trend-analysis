"""vision_normalize pinning — Gemini raw → enum 매핑 + case D drop."""
from __future__ import annotations

import pytest

from aggregation.vision_normalize import (
    normalize_fabric,
    normalize_garment_for_cluster,
    normalize_technique,
)
from contracts.common import Fabric, GarmentType, Technique
from contracts.vision import EthnicOutfit


def _outfit(
    *,
    upper: str | None = "kurta",
    lower: str | None = "palazzo",
    fabric: str | None = "cotton",
    technique: str | None = "block_print",
    upper_is_ethnic: bool | None = True,
    lower_is_ethnic: bool | None = True,
    dress_as_single: bool = False,
) -> EthnicOutfit:
    return EthnicOutfit(
        person_bbox=(0.1, 0.1, 0.5, 0.7),
        person_bbox_area_ratio=0.35,
        upper_garment_type=upper,
        upper_is_ethnic=upper_is_ethnic,
        lower_garment_type=lower,
        lower_is_ethnic=lower_is_ethnic,
        dress_as_single=dress_as_single,
        fabric=fabric,
        technique=technique,
        color_preset_picks_top3=[],
    )


# ─────────────── garment_type ───────────────

@pytest.mark.parametrize("raw,expected", [
    ("kurta", GarmentType.STRAIGHT_KURTA),
    ("kurti", GarmentType.STRAIGHT_KURTA),
    ("saree", GarmentType.CASUAL_SAREE),
    ("choli", GarmentType.CASUAL_SAREE),
    ("blouse", GarmentType.CASUAL_SAREE),
    ("anarkali", GarmentType.ANARKALI),
    ("tunic", GarmentType.TUNIC),
    ("kaftan", GarmentType.ETHNIC_DRESS),
    ("lehenga", GarmentType.ETHNIC_DRESS),
    ("dress", GarmentType.ETHNIC_DRESS),
    ("kameez", GarmentType.KURTA_SET),
    ("sherwani", GarmentType.ETHNIC_SHIRT),
    ("Co-Ord", GarmentType.CO_ORD),  # case-insensitive + hyphen normalize
])
def test_garment_normalize_ethnic_upper(raw, expected) -> None:
    out = _outfit(upper=raw, upper_is_ethnic=True)
    assert normalize_garment_for_cluster(out) == expected


def test_garment_case_d_lower_only_ethnic_drops() -> None:
    """case D — upper 비-ethnic + lower ethnic → cluster fan-out 미참여."""
    out = _outfit(upper="t_shirt", upper_is_ethnic=False, lower="palazzo", lower_is_ethnic=True)
    assert normalize_garment_for_cluster(out) is None


def test_garment_unmapped_word_returns_none() -> None:
    """매핑 외 단어 (`top`, `crop_top` 등) → None."""
    out = _outfit(upper="crop_top", upper_is_ethnic=False)
    assert normalize_garment_for_cluster(out) is None


def test_garment_dress_as_single_uses_upper_only() -> None:
    out = _outfit(
        upper="anarkali", lower=None,
        upper_is_ethnic=True, lower_is_ethnic=None,
        dress_as_single=True,
    )
    assert normalize_garment_for_cluster(out) == GarmentType.ANARKALI


def test_garment_dress_as_single_non_ethnic_drops() -> None:
    out = _outfit(
        upper="cocktail_dress", lower=None,
        upper_is_ethnic=False, lower_is_ethnic=None,
        dress_as_single=True,
    )
    assert normalize_garment_for_cluster(out) is None


def test_garment_none_upper_returns_none() -> None:
    out = _outfit(upper=None, upper_is_ethnic=None)
    assert normalize_garment_for_cluster(out) is None


# ─────────────── fabric ───────────────

@pytest.mark.parametrize("raw,expected", [
    ("cotton", Fabric.COTTON),
    ("silk", Fabric.SILK),
    ("organza", Fabric.ORGANZA),
    ("satin", Fabric.SATIN),
    ("net", Fabric.NET),
    ("velvet", Fabric.VELVET),
    ("georgette", Fabric.GEORGETTE),
    ("chiffon", Fabric.CHIFFON),
])
def test_fabric_normalize_known(raw, expected) -> None:
    out = _outfit(fabric=raw)
    assert normalize_fabric(out) == expected


@pytest.mark.parametrize("raw", ["denim", "knit", "lace", "leather", "polyester"])
def test_fabric_unmapped_returns_none(raw) -> None:
    out = _outfit(fabric=raw)
    assert normalize_fabric(out) is None


# ─────────────── technique ───────────────

@pytest.mark.parametrize("raw,expected", [
    ("solid", Technique.SOLID),
    ("plain", Technique.SOLID),
    ("embroidery", Technique.THREAD_EMBROIDERY),
    ("thread_embroidery", Technique.THREAD_EMBROIDERY),
    ("print", Technique.DIGITAL_PRINT),
    ("printed", Technique.DIGITAL_PRINT),
    ("block_print", Technique.BLOCK_PRINT),
    ("chikankari", Technique.CHIKANKARI),
    ("mirror_work", Technique.MIRROR_WORK),
    ("schiffli", Technique.SCHIFFLI),
    ("ikat", Technique.IKAT),
    ("brocade", Technique.BROCADE),
    ("beadwork", Technique.BEADWORK),
    ("bandhani", Technique.BANDHANI),
    ("zardosi", Technique.ZARI_ZARDOSI),
    ("zardozi", Technique.ZARI_ZARDOSI),
    ("zari", Technique.ZARI_ZARDOSI),
    ("kalamkari", Technique.KALAMKARI),
    ("sequins", Technique.SEQUIN_WORK),
    ("sequin_work", Technique.SEQUIN_WORK),
    ("woven", Technique.SELF_TEXTURE),
    ("Gota Patti", Technique.GOTA_PATTI),  # case + space normalize
])
def test_technique_normalize_known(raw, expected) -> None:
    out = _outfit(technique=raw)
    assert normalize_technique(out) == expected


@pytest.mark.parametrize("raw", ["crochet", "batik", "tie_dye", "pleating"])
def test_technique_unmapped_returns_none(raw) -> None:
    out = _outfit(technique=raw)
    assert normalize_technique(out) is None
