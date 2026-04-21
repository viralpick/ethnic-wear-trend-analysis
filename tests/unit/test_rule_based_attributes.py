from __future__ import annotations

from datetime import datetime

from attributes.extract_text_attributes import extract_rule_based
from contracts.common import (
    ClassificationMethod,
    ContentSource,
    EmbellishmentIntensity,
    Fabric,
    GarmentType,
    Occasion,
    StylingCombo,
    Technique,
)
from contracts.normalized import NormalizedContentItem


def _make(hashtags: list[str], text_blob: str = "", post_id: str = "t") -> NormalizedContentItem:
    return NormalizedContentItem(
        source=ContentSource.INSTAGRAM,
        source_post_id=post_id,
        text_blob=text_blob,
        hashtags=hashtags,
        image_urls=[],
        post_date=datetime(2026, 4, 21),
        engagement_raw=100,
    )


def test_kurta_set_chikankari_cotton_rule_solvable() -> None:
    item = _make(["#kurtaset", "#chikankari", "#cottonkurta"], "office look for monday")
    state = extract_rule_based(item)

    assert state.garment_type == GarmentType.KURTA_SET
    assert state.technique == Technique.CHIKANKARI
    assert state.fabric == Fabric.COTTON
    # embellishment_intensity 는 technique 로부터 파생 (spec §4.1 ③).
    assert state.embellishment_intensity == EmbellishmentIntensity.EVERYDAY
    assert state.method_per_attribute["garment_type"] == ClassificationMethod.RULE


def test_co_ord_block_print_linen_rule_solvable() -> None:
    item = _make(["#coordset", "#handblockprint", "#linen"], "summer coordinate set")
    state = extract_rule_based(item)

    assert state.garment_type == GarmentType.CO_ORD
    assert state.technique == Technique.BLOCK_PRINT
    assert state.fabric == Fabric.LINEN


def test_office_occasion_detected_via_hashtag() -> None:
    item = _make(["#officewear"], "")
    state = extract_rule_based(item)

    assert state.occasion == Occasion.OFFICE
    assert state.method_per_attribute["occasion"] == ClassificationMethod.RULE


def test_with_palazzo_styling_detected() -> None:
    item = _make(["#palazzoset"], "")
    state = extract_rule_based(item)

    assert state.styling_combo == StylingCombo.WITH_PALAZZO


def test_emoji_only_caption_unclassified() -> None:
    # sample_data 의 ig_unclassifiable_1 이 대표하는 케이스.
    item = _make([], "💙✨🌸")
    state = extract_rule_based(item)

    assert state.garment_type is None
    assert state.technique is None
    assert state.fabric is None
    assert state.method_per_attribute == {}
