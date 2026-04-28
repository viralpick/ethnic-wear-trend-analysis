from __future__ import annotations

from datetime import datetime

from attributes.brand_registry import BrandRegistry, load_brand_registry
from attributes.extract_text_attributes import extract_rule_based
from contracts.common import (
    BrandTier,
    ClassificationMethod,
    ContentSource,
    EmbellishmentIntensity,
    Fabric,
    GarmentType,
    Occasion,
    StylingCombo,
    Technique,
)
from contracts.enriched import BrandInfo
from contracts.normalized import NormalizedContentItem


def _make(
    hashtags: list[str],
    text_blob: str = "",
    post_id: str = "t",
    account_handle: str | None = None,
) -> NormalizedContentItem:
    return NormalizedContentItem(
        source=ContentSource.INSTAGRAM,
        source_post_id=post_id,
        text_blob=text_blob,
        hashtags=hashtags,
        image_urls=[],
        post_date=datetime(2026, 4, 21),
        engagement_raw=100,
        account_handle=account_handle,
    )


def _build_test_registry(tmp_path) -> BrandRegistry:  # type: ignore[no-untyped-def]
    import json
    payload = {
        "brands": [
            {
                "id": "myntra",
                "primary_handle": "myntra",
                "aliases": ["myntrafashion"],
                "display_name": "Myntra",
                "category": "marketplace",
                "tier": "mid",
                "country": "IN",
                "notes": "",
            },
            {
                "id": "manish-malhotra",
                "primary_handle": "manishmalhotraworld",
                "aliases": [],
                "display_name": "Manish Malhotra",
                "category": "designer",
                "tier": "premium_everyday",
                "country": "IN",
                "notes": "",
            },
        ],
    }
    p = tmp_path / "br.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return load_brand_registry(p)


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


def test_brand_filled_from_account_handle(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """IG account_handle lookup 으로 brands 채움 (single brand 케이스)."""
    registry = _build_test_registry(tmp_path)
    item = _make([], "stylish look", account_handle="myntra")
    state = extract_rule_based(item, registry)

    assert state.brands == [BrandInfo(name="Myntra", tier=BrandTier.MID)]
    assert state.method_per_attribute["brand"] == ClassificationMethod.RULE


def test_brand_filled_from_caption_mention_when_handle_unknown(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """account_handle 이 unknown 이어도 caption text 의 brand mention 모두 수집."""
    registry = _build_test_registry(tmp_path)
    item = _make(
        [],
        "outfit by @manishmalhotraworld 💃 styled by @somebody",
        account_handle="random_creator",
    )
    state = extract_rule_based(item, registry)

    assert state.brands == [
        BrandInfo(name="Manish Malhotra", tier=BrandTier.PREMIUM_EVERYDAY),
    ]


def test_brand_collects_handle_and_caption_dedup(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """account_handle + caption mention 양쪽 — 각 brand 1번 (dedup), 순서 보존."""
    registry = _build_test_registry(tmp_path)
    item = _make(
        [],
        "haul drop @manishmalhotraworld + @myntrafashion + @myntra repeat",
        account_handle="myntra",
    )
    state = extract_rule_based(item, registry)

    # account_handle 의 myntra 가 첫 brand → 그 다음 caption 순서. myntra 중복 dedup.
    assert state.brands == [
        BrandInfo(name="Myntra", tier=BrandTier.MID),
        BrandInfo(name="Manish Malhotra", tier=BrandTier.PREMIUM_EVERYDAY),
    ]


def test_brand_empty_when_no_match(tmp_path) -> None:  # type: ignore[no-untyped-def]
    registry = _build_test_registry(tmp_path)
    item = _make([], "regular post no brands", account_handle="random_user")
    state = extract_rule_based(item, registry)

    assert state.brands == []
    assert "brand" not in state.method_per_attribute


def test_brand_skipped_when_registry_none() -> None:
    """registry=None → brand 추출 skip (backwards-compat)."""
    item = _make([], "look @myntra", account_handle="myntra")
    state = extract_rule_based(item, brand_registry=None)
    assert state.brands == []
