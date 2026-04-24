from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest
from pydantic import ValidationError

from contracts.common import InstagramSourceType
from contracts.enriched import EnrichedContentItem
from contracts.raw import RawInstagramPost


def _valid_ig_payload() -> dict[str, Any]:
    return {
        "post_id": "ig_test_1",
        "source_type": "influencer_fixed",
        "account_handle": "@test",
        "account_followers": 1_000_000,
        "image_urls": ["https://example.com/1.jpg"],
        "caption_text": "test",
        "hashtags": ["#test"],
        "likes": 100,
        "comments_count": 5,
        "saves": 10,
        "post_date": "2026-04-21T10:00:00Z",
        "collected_at": "2026-04-21T11:00:00Z",
    }


def _valid_normalized_payload() -> dict[str, Any]:
    return {
        "source": "instagram",
        "source_post_id": "ig_x",
        "text_blob": "",
        "hashtags": [],
        "image_urls": [],
        "post_date": "2026-04-21T00:00:00Z",
        "engagement_raw": 0,
    }


def test_raw_instagram_post_happy_path() -> None:
    post = RawInstagramPost.model_validate(_valid_ig_payload())

    assert post.post_id == "ig_test_1"
    assert isinstance(post.post_date, datetime)
    assert post.source_type == InstagramSourceType.INFLUENCER_FIXED


def test_raw_instagram_post_missing_required_field_raises() -> None:
    payload = _valid_ig_payload()
    del payload["post_id"]

    with pytest.raises(ValidationError):
        RawInstagramPost.model_validate(payload)


def test_raw_instagram_post_rejects_unknown_field() -> None:
    # extra="forbid" 가 크롤러 스키마 드리프트를 조기에 잡는다.
    payload = _valid_ig_payload()
    payload["new_crawler_field"] = "oops"

    with pytest.raises(ValidationError):
        RawInstagramPost.model_validate(payload)


def test_enriched_item_allows_all_attrs_null() -> None:
    # 8개 속성 + embellishment_intensity 가 전부 null 이어도 통과해야 한다 (unclassifiable 케이스).
    item = EnrichedContentItem.model_validate(
        {"normalized": _valid_normalized_payload()}
    )

    assert item.garment_type is None
    assert item.brand is None
    assert item.trend_cluster_key is None
    assert item.classification_method_per_attribute == {}


def test_enriched_item_rejects_invalid_enum_value() -> None:
    # LLM/VLM 추출기가 enum 에 없는 값을 내면 ValidationError — 이건 계약의 핵심.
    with pytest.raises(ValidationError):
        EnrichedContentItem.model_validate(
            {
                "normalized": _valid_normalized_payload(),
                "garment_type": "not_a_valid_garment",
            }
        )


def test_enriched_item_rejects_invalid_classification_method_value() -> None:
    with pytest.raises(ValidationError):
        EnrichedContentItem.model_validate(
            {
                "normalized": _valid_normalized_payload(),
                "classification_method_per_attribute": {"garment_type": "magic"},
            }
        )


def test_enriched_item_accepts_partial_method_map() -> None:
    # 추출된 속성만 맵에 들어가고, null 속성은 키가 없다.
    item = EnrichedContentItem.model_validate(
        {
            "normalized": _valid_normalized_payload(),
            "garment_type": "kurta_set",
            "classification_method_per_attribute": {"garment_type": "rule"},
        }
    )

    assert item.classification_method_per_attribute == {"garment_type": "rule"}
    assert item.fabric is None
