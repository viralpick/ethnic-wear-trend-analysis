"""StarRocksRawLoader 단위 — StarRocksReader 를 MagicMock 으로 대체.

실 DB 호출 없음. _load_followers / _load_posting_rows / _load_hashtag_rows /
_load_youtube_rows 가 dict row 를 contract 로 변환하는 로직 검증.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from unittest.mock import MagicMock

import pytest

pytest.importorskip("pymysql", reason="starrocks extras required")
pytest.importorskip("dotenv", reason="starrocks extras required")

from contracts.common import InstagramSourceType  # noqa: E402
from loaders.starrocks_raw_loader import (  # noqa: E402
    _HASHTAG_SEARCH_PLACEHOLDER_DATE,
    StarRocksRawLoader,
)


def _fake_reader(responses_by_table: dict[str, list[dict]]) -> MagicMock:
    """table 이름별로 미리 정의한 dict row 를 반환하는 reader mock.

    table 키는 쿼리에 들어간 table 이름 (e.g., `india_ai_fashion_inatagram_posting`) 으로 match.
    """
    reader = MagicMock()

    def select(query: str, params: tuple = ()):  # noqa: ARG001
        for table, rows in responses_by_table.items():
            if table in query:
                return rows
        return []

    reader.select.side_effect = select
    return reader


# --------------------------------------------------------------------------- #
# followers JOIN
# --------------------------------------------------------------------------- #

def test_load_followers_builds_handle_dict() -> None:
    reader = _fake_reader({
        "india_ai_fashion_inatagram_profile": [
            {"account_handle": "masoomminawala", "followers": 1_000_000},
            {"account_handle": "juhigodambe", "followers": 920_000},
            {"account_handle": None, "followers": 100},  # handle 없으면 skip
        ],
    })
    loader = StarRocksRawLoader(reader)
    out = loader._load_followers()  # noqa: SLF001
    assert out == {"masoomminawala": 1_000_000, "juhigodambe": 920_000}


# --------------------------------------------------------------------------- #
# posting
# --------------------------------------------------------------------------- #

def _posting_row(**overrides) -> dict:
    base = {
        "ulid": "01KPNKJ80633TEST",
        "account_handle": "masoomminawala",
        "caption": "Chikankari cotton kurta set #chikankari #office",
        "likes": 3474,
        "comments_count": 34,
        "image_paths": "collectify/poc/a.jpg,collectify/poc/b.jpg",
        "post_date": datetime(2026, 4, 19, 12, 0, 0, tzinfo=timezone.utc),
        "created_at": datetime(2026, 4, 20, 12, 0, 0, tzinfo=timezone.utc),
    }
    base.update(overrides)
    return base


def test_load_posting_rows_maps_fields() -> None:
    reader = _fake_reader({
        "india_ai_fashion_inatagram_profile": [
            {"account_handle": "masoomminawala", "followers": 1_000_000},
        ],
        "india_ai_fashion_inatagram_posting": [_posting_row()],
    })
    loader = StarRocksRawLoader(reader)
    batch = loader.load_batch(date(2026, 4, 20))
    assert len(batch.instagram) == 1
    p = batch.instagram[0]
    assert p.post_id == "01KPNKJ80633TEST"
    assert p.source_type == InstagramSourceType.INFLUENCER_FIXED
    assert p.account_handle == "masoomminawala"
    assert p.account_followers == 1_000_000
    assert p.hashtags == ["#chikankari", "#office"]
    assert len(p.image_urls) == 2
    assert p.likes == 3474
    assert p.saves is None


def test_load_posting_profile_miss_yields_zero_followers() -> None:
    reader = _fake_reader({
        "india_ai_fashion_inatagram_profile": [],
        "india_ai_fashion_inatagram_posting": [_posting_row(account_handle="unknown_user")],
    })
    loader = StarRocksRawLoader(reader)
    batch = loader.load_batch(date(2026, 4, 20))
    assert batch.instagram[0].account_followers == 0


def test_load_posting_bad_row_skipped_not_raise() -> None:
    reader = _fake_reader({
        "india_ai_fashion_inatagram_profile": [],
        "india_ai_fashion_inatagram_posting": [
            _posting_row(),
            {"ulid": "BAD", "created_at": "not a datetime"},  # post_date 누락 + 타입 불일치
        ],
    })
    loader = StarRocksRawLoader(reader)
    batch = loader.load_batch(date(2026, 4, 20))
    # 정상 1건만 통과.
    assert len(batch.instagram) == 1


# --------------------------------------------------------------------------- #
# hashtag_search
# --------------------------------------------------------------------------- #

def _hashtag_row(**overrides) -> dict:
    base = {
        "ulid": "01KPNKGH7AG0TEST",
        "hashtag": "ethnicwear",
        "image_url": "https://cdn.example/ht.jpg",
        "caption": "Photo by someone.",
        "likes": 0,
        "comments_count": 0,
        "created_at": datetime(2026, 4, 20, 11, 0, 0, tzinfo=timezone.utc),
    }
    base.update(overrides)
    return base


def test_load_hashtag_uses_placeholder_date_and_none_handle() -> None:
    reader = _fake_reader({
        "india_ai_fashion_inatagram_profile": [],
        "india_ai_fashion_inatagram_posting": [],
        "india_ai_fashionash_tag_search_result": [_hashtag_row()],
    })
    loader = StarRocksRawLoader(reader)
    batch = loader.load_batch(date(2026, 4, 20))
    assert len(batch.instagram) == 1
    h = batch.instagram[0]
    assert h.source_type == InstagramSourceType.HASHTAG_TRACKING
    assert h.account_handle is None
    assert h.account_followers == 0
    assert h.post_date == _HASHTAG_SEARCH_PLACEHOLDER_DATE
    assert h.hashtags == ["#ethnicwear"]
    assert h.image_urls == ["https://cdn.example/ht.jpg"]


# --------------------------------------------------------------------------- #
# youtube
# --------------------------------------------------------------------------- #

def _yt_row(**overrides) -> dict:
    base = {
        "ulid": "01KPNM3TA3TEST",
        "video_url": "https://www.youtube.com/watch?v=TESTVIDEO01",
        "channel": "Jhanvi Bhatia",
        "title": "Office Kurta Haul",
        "description": "desc",
        "tags": "office kurta,workwear",
        "thumbnail_url": "https://i.ytimg.com/vi/TESTVIDEO01/3.jpg",
        "view_count": 5000,
        "like_count": 200,
        "comment_count": 20,
        "top_comments": "a|b|c",
        "published_at": datetime(2026, 3, 4, tzinfo=timezone.utc),
        "created_at": datetime(2026, 4, 20, 13, 0, 0, tzinfo=timezone.utc),
    }
    base.update(overrides)
    return base


def test_load_youtube_extracts_video_id() -> None:
    reader = _fake_reader({
        "india_ai_fashion_inatagram_profile": [],
        "india_ai_fashion_inatagram_posting": [],
        "india_ai_fashionash_tag_search_result": [],
        "india_ai_fashion_youtube_posting": [_yt_row()],
    })
    loader = StarRocksRawLoader(reader)
    batch = loader.load_batch(date(2026, 4, 20))
    assert len(batch.youtube) == 1
    v = batch.youtube[0]
    assert v.video_id == "TESTVIDEO01"
    assert v.tags == ["office kurta", "workwear"]
    assert v.top_comments == ["a", "b", "c"]


def test_load_youtube_skips_when_video_id_missing() -> None:
    reader = _fake_reader({
        "india_ai_fashion_inatagram_profile": [],
        "india_ai_fashion_inatagram_posting": [],
        "india_ai_fashionash_tag_search_result": [],
        "india_ai_fashion_youtube_posting": [_yt_row(video_url="https://invalid")],
    })
    loader = StarRocksRawLoader(reader)
    batch = loader.load_batch(date(2026, 4, 20))
    assert batch.youtube == []


# --------------------------------------------------------------------------- #
# 복합 — posting + hashtag + youtube 모두
# --------------------------------------------------------------------------- #

def test_load_batch_merges_all_sources() -> None:
    reader = _fake_reader({
        "india_ai_fashion_inatagram_profile": [
            {"account_handle": "masoomminawala", "followers": 1_000_000},
        ],
        "india_ai_fashion_inatagram_posting": [_posting_row()],
        "india_ai_fashionash_tag_search_result": [_hashtag_row()],
        "india_ai_fashion_youtube_posting": [_yt_row()],
    })
    loader = StarRocksRawLoader(reader)
    batch = loader.load_batch(date(2026, 4, 20))
    assert len(batch.instagram) == 2  # 1 posting + 1 hashtag
    assert len(batch.youtube) == 1
