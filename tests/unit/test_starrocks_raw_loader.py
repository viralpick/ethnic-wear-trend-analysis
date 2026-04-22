"""StarRocksRawLoader 단위 — StarRocksReader 를 MagicMock 으로 대체.

실 DB 호출 없음. _load_followers / posting / hashtag / youtube 가 dict row → contract
변환하는 로직 검증 (실 스키마 기준 컬럼명: user / content / posting_at / like_count 등).
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
    _parse_varchar_datetime,
)


def _fake_reader(
    rows_by_table: dict[str, list[dict]],
    tables: list[str] | None = None,
) -> MagicMock:
    """table 이름별로 미리 정의한 dict row 를 반환하는 reader mock."""
    reader = MagicMock()
    reader.list_tables.return_value = tables if tables is not None else list(rows_by_table.keys())

    def select(query: str, params: tuple = ()):  # noqa: ARG001
        for table, rows in rows_by_table.items():
            if table in query:
                return rows
        return []

    reader.select.side_effect = select
    return reader


# --------------------------------------------------------------------------- #
# _parse_varchar_datetime
# --------------------------------------------------------------------------- #

def test_parse_varchar_iso_with_z() -> None:
    dt = _parse_varchar_datetime("2026-04-19T23:49:20Z")
    assert dt == datetime(2026, 4, 19, 23, 49, 20, tzinfo=timezone.utc)


def test_parse_varchar_iso_space_separated() -> None:
    dt = _parse_varchar_datetime("2026-04-20 12:00:00")
    assert dt.replace(tzinfo=None) == datetime(2026, 4, 20, 12, 0, 0)


def test_parse_varchar_yyyymmdd() -> None:
    dt = _parse_varchar_datetime("20260304")
    assert dt == datetime(2026, 3, 4, tzinfo=timezone.utc)


def test_parse_varchar_invalid_raises() -> None:
    with pytest.raises(ValueError, match="unrecognized"):
        _parse_varchar_datetime("not a date")


# --------------------------------------------------------------------------- #
# followers JOIN (profile 테이블)
# --------------------------------------------------------------------------- #

def test_load_followers_builds_handle_dict() -> None:
    reader = _fake_reader({
        "india_ai_fashion_inatagram_profile": [
            {"user": "masoomminawala", "follower_count": 1_000_000},
            {"user": "juhigodambe", "follower_count": 920_000},
            {"user": None, "follower_count": 100},  # handle 없으면 skip
        ],
    })
    loader = StarRocksRawLoader(reader)
    batch = loader.load_batch(date(2026, 4, 20))
    # followers 테이블만 있고 나머지는 empty → ig=0.
    assert batch.instagram == []
    # followers 는 내부 dict 로만 쓰이고 직접 노출되지 않음. 간접 검증은 posting 테스트.


# --------------------------------------------------------------------------- #
# posting (실 컬럼명: id / user / content / like_count / posting_at / download_urls 등)
# --------------------------------------------------------------------------- #

def _posting_row(**overrides) -> dict:
    base = {
        "id": "01KPNKJ80633TEST",
        "user": "masoomminawala",
        "content": "Chikankari cotton kurta set #chikankari #office",
        "like_count": 3474,
        "comment_count": 34,
        "download_urls": "collectify/poc/a.jpg,collectify/poc/b.jpg",
        "posting_at": "2026-04-19T12:00:00Z",  # varchar — 실 스키마 반영
        "created_at": datetime(2026, 4, 20, 12, 0, 0, tzinfo=timezone.utc),
    }
    base.update(overrides)
    return base


def test_load_posting_rows_maps_fields() -> None:
    reader = _fake_reader({
        "india_ai_fashion_inatagram_profile": [
            {"user": "masoomminawala", "follower_count": 1_000_000},
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
        "india_ai_fashion_inatagram_posting": [_posting_row(user="unknown_user")],
    })
    loader = StarRocksRawLoader(reader)
    batch = loader.load_batch(date(2026, 4, 20))
    assert batch.instagram[0].account_followers == 0


def test_load_posting_bad_row_skipped_not_raise() -> None:
    reader = _fake_reader({
        "india_ai_fashion_inatagram_profile": [],
        "india_ai_fashion_inatagram_posting": [
            _posting_row(),
            {"id": "BAD", "created_at": datetime(2026, 4, 20, tzinfo=timezone.utc)},
            # 필수 posting_at 누락 + 타입 불일치 → skip
        ],
    })
    loader = StarRocksRawLoader(reader)
    batch = loader.load_batch(date(2026, 4, 20))
    assert len(batch.instagram) == 1


def test_posting_missing_table_yields_empty() -> None:
    # list_tables 에 posting 없음 → 쿼리 skip.
    reader = _fake_reader({}, tables=["india_ai_fashion_inatagram_profile"])
    loader = StarRocksRawLoader(reader)
    batch = loader.load_batch(date(2026, 4, 20))
    assert batch.instagram == []


# --------------------------------------------------------------------------- #
# hashtag_search — 3 테이블 이름 후보 fallback
# --------------------------------------------------------------------------- #

def _hashtag_row(**overrides) -> dict:
    base = {
        "id": "01KPNKGH7AG0TEST",
        "hash_tag": "ethnicwear",
        "thumbnail_url": "https://cdn.example/ht.jpg",
        "content": "Photo by someone.",
        "like_count": 0,
        "comment_count": 0,
        "created_at": datetime(2026, 4, 20, 11, 0, 0, tzinfo=timezone.utc),
    }
    base.update(overrides)
    return base


def test_hashtag_table_absent_yields_empty() -> None:
    # posting 만 있고 어떤 hashtag 후보 테이블도 없음 → 빈 리스트.
    reader = _fake_reader({}, tables=["india_ai_fashion_inatagram_posting"])
    loader = StarRocksRawLoader(reader)
    batch = loader.load_batch(date(2026, 4, 20))
    # instagram 은 posting 만 이고, 그것도 row 는 비어있음.
    assert batch.instagram == []


def test_hashtag_uses_placeholder_date_and_none_handle() -> None:
    # 실 DB 테이블명 (2026-04-22 확인).
    reader = _fake_reader({
        "india_ai_fashion_inatagram_hash_tag_search_result": [_hashtag_row()],
    })
    loader = StarRocksRawLoader(reader)
    batch = loader.load_batch(date(2026, 4, 20))
    assert len(batch.instagram) == 1
    h = batch.instagram[0]
    assert h.source_type == InstagramSourceType.HASHTAG_TRACKING
    assert h.account_handle is None
    assert h.post_date == _HASHTAG_SEARCH_PLACEHOLDER_DATE
    assert h.hashtags == ["#ethnicwear"]


# --------------------------------------------------------------------------- #
# youtube (실 컬럼: url / upload_date(varchar) / comments)
# --------------------------------------------------------------------------- #

def _yt_row(**overrides) -> dict:
    base = {
        "id": "01KPNM3TA3TEST",
        "url": "https://www.youtube.com/watch?v=TESTVIDEO01",
        "channel": "Jhanvi Bhatia",
        "title": "Office Kurta Haul",
        "description": "desc",
        "tags": "office kurta,workwear",
        "thumbnail_url": "https://i.ytimg.com/vi/TESTVIDEO01/3.jpg",
        "view_count": 5000,
        "like_count": 200,
        "comment_count": 20,
        "comments": "a|b|c",
        "upload_date": "20260304",  # varchar — 실 스키마 반영
        "created_at": datetime(2026, 4, 20, 13, 0, 0, tzinfo=timezone.utc),
    }
    base.update(overrides)
    return base


def test_load_youtube_extracts_video_id() -> None:
    reader = _fake_reader({
        "india_ai_fashion_youtube_posting": [_yt_row()],
    })
    loader = StarRocksRawLoader(reader)
    batch = loader.load_batch(date(2026, 4, 20))
    assert len(batch.youtube) == 1
    v = batch.youtube[0]
    assert v.video_id == "TESTVIDEO01"
    assert v.tags == ["office kurta", "workwear"]
    assert v.top_comments == ["a", "b", "c"]
    assert v.published_at == datetime(2026, 3, 4, tzinfo=timezone.utc)


def test_load_youtube_skips_when_video_id_missing() -> None:
    reader = _fake_reader({
        "india_ai_fashion_youtube_posting": [_yt_row(url="https://invalid")],
    })
    loader = StarRocksRawLoader(reader)
    batch = loader.load_batch(date(2026, 4, 20))
    assert batch.youtube == []


# --------------------------------------------------------------------------- #
# 복합 — posting + hashtag + youtube 모두 존재
# --------------------------------------------------------------------------- #

def test_load_batch_merges_all_sources() -> None:
    reader = _fake_reader({
        "india_ai_fashion_inatagram_profile": [
            {"user": "masoomminawala", "follower_count": 1_000_000},
        ],
        "india_ai_fashion_inatagram_posting": [_posting_row()],
        "india_ai_fashion_inatagram_hash_tag_search_result": [_hashtag_row()],
        "india_ai_fashion_youtube_posting": [_yt_row()],
    })
    loader = StarRocksRawLoader(reader)
    batch = loader.load_batch(date(2026, 4, 20))
    assert len(batch.instagram) == 2
    assert len(batch.youtube) == 1
