"""TsvRawLoader 단위 테스트 — fixture TSV 4종을 RawDailyBatch 로 변환.

hashtag_search 의 placeholder post_date / account_handle=None 처리 + youtube video_id 추출 +
profile.tsv JOIN 동작 + 잘못된 row skip 검증.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from contracts.common import InstagramSourceType
from loaders.tsv_raw_loader import (
    _HASHTAG_SEARCH_PLACEHOLDER_DATE,
    TsvRawLoader,
)

_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "tsv"
_TARGET_DATE = date(2026, 4, 21)


@pytest.fixture
def batch():
    return TsvRawLoader(_FIXTURE_DIR).load_batch(_TARGET_DATE)


# --------------------------------------------------------------------------- #
# Instagram — posting + hashtag_search 합계
# --------------------------------------------------------------------------- #

def test_loads_instagram_posting_two_valid_rows(batch) -> None:
    # posting fixture 3 rows 중 [2] row 는 likes=NOT_AN_INT 라 skip → 2 남음.
    # + hashtag_search 2 rows = 총 4 ig.
    assert len(batch.instagram) == 4


def test_posting_row_one_maps_all_fields(batch) -> None:
    # post_id / handle / followers (profile JOIN) / image_urls / caption / hashtags / likes / date.
    post = next(p for p in batch.instagram if p.post_id == "01KFIXTUREPOSTING000000001")
    assert post.source_type == InstagramSourceType.INFLUENCER_FIXED
    assert post.account_handle == "test_handle_a"
    assert post.account_followers == 123456  # profile.tsv JOIN
    assert post.caption_text.startswith("Chikankari cotton kurta set")
    assert post.hashtags == ["#chikankari", "#office"]
    assert len(post.image_urls) == 2
    assert post.image_urls[0] == "collectify/fixture/images/01KFIX_A_001.jpg"
    assert post.likes == 1200
    assert post.comments_count == 30
    assert post.saves is None
    assert post.post_date == datetime(2026, 4, 19, 12, 0, 0, tzinfo=timezone.utc)


def test_posting_profile_miss_yields_zero_followers(batch) -> None:
    # test_handle_b 는 profile.tsv 에 없음 → followers=0.
    post = next(p for p in batch.instagram if p.post_id == "01KFIXTUREPOSTING000000002")
    assert post.account_followers == 0
    assert post.account_handle == "test_handle_b"


def test_posting_bad_row_skipped_not_raise(batch) -> None:
    # likes=NOT_AN_INT row 는 ValueError 발생 → skip. load_batch 자체는 raise 안 함.
    ids = [p.post_id for p in batch.instagram]
    assert "01KFIXTUREBADROW" not in ids


# --------------------------------------------------------------------------- #
# hashtag_search — account_handle None + post_date placeholder + hashtag 보존
# --------------------------------------------------------------------------- #

def test_hashtag_search_account_handle_is_none(batch) -> None:
    ht = next(p for p in batch.instagram if p.post_id == "01KFIXTUREHASHTAG00000001")
    assert ht.account_handle is None
    assert ht.account_followers == 0
    assert ht.source_type == InstagramSourceType.HASHTAG_TRACKING


def test_hashtag_search_post_date_placeholder(batch) -> None:
    ht = next(p for p in batch.instagram if p.post_id == "01KFIXTUREHASHTAG00000001")
    assert ht.post_date == _HASHTAG_SEARCH_PLACEHOLDER_DATE
    assert ht.post_date == datetime(2026, 4, 15, tzinfo=timezone.utc)


def test_hashtag_search_preserves_source_tag(batch) -> None:
    # 어느 해시태그에서 온 포스팅인지 hashtags 리스트에 반영됨.
    ht_eth = next(p for p in batch.instagram if p.post_id == "01KFIXTUREHASHTAG00000001")
    ht_kurti = next(p for p in batch.instagram if p.post_id == "01KFIXTUREHASHTAG00000002")
    assert ht_eth.hashtags == ["#ethnicwear"]
    assert ht_kurti.hashtags == ["#kurti"]


def test_hashtag_search_uses_cdn_url_for_images(batch) -> None:
    # blob path 가 없으니 CDN URL 을 image_urls 에 (pipeline_b_adapter 는 이 URL 로 로컬 매핑 실패).
    ht = next(p for p in batch.instagram if p.post_id == "01KFIXTUREHASHTAG00000001")
    assert ht.image_urls == ["https://cdn.example/ig/ht1.jpg"]


# --------------------------------------------------------------------------- #
# YouTube
# --------------------------------------------------------------------------- #

def test_loads_youtube_two_rows(batch) -> None:
    assert len(batch.youtube) == 2


def test_youtube_video_id_extracted_from_url(batch) -> None:
    ids = {v.video_id for v in batch.youtube}
    assert ids == {"TESTVIDEO01", "TESTVIDEO02"}


def test_youtube_published_at_from_yyyymmdd(batch) -> None:
    v = next(v for v in batch.youtube if v.video_id == "TESTVIDEO01")
    assert v.published_at == datetime(2026, 3, 4, tzinfo=timezone.utc)


def test_youtube_tags_csv_split(batch) -> None:
    v = next(v for v in batch.youtube if v.video_id == "TESTVIDEO01")
    assert v.tags == ["office kurta", "workwear", "test"]


def test_youtube_top_comments_pipe_split(batch) -> None:
    v = next(v for v in batch.youtube if v.video_id == "TESTVIDEO01")
    assert v.top_comments == ["comment one", "comment two", "comment three"]


def test_youtube_empty_top_comments_yields_empty_list(batch) -> None:
    v = next(v for v in batch.youtube if v.video_id == "TESTVIDEO02")
    assert v.top_comments == []


# --------------------------------------------------------------------------- #
# Missing file safety
# --------------------------------------------------------------------------- #

def test_missing_directory_returns_empty_batch(tmp_path: Path) -> None:
    # 빈 디렉토리 (TSV 없음) → load_batch 는 빈 배치 반환 (raise 안 함).
    loader = TsvRawLoader(tmp_path)
    batch = loader.load_batch(_TARGET_DATE)
    assert batch.instagram == []
    assert batch.youtube == []
