"""StarRocksRawLoader — RawLoader Protocol 구현체. `png` 스키마 → RawDailyBatch.

daily 파이프라인에서 TsvRawLoader 를 대체해 실시간 DB 쿼리로 raw 수집. 필터 기준은
`created_at` (수집 시점) — spec §10.1 의 "당일 수집된 신규 포스트만" 과 정합.

처리 대상 테이블 (남궁현님 안내 기준):
  - `india_ai_fashion_inatagram_posting`    — IG 포스트 본문 (13 col)
  - `india_ai_fashion_inatagram_profile`    — 계정 프로필 (followers JOIN)
  - `india_ai_fashionash_tag_search_result` — 해시태그 검색 결과 (10 col)
  - `india_ai_fashion_youtube_posting`      — YT 영상 (18 col)

컬럼명 가정 (TSV 매핑 기반 — 실 스키마와 다르면 `describe()` 로 확인 후 조정):
  - posting / profile / hashtag / youtube 의 field 이름은 TSV 의 의미 매핑 그대로
  - 예: posting 에 `ulid`, `account_handle`, `caption`, `image_paths`, `created_at` 등
  - 컬럼명 mismatch 시 KeyError 로그 + 해당 row skip (raise 안 함 — batch 계속 진행)

`[starrocks]` extras 필요.
"""
from __future__ import annotations

import re
from datetime import date, datetime, time, timezone
from typing import Any

from contracts.common import InstagramSourceType
from contracts.raw import RawInstagramPost, RawYouTubeVideo
from loaders.raw_loader import RawDailyBatch
from loaders.starrocks_reader import StarRocksReader
from utils.logging import get_logger

logger = get_logger(__name__)

_POSTING_TABLE = "india_ai_fashion_inatagram_posting"
_PROFILE_TABLE = "india_ai_fashion_inatagram_profile"
_HASHTAG_TABLE = "india_ai_fashionash_tag_search_result"
_YOUTUBE_TABLE = "india_ai_fashion_youtube_posting"

_HASHTAG_RE = re.compile(r"#\w+")
_YT_VIDEO_ID_RE = re.compile(r"[?&]v=([\w-]+)")
_HASHTAG_SEARCH_PLACEHOLDER_DATE = datetime(2026, 4, 15, tzinfo=timezone.utc)


def _day_range(target_date: date) -> tuple[datetime, datetime]:
    start = datetime.combine(target_date, time.min).replace(tzinfo=timezone.utc)
    end = datetime.combine(target_date, time.max).replace(tzinfo=timezone.utc)
    return start, end


def _extract_hashtags(caption: str | None) -> list[str]:
    return _HASHTAG_RE.findall(caption or "")


def _split_csv(cell: Any) -> list[str]:
    if cell is None:
        return []
    return [x for x in str(cell).split(",") if x]


def _ensure_utc(value: Any) -> datetime:
    """StarRocks datetime → UTC tz-aware."""
    if isinstance(value, datetime):
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value
    raise ValueError(f"expected datetime, got {type(value).__name__}: {value!r}")


def _build_posting(
    row: dict, followers_by_handle: dict[str, int],
) -> RawInstagramPost | None:
    """posting row (dict) → RawInstagramPost. 필수 필드 누락 시 None + log."""
    try:
        handle = row.get("account_handle") or None
        caption = row.get("caption") or ""
        return RawInstagramPost(
            post_id=str(row["ulid"]),
            source_type=InstagramSourceType.INFLUENCER_FIXED,
            account_handle=handle,
            account_followers=followers_by_handle.get(handle or "", 0),
            image_urls=_split_csv(row.get("image_paths")),
            caption_text=caption,
            hashtags=_extract_hashtags(caption),
            likes=int(row.get("likes") or 0),
            comments_count=int(row.get("comments_count") or 0),
            saves=None,
            post_date=_ensure_utc(row["post_date"]),
            collected_at=_ensure_utc(row["created_at"]),
        )
    except (KeyError, ValueError, TypeError) as exc:
        logger.info("starrocks_posting_skip reason=%s", exc)
        return None


def _build_hashtag_search(row: dict) -> RawInstagramPost | None:
    """hashtag_search row → RawInstagramPost. account_handle=None, post_date placeholder."""
    try:
        tag = row.get("hashtag") or row.get("tag")
        cdn_url = row.get("image_url")
        return RawInstagramPost(
            post_id=str(row["ulid"]),
            source_type=InstagramSourceType.HASHTAG_TRACKING,
            account_handle=None,
            account_followers=0,
            image_urls=[cdn_url] if cdn_url else [],
            caption_text=row.get("caption") or "",
            hashtags=[f"#{tag}"] if tag else [],
            likes=int(row.get("likes") or 0),
            comments_count=int(row.get("comments_count") or 0),
            saves=None,
            post_date=_HASHTAG_SEARCH_PLACEHOLDER_DATE,  # TSV 와 동일 처리 (agenda §1.6)
            collected_at=_ensure_utc(row["created_at"]),
        )
    except (KeyError, ValueError, TypeError) as exc:
        logger.info("starrocks_hashtag_skip reason=%s", exc)
        return None


def _build_youtube(row: dict) -> RawYouTubeVideo | None:
    """youtube_posting row → RawYouTubeVideo. video_id 는 URL 에서 regex 추출."""
    try:
        url = row.get("video_url") or row.get("url") or ""
        match = _YT_VIDEO_ID_RE.search(url)
        if match is None:
            return None
        return RawYouTubeVideo(
            video_id=match.group(1),
            channel=row.get("channel") or row.get("channel_name") or "",
            title=row.get("title") or "",
            description=row.get("description") or "",
            tags=_split_csv(row.get("tags")),
            thumbnail_url=row.get("thumbnail_url") or "",
            view_count=int(row.get("view_count") or 0),
            like_count=int(row.get("like_count") or 0),
            comment_count=int(row.get("comment_count") or 0),
            top_comments=[
                x for x in (row.get("top_comments") or "").split("|") if x
            ],
            published_at=_ensure_utc(row.get("published_at") or row["post_date"]),
        )
    except (KeyError, ValueError, TypeError) as exc:
        logger.info("starrocks_yt_skip reason=%s", exc)
        return None


class StarRocksRawLoader:
    """RawLoader 구현체. target_date 기준 당일 수집 데이터 batch."""

    def __init__(self, reader: StarRocksReader) -> None:
        self._reader = reader

    @classmethod
    def from_env(cls) -> "StarRocksRawLoader":
        return cls(StarRocksReader.from_env())

    def load_batch(self, target_date: date) -> RawDailyBatch:
        followers = self._load_followers()
        ig_posting = self._load_posting_rows(target_date, followers)
        ig_hashtag = self._load_hashtag_rows(target_date)
        yt = self._load_youtube_rows(target_date)
        logger.info(
            "starrocks_loaded target=%s ig_posting=%d ig_hashtag=%d yt=%d",
            target_date, len(ig_posting), len(ig_hashtag), len(yt),
        )
        return RawDailyBatch(
            instagram=[*ig_posting, *ig_hashtag],
            youtube=yt,
        )

    def _load_followers(self) -> dict[str, int]:
        rows = self._reader.select(
            f"SELECT account_handle, followers FROM {_PROFILE_TABLE}"
        )
        out: dict[str, int] = {}
        for r in rows:
            handle = r.get("account_handle")
            if not handle:
                continue
            try:
                out[handle] = int(r.get("followers") or 0)
            except (ValueError, TypeError):
                continue
        return out

    def _load_posting_rows(
        self, target_date: date, followers: dict[str, int],
    ) -> list[RawInstagramPost]:
        start, end = _day_range(target_date)
        rows = self._reader.select(
            f"SELECT * FROM {_POSTING_TABLE} "
            "WHERE created_at >= %s AND created_at <= %s",
            (start, end),
        )
        return [p for r in rows if (p := _build_posting(r, followers)) is not None]

    def _load_hashtag_rows(self, target_date: date) -> list[RawInstagramPost]:
        start, end = _day_range(target_date)
        rows = self._reader.select(
            f"SELECT * FROM {_HASHTAG_TABLE} "
            "WHERE created_at >= %s AND created_at <= %s",
            (start, end),
        )
        return [p for r in rows if (p := _build_hashtag_search(r)) is not None]

    def _load_youtube_rows(self, target_date: date) -> list[RawYouTubeVideo]:
        start, end = _day_range(target_date)
        rows = self._reader.select(
            f"SELECT * FROM {_YOUTUBE_TABLE} "
            "WHERE created_at >= %s AND created_at <= %s",
            (start, end),
        )
        return [v for r in rows if (v := _build_youtube(r)) is not None]
