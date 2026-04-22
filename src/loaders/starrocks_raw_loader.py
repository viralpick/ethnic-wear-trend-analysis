"""StarRocksRawLoader — RawLoader Protocol 구현체. `png` 스키마 → RawDailyBatch.

daily 파이프라인에서 TsvRawLoader 를 대체해 실시간 DB 쿼리로 raw 수집. 필터 기준은
`created_at` (수집 시점) — spec §10.1 의 "당일 수집된 신규 포스트만" 과 정합.

처리 대상 테이블 (2026-04-22 실 StarRocks 스키마 확인 후 반영):
  - `india_ai_fashion_inatagram_posting`    — IG 포스트 (id/user/content/like_count 등)
  - `india_ai_fashion_inatagram_profile`    — 계정 프로필 (user/follower_count JOIN)
  - `india_ai_fashion_youtube_posting`      — YT 영상 (url/upload_date/comments 등)
  - hashtag_search 전용 테이블은 아직 확정 안 됨 (agenda §1.6 업데이트) — 존재하지 않으면
    load_batch 가 빈 리스트로 safe skip

컬럼명 매핑 (실 스키마 기준, TSV 명 ↔ DB 명):
  posting:  id / user / content / posting_at(varchar) / like_count / comment_count
            / download_urls / created_at(datetime)
  profile:  user / follower_count
  youtube:  url / upload_date(varchar) / comments / view_count / like_count / comment_count

varchar datetime (posting_at / upload_date) 은 ISO 또는 YYYYMMDD 포맷 가정 — 실제 포맷에 따라
_parse_varchar_datetime 에서 확장.

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
_YOUTUBE_TABLE = "india_ai_fashion_youtube_posting"
# hashtag 테이블 실 이름 (2026-04-22 DB 확인). TSV export 파일명은 중간에 `_inatagram_` 을
# 건너뛰어 혼동을 유발했었음 (agenda §1.1 오타 기록).
_HASHTAG_TABLE_CANDIDATES = (
    "india_ai_fashion_inatagram_hash_tag_search_result",  # 실 DB 이름
    "india_ai_fashionash_tag_search_result",              # TSV export 오타 (fallback)
)

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
    """datetime 객체 → UTC tz-aware. string 이면 _parse_varchar_datetime 경유."""
    if isinstance(value, datetime):
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value
    if isinstance(value, str):
        return _parse_varchar_datetime(value)
    raise ValueError(f"expected datetime|str, got {type(value).__name__}: {value!r}")


def _parse_varchar_datetime(value: str) -> datetime:
    """varchar datetime → UTC. ISO / YYYYMMDD / MySQL-style 시도."""
    s = value.strip()
    # ISO (2026-04-19T23:49:20Z or 2026-04-19 23:49:20)
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
    except ValueError:
        pass
    # YYYYMMDD
    if len(s) == 8 and s.isdigit():
        return datetime.strptime(s, "%Y%m%d").replace(tzinfo=timezone.utc)
    raise ValueError(f"unrecognized datetime format: {value!r}")


def _build_posting(
    row: dict, followers_by_handle: dict[str, int],
) -> RawInstagramPost | None:
    """posting row → RawInstagramPost. 실 컬럼: id/user/content/posting_at/like_count/..."""
    try:
        handle = row.get("user") or None
        content = row.get("content") or ""
        return RawInstagramPost(
            post_id=str(row["id"]),
            source_type=InstagramSourceType.INFLUENCER_FIXED,
            account_handle=handle,
            account_followers=followers_by_handle.get(handle or "", 0),
            image_urls=_split_csv(row.get("download_urls")),
            caption_text=content,
            hashtags=_extract_hashtags(content),
            likes=int(row.get("like_count") or 0),
            comments_count=int(row.get("comment_count") or 0),
            saves=None,
            post_date=_ensure_utc(row["posting_at"]),
            collected_at=_ensure_utc(row["created_at"]),
        )
    except (KeyError, ValueError, TypeError) as exc:
        logger.info("starrocks_posting_skip id=%s reason=%s", row.get("id"), exc)
        return None


def _build_hashtag_search(row: dict) -> RawInstagramPost | None:
    """hash_tag_search row → RawInstagramPost. 실 컬럼: id/hash_tag/url/thumbnail_url/content."""
    try:
        tag = row.get("hash_tag")
        return RawInstagramPost(
            post_id=str(row["id"]),
            source_type=InstagramSourceType.HASHTAG_TRACKING,
            account_handle=None,
            account_followers=0,
            image_urls=[row["thumbnail_url"]] if row.get("thumbnail_url") else [],
            caption_text=row.get("content") or "",
            hashtags=[f"#{tag}"] if tag else [],
            likes=int(row.get("like_count") or 0),
            comments_count=int(row.get("comment_count") or 0),
            saves=None,
            post_date=_HASHTAG_SEARCH_PLACEHOLDER_DATE,
            collected_at=_ensure_utc(row["created_at"]),
        )
    except (KeyError, ValueError, TypeError) as exc:
        logger.info("starrocks_hashtag_skip id=%s reason=%s", row.get("id"), exc)
        return None


def _build_youtube(row: dict) -> RawYouTubeVideo | None:
    """youtube_posting row → RawYouTubeVideo. video_id 는 url regex 추출."""
    try:
        url = row.get("url") or ""
        match = _YT_VIDEO_ID_RE.search(url)
        if match is None:
            return None
        return RawYouTubeVideo(
            video_id=match.group(1),
            channel=row.get("channel") or "",
            title=row.get("title") or "",
            description=row.get("description") or "",
            tags=_split_csv(row.get("tags")),
            thumbnail_url=row.get("thumbnail_url") or "",
            view_count=int(row.get("view_count") or 0),
            like_count=int(row.get("like_count") or 0),
            comment_count=int(row.get("comment_count") or 0),
            top_comments=[
                x for x in (row.get("comments") or "").split("|") if x
            ],
            published_at=_ensure_utc(row.get("upload_date") or row["created_at"]),
        )
    except (KeyError, ValueError, TypeError) as exc:
        logger.info("starrocks_yt_skip id=%s reason=%s", row.get("id"), exc)
        return None


class StarRocksRawLoader:
    """RawLoader 구현체. target_date 기준 당일 수집 데이터 batch."""

    def __init__(self, reader: StarRocksReader) -> None:
        self._reader = reader

    @classmethod
    def from_env(cls) -> "StarRocksRawLoader":
        return cls(StarRocksReader.from_env())

    def load_batch(self, target_date: date) -> RawDailyBatch:
        available = set(self._reader.list_tables())
        followers = self._load_followers(available)
        ig_posting = self._load_posting_rows(target_date, followers, available)
        ig_hashtag = self._load_hashtag_rows(target_date, available)
        yt = self._load_youtube_rows(target_date, available)
        logger.info(
            "starrocks_loaded target=%s ig_posting=%d ig_hashtag=%d yt=%d",
            target_date, len(ig_posting), len(ig_hashtag), len(yt),
        )
        return RawDailyBatch(
            instagram=[*ig_posting, *ig_hashtag],
            youtube=yt,
        )

    def _load_followers(self, available: set[str]) -> dict[str, int]:
        if _PROFILE_TABLE not in available:
            return {}
        rows = self._reader.select(
            f"SELECT user, follower_count FROM {_PROFILE_TABLE}"
        )
        out: dict[str, int] = {}
        for r in rows:
            handle = r.get("user")
            if not handle:
                continue
            try:
                out[handle] = int(r.get("follower_count") or 0)
            except (ValueError, TypeError):
                continue
        return out

    def _load_posting_rows(
        self, target_date: date, followers: dict[str, int], available: set[str],
    ) -> list[RawInstagramPost]:
        if _POSTING_TABLE not in available:
            return []
        start, end = _day_range(target_date)
        rows = self._reader.select(
            f"SELECT * FROM {_POSTING_TABLE} "
            "WHERE created_at >= %s AND created_at <= %s",
            (start, end),
        )
        return [p for r in rows if (p := _build_posting(r, followers)) is not None]

    def _load_hashtag_rows(
        self, target_date: date, available: set[str],
    ) -> list[RawInstagramPost]:
        table = next((t for t in _HASHTAG_TABLE_CANDIDATES if t in available), None)
        if table is None:
            return []
        start, end = _day_range(target_date)
        rows = self._reader.select(
            f"SELECT * FROM {table} "
            "WHERE created_at >= %s AND created_at <= %s",
            (start, end),
        )
        return [p for r in rows if (p := _build_hashtag_search(r)) is not None]

    def _load_youtube_rows(
        self, target_date: date, available: set[str],
    ) -> list[RawYouTubeVideo]:
        if _YOUTUBE_TABLE not in available:
            return []
        start, end = _day_range(target_date)
        rows = self._reader.select(
            f"SELECT * FROM {_YOUTUBE_TABLE} "
            "WHERE created_at >= %s AND created_at <= %s",
            (start, end),
        )
        return [v for r in rows if (v := _build_youtube(r)) is not None]
