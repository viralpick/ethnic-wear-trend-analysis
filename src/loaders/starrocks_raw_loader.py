"""StarRocks raw loader — png DB 직접 쿼리로 RawDailyBatch 를 반환.

크리덴셜 (.env 또는 환경 변수):
  STARROCKS_HOST, STARROCKS_PORT, STARROCKS_USER, STARROCKS_PASSWORD, STARROCKS_RAW_DATABASE

로드 모드 (--window-mode):
  count: posting_at 오름차순 정렬 후 LIMIT/OFFSET. batch_index = (target_date - collection_start).days.
         현재 POC 데이터처럼 하루에 몰려있는 수집 데이터를 가상 day로 나눌 때 사용.
  date:  posting_at >= target_date - window_days AND posting_at <= target_date.
         향후 운영 기준 — 매일 실제 수집분을 rolling window로 분석할 때 사용.

Bollywood 계정 판정:
  spec §3.1 C 의 5개 고정 계정 handle 과 일치하면 BOLLYWOOD, profile 나머지는 INFLUENCER_FIXED.

top-level import 로 pymysql / python-dotenv 사용. `[db]` extras 미설치 환경에서는
이 모듈 import 자체가 ImportError — core 는 절대 top-level import 하지 말 것.
"""
from __future__ import annotations

import logging
import os
import re
from datetime import date, datetime, timedelta, timezone
from typing import Any

import pymysql
from dotenv import load_dotenv

from contracts.common import InstagramSourceType
from contracts.raw import RawInstagramPost, RawYouTubeVideo
from loaders.raw_loader import RawDailyBatch

logger = logging.getLogger(__name__)

_HASHTAG_RE = re.compile(r"#\w+")
_YT_VIDEO_ID_RE = re.compile(r"[?&]v=([\w-]+)")

# IG download_urls 가 image+video 혼입 (jpg/mp4 한 row 안에 섞임 — IG carousel 구조).
# loader 에서 확장자 기반으로 split. 알려지지 않은 확장자는 image_urls 로 분류 (보수).
_VIDEO_EXTENSIONS: tuple[str, ...] = (".mp4", ".mov", ".webm", ".m4v")

# spec §3.1 C — Bollywood 디코딩 계정 5개 (소문자 handle, @ 제외)
_BOLLYWOOD_HANDLES: frozenset[str] = frozenset({
    "bollywoodwomencloset",
    "celebrity_fashion_decode",
    "bollywoodalmari",
    "celebrities_outfit_decode",
    "bollydressdecode",
})


def _parse_iso_z(raw: str) -> datetime:
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


def _parse_yyyymmdd(raw: str) -> datetime:
    return datetime.strptime(raw, "%Y%m%d").replace(tzinfo=timezone.utc)


def _split_csv(cell: str) -> list[str]:
    return [x.strip() for x in (cell or "").split(",") if x.strip()]


def _split_pipe(cell: str) -> list[str]:
    return [x.strip() for x in (cell or "").split("|") if x.strip()]


def _split_image_video(urls: list[str]) -> tuple[list[str], list[str]]:
    """확장자 기반으로 (images, videos) 분리. unknown 확장자는 images 쪽 (보수)."""
    images: list[str] = []
    videos: list[str] = []
    for url in urls:
        # query string 제거 후 lowercase 비교 (Azure Blob SAS query 등 대응).
        path = url.split("?", 1)[0].lower()
        if path.endswith(_VIDEO_EXTENSIONS):
            videos.append(url)
        else:
            images.append(url)
    return images, videos


def _source_type(entry: str, user: str) -> InstagramSourceType:
    if entry == "hashtag":
        return InstagramSourceType.HASHTAG_TRACKING
    if user.lstrip("@").lower() in _BOLLYWOOD_HANDLES:
        return InstagramSourceType.BOLLYWOOD_DECODE
    return InstagramSourceType.INFLUENCER_FIXED


def _build_ig_post(row: dict[str, Any]) -> RawInstagramPost | None:
    try:
        images, videos = _split_image_video(_split_csv(row["download_urls"] or ""))
        return RawInstagramPost(
            post_id=row["id"],
            source_type=_source_type(row["entry"] or "profile", row["user"] or ""),
            post_url=row.get("url") or None,
            account_handle=row["user"] or None,
            account_followers=int(row["follower_count"] or 0),
            image_urls=images,
            video_urls=videos,
            caption_text=row["content"] or "",
            hashtags=_HASHTAG_RE.findall(row["content"] or ""),
            likes=int(row["like_count"] or 0),
            comments_count=int(row["comment_count"] or 0),
            saves=None,
            post_date=_parse_iso_z(row["posting_at"]),
            collected_at=row["created_at"].replace(tzinfo=timezone.utc)
            if row["created_at"]
            else datetime.now(timezone.utc),
        )
    except Exception as exc:
        logger.info("starrocks_ig_skip id=%s reason=%s", row.get("id"), exc)
        return None


def _build_yt_video(row: dict[str, Any]) -> RawYouTubeVideo | None:
    url = row.get("url") or ""
    match = _YT_VIDEO_ID_RE.search(url)
    if not match:
        logger.info("starrocks_yt_skip id=%s reason=no_video_id url=%s", row.get("id"), url[:60])
        return None
    try:
        # M3.H — YT 도 download_urls 가 mp4. IG 와 달리 video 만 (image 혼입 없음).
        # 확장자 검사로 video 만 필터 (보수: 알 수 없는 확장자는 drop).
        _, videos = _split_image_video(_split_csv(row.get("download_urls") or ""))
        return RawYouTubeVideo(
            video_id=match.group(1),
            video_url=url,
            channel=row["channel"] or "",
            channel_follower_count=int(row.get("channel_follower_count") or 0),
            title=row["title"] or "",
            description=row["description"] or "",
            tags=_split_csv(row["tags"] or ""),
            thumbnail_url=row["thumbnail_url"] or "",
            view_count=int(row["view_count"] or 0),
            like_count=int(row["like_count"] or 0),
            comment_count=int(row["comment_count"] or 0),
            top_comments=_split_pipe(row["comments"] or ""),
            published_at=_parse_yyyymmdd(row["upload_date"])
            if row.get("upload_date")
            else datetime.now(timezone.utc),
            collected_at=row["created_at"].replace(tzinfo=timezone.utc)
            if row.get("created_at")
            else datetime.now(timezone.utc),
            video_urls=videos,
        )
    except Exception as exc:
        logger.info("starrocks_yt_skip id=%s reason=%s", row.get("id"), exc)
        return None


class StarRocksRawLoader:
    """RawLoader Protocol 구현체. StarRocks png DB 에서 직접 쿼리.

    window_mode='count': posting_at 오름차순으로 정렬 후 page_size 단위 LIMIT/OFFSET.
                         batch_index = (target_date - collection_start).days.
    window_mode='date':  posting_at >= target_date - window_days AND posting_at <= target_date.
    """

    def __init__(
        self,
        host: str,
        port: int,
        user: str,
        password: str,
        database: str,
        window_mode: str = "count",
        page_size: int = 200,
        window_days: int = 30,
        collection_start: date | None = None,
    ) -> None:
        self._conn_kwargs = {
            "host": host, "port": port, "user": user,
            "password": password, "database": database,
            "connect_timeout": 15,
            "cursorclass": pymysql.cursors.DictCursor,
        }
        if window_mode not in ("count", "date"):
            raise ValueError(f"window_mode must be 'count' or 'date', got {window_mode!r}")
        self._window_mode = window_mode
        self._page_size = page_size
        self._window_days = window_days
        self._collection_start = collection_start or date.today()

    @classmethod
    def from_env(
        cls,
        window_mode: str = "count",
        page_size: int = 200,
        window_days: int = 30,
        collection_start: date | None = None,
    ) -> "StarRocksRawLoader":
        """환경 변수 / .env 에서 크리덴셜 읽기."""
        load_dotenv()
        return cls(
            host=os.environ["STARROCKS_HOST"],
            port=int(os.environ.get("STARROCKS_PORT", "9030")),
            user=os.environ["STARROCKS_USER"],
            password=os.environ["STARROCKS_PASSWORD"],
            database=os.environ.get("STARROCKS_RAW_DATABASE", "png"),
            window_mode=window_mode,
            page_size=page_size,
            window_days=window_days,
            collection_start=collection_start,
        )

    def _connect(self) -> pymysql.Connection:
        return pymysql.connect(**self._conn_kwargs)

    def load_batch(self, target_date: date) -> RawDailyBatch:
        if self._window_mode == "count":
            batch_index = (target_date - self._collection_start).days
            ig = self._load_instagram_count(batch_index, self._page_size)
            yt = self._load_youtube_count(batch_index, self._page_size)
        else:
            window_start = target_date - timedelta(days=self._window_days - 1)
            ig = self._load_instagram_date(window_start, target_date)
            yt = self._load_youtube_date(window_start, target_date)
        logger.info(
            "starrocks_loaded mode=%s ig=%d yt=%d date=%s",
            self._window_mode, len(ig), len(yt), target_date,
        )
        return RawDailyBatch(instagram=ig, youtube=yt)

    # ------------------------------------------------------------------ #
    # count 모드
    # ------------------------------------------------------------------ #

    def _load_instagram_count(self, batch_index: int, page_size: int) -> list[RawInstagramPost]:
        # profile 테이블에 동일 user 중복 행 존재 → GROUP BY 로 먼저 집계 후 JOIN (fan-out 방지).
        sql = """
            SELECT
                p.id, p.user, p.url, p.posting_at, p.content,
                p.like_count, p.comment_count, p.download_urls,
                p.created_at, p.entry,
                COALESCE(pr.follower_count, 0) AS follower_count
            FROM india_ai_fashion_inatagram_posting p
            LEFT JOIN (
                SELECT user, MAX(follower_count) AS follower_count
                FROM india_ai_fashion_inatagram_profile
                GROUP BY user
            ) pr ON p.user = pr.user
            WHERE p.posting_at IS NOT NULL AND p.posting_at != ''
            ORDER BY p.posting_at ASC
            LIMIT %s OFFSET %s
        """
        offset = batch_index * page_size
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (page_size, offset))
                rows = cur.fetchall()
        return [p for p in (_build_ig_post(r) for r in rows) if p is not None]

    def _load_youtube_count(self, batch_index: int, page_size: int) -> list[RawYouTubeVideo]:
        sql = """
            SELECT id, url, channel, channel_follower_count, title, description, tags,
                   thumbnail_url, upload_date, view_count, like_count,
                   comment_count, comments, download_urls
            FROM india_ai_fashion_youtube_posting
            WHERE upload_date IS NOT NULL AND upload_date != ''
            ORDER BY upload_date ASC, created_at ASC
            LIMIT %s OFFSET %s
        """
        offset = batch_index * page_size
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (page_size, offset))
                rows = cur.fetchall()
        return [v for v in (_build_yt_video(r) for r in rows) if v is not None]

    # ------------------------------------------------------------------ #
    # date 모드
    # ------------------------------------------------------------------ #

    def _load_instagram_date(
        self, window_start: date, window_end: date
    ) -> list[RawInstagramPost]:
        sql = """
            SELECT
                p.id, p.user, p.url, p.posting_at, p.content,
                p.like_count, p.comment_count, p.download_urls,
                p.created_at, p.entry,
                COALESCE(pr.follower_count, 0) AS follower_count
            FROM india_ai_fashion_inatagram_posting p
            LEFT JOIN (
                SELECT user, MAX(follower_count) AS follower_count
                FROM india_ai_fashion_inatagram_profile
                GROUP BY user
            ) pr ON p.user = pr.user
            WHERE DATE(p.posting_at) >= %s AND DATE(p.posting_at) <= %s
            ORDER BY p.posting_at ASC
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (window_start.isoformat(), window_end.isoformat()))
                rows = cur.fetchall()
        return [p for p in (_build_ig_post(r) for r in rows) if p is not None]

    def _load_youtube_date(
        self, window_start: date, window_end: date
    ) -> list[RawYouTubeVideo]:
        sql = """
            SELECT id, url, channel, channel_follower_count, title, description, tags,
                   thumbnail_url, upload_date, view_count, like_count,
                   comment_count, comments, download_urls
            FROM india_ai_fashion_youtube_posting
            WHERE upload_date >= %s AND upload_date <= %s
            ORDER BY upload_date ASC, created_at ASC
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (
                    window_start.strftime("%Y%m%d"),
                    window_end.strftime("%Y%m%d"),
                ))
                rows = cur.fetchall()
        return [v for v in (_build_yt_video(r) for r in rows) if v is not None]
