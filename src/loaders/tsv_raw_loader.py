"""TSV raw loader — 크롤러가 DB export 한 TSV 를 Raw contract 로 변환.

처리 대상 TSV (sample_data/ 의 실 파일 또는 fixture):
  - png_india_ai_fashion_inatagram_posting.tsv       (13 cols, IG 포스트 본문)
  - png_india_ai_fashion_inatagram_profile.tsv       (9 cols, 계정 프로필 — followers JOIN)
  - png_india_ai_fashionash_tag_search_result.tsv    (10 cols, 해시태그 검색 결과)
  - png_india_ai_fashion_youtube_posting.tsv         (18 cols, YT 영상)

제외:
  - fashionagram_profile_posting.tsv: posting.tsv 와 URL 100% 중복 (4/24 agenda 에 기록)

가정 / 합의 필요 (4/24 sync):
  - post_id 는 TSV col [2] ULID 사용 (BE/DW 팀 primary-key 기대 값 확인 필요)
  - posting.tsv source_type 은 INFLUENCER_FIXED 로 가정 (Top-10 profile-centered)
  - hashtag_search.tsv: account_handle 없음 → None 저장. post_date 없음 → 2026-04-15 placeholder.
  - image_urls 는 blob path (container-relative). pipeline_b_adapter 가 image_root+basename 매칭.
"""
from __future__ import annotations

import csv
import logging
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterator

from contracts.common import InstagramSourceType
from contracts.raw import RawInstagramPost, RawYouTubeVideo
from loaders._datetime import parse_db_timestamp, parse_iso_z, parse_yyyymmdd
from loaders.raw_loader import RawDailyBatch
from loaders.url_parsing import extract_yt_video_id

logger = logging.getLogger(__name__)

_POSTING_FILE = "png_india_ai_fashion_inatagram_posting.tsv"
_PROFILE_FILE = "png_india_ai_fashion_inatagram_profile.tsv"
_HASHTAG_SEARCH_FILE = "png_india_ai_fashionash_tag_search_result.tsv"
_YOUTUBE_FILE = "png_india_ai_fashion_youtube_posting.tsv"

_HASHTAG_RE = re.compile(r"#\w+")
_HASHTAG_SEARCH_PLACEHOLDER_DATE = datetime(2026, 4, 15, tzinfo=timezone.utc)


def _extract_hashtags(caption: str) -> list[str]:
    return _HASHTAG_RE.findall(caption or "")


def _split_csv_cell(cell: str) -> list[str]:
    return [x for x in (cell or "").split(",") if x]


def _read_tsv(path: Path) -> Iterator[list[str]]:
    """누락 파일은 빈 iterator. QUOTE_NONE — 따옴표를 literal 로 처리 (TSV 관례)."""
    if not path.exists():
        logger.info("tsv_missing path=%s", path)
        return iter(())
    handle = path.open(newline="", encoding="utf-8")
    return csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)


def _load_followers_index(path: Path) -> dict[str, int]:
    """profile.tsv → {account_handle → followers}. JOIN 테이블."""
    out: dict[str, int] = {}
    for row in _read_tsv(path):
        if len(row) < 5:
            continue
        try:
            out[row[2]] = int(row[4])
        except ValueError:
            continue
    return out


def _build_posting(
    row: list[str], followers_by_handle: dict[str, int],
) -> RawInstagramPost | None:
    """posting.tsv row → RawInstagramPost. 실패 시 None."""
    if len(row) < 13:
        return None
    handle = row[2] or None
    try:
        return RawInstagramPost(
            post_id=row[1],
            source_type=InstagramSourceType.INFLUENCER_FIXED,
            account_handle=handle,
            account_followers=followers_by_handle.get(handle or "", 0),
            image_urls=_split_csv_cell(row[10]),
            caption_text=row[6],
            hashtags=_extract_hashtags(row[6]),
            likes=int(row[8]),
            comments_count=int(row[9]),
            saves=None,
            post_date=parse_iso_z(row[5]),
            collected_at=parse_db_timestamp(row[11]),
        )
    except (ValueError, KeyError) as exc:
        logger.info("tsv_posting_skip post_id=%s reason=%s", row[1] if len(row) > 1 else "?", exc)
        return None


def _build_hashtag_search(row: list[str]) -> RawInstagramPost | None:
    """hashtag_search.tsv row → RawInstagramPost. account_handle=None, post_date=placeholder."""
    if len(row) < 10:
        return None
    tag = row[2]
    try:
        return RawInstagramPost(
            post_id=row[1],
            source_type=InstagramSourceType.HASHTAG_TRACKING,
            account_handle=None,
            account_followers=0,
            image_urls=[row[4]] if row[4] else [],
            caption_text=row[5],
            # spec §4.2: hashtag source 를 hashtags 리스트에 보존 — 어느 태그에서 왔는지 추적.
            hashtags=[f"#{tag}"] if tag else [],
            likes=int(row[6] or 0),
            comments_count=int(row[7] or 0),
            saves=None,
            # TSV 에 post_date 없음 — 크롤러 팀에 요청. 일단 placeholder (agenda §1.x).
            post_date=_HASHTAG_SEARCH_PLACEHOLDER_DATE,
            collected_at=parse_db_timestamp(row[8]),
        )
    except (ValueError, KeyError) as exc:
        logger.info("tsv_hashtag_skip post_id=%s reason=%s", row[1] if len(row) > 1 else "?", exc)
        return None


def _build_youtube(row: list[str]) -> RawYouTubeVideo | None:
    """youtube_posting.tsv row → RawYouTubeVideo. 실패 시 None."""
    if len(row) < 18:
        return None
    video_id = extract_yt_video_id(row[3])
    if video_id is None:
        logger.warning("tsv_yt_skip ulid=%s reason=no_video_id_in_url", row[1])
        return None
    try:
        published = parse_yyyymmdd(row[10])
        return RawYouTubeVideo(
            video_id=video_id,
            channel=row[4],
            title=row[6],
            description=row[7],
            tags=_split_csv_cell(row[8]),
            thumbnail_url=row[9],
            view_count=int(row[11]),
            like_count=int(row[12]),
            comment_count=int(row[13]),
            top_comments=[x for x in (row[14] or "").split("|") if x],
            published_at=published,
            # TSV 에 collected_at 컬럼 없음 — published_at 으로 fallback (TSV path 는
            # deprecated, growth_rate 시계열 의미 없음).
            collected_at=published,
        )
    except (ValueError, KeyError) as exc:
        logger.info("tsv_yt_skip ulid=%s reason=%s", row[1] if len(row) > 1 else "?", exc)
        return None


class TsvRawLoader:
    """RawLoader Protocol 구현체. sample_data/ 의 실 TSV 또는 fixture 디렉토리를 받는다.

    target_date 는 현재 파티션 필터로 쓰지 않는다 (TSV 가 단일 날짜 export 를 전제).
    M3 에서 Blob 연동 시 date 기반 파티션 로드 예정.
    """

    def __init__(self, tsv_dir: Path) -> None:
        self._tsv_dir = Path(tsv_dir)
        self._followers_by_handle = _load_followers_index(self._tsv_dir / _PROFILE_FILE)

    def load_batch(self, target_date: date) -> RawDailyBatch:  # noqa: ARG002
        ig: list[RawInstagramPost] = []
        for row in _read_tsv(self._tsv_dir / _POSTING_FILE):
            post = _build_posting(row, self._followers_by_handle)
            if post is not None:
                ig.append(post)
        for row in _read_tsv(self._tsv_dir / _HASHTAG_SEARCH_FILE):
            post = _build_hashtag_search(row)
            if post is not None:
                ig.append(post)
        yt: list[RawYouTubeVideo] = []
        for row in _read_tsv(self._tsv_dir / _YOUTUBE_FILE):
            video = _build_youtube(row)
            if video is not None:
                yt.append(video)
        logger.info("tsv_loaded ig=%d yt=%d from=%s", len(ig), len(yt), self._tsv_dir)
        return RawDailyBatch(instagram=ig, youtube=yt)
