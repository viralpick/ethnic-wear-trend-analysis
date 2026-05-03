"""URL → unique short tag 추출 (Phase 3, 2026-04-30 sync).

raw DB 의 source_post_id (ULID) 는 같은 외부 게시물에 대해 매 crawl 마다 새로
발급됨 → ULID 기준 dedup 시 같은 게시물의 시계열 snapshot 도 합쳐버림.

URL short tag 가 외부 게시물의 진짜 unique key:
- IG: `instagram.com/p/{shortcode}/` — shortcode 가 11자 base64-like
- YT: `youtube.com/watch?v={video_id}` — video_id 가 11자

같은 short tag 의 multiple snapshot = 시계열 (좋아요 / 조회수 변화) → growth rate 시그널.
"""
from __future__ import annotations

import re

_IG_SHORTCODE_RE = re.compile(r"instagram\.com/p/([A-Za-z0-9_-]+)")
_IG_REEL_RE = re.compile(r"instagram\.com/reel/([A-Za-z0-9_-]+)")
_YT_VIDEO_ID_RE = re.compile(r"[?&]v=([A-Za-z0-9_-]+)")
_YT_SHORT_RE = re.compile(r"youtu\.be/([A-Za-z0-9_-]+)")

# IG download_urls 는 image+video 혼입 (jpg/mp4 carousel). 확장자 기반 분류 — single source.
# 호출처: `loaders/starrocks_raw_loader._split_image_video` + `scripts/cost_audit._classify_ig_urls`
# 등. 정책 변경은 여기서만 (drift 방지).
VIDEO_EXTENSIONS: tuple[str, ...] = (".mp4", ".mov", ".webm", ".m4v")


def split_image_video_urls(urls: list[str]) -> tuple[list[str], list[str]]:
    """확장자 기반으로 (image_urls, video_urls) 분리. unknown 확장자는 images 쪽 (보수).

    URL query string (`?sig=...`) 은 분류 시 무시 (`split("?", 1)[0]`). lowercase 비교.
    `feedback_resolve_video_paths_query_string` 에 명시된 query string 함정 처리.
    """
    images: list[str] = []
    videos: list[str] = []
    for url in urls:
        path = url.split("?", 1)[0].lower()
        if path.endswith(VIDEO_EXTENSIONS):
            videos.append(url)
        else:
            images.append(url)
    return images, videos


def extract_yt_video_id(url: str | None) -> str | None:
    """YT URL → video_id (`watch?v=` 또는 `youtu.be/`). 매칭 실패 시 None.

    loaders/* 가 raw row 의 url 컬럼에서 video_id 를 뽑을 때 사용. 자체 정규식 정의 금지 —
    이 함수가 single source 이므로 패턴 변경은 여기에서만.
    """
    if not url:
        return None
    m = _YT_VIDEO_ID_RE.search(url) or _YT_SHORT_RE.search(url)
    return m.group(1) if m else None


def extract_url_short_tag(url: str | None) -> str | None:
    """외부 게시물 unique key — IG shortcode / YT video_id.

    IG 패턴:
    - https://www.instagram.com/p/{shortcode}/
    - https://www.instagram.com/reel/{shortcode}/

    YT 패턴:
    - https://www.youtube.com/watch?v={video_id}
    - https://youtu.be/{video_id}

    매칭 실패 시 None — caller 가 fallback (raw post_id 사용).
    """
    if not url:
        return None
    m = _IG_SHORTCODE_RE.search(url) or _IG_REEL_RE.search(url)
    if m:
        return m.group(1)
    m = _YT_VIDEO_ID_RE.search(url) or _YT_SHORT_RE.search(url)
    if m:
        return m.group(1)
    return None
