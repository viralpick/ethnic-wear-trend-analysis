"""M3.H — YT video_urls 매핑 pinning.

목표:
- `_build_yt_video` 가 `download_urls` (CSV) 를 video_urls 로 매핑
- IG 와 달리 image 혼입 없음 (mp4 만) — image 는 drop
- `normalize_youtube_video` 가 video_urls 를 NormalizedContentItem 으로 전달
- backwards-compat: download_urls=None / 빈 문자열 → video_urls=[]
"""
from __future__ import annotations

from datetime import datetime, timezone

from contracts.raw import RawYouTubeVideo
from loaders.starrocks_raw_loader import _build_yt_video
from normalization.normalize_content import normalize_youtube_video


def _yt_row(download_urls: str | None = "") -> dict:
    return {
        "id": "01YT_TESTROW",
        "url": "https://www.youtube.com/watch?v=aNbx37Vzd88",
        "channel": "Jhanvi Bhatia",
        "title": "Saree Haul",
        "description": "ethnic wear haul",
        "tags": "saree,haul,ethnic",
        "thumbnail_url": "https://example.com/thumb.jpg",
        "upload_date": "20260420",
        "view_count": 1000,
        "like_count": 50,
        "comment_count": 10,
        "comments": "comment1|comment2",
        "download_urls": download_urls,
        "created_at": datetime(2026, 4, 20, tzinfo=timezone.utc),
    }


def test_build_yt_video_maps_download_urls_to_video_urls() -> None:
    """download_urls 의 mp4 URL → video_urls 매핑."""
    row = _yt_row(
        "collectify/poc/ai_fashion/youtube/UCXFRIXunsSkK_5cMOVK3M6A/aNbx37Vzd88.mp4"
    )
    video = _build_yt_video(row)
    assert video is not None
    assert video.video_urls == [
        "collectify/poc/ai_fashion/youtube/UCXFRIXunsSkK_5cMOVK3M6A/aNbx37Vzd88.mp4"
    ]


def test_build_yt_video_empty_download_urls_yields_empty_list() -> None:
    """download_urls 빈 문자열/None → video_urls=[] (backwards-compat)."""
    video = _build_yt_video(_yt_row(""))
    assert video is not None
    assert video.video_urls == []

    video2 = _build_yt_video(_yt_row(None))
    assert video2 is not None
    assert video2.video_urls == []


def test_build_yt_video_drops_non_video_extensions() -> None:
    """download_urls 에 image 확장자가 섞이면 drop (보수). YT 는 mp4 만."""
    row = _yt_row(
        "stem/abc.mp4,stem/thumb.jpg,stem/intro.webm,stem/notes.txt"
    )
    video = _build_yt_video(row)
    assert video is not None
    # .mp4 + .webm 만 keep, .jpg / .txt drop.
    assert video.video_urls == ["stem/abc.mp4", "stem/intro.webm"]


def test_normalize_youtube_video_propagates_video_urls() -> None:
    """RawYouTubeVideo.video_urls → NormalizedContentItem.video_urls."""
    raw = RawYouTubeVideo(
        video_id="aNbx37Vzd88",
        channel="Jhanvi Bhatia",
        title="Saree Haul",
        description="ethnic wear haul",
        tags=["saree", "haul"],
        thumbnail_url="",
        view_count=1000,
        like_count=50,
        comment_count=10,
        top_comments=[],
        published_at=datetime(2026, 4, 20, tzinfo=timezone.utc),
        collected_at=datetime(2026, 4, 20, tzinfo=timezone.utc),
        video_urls=["stem/abc.mp4"],
    )
    item = normalize_youtube_video(raw)
    assert item.video_urls == ["stem/abc.mp4"]
    # YT 는 image_urls 는 여전히 비움 — thumbnail 은 본 컬러 추출에 안 씀.
    assert item.image_urls == []


def test_normalize_youtube_video_default_empty_video_urls() -> None:
    """raw.video_urls=[] (default) → item.video_urls=[]."""
    raw = RawYouTubeVideo(
        video_id="xyz",
        channel="ch",
        title="t",
        description="",
        tags=[],
        thumbnail_url="",
        view_count=0,
        like_count=0,
        comment_count=0,
        top_comments=[],
        published_at=datetime(2026, 4, 20, tzinfo=timezone.utc),
        collected_at=datetime(2026, 4, 20, tzinfo=timezone.utc),
    )
    item = normalize_youtube_video(raw)
    assert item.video_urls == []
