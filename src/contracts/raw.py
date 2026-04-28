"""크롤러 출력 contract.

크롤러(별도 레포)가 우리에게 건네주는 shape. 우리는 크롤러에 맞추지 않고, 크롤러가 여기에
맞춘다. `extra="forbid"` 로 스키마 드리프트를 조기에 잡는다.

IG 와 YT 의 필드명 비대칭은 현재 domain spec 입력 그대로다 (likes vs like_count,
collected_at 유무 등). 4/24 싱크 전까지 크롤러 팀과 통일 여부를 협상할 필드는 README 의
recommendation 섹션에 명시.
"""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from contracts.common import InstagramSourceType


class RawInstagramPost(BaseModel):
    """
    purpose: Instagram 포스트 1건의 크롤러 수집 결과
    stage: raw
    ownership: crawler-owned
    stability: negotiable (4/24 이전 필드 정합 조율 — README 참조)
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    post_id: str
    source_type: InstagramSourceType
    # hashtag_search 소스의 포스트는 계정 정보 없이 수집될 수 있음 → None 허용 (2026-04-22).
    # 크롤러 팀과 4/24 에 "항상 account 해석" 가능 여부 협의 후 재고.
    account_handle: str | None = None
    account_followers: int

    image_urls: list[str]
    # IG Reel / IG carousel 의 영상 URL (.mp4/.mov/.webm/.m4v). 크롤러 raw 에서는
    # download_urls 가 image+video 혼입 단일 CSV — loader 가 확장자로 split. M3.G
    # (2026-04-28) 추가. 기존 fixture / FakeIngestor 호환 위해 default `[]`.
    video_urls: list[str] = []
    caption_text: str
    hashtags: list[str]

    likes: int
    comments_count: int
    saves: int | None  # 수집 실패 시 null 허용 (spec §8.1)

    post_date: datetime
    collected_at: datetime


class RawYouTubeVideo(BaseModel):
    """
    purpose: YouTube 영상 1건의 크롤러 수집 결과 (spec §3.2 D)
    stage: raw
    ownership: crawler-owned
    stability: negotiable (IG 와의 필드명 비대칭 — README 참조)
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    video_id: str
    channel: str

    title: str
    description: str
    tags: list[str]
    thumbnail_url: str

    view_count: int
    like_count: int
    comment_count: int
    top_comments: list[str]  # spec §3.2 — 상위 50개, 텍스트만

    published_at: datetime

    # M3.H — YT 영상 mp4 Azure Blob URL. 보통 1개. backwards-compat 위해 default `[]`.
    video_urls: list[str] = []
