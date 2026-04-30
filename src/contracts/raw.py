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
    # 외부 IG URL (Phase 3, 2026-04-30): unique 게시물 식별 + 시계열 dedup용. raw DB
    # `url` 컬럼. backwards-compat default None (옛 fixture / FakeIngestor 호환).
    post_url: str | None = None
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
    # raw DB 미수집 → 항상 None (2026-04-30 sync). Engagement Score (rate-based) 에서도
    # saves 항 제외됨. 유지는 backwards-compat 만 (tsv fixture 등). 추후 cleanup 후보.
    saves: int | None = None

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
    # 외부 YT URL (Phase 3, 2026-04-30): video_id 와 같은 unique 식별이지만 IG 와
    # 일관성 위해 별도 보존. backwards-compat default None.
    video_url: str | None = None
    channel: str
    # 채널 구독자 수 (rate-based engagement 분모, 2026-04-30). raw DB
    # `channel_follower_count` 컬럼. 미수집/0 → engagement_score 가 _MIN_FOLLOWERS=100
    # fallback. backwards-compat default 0.
    channel_follower_count: int = 0

    title: str
    description: str
    tags: list[str]
    thumbnail_url: str

    view_count: int
    like_count: int
    comment_count: int
    top_comments: list[str]  # spec §3.2 — 상위 50개, 텍스트만

    published_at: datetime
    # 크롤 수집 시점. 시계열 growth rate 계산의 Δ days 분모로 사용 (published_at 은 게시일
    # 불변이라 multi-snapshot 에 동일하므로 부적합). raw DB `created_at` 컬럼 매핑.
    # 2026-04-30 추가.
    collected_at: datetime

    # M3.H — YT 영상 mp4 Azure Blob URL. 보통 1개. backwards-compat 위해 default `[]`.
    video_urls: list[str] = []
