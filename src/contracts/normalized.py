"""NormalizedContentItem — attribute extraction / clustering 이 먹기 좋은 최소 필드 뷰.

NOT a canonical domain object — lossy by design. 의도적으로 IG/YT 대칭을 강제하지 않는다.
downstream 단계가 source-specific 필드 (예: account_followers, view_count) 가 필요하면
이 뷰가 아니라 raw record 에서 직접 가져온다. 이 뷰는 "hint" 이지 "bottleneck" 이 아니다.
"""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from contracts.common import ContentSource


class NormalizedContentItem(BaseModel):
    """
    purpose: Raw 를 룰 기반 + LLM 분류가 먹기 좋은 최소 필드로 요약 (lossy)
    stage: normalized
    ownership: analysis-owned
    stability: experimental (step 3-A extraction 모듈 요구에 따라 축소/확대 가능)
    """
    model_config = ConfigDict(frozen=True)

    source: ContentSource
    source_post_id: str

    # 룰/LLM 매칭 대상. IG: caption + hashtag 문자열, YT: title + description + tags.
    text_blob: str

    # 해시태그 기반 정확 매칭(spec §6.2)을 text_blob 토큰 분석보다 우선 사용하므로 분리 유지.
    hashtags: list[str]

    # VLM 대상 이미지. IG 포스트는 여러 장, YT 는 [] (spec §7.2 — YouTube 컬러 추출 안 함)
    image_urls: list[str]

    # 영상 URL (.mp4/.mov/.webm/.m4v). IG Reel / IG carousel video 는 raw download_urls
    # 를 확장자로 split 후 여기에. YT 영상은 M3.H 까지 [] (raw row 에 별도 매핑 미구현).
    # M3.G (2026-04-28) 추가, 기존 fixture 호환 default `[]`.
    video_urls: list[str] = []

    # IG: post_date, YT: published_at
    post_date: datetime

    # Engagement Score (rate-based, 2026-04-30 sync 결정):
    #   engagement_score = (likes / max(followers, 100)) × 1
    #                    + (comments / max(followers, 100)) × 2
    # YT 는 channel_follower_count 사용. saves 는 raw DB 미수집 → 제외.
    # 절대량 (likes/views/comments) 자체 정렬이 필요한 경우 (예: top_posts in cluster
    # drilldown) 는 engagement_raw_count 를 별도 사용.
    engagement_score: float = 0.0

    # 절대량 합산 (top N 정렬, 영상 view_count 부각 등). IG: likes + comments*2,
    # YT: view_count + like_count + comment_count*2. 옛 engagement_raw 의미 유지.
    engagement_raw_count: int = 0

    # IG 전용 scoring 입력 — YT 는 None / 0 (§9.2 Social / Cultural 계산에 직접 사용).
    account_followers: int = 0
    ig_source_type: str | None = None   # InstagramSourceType.value 문자열, YT 는 None
    account_handle: str | None = None   # new_account_ratio 추적 + top_influencers 표시용
