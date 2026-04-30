"""Raw (IG/YT) → NormalizedContentItem 변환 (spec §5.2 전처리 단계).

Engagement Score (rate-based, 2026-04-30 sync 결정):
- IG: likes / max(followers, 100) × 1 + comments / max(followers, 100) × 2
- YT: like_count / max(subscriber, 100) × 1 + comment_count / max(subscriber, 100) × 2
  + view_count 는 절대량 (engagement_raw_count) 으로만 보존
- saves 항목 raw DB 미수집 → 제외
- min_followers=100 — 작은 계정 viral 보존 + zero-division/huge-inflation 차단
"""
from __future__ import annotations

from contracts.common import ContentSource, InstagramSourceType
from contracts.normalized import NormalizedContentItem
from contracts.raw import RawInstagramPost, RawYouTubeVideo
from loaders.url_parsing import extract_url_short_tag

_MIN_FOLLOWERS = 100  # rate normalize lower bound (2026-04-30 sync)


def _engagement_score(likes: int, comments: int, followers: int | None) -> float:
    """rate-based engagement: likes/F × 1 + comments/F × 2. F = max(followers, 100)."""
    f = max(followers or 0, _MIN_FOLLOWERS)
    like_rate = likes / f
    comment_rate = comments / f
    return like_rate * 1.0 + comment_rate * 2.0


def _as_hashtag_token(tag: str) -> str:
    """YouTube tags 는 공백 있는 자유 문자열. IG 스타일 '#nospace' 로 정규화."""
    cleaned = tag.strip().lower().replace(" ", "")
    if not cleaned:
        return ""
    return cleaned if cleaned.startswith("#") else f"#{cleaned}"


def _classify_ig_source_type(
    post: RawInstagramPost,
    haul_tags: frozenset[str],
) -> InstagramSourceType:
    """M3.E — HASHTAG_TRACKING post 의 해시태그가 haul_tags 와 겹치면 HASHTAG_HAUL 로 승격.

    raw contract 의 source_type 은 크롤러 원본값 그대로 유지한다. 분석 파생 분류는
    NormalizedContentItem.ig_source_type 에만 기록 → 재수집 없이 분류 규칙만 갱신 가능.
    비교 기준: lowercase, leading `#` 제거.
    """
    if post.source_type != InstagramSourceType.HASHTAG_TRACKING or not haul_tags:
        return post.source_type
    post_tags = {t.lstrip("#").lower() for t in post.hashtags}
    if post_tags & haul_tags:
        return InstagramSourceType.HASHTAG_HAUL
    return post.source_type


def normalize_instagram_post(
    post: RawInstagramPost,
    haul_tags: frozenset[str] = frozenset(),
) -> NormalizedContentItem:
    """IG: text_blob = caption + hashtags, hashtags = post.hashtags, images = post.image_urls."""
    text_blob = post.caption_text
    if post.hashtags:
        text_blob = f"{text_blob} {' '.join(post.hashtags)}".strip()

    # rate-based engagement score (2026-04-30 sync). saves 는 raw DB 미수집 → 제외.
    # 절대량 합산은 별도 engagement_raw_count 로 보존 (top N 정렬용).
    engagement_score = _engagement_score(
        post.likes, post.comments_count, post.account_followers,
    )
    engagement_raw_count = post.likes + post.comments_count * 2

    return NormalizedContentItem(
        source=ContentSource.INSTAGRAM,
        source_post_id=post.post_id,
        url_short_tag=extract_url_short_tag(post.post_url),
        text_blob=text_blob,
        hashtags=list(post.hashtags),
        image_urls=list(post.image_urls),
        video_urls=list(post.video_urls),
        post_date=post.post_date,
        engagement_score=engagement_score,
        engagement_raw_count=engagement_raw_count,
        account_followers=post.account_followers,
        ig_source_type=_classify_ig_source_type(post, haul_tags).value,
        account_handle=post.account_handle,
    )


def normalize_youtube_video(video: RawYouTubeVideo) -> NormalizedContentItem:
    """YT: text_blob = title + description + tags, hashtags = tags as #noshape.

    M3.H — image_urls 는 빈 리스트 (YT 는 thumbnail 만, 본 컬러 추출엔 안 씀).
    video_urls 는 download_urls 의 mp4 매핑 → VideoFrameSource 가 frame 추출 후
    Pipeline B 가 IG 와 동일한 흐름으로 처리.

    B-2 (M3.G/H 후): YT channel 을 account_handle 에 매핑 — momentum sub-signal
    `new_yt_channel_ratio` 추적의 entity 식별자. IG handle 과 단위가 다르지만 동일
    필드로 흐른 뒤 score_history bucket 의 accounts/channels 분리 list 로 누적된다.
    """
    text_blob = " ".join([video.title, video.description, *video.tags]).strip()
    tag_tokens = [t for t in (_as_hashtag_token(t) for t in video.tags) if t]

    # rate-based engagement: like/sub × 1 + comment/sub × 2. saves 미수집 → 제외.
    # view_count 는 절대량 (engagement_raw_count) 으로만 보존 (rate 의미 약함, 노출 ≠ 호응).
    engagement_score = _engagement_score(
        video.like_count, video.comment_count, video.channel_follower_count,
    )
    engagement_raw_count = (
        video.view_count + video.like_count + video.comment_count * 2
    )

    return NormalizedContentItem(
        source=ContentSource.YOUTUBE,
        source_post_id=video.video_id,
        url_short_tag=extract_url_short_tag(video.video_url) or video.video_id,
        text_blob=text_blob,
        hashtags=tag_tokens,
        image_urls=[],
        video_urls=list(video.video_urls),
        post_date=video.published_at,
        engagement_score=engagement_score,
        engagement_raw_count=engagement_raw_count,
        account_followers=video.channel_follower_count,
        account_handle=video.channel or None,
    )


def normalize_batch(
    instagram_posts: list[RawInstagramPost],
    youtube_videos: list[RawYouTubeVideo],
    haul_tags: frozenset[str] = frozenset(),
) -> list[NormalizedContentItem]:
    ig = [normalize_instagram_post(p, haul_tags) for p in instagram_posts]
    yt = [normalize_youtube_video(v) for v in youtube_videos]
    return ig + yt
