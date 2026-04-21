"""Raw (IG/YT) → NormalizedContentItem 변환 (spec §5.2 전처리 단계)."""
from __future__ import annotations

from contracts.common import ContentSource
from contracts.normalized import NormalizedContentItem
from contracts.raw import RawInstagramPost, RawYouTubeVideo


def _as_hashtag_token(tag: str) -> str:
    """YouTube tags 는 공백 있는 자유 문자열. IG 스타일 '#nospace' 로 정규화."""
    cleaned = tag.strip().lower().replace(" ", "")
    if not cleaned:
        return ""
    return cleaned if cleaned.startswith("#") else f"#{cleaned}"


def normalize_instagram_post(post: RawInstagramPost) -> NormalizedContentItem:
    """IG: text_blob = caption + hashtags, hashtags = post.hashtags, images = post.image_urls."""
    text_blob = post.caption_text
    if post.hashtags:
        text_blob = f"{text_blob} {' '.join(post.hashtags)}".strip()

    # spec §9.2 공식의 단순 합 (단, influencer_weight 없이). scoring 에서 raw 로부터 재계산.
    engagement = post.likes + post.comments_count * 2 + (post.saves or 0) * 3

    return NormalizedContentItem(
        source=ContentSource.INSTAGRAM,
        source_post_id=post.post_id,
        text_blob=text_blob,
        hashtags=list(post.hashtags),
        image_urls=list(post.image_urls),
        post_date=post.post_date,
        engagement_raw=engagement,
    )


def normalize_youtube_video(video: RawYouTubeVideo) -> NormalizedContentItem:
    """YT: text_blob = title + description + tags, hashtags = tags as #noshape, images = []."""
    text_blob = " ".join([video.title, video.description, *video.tags]).strip()
    tag_tokens = [t for t in (_as_hashtag_token(t) for t in video.tags) if t]

    # spec §7.2 — YouTube 는 컬러 추출 안 함, 이미지 리스트는 비움.
    # engagement 프리-랭킹: view_count 가 dominant signal (추후 scoring 에서 정교화).
    engagement = video.view_count + video.like_count + video.comment_count * 2

    return NormalizedContentItem(
        source=ContentSource.YOUTUBE,
        source_post_id=video.video_id,
        text_blob=text_blob,
        hashtags=tag_tokens,
        image_urls=[],
        post_date=video.published_at,
        engagement_raw=engagement,
    )


def normalize_batch(
    instagram_posts: list[RawInstagramPost],
    youtube_videos: list[RawYouTubeVideo],
) -> list[NormalizedContentItem]:
    ig = [normalize_instagram_post(p) for p in instagram_posts]
    yt = [normalize_youtube_video(v) for v in youtube_videos]
    return ig + yt
