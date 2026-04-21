from __future__ import annotations

from pathlib import Path

from contracts.common import InstagramSourceType
from loaders.sample_loader import load_instagram_samples, load_youtube_samples

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "sample_data"


def test_load_instagram_samples_count() -> None:
    posts = load_instagram_samples(SAMPLE_DIR / "sample_instagram_posts.json")

    assert len(posts) == 8


def test_load_youtube_samples_count() -> None:
    videos = load_youtube_samples(SAMPLE_DIR / "sample_youtube_videos.json")

    assert len(videos) == 3


def test_instagram_source_type_coverage() -> None:
    # 샘플 설계(§3.1 A/B/C)가 세 source_type 을 모두 담아야 하므로 탈락하면 조기에 잡힌다.
    posts = load_instagram_samples(SAMPLE_DIR / "sample_instagram_posts.json")
    source_types = {post.source_type for post in posts}

    assert source_types == {
        InstagramSourceType.INFLUENCER_FIXED,
        InstagramSourceType.HASHTAG_TRACKING,
        InstagramSourceType.BOLLYWOOD_DECODE,
    }


def test_saves_nullability() -> None:
    # 이모지-only 포스트(ig_unclassifiable_1)가 saves=null 케이스를 대표한다.
    posts = load_instagram_samples(SAMPLE_DIR / "sample_instagram_posts.json")
    null_saves = [post for post in posts if post.saves is None]

    assert len(null_saves) == 1
    assert null_saves[0].post_id == "ig_unclassifiable_1"


def test_youtube_fields_present() -> None:
    # YT raw contract 는 channel / like_count / comment_count 명명을 쓴다 (IG 와 비대칭).
    videos = load_youtube_samples(SAMPLE_DIR / "sample_youtube_videos.json")

    first = videos[0]
    assert first.channel.startswith("@")
    assert isinstance(first.like_count, int)
    assert isinstance(first.comment_count, int)
