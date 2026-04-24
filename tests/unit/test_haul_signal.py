"""M3.E — 하울 파생 분류 + source_type_weight 가중치 단위 테스트."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from contracts.common import InstagramSourceType
from contracts.raw import RawInstagramPost
from normalization.normalize_content import (
    _classify_ig_source_type,
    normalize_instagram_post,
)
from pipelines.run_scoring_pipeline import _source_type_weight
from settings import ScoringConfig, SourceTypeWeights


def _make_post(
    source_type: InstagramSourceType,
    hashtags: list[str],
) -> RawInstagramPost:
    return RawInstagramPost(
        post_id="p1",
        source_type=source_type,
        account_handle="tester",
        account_followers=1000,
        image_urls=[],
        caption_text="",
        hashtags=hashtags,
        likes=10,
        comments_count=2,
        saves=None,
        post_date=datetime(2026, 4, 24, tzinfo=timezone.utc),
        collected_at=datetime(2026, 4, 24, tzinfo=timezone.utc),
    )


# --------------------------------------------------------------------------- #
# _classify_ig_source_type
# --------------------------------------------------------------------------- #

def test_hashtag_tracking_with_haul_tag_flips_to_haul() -> None:
    post = _make_post(InstagramSourceType.HASHTAG_TRACKING, ["myntrahaul", "ootd"])
    result = _classify_ig_source_type(post, frozenset({"myntrahaul"}))
    assert result == InstagramSourceType.HASHTAG_HAUL


def test_hashtag_tracking_without_haul_tag_stays_tracking() -> None:
    post = _make_post(InstagramSourceType.HASHTAG_TRACKING, ["ootd", "fashion"])
    result = _classify_ig_source_type(post, frozenset({"myntrahaul"}))
    assert result == InstagramSourceType.HASHTAG_TRACKING


def test_influencer_fixed_never_flips_even_with_haul_tag() -> None:
    # profile 수집 post 는 크롤러 분류 그대로 유지 — haul 은 hashtag 수집 대상에서만 의미.
    post = _make_post(InstagramSourceType.INFLUENCER_FIXED, ["myntrahaul"])
    result = _classify_ig_source_type(post, frozenset({"myntrahaul"}))
    assert result == InstagramSourceType.INFLUENCER_FIXED


def test_bollywood_decode_never_flips_even_with_haul_tag() -> None:
    post = _make_post(InstagramSourceType.BOLLYWOOD_DECODE, ["myntrahaul"])
    result = _classify_ig_source_type(post, frozenset({"myntrahaul"}))
    assert result == InstagramSourceType.BOLLYWOOD_DECODE


def test_empty_haul_tags_is_noop() -> None:
    post = _make_post(InstagramSourceType.HASHTAG_TRACKING, ["myntrahaul"])
    result = _classify_ig_source_type(post, frozenset())
    assert result == InstagramSourceType.HASHTAG_TRACKING


def test_hashtag_matching_strips_leading_hash_and_is_case_insensitive() -> None:
    # 크롤러가 `#` 포함 저장해도 / 대소문자 섞여도 매칭되어야 함.
    post = _make_post(InstagramSourceType.HASHTAG_TRACKING, ["#MyntraHaul"])
    result = _classify_ig_source_type(post, frozenset({"myntrahaul"}))
    assert result == InstagramSourceType.HASHTAG_HAUL


# --------------------------------------------------------------------------- #
# normalize_instagram_post end-to-end
# --------------------------------------------------------------------------- #

def test_normalize_instagram_post_records_haul_in_ig_source_type() -> None:
    post = _make_post(InstagramSourceType.HASHTAG_TRACKING, ["myntrahaul"])
    normalized = normalize_instagram_post(post, frozenset({"myntrahaul"}))
    assert normalized.ig_source_type == InstagramSourceType.HASHTAG_HAUL.value


def test_normalize_instagram_post_raw_contract_untouched() -> None:
    # raw source_type 은 파생 분류로 덮어쓰지 않음 (크롤러 원본값 보존 원칙).
    post = _make_post(InstagramSourceType.HASHTAG_TRACKING, ["myntrahaul"])
    _ = normalize_instagram_post(post, frozenset({"myntrahaul"}))
    assert post.source_type == InstagramSourceType.HASHTAG_TRACKING


# --------------------------------------------------------------------------- #
# _source_type_weight
# --------------------------------------------------------------------------- #

def _make_scoring_cfg() -> ScoringConfig:
    from settings import (
        CulturalFactorWeights,
        DataMaturityConfig,
        InfluencerTierThresholds,
        InfluencerWeights,
        LifecycleConfig,
        MomentumFactorWeights,
        ScoringWeights,
        YouTubeFactorWeights,
    )
    return ScoringConfig(
        weights=ScoringWeights(social=40.0, youtube=25.0, cultural=15.0, momentum=20.0),
        normalization_method="minmax_same_run",
        direction_threshold_pct=5.0,
        lifecycle=LifecycleConfig(
            early_below=30.0, growth_until=65.0, early_post_count_threshold=10
        ),
        influencer_weights=InfluencerWeights(mega=3.0, macro=2.0, mid=1.5, micro=1.0),
        influencer_tier_thresholds=InfluencerTierThresholds(
            mega=1000000, macro=100000, mid=10000
        ),
        source_type_weights=SourceTypeWeights(
            influencer_fixed=1.0,
            hashtag_tracking=1.2,
            hashtag_haul=1.5,
            bollywood_decode=1.1,
        ),
        youtube_factor_weights=YouTubeFactorWeights(
            video_count=0.3, views=0.4, view_growth=0.3
        ),
        cultural_factor_weights=CulturalFactorWeights(festival=0.6, bollywood=0.4),
        cultural_festival_boost=1.5,
        cultural_bollywood_bonus=0.3,
        momentum_factor_weights=MomentumFactorWeights(
            post_growth=0.4, hashtag_velocity=0.3, new_account_ratio=0.3
        ),
        momentum_window_days=7,
        data_maturity=DataMaturityConfig(bootstrap_below_days=3, full_from_days=7),
    )


@pytest.mark.parametrize("source_type,expected", [
    (InstagramSourceType.INFLUENCER_FIXED.value, 1.0),
    (InstagramSourceType.HASHTAG_TRACKING.value, 1.2),
    (InstagramSourceType.HASHTAG_HAUL.value, 1.5),
    (InstagramSourceType.BOLLYWOOD_DECODE.value, 1.1),
])
def test_source_type_weight_returns_configured_multiplier(
    source_type: str, expected: float
) -> None:
    cfg = _make_scoring_cfg()
    assert _source_type_weight(source_type, cfg) == expected


def test_source_type_weight_unknown_value_falls_back_to_one() -> None:
    # YouTube 같이 ig_source_type 이 None 이거나 예상 외 값이면 1.0 — no-op.
    cfg = _make_scoring_cfg()
    assert _source_type_weight(None, cfg) == 1.0
    assert _source_type_weight("something_else", cfg) == 1.0
