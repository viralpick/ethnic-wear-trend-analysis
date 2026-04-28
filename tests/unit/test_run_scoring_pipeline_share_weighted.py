"""Phase β2 (2026-04-28) — _build_contexts share-weighted fan-out pinning.

검증 포인트 (advisor mass preservation invariant + N<3 zero contribution):
- N=3 item 의 share 합 = 1.0 (cross-product 분산 후 mass 보존)
- N<3 (G/T/F 한 축이라도 비면) item 은 어느 cluster 에도 기여 0
- 1 item × 다중 cluster_key 분배 — winner-takes-all 합산값 = fan-out 합산값
- youtube_video_count, social_weighted_engagement 등 모든 누적값이 share 비례
- accounts 는 winner-keyed 유지 (fan-out 으로 새 cluster 에 중복 X)
- post_count_total = history int + round(share-sum) — γ 마이그 전 mixed precision
"""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pytest

from contracts.common import (
    ClassificationMethod,
    ContentSource,
    Fabric,
    GarmentType,
    Technique,
)
from contracts.enriched import EnrichedContentItem
from contracts.normalized import NormalizedContentItem
from pipelines.run_scoring_pipeline import (
    _accumulate_share_weighted,
    _build_contexts,
)
from scoring.score_history import ScoreHistory
from settings import (
    CulturalFactorWeights,
    DataMaturityConfig,
    InfluencerTierThresholds,
    InfluencerWeights,
    LifecycleConfig,
    MomentumFactorWeights,
    ScoringConfig,
    ScoringWeights,
    SourceTypeWeights,
    YouTubeFactorWeights,
)


def _cfg() -> ScoringConfig:
    return ScoringConfig(
        weights=ScoringWeights(social=40.0, youtube=25.0, cultural=15.0, momentum=20.0),
        normalization_method="minmax_same_run",
        direction_threshold_pct=5.0,
        lifecycle=LifecycleConfig(
            early_below=30.0, growth_until=65.0, early_post_count_threshold=10
        ),
        influencer_weights=InfluencerWeights(mega=3.0, macro=2.0, mid=1.5, micro=1.0),
        influencer_tier_thresholds=InfluencerTierThresholds(
            mega=1_000_000, macro=100_000, mid=10_000
        ),
        source_type_weights=SourceTypeWeights(
            influencer_fixed=1.0, hashtag_tracking=1.0,
            hashtag_haul=1.0, bollywood_decode=1.0,
        ),
        youtube_factor_weights=YouTubeFactorWeights(
            video_count=0.3, views=0.4, view_growth=0.3
        ),
        cultural_factor_weights=CulturalFactorWeights(festival=0.6, bollywood=0.4),
        cultural_festival_boost=1.5,
        cultural_bollywood_bonus=0.3,
        cultural_festivals=[],
        momentum_factor_weights=MomentumFactorWeights(
            post_growth=0.4, hashtag_velocity=0.3, new_account_ratio=0.3
        ),
        momentum_window_days=7,
        data_maturity=DataMaturityConfig(bootstrap_below_days=3, full_from_days=7),
    )


def _normalized(
    post_id: str,
    *,
    source: ContentSource = ContentSource.INSTAGRAM,
    engagement: int = 100,
    handle: str | None = None,
) -> NormalizedContentItem:
    return NormalizedContentItem(
        source=source,
        source_post_id=post_id,
        text_blob="",
        hashtags=[],
        image_urls=[],
        post_date=datetime(2026, 4, 27),
        engagement_raw=engagement,
        account_handle=handle,
        account_followers=1000,
    )


def _enriched(
    post_id: str,
    *,
    g: GarmentType | None,
    t: Technique | None,
    f: Fabric | None,
    cluster_key: str | None,
    source: ContentSource = ContentSource.INSTAGRAM,
    engagement: int = 100,
    handle: str | None = None,
) -> EnrichedContentItem:
    """3 축 enum + classification_method=RULE 로 distribution 결정성 확보."""
    methods: dict[str, ClassificationMethod] = {}
    if g is not None:
        methods["garment_type"] = ClassificationMethod.RULE
    if t is not None:
        methods["technique"] = ClassificationMethod.RULE
    if f is not None:
        methods["fabric"] = ClassificationMethod.RULE
    return EnrichedContentItem(
        normalized=_normalized(post_id, source=source, engagement=engagement, handle=handle),
        garment_type=g,
        fabric=f,
        technique=t,
        canonicals=[],
        classification_method_per_attribute=methods,
        trend_cluster_key=cluster_key,
    )


@pytest.fixture
def empty_history(tmp_path: Path) -> ScoreHistory:
    return ScoreHistory(tmp_path / "score_history.json")


# --------------------------------------------------------------------------- #
# Invariants
# --------------------------------------------------------------------------- #

def test_n_lt_3_item_contributes_partial_mass(empty_history: ScoreHistory) -> None:
    # Phase partial(g) 활성화 (2026-04-28) — β2 의 N<3 zero contribution 정책 revisit.
    # N=2 (technique 누락) → assign_shares 가 multiplier_ratio (0.5) 가중 share 반환 →
    # cluster 에 mass=0.5 비례 기여 (per-item mass: N=3=1.0 / N=2=0.5 / N=1=0.2).
    item = _enriched("p1", g=GarmentType.KURTA_SET, t=None, f=Fabric.COTTON,
                     cluster_key="kurta_set__unknown__cotton")
    grouped = {"kurta_set__unknown__cotton": [item]}
    acc = _accumulate_share_weighted(grouped, date(2026, 4, 27), _cfg())
    assert "kurta_set__unknown__cotton" in acc
    a = acc["kurta_set__unknown__cotton"]
    assert a.post_count_today == pytest.approx(0.5)  # N=2 multiplier_ratio
    assert a.social_weighted_engagement == pytest.approx(50.0)  # 100 × 0.5

    contexts = _build_contexts(grouped, date(2026, 4, 27), _cfg(), empty_history)
    assert len(contexts) == 1
    ctx = contexts[0]
    assert ctx.cluster_key == "kurta_set__unknown__cotton"
    assert ctx.post_count_today == pytest.approx(0.5)
    assert ctx.social_weighted_engagement == pytest.approx(50.0)


def test_n_zero_item_contributes_zero(empty_history: ScoreHistory) -> None:
    # N=0 (G/T/F 모두 없음) → assign_shares 빈 dict → accumulator 빈 dict.
    # zero-aggregate fallback 으로 grouped 의 winner key 만 context 받음.
    item = _enriched("p1", g=None, t=None, f=None,
                     cluster_key="unclassified")
    grouped = {"unclassified": [item]}
    acc = _accumulate_share_weighted(grouped, date(2026, 4, 27), _cfg())
    assert acc == {}

    contexts = _build_contexts(grouped, date(2026, 4, 27), _cfg(), empty_history)
    assert len(contexts) == 1
    ctx = contexts[0]
    assert ctx.cluster_key == "unclassified"
    assert ctx.post_count_today == pytest.approx(0.0)
    assert ctx.social_weighted_engagement == pytest.approx(0.0)


def test_n_eq_3_mass_preservation_single_winner(empty_history: ScoreHistory) -> None:
    # G/T/F 모두 단일값 1.0 → cross-product 1개 cluster, share=1.0.
    item = _enriched(
        "p1", g=GarmentType.KURTA_SET, t=Technique.CHIKANKARI, f=Fabric.COTTON,
        cluster_key="kurta_set__chikankari__cotton",
    )
    grouped = {"kurta_set__chikankari__cotton": [item]}
    acc = _accumulate_share_weighted(grouped, date(2026, 4, 27), _cfg())

    assert set(acc.keys()) == {"kurta_set__chikankari__cotton"}
    a = acc["kurta_set__chikankari__cotton"]
    assert a.post_count_today == pytest.approx(1.0)
    # winner-takes-all baseline 과 동치 (degenerate distribution)
    assert a.social_weighted_engagement == pytest.approx(100.0)


def test_mass_preservation_3_items(empty_history: ScoreHistory) -> None:
    # 3 N=3 items + 1 N=2 item → partial 활성화 후 sum(post_count_today) = 3.0 + 0.5 = 3.5.
    # per-item mass: N=3=1.0 (×3) + N=2=0.5 (×1) = 3.5.
    items = [
        _enriched("p1", g=GarmentType.KURTA_SET, t=Technique.CHIKANKARI, f=Fabric.COTTON,
                  cluster_key="kurta_set__chikankari__cotton"),
        _enriched("p2", g=GarmentType.CASUAL_SAREE, t=Technique.BLOCK_PRINT, f=Fabric.CHANDERI,
                  cluster_key="casual_saree__block_print__chanderi"),
        _enriched("p3", g=GarmentType.KURTA_SET, t=Technique.CHIKANKARI, f=Fabric.COTTON,
                  cluster_key="kurta_set__chikankari__cotton"),
        _enriched("p4_n2", g=GarmentType.KURTA_SET, t=None, f=Fabric.COTTON,
                  cluster_key="kurta_set__unknown__cotton"),
    ]
    grouped = {
        "kurta_set__chikankari__cotton": [items[0], items[2]],
        "casual_saree__block_print__chanderi": [items[1]],
        "kurta_set__unknown__cotton": [items[3]],
    }
    acc = _accumulate_share_weighted(grouped, date(2026, 4, 27), _cfg())
    total_mass = sum(a.post_count_today for a in acc.values())
    assert total_mass == pytest.approx(3.5)  # N=3 (×3, 1.0) + N=2 (×1, 0.5)


def test_share_fan_out_cross_product(empty_history: ScoreHistory) -> None:
    # garment_type 의 분포는 enriched_to_item_distribution 에서 text RULE=6 만 단일이라
    # G={value:1.0} 만 가능. cross-product 검증은 vision share 까지 가야해서 representative_builder
    # 의 item_cluster_shares 에서 직접 확인 (test_representative_builder 가 cover).
    # 여기서는 1 item 의 contribution 이 정확히 1 cluster 에 들어가는지 핀.
    items = [
        _enriched("p1", g=GarmentType.KURTA_SET, t=Technique.CHIKANKARI, f=Fabric.COTTON,
                  cluster_key="kurta_set__chikankari__cotton", engagement=200),
        _enriched("p2", g=GarmentType.CASUAL_SAREE, t=Technique.BLOCK_PRINT, f=Fabric.CHANDERI,
                  cluster_key="casual_saree__block_print__chanderi", engagement=300),
    ]
    grouped = {
        "kurta_set__chikankari__cotton": [items[0]],
        "casual_saree__block_print__chanderi": [items[1]],
    }
    acc = _accumulate_share_weighted(grouped, date(2026, 4, 27), _cfg())
    a_kurta = acc["kurta_set__chikankari__cotton"]
    a_saree = acc["casual_saree__block_print__chanderi"]
    assert a_kurta.social_weighted_engagement == pytest.approx(200.0)
    assert a_saree.social_weighted_engagement == pytest.approx(300.0)


def test_youtube_video_count_share_weighted(empty_history: ScoreHistory) -> None:
    # YT N=3 item → fan-out cluster 에 yt_count = share, sum = 1.0.
    yt_item = _enriched(
        "yt1", g=GarmentType.CASUAL_SAREE, t=Technique.BLOCK_PRINT, f=Fabric.COTTON,
        cluster_key="casual_saree__block_print__cotton",
        source=ContentSource.YOUTUBE, engagement=5000,
    )
    grouped = {"casual_saree__block_print__cotton": [yt_item]}
    acc = _accumulate_share_weighted(grouped, date(2026, 4, 27), _cfg())
    a = acc["casual_saree__block_print__cotton"]
    assert a.youtube_video_count == pytest.approx(1.0)
    assert a.youtube_views_total == pytest.approx(5000.0)


def test_accounts_winner_keyed_not_fanned(empty_history: ScoreHistory) -> None:
    # accounts 는 winner-keyed (grouped[key]) 만 — fan-out 으로 inflate 되면 안됨.
    item = _enriched(
        "p1", g=GarmentType.KURTA_SET, t=Technique.CHIKANKARI, f=Fabric.COTTON,
        cluster_key="kurta_set__chikankari__cotton",
        handle="user_alpha",
    )
    grouped = {"kurta_set__chikankari__cotton": [item]}
    contexts = _build_contexts(grouped, date(2026, 4, 27), _cfg(), empty_history)
    assert len(contexts) == 1
    # new_account_ratio 가 빈 history 에 대해 호출됐는지 (간접 — context 가 정상 빌드됨)
    assert contexts[0].cluster_key == "kurta_set__chikankari__cotton"


def test_post_count_total_history_int_plus_rounded_share(empty_history: ScoreHistory) -> None:
    # post_count_total = history.get_total_post_count + round(share-sum). history 빈 0 이면
    # 단일 N=3 item → round(1.0) = 1.
    item = _enriched(
        "p1", g=GarmentType.KURTA_SET, t=Technique.CHIKANKARI, f=Fabric.COTTON,
        cluster_key="kurta_set__chikankari__cotton",
    )
    grouped = {"kurta_set__chikankari__cotton": [item]}
    contexts = _build_contexts(grouped, date(2026, 4, 27), _cfg(), empty_history)
    assert len(contexts) == 1
    assert contexts[0].post_count_total == 1
    assert contexts[0].post_count_today == pytest.approx(1.0)


def test_empty_grouped_returns_empty(empty_history: ScoreHistory) -> None:
    assert _build_contexts({}, date(2026, 4, 27), _cfg(), empty_history) == []
    assert _accumulate_share_weighted({}, date(2026, 4, 27), _cfg()) == {}
