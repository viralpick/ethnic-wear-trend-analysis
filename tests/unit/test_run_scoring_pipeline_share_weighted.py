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
from aggregation.build_cluster_summary import group_by_cluster
from contracts.enriched import EnrichedContentItem
from contracts.normalized import NormalizedContentItem
from contracts.vision import CanonicalOutfit, EthnicOutfit, OutfitMember
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
    canonicals: list[CanonicalOutfit] | None = None,
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
        canonicals=canonicals or [],
        classification_method_per_attribute=methods,
        trend_cluster_key=cluster_key,
    )


def _canonical_with_garment(
    upper: str,
    *,
    fabric: str = "cotton",
    technique: str = "chikankari",
    bbox: tuple[float, float, float, float] = (0.1, 0.1, 0.5, 0.7),
    area_ratio: float = 0.35,
) -> CanonicalOutfit:
    """multi-cluster fan-out 검증용 — text 와 다른 garment_type 의 canonical."""
    outfit = EthnicOutfit(
        person_bbox=bbox,
        person_bbox_area_ratio=area_ratio,
        upper_garment_type=upper,
        upper_is_ethnic=True,
        lower_garment_type="palazzo",
        lower_is_ethnic=True,
        dress_as_single=False,
        fabric=fabric,
        technique=technique,
        color_preset_picks_top3=[],
    )
    return CanonicalOutfit(
        canonical_index=0,
        representative=outfit,
        members=[OutfitMember(image_id="img_0", outfit_index=0, person_bbox=bbox)],
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
    grouped = group_by_cluster([item])
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
    # N=0 (G/T/F 모두 없음) → group_by_cluster 빈 dict → context [].
    # β4 후 "unclassified" placeholder 자체가 grouped 에 진입하지 않음 (assign_shares 빈 dict).
    item = _enriched("p1", g=None, t=None, f=None,
                     cluster_key="unclassified")
    grouped = group_by_cluster([item])
    assert grouped == {}
    acc = _accumulate_share_weighted(grouped, date(2026, 4, 27), _cfg())
    assert acc == {}
    contexts = _build_contexts(grouped, date(2026, 4, 27), _cfg(), empty_history)
    assert contexts == []


def test_n_eq_3_mass_preservation_single_winner(empty_history: ScoreHistory) -> None:
    # G/T/F 모두 단일값 1.0 → cross-product 1개 cluster, share=1.0.
    item = _enriched(
        "p1", g=GarmentType.KURTA_SET, t=Technique.CHIKANKARI, f=Fabric.COTTON,
        cluster_key="kurta_set__chikankari__cotton",
    )
    grouped = group_by_cluster([item])
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
    grouped = group_by_cluster(items)
    acc = _accumulate_share_weighted(grouped, date(2026, 4, 27), _cfg())
    total_mass = sum(a.post_count_today for a in acc.values())
    assert total_mass == pytest.approx(3.5)  # N=3 (×3, 1.0) + N=2 (×1, 0.5)


def test_share_fan_out_cross_product(empty_history: ScoreHistory) -> None:
    # 1 item 의 contribution 이 정확히 1 cluster 에 들어가는지 핀 (text RULE 단일값
    # fixture 라 cross-product 결과도 1 cluster).
    items = [
        _enriched("p1", g=GarmentType.KURTA_SET, t=Technique.CHIKANKARI, f=Fabric.COTTON,
                  cluster_key="kurta_set__chikankari__cotton", engagement=200),
        _enriched("p2", g=GarmentType.CASUAL_SAREE, t=Technique.BLOCK_PRINT, f=Fabric.CHANDERI,
                  cluster_key="casual_saree__block_print__chanderi", engagement=300),
    ]
    grouped = group_by_cluster(items)
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
    grouped = group_by_cluster([yt_item])
    acc = _accumulate_share_weighted(grouped, date(2026, 4, 27), _cfg())
    a = acc["casual_saree__block_print__cotton"]
    assert a.youtube_video_count == pytest.approx(1.0)
    assert a.youtube_views_total == pytest.approx(5000.0)


def test_accounts_in_cluster_includes_ig_handles(empty_history: ScoreHistory) -> None:
    # β4 (2026-04-28): accounts 는 cluster 안 IG account_handle (share>0 인 entry).
    # winner-keyed 시절과 달리 multi-fan-out 도 자연 cluster 별 분리 (multi-fan-out
    # 자체는 별도 multi_cluster_fan_out 핀이 검증).
    item = _enriched(
        "p1", g=GarmentType.KURTA_SET, t=Technique.CHIKANKARI, f=Fabric.COTTON,
        cluster_key="kurta_set__chikankari__cotton",
        handle="user_alpha",
    )
    grouped = group_by_cluster([item])
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
    grouped = group_by_cluster([item])
    contexts = _build_contexts(grouped, date(2026, 4, 27), _cfg(), empty_history)
    assert len(contexts) == 1
    assert contexts[0].post_count_total == 1
    assert contexts[0].post_count_today == pytest.approx(1.0)


def test_empty_grouped_returns_empty(empty_history: ScoreHistory) -> None:
    assert _build_contexts({}, date(2026, 4, 27), _cfg(), empty_history) == []
    assert _accumulate_share_weighted({}, date(2026, 4, 27), _cfg()) == {}


# --------------------------------------------------------------------------- #
# Multi-cluster fan-out — mass preservation (β4 signature 후 자연)
# --------------------------------------------------------------------------- #

def test_multi_cluster_fan_out_mass_preserved(empty_history: ScoreHistory) -> None:
    """1 item 이 multi-cluster 에 fan-out 되어도 per-item mass=1.0 보존.

    β4 signature (grouped entry = (item, share)) 에서는 outer loop 가 cluster 단위라
    같은 item 도 cluster 마다 자기 share 만큼만 1번씩 기여 — over-count 자연 차단.
    재현: text RULE garment + vision canonical 다른 upper_garment_type → G dist multi-key.
    """
    canonical = _canonical_with_garment(
        upper="casual_saree", fabric="cotton", technique="chikankari",
    )
    item = _enriched(
        "p_multi", g=GarmentType.KURTA_SET, t=Technique.CHIKANKARI, f=Fabric.COTTON,
        cluster_key="kurta_set__chikankari__cotton", engagement=100,
        canonicals=[canonical],
    )
    grouped = group_by_cluster([item])
    # multi-fan-out 확인 — cluster 2 개 등장.
    assert set(grouped.keys()) == {
        "kurta_set__chikankari__cotton", "casual_saree__chikankari__cotton",
    }
    acc = _accumulate_share_weighted(grouped, date(2026, 4, 27), _cfg())

    total_mass = sum(a.post_count_today for a in acc.values())
    assert total_mass == pytest.approx(1.0)
    total_engagement = sum(a.social_weighted_engagement for a in acc.values())
    assert total_engagement == pytest.approx(100.0)


def test_multi_cluster_fan_out_build_contexts_mass_consistent(
    empty_history: ScoreHistory,
) -> None:
    """`_build_contexts` 도 fan-out 후 post_count_today 합 = per-item mass."""
    canonical = _canonical_with_garment(upper="casual_saree")
    item = _enriched(
        "p_multi", g=GarmentType.KURTA_SET, t=Technique.CHIKANKARI, f=Fabric.COTTON,
        cluster_key="kurta_set__chikankari__cotton", engagement=100,
        canonicals=[canonical],
    )
    grouped = group_by_cluster([item])
    contexts = _build_contexts(grouped, date(2026, 4, 27), _cfg(), empty_history)

    by_key = {c.cluster_key: c for c in contexts}
    total_mass = sum(c.post_count_today for c in contexts)
    assert total_mass == pytest.approx(1.0)
    assert by_key["kurta_set__chikankari__cotton"].post_count_today > 0
    assert by_key["casual_saree__chikankari__cotton"].post_count_today > 0
