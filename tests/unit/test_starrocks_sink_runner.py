"""sink_runner pinning — Step 7.7 emit_to_starrocks 4-table 적재 invariant.

검증:
- 적재 순서: item → canonical_group → canonical_object → representative_weekly.
- empty enriched → 4 table 모두 0, no HTTP 호출.
- N=3 cluster 만 representative_weekly 적재 (build_contributions filter).
- representative row 의 score/lifecycle/weekly_direction/color_palette 가 summary 에서 채워짐.
- evidence_ig_post_ids / yt_video_ids = top_evidence_per_source 결과 source_post_id 매핑.
- distributions: silhouette/occasion/styling_combo 만 채움 (drilldown), garment/fabric/technique NULL.
- weekly_history.update_weekly + save 가 score_and_export 에서 끝났다고 가정 (separate path).
- trajectory: weekly_history get_trajectory_12w(target_date) 결과 길이 12.
"""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import pytest

from contracts.common import (
    ClassificationMethod,
    ColorFamily,
    ContentSource,
    DataMaturity,
    Direction,
    Fabric,
    GarmentType,
    LifecycleStage,
    Occasion,
    PaletteCluster,
    Silhouette,
    Technique,
)
from contracts.enriched import EnrichedContentItem
from contracts.normalized import NormalizedContentItem
from contracts.output import (
    DrilldownPayload,
    MomentumComponents,
    ScoreBreakdown,
    TrendClusterSummary,
)
from contracts.vision import (
    CanonicalOutfit,
    EthnicOutfit,
    OutfitMember,
)
from exporters.starrocks.fake_writer import FakeStarRocksWriter
from exporters.starrocks.row_builder import representative_id
from exporters.starrocks.sink_runner import (
    GROUP_TABLE,
    ITEM_TABLE,
    OBJECT_TABLE,
    REPRESENTATIVE_TABLE,
    emit_to_starrocks,
)


_FIXED_DATE = date(2026, 4, 27)
_COMPUTED_AT = "2026-04-27 09:00:00"


# --------------------------------------------------------------------------- #
# 픽스처 헬퍼

def _normalized(
    post_id: str,
    *,
    source: ContentSource = ContentSource.INSTAGRAM,
    handle: str | None = "brand_x",
) -> NormalizedContentItem:
    return NormalizedContentItem(
        source=source,
        source_post_id=post_id,
        text_blob="cotton kurta in jaipur",
        hashtags=["#kurta"],
        image_urls=["https://example.com/img.jpg"],
        post_date=datetime(2026, 4, 26, 12, 0, 0),
        engagement_raw_count=4321,
        account_followers=1500,
        account_handle=handle,
    )


def _outfit(
    *,
    upper: str = "kurta_set",
    lower: str = "palazzo",
    fabric: str = "cotton",
    technique: str = "block_print",
    silhouette: Silhouette = Silhouette.A_LINE,
) -> EthnicOutfit:
    return EthnicOutfit(
        person_bbox=(0.1, 0.1, 0.5, 0.7),
        person_bbox_area_ratio=0.35,
        upper_garment_type=upper,
        lower_garment_type=lower,
        fabric=fabric,
        technique=technique,
        silhouette=silhouette,
        color_preset_picks_top3=[],
    )


def _canonical(index: int, outfit: EthnicOutfit) -> CanonicalOutfit:
    return CanonicalOutfit(
        canonical_index=index,
        representative=outfit,
        members=[
            OutfitMember(
                image_id="img_0",
                outfit_index=0,
                person_bbox=outfit.person_bbox,
                garment_type=outfit.upper_garment_type,
                fabric=outfit.fabric,
                technique=outfit.technique,
                silhouette=outfit.silhouette,
            )
        ],
        palette=[],
    )


def _enriched(
    post_id: str,
    *,
    cluster_key: str = "kurta_set__cotton",
    garment_type: GarmentType | None = GarmentType.KURTA_SET,
    fabric: Fabric | None = Fabric.COTTON,
    technique: Technique | None = Technique.BLOCK_PRINT,
    source: ContentSource = ContentSource.INSTAGRAM,
) -> EnrichedContentItem:
    canonical = _canonical(0, _outfit())
    method_map = {
        "garment_type": ClassificationMethod.RULE,
        "fabric": ClassificationMethod.RULE,
        "technique": ClassificationMethod.LLM,
    }
    return EnrichedContentItem(
        normalized=_normalized(post_id, source=source),
        garment_type=garment_type,
        fabric=fabric,
        technique=technique,
        occasion=Occasion.CASUAL,
        canonicals=[canonical],
        post_palette=[
            PaletteCluster(hex="#aabbcc", share=1.0, family=ColorFamily.NEUTRAL),
        ],
        classification_method_per_attribute=method_map,
        trend_cluster_key=cluster_key,
    )


def _summary(
    cluster_key: str = "kurta_set__cotton",
    *,
    score: float = 80.0,
) -> TrendClusterSummary:
    return TrendClusterSummary(
        cluster_key=cluster_key,
        display_name="kurta set / block print / cotton",
        date=_FIXED_DATE,
        score=score,
        score_breakdown=ScoreBreakdown(
            social=30.0, youtube=20.0, cultural=10.0, momentum=20.0,
            momentum_components=MomentumComponents(
                post_growth=0.5, hashtag_velocity=0.2,
                new_ig_account_ratio=0.1, new_yt_channel_ratio=0.0,
            ),
        ),
        daily_direction=Direction.UP,
        weekly_direction=Direction.UP,
        daily_change_pct=10.0,
        weekly_change_pct=15.0,
        lifecycle_stage=LifecycleStage.GROWTH,
        data_maturity=DataMaturity.FULL,
        drilldown=DrilldownPayload(
            color_palette=[
                PaletteCluster(hex="#aabbcc", share=0.7, family=ColorFamily.NEUTRAL),
                PaletteCluster(hex="#112233", share=0.3, family=ColorFamily.EARTH),
            ],
            silhouette_distribution={"a_line": 0.6, "straight": 0.4},
            occasion_distribution={"casual": 1.0},
            styling_distribution={},
            top_posts=["p1"],
            top_videos=[],
            top_influencers=["brand_x"],
        ),
        post_count_total=10,
        post_count_today=2,
        avg_engagement_rate=12.5,
        total_video_views=0,
    )


@pytest.fixture
def empty_weekly_history(tmp_path: Path) -> Path:
    """비어 있는 weekly history JSON. trajectory 는 12개 모두 0.0."""
    path = tmp_path / "score_history_weekly.json"
    return path  # 파일 없음 → WeeklyScoreHistory 가 빈 dict 로 시작


@pytest.fixture
def populated_weekly_history(tmp_path: Path) -> Path:
    """target_date 가 속한 주에 score 가 있는 weekly history."""
    from scoring.score_history_weekly import (
        WeeklyScoreHistory,
        iso_week_key,
        week_start_monday,
    )

    path = tmp_path / "score_history_weekly.json"
    history = WeeklyScoreHistory(path)
    history.update_weekly("kurta_set__cotton", _FIXED_DATE, 80.0, post_count=2)
    history.save()
    # sanity
    raw = json.loads(path.read_text())
    assert iso_week_key(_FIXED_DATE) in raw["kurta_set__cotton"]
    return path


# --------------------------------------------------------------------------- #
# emit_to_starrocks

def test_emit_empty_enriched_returns_zero_for_all_tables(empty_weekly_history) -> None:
    writer = FakeStarRocksWriter()
    counts = emit_to_starrocks(
        enriched=[],
        summaries=[],
        target_date=_FIXED_DATE,
        writer=writer,
        weekly_history_path=empty_weekly_history,
    )
    assert counts == {
        ITEM_TABLE: 0, GROUP_TABLE: 0, OBJECT_TABLE: 0, REPRESENTATIVE_TABLE: 0,
    }
    # write_batch 호출 자체가 없어야 함 (HTTP 비용 회피).
    assert writer.call_count == 0
    assert writer.total_rows() == 0


def test_emit_writes_four_tables_in_order(populated_weekly_history) -> None:
    writer = FakeStarRocksWriter()
    enriched = [_enriched("p1"), _enriched("p2")]
    summary = _summary()

    counts = emit_to_starrocks(
        enriched=enriched,
        summaries=[summary],
        target_date=_FIXED_DATE,
        writer=writer,
        weekly_history_path=populated_weekly_history,
        computed_at=_COMPUTED_AT,
    )

    # 4 table 모두 row 가 있어야 함.
    assert counts[ITEM_TABLE] == 2
    assert counts[GROUP_TABLE] == 2  # canonical 1개씩 * 2 post
    assert counts[OBJECT_TABLE] == 2  # member 1개씩 * 2 post
    assert counts[REPRESENTATIVE_TABLE] == 1  # cluster 1개

    # 적재 순서 검증 — write_batch 4 회, table 순서 동결 (FK 없지만 의미상 의존).
    assert writer.call_count == 4
    assert writer.call_order == [
        ITEM_TABLE, GROUP_TABLE, OBJECT_TABLE, REPRESENTATIVE_TABLE,
    ]


def test_emit_representative_row_pulls_from_summary(populated_weekly_history) -> None:
    writer = FakeStarRocksWriter()
    enriched = [_enriched("p1")]
    summary = _summary(score=88.5)

    emit_to_starrocks(
        enriched=enriched,
        summaries=[summary],
        target_date=_FIXED_DATE,
        writer=writer,
        weekly_history_path=populated_weekly_history,
        computed_at=_COMPUTED_AT,
    )

    rep_rows = writer.batches[REPRESENTATIVE_TABLE]
    assert len(rep_rows) == 1
    row = rep_rows[0]

    assert row["representative_key"] == "kurta_set__cotton"
    assert row["representative_id"] == representative_id("kurta_set__cotton")
    assert row["score_total"] == 88.5
    assert row["lifecycle_stage"] == LifecycleStage.GROWTH.value
    assert row["weekly_direction"] == Direction.UP.value
    assert row["weekly_change_pct"] == 15.0
    assert row["display_name"] == "kurta set / block print / cotton"
    # score_breakdown 4 키 unwrap.
    assert row["score_breakdown"] == {
        "social": 30.0, "youtube": 20.0, "cultural": 10.0, "momentum": 20.0,
    }
    # color_palette = drilldown 그대로 (cluster-level palette = representative palette).
    assert row["color_palette"] is not None
    assert len(row["color_palette"]) == 2


def test_emit_writes_effective_item_count_pin(populated_weekly_history) -> None:
    """β1 (2026-04-28): batch 분모가 모든 representative row 에 동일 값 주입.

    enriched 2 post 모두 G/T/F 결정 (N=3 → weight 1.0 each) → batch 분모 = 2.0.
    """
    writer = FakeStarRocksWriter()
    enriched = [_enriched("p1"), _enriched("p2")]

    emit_to_starrocks(
        enriched=enriched,
        summaries=[_summary()],
        target_date=_FIXED_DATE,
        writer=writer,
        weekly_history_path=populated_weekly_history,
        computed_at=_COMPUTED_AT,
    )

    rep_rows = writer.batches[REPRESENTATIVE_TABLE]
    assert all(row["effective_item_count"] == 2.0 for row in rep_rows)


def test_emit_effective_item_count_partial_n_proportional(populated_weekly_history) -> None:
    """N<2 item 은 multiplier-scaled 비율로만 분모 기여 (1=0.4 / 2=1.0)."""
    writer = FakeStarRocksWriter()
    enriched = [
        _enriched("p1"),  # G/F resolved → 1.0
        _enriched("p2", fabric=None),  # garment only → 0.4
        _enriched("p3", fabric=None, technique=None),  # garment only → 0.4
    ]

    emit_to_starrocks(
        enriched=enriched,
        summaries=[_summary()],
        target_date=_FIXED_DATE,
        writer=writer,
        weekly_history_path=populated_weekly_history,
        computed_at=_COMPUTED_AT,
    )

    rep_rows = writer.batches[REPRESENTATIVE_TABLE]
    assert all(row["effective_item_count"] == pytest.approx(1.8) for row in rep_rows)


def test_emit_distribution_keys_pin(populated_weekly_history) -> None:
    """6 key 중 silhouette/occasion 채움, styling_combo 빈 dict 도 NULL,
    garment/fabric/technique 는 항상 NULL (representative 단일값이라 redundant)."""
    writer = FakeStarRocksWriter()
    emit_to_starrocks(
        enriched=[_enriched("p1")],
        summaries=[_summary()],
        target_date=_FIXED_DATE,
        writer=writer,
        weekly_history_path=populated_weekly_history,
        computed_at=_COMPUTED_AT,
    )
    row = writer.batches[REPRESENTATIVE_TABLE][0]
    assert row["silhouette_distribution"] == {"a_line": 0.6, "straight": 0.4}
    assert row["occasion_distribution"] == {"casual": 1.0}
    # 빈 dict → NULL 적재 (Stream Load JSON null).
    assert row["styling_combo_distribution"] is None
    # representative 단위 단일값이라 NULL.
    assert row["garment_type_distribution"] is None
    assert row["fabric_distribution"] is None
    assert row["technique_distribution"] is None


def test_emit_evidence_resolves_source_post_id(populated_weekly_history) -> None:
    """evidence_ig_post_ids / yt_video_ids 가 item_id 가 아닌 source_post_id 로 채워짐."""
    writer = FakeStarRocksWriter()
    enriched = [
        _enriched("igpost1", source=ContentSource.INSTAGRAM),
        _enriched("igpost2", source=ContentSource.INSTAGRAM),
    ]
    emit_to_starrocks(
        enriched=enriched,
        summaries=[_summary()],
        target_date=_FIXED_DATE,
        writer=writer,
        weekly_history_path=populated_weekly_history,
        computed_at=_COMPUTED_AT,
    )
    row = writer.batches[REPRESENTATIVE_TABLE][0]
    # IG 2개 → top-4 안에 둘 다 들어감 (사실 2개 뿐).
    assert set(row["evidence_ig_post_ids"]) == {"igpost1", "igpost2"}
    # YT 없음.
    assert row["evidence_yt_video_ids"] == []


def test_emit_partial_cluster_representative_partial_activation(populated_weekly_history) -> None:
    """Phase partial(g) 활성화 (2026-04-28) — N<3 도 representative emit.

    text + vision 양쪽 technique=None → ItemDistribution 의 technique dict 가 빈
    상태 (N=2). 이전엔 representative emit 안 됐으나, partial 활성화 후 unknown
    technique 누락이어도 G/F exact representative (`kurta_set__cotton`) emit 됨.
    """
    writer = FakeStarRocksWriter()
    # technique=None + canonical 의 representative.technique=None 양쪽 모두 비움.
    canonical_no_tech = CanonicalOutfit(
        canonical_index=0,
        representative=_outfit(technique=None),  # vision 쪽 technique 도 None
        members=[
            OutfitMember(
                image_id="img_0",
                outfit_index=0,
                person_bbox=(0.1, 0.1, 0.5, 0.7),
                garment_type="kurta_set",
                fabric="cotton",
                technique=None,
                silhouette=Silhouette.A_LINE,
            )
        ],
        palette=[],
    )
    enriched_partial = EnrichedContentItem(
        normalized=_normalized("p1"),
        garment_type=GarmentType.KURTA_SET,
        fabric=Fabric.COTTON,
        technique=None,  # text 쪽 technique 도 None
        occasion=Occasion.CASUAL,
        canonicals=[canonical_no_tech],
        post_palette=[],
        classification_method_per_attribute={
            "garment_type": ClassificationMethod.RULE,
            "fabric": ClassificationMethod.RULE,
        },
        trend_cluster_key="kurta_set__cotton",
    )
    counts = emit_to_starrocks(
        enriched=[enriched_partial],
        summaries=[],
        target_date=_FIXED_DATE,
        writer=writer,
        weekly_history_path=populated_weekly_history,
        computed_at=_COMPUTED_AT,
    )
    # item / group / object 적재 + partial cluster representative 1 row 적재.
    assert counts[ITEM_TABLE] == 1
    assert counts[REPRESENTATIVE_TABLE] == 1
    rep_row = writer.batches[REPRESENTATIVE_TABLE][0]
    assert rep_row["representative_key"] == "kurta_set__cotton"


def test_emit_trajectory_length_is_12(populated_weekly_history) -> None:
    writer = FakeStarRocksWriter()
    emit_to_starrocks(
        enriched=[_enriched("p1")],
        summaries=[_summary()],
        target_date=_FIXED_DATE,
        writer=writer,
        weekly_history_path=populated_weekly_history,
        computed_at=_COMPUTED_AT,
    )
    row = writer.batches[REPRESENTATIVE_TABLE][0]
    assert len(row["trajectory"]) == 12
    # 마지막 (= 현재 주) 는 populated 에서 80.0.
    assert row["trajectory"][-1] == 80.0
    # 그 외 11 주는 0 (history 비어 있음).
    assert all(v == 0.0 for v in row["trajectory"][:-1])


def test_emit_week_start_date_is_monday(populated_weekly_history) -> None:
    writer = FakeStarRocksWriter()
    emit_to_starrocks(
        enriched=[_enriched("p1")],
        summaries=[_summary()],
        target_date=_FIXED_DATE,  # 2026-04-27 is Monday
        writer=writer,
        weekly_history_path=populated_weekly_history,
        computed_at=_COMPUTED_AT,
    )
    row = writer.batches[REPRESENTATIVE_TABLE][0]
    # 2026-04-27 자체가 ISO Monday → week_start = same date.
    assert row["week_start_date"] == "2026-04-27"


def test_emit_computed_at_uniform_across_tables(populated_weekly_history) -> None:
    writer = FakeStarRocksWriter()
    emit_to_starrocks(
        enriched=[_enriched("p1")],
        summaries=[_summary()],
        target_date=_FIXED_DATE,
        writer=writer,
        weekly_history_path=populated_weekly_history,
        computed_at=_COMPUTED_AT,
    )
    item_row = writer.batches[ITEM_TABLE][0]
    group_row = writer.batches[GROUP_TABLE][0]
    object_row = writer.batches[OBJECT_TABLE][0]
    rep_row = writer.batches[REPRESENTATIVE_TABLE][0]
    # 같은 run 의 4 table 모두 동일 computed_at.
    assert item_row["computed_at"] == _COMPUTED_AT
    assert group_row["computed_at"] == _COMPUTED_AT
    assert object_row["computed_at"] == _COMPUTED_AT
    assert rep_row["computed_at"] == _COMPUTED_AT


def test_emit_handles_summary_missing_for_cluster(populated_weekly_history) -> None:
    """summary 가 없는 cluster → score/lifecycle 등 NULL, color_palette 빈 list (None)."""
    writer = FakeStarRocksWriter()
    emit_to_starrocks(
        enriched=[_enriched("p1")],
        summaries=[],  # summary 비어 있음
        target_date=_FIXED_DATE,
        writer=writer,
        weekly_history_path=populated_weekly_history,
        computed_at=_COMPUTED_AT,
    )
    row = writer.batches[REPRESENTATIVE_TABLE][0]
    assert row["score_total"] is None
    assert row["score_breakdown"] is None
    assert row["lifecycle_stage"] is None
    assert row["weekly_direction"] is None
    assert row["weekly_change_pct"] is None
    assert row["display_name"] is None
    # 빈 palette → NULL.
    assert row["color_palette"] is None
    # distribution 6 키 모두 NULL.
    assert row["silhouette_distribution"] is None
    assert row["occasion_distribution"] is None
    assert row["styling_combo_distribution"] is None
