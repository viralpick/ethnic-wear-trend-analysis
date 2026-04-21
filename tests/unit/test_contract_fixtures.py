"""sample_data/contract_fixtures/ 의 각 JSON 이 해당 contract 로 통과하는지 검증.

Contract 가 도메인 스펙과 정합하는지 확인하는 회귀 테스트. 새 속성 값이나 필드를 바꿀 때
여기 fixture 가 먼저 깨지면서 경고를 준다.
"""
from __future__ import annotations

from pathlib import Path

from contracts.common import CONTRACT_VERSION, DataMaturity, GarmentType
from contracts.enriched import EnrichedContentItem
from contracts.normalized import NormalizedContentItem
from contracts.output import TrendClusterSummary
from contracts.raw import RawInstagramPost, RawYouTubeVideo

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "sample_data" / "contract_fixtures"


def test_raw_instagram_fixture_validates() -> None:
    post = RawInstagramPost.model_validate_json(
        (FIXTURE_DIR / "raw_instagram.json").read_bytes()
    )

    assert post.post_id == "ig_fixture_chikankari_1"
    assert "#chikankari" in post.hashtags


def test_raw_youtube_fixture_validates() -> None:
    video = RawYouTubeVideo.model_validate_json(
        (FIXTURE_DIR / "raw_youtube.json").read_bytes()
    )

    assert video.channel == "@jhanvibhatia"
    assert video.like_count > 0


def test_normalized_fixture_validates() -> None:
    item = NormalizedContentItem.model_validate_json(
        (FIXTURE_DIR / "normalized.json").read_bytes()
    )

    assert item.source.value == "instagram"
    assert item.engagement_raw == 59580


def test_enriched_fixture_has_all_eight_attrs_and_cluster() -> None:
    item = EnrichedContentItem.model_validate_json(
        (FIXTURE_DIR / "enriched.json").read_bytes()
    )

    # 8 속성 + embellishment_intensity 전부 채워진 상태여야 한다.
    assert item.garment_type == GarmentType.KURTA_SET
    assert item.fabric is not None
    assert item.technique is not None
    assert item.embellishment_intensity is not None
    assert item.color is not None
    assert item.silhouette is not None
    assert item.occasion is not None
    assert item.styling_combo is not None
    assert item.brand is not None

    assert item.trend_cluster_key == "kurta_set__chikankari__cotton"

    # classification_method_per_attribute 가 rule/llm/vlm 을 섞어 채운다.
    methods = set(item.classification_method_per_attribute.values())
    assert {m.value for m in methods} == {"rule", "llm", "vlm"}


def test_trend_cluster_summary_fixture_bootstrap() -> None:
    summary = TrendClusterSummary.model_validate_json(
        (FIXTURE_DIR / "trend_cluster_summary.json").read_bytes()
    )

    assert summary.schema_version == CONTRACT_VERSION
    assert summary.cluster_key == "kurta_set__chikankari__cotton"
    assert summary.data_maturity == DataMaturity.BOOTSTRAP
    assert summary.daily_direction.value == "flat"
    assert summary.weekly_direction.value == "flat"
