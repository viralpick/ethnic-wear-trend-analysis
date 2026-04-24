"""Drill-down / 스코어보드용 output contract.

TrendClusterSummary 가 대시보드의 데이터 소스. schema_version 필드로 consumer 가 깨지는
변경을 탐지할 수 있게 한다 (backward-compat shim 은 v2 에서 검토 — v1 은 단순 플래그).
"""
from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict

from contracts.common import (
    CONTRACT_VERSION,
    DataMaturity,
    Direction,
    DistributionMap,
    LifecycleStage,
    PaletteCluster,
)


class ScoreBreakdown(BaseModel):
    """
    purpose: 종합 스코어 0~100 의 4개 팩터 분해 (spec §9.1~§9.2)
    stage: output
    ownership: analysis-owned
    stability: locked (가중치는 configs/local.yaml, 필드는 고정)
    """
    model_config = ConfigDict(frozen=True)

    social: float      # 0~40
    youtube: float     # 0~25
    cultural: float    # 0~15
    momentum: float    # 0~20


class DrilldownPayload(BaseModel):
    """
    purpose: 트렌드 상세 화면의 증거 묶음 (spec §2 화면 2, §5.4)
    stage: output
    ownership: analysis-owned
    stability: locked
    """
    model_config = ConfigDict(frozen=True)

    color_palette: list[PaletteCluster]
    silhouette_distribution: DistributionMap
    occasion_distribution: DistributionMap
    styling_distribution: DistributionMap
    top_posts: list[str]
    top_videos: list[str]
    top_influencers: list[str]


class TrendClusterSummary(BaseModel):
    """
    purpose: 클러스터 1개의 하루 상태 (스코어 + 방향 + 라이프사이클 + 드릴다운)
    stage: output
    ownership: analysis-owned
    stability: locked (schema_version 필드로 변경 탐지)
    """
    model_config = ConfigDict(frozen=True)

    schema_version: str = CONTRACT_VERSION

    # 식별
    cluster_key: str
    display_name: str
    date: date

    # 스코어 (spec §9)
    score: float                          # 0~100 총합
    score_breakdown: ScoreBreakdown

    # 방향성 + 라이프사이클 + 데이터 성숙도 (spec §9.3, §9.4)
    daily_direction: Direction
    weekly_direction: Direction
    daily_change_pct: float
    weekly_change_pct: float
    lifecycle_stage: LifecycleStage
    data_maturity: DataMaturity

    # Drill-down 묶음
    drilldown: DrilldownPayload

    # 집계 지표 (spec §8.2)
    post_count_total: int
    post_count_today: int
    avg_engagement_rate: float
    total_video_views: int


class UnknownAttributeSignal(BaseModel):
    """
    purpose: 매핑에 없는 새 해시태그 자동 감지 (spec §4.2)
    stage: output
    ownership: analysis-owned
    stability: locked
    """
    model_config = ConfigDict(frozen=True)

    tag: str                                # 예: "#bandhani"
    count_3day: int
    first_seen: date
    likely_category: str | None = None      # 예: "technique?"
    reviewed: bool = False
