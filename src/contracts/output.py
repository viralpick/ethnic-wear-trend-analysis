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


class MomentumComponents(BaseModel):
    """Momentum sub-signal raw 값 — score_breakdown.momentum 의 분해 추적용 (B-2).

    M3.J + M3.G/H (2026-04-29): IG/YT 신규 entity 시그널을 분리해 breakdown 에서
    개별 추적. 합산은 ScoreBreakdown.momentum 필드 (weighted sum), 여기는 raw 값만.
    """
    model_config = ConfigDict(frozen=True)

    post_growth: float                  # (오늘 - 7일평균) / 7일평균
    hashtag_velocity: float             # 주간 hashtag 사용량 증가율
    new_ig_account_ratio: float         # 0~1, IG handle 신규 비율
    new_yt_channel_ratio: float         # 0~1, YT channel 신규 비율


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
    # B-2 (M3.G/H 후): momentum sub-signal raw 가시화. weighted sum 은 momentum 필드.
    momentum_components: MomentumComponents


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
    # Phase 2 (2026-04-30): cluster_key 가 g__f 로 단순화되면서 technique 은 cluster
    # 의 drilldown distribution 으로만 노출. share-weighted vote (item 의 technique
    # distribution × cluster share 합산 → 정규화). 빈 dict = 미기여.
    technique_distribution: DistributionMap = {}
    # 로직 C (2026-04-29) — log-scale 균등 분배 + threshold/top-N cap. 합=1.0 또는 빈 dict.
    # insertion order = share desc (JSON 직렬화 시 순서 보존).
    brand_distribution: DistributionMap = {}
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
    # daily_* 폐기 (2026-05-01) — weekly cadence only. spec §3.4 / pipeline_spec.
    weekly_direction: Direction
    weekly_change_pct: float
    lifecycle_stage: LifecycleStage
    data_maturity: DataMaturity

    # Drill-down 묶음
    drilldown: DrilldownPayload

    # 집계 지표 (spec §8.2)
    # 옵션 C (2026-04-29): share-weighted fan-out (β2/β3/β4) 으로 fractional. γ 의
    # ScoreHistory.post_count + ClusterScoringContext.post_count_today 와 단위 정합.
    post_count_total: float
    post_count_today: float
    avg_engagement_rate: float
    total_video_views: int


class UnknownAttributeSignal(BaseModel):
    """
    purpose: 매핑에 없는 새 해시태그 자동 감지 (spec §4.2 / §8.3 v2.2)
    stage: output
    ownership: analysis-owned
    stability: locked

    v2.2 (2026-05-01) emergence rule:
    - bucket key = post_date IST. weekly cadence 로 매 주 anchor 별 평가.
    - surface(tag) := baseline_window 부재 + spike_window K회 + ethnic_co_share R 통과.
    - 옛 `count_3day` (≥10 / 3일 단순 룰) 폐기 — `count_recent_window` 으로 대체.
    """
    model_config = ConfigDict(frozen=True)

    tag: str                                # 예: "#bandhani"
    week_start_date: date                   # weekly anchor 의 주 시작일 (월요일)
    count_recent_window: int                # spike window 안 등장 횟수
    first_seen: date
    likely_category: str | None = None      # 예: "technique?"
    reviewed: bool = False
