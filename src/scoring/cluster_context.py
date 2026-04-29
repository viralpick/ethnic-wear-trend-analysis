"""ClusterScoringContext — 스코어링에 필요한 raw 집계값만 담은 작은 타입드 구조.

설계 목적:
- score_*.compute(ctx, cfg) 는 pure 함수가 되어야 한다. 그러려면 입력이 자기 완결적이어야 한다.
- EnrichedContentItem 리스트 전체를 넘기지 않는다 — 사전 집계된 숫자 몇 개만 넘긴다.
- 이 구조를 보면 "이 sub-score 가 어떤 raw 값을 먹고 어떻게 설명 가능한 점수를 내는지"가
  한눈에 보인다. 설명력(spec §9 mandate)의 핵심.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ClusterScoringContext(BaseModel):
    """
    purpose: 한 클러스터 × 하루에 대한 스코어링 입력 aggregate
    stage: scoring
    ownership: analysis-owned
    stability: locked (Step 3-B 동결 — 필드 추가는 migration 로직 필요)
    """
    model_config = ConfigDict(frozen=True)

    cluster_key: str

    # --- Social (spec §9.2 Social) ---
    # 각 포스트 weighted_engagement = (likes + comments*2 + saves*3) × influencer_weight 합.
    social_weighted_engagement: float

    # --- YouTube (spec §9.2 YouTube) ---
    youtube_video_count: float            # V (Phase β2: share-weighted fan-out 으로 fractional)
    youtube_views_total: float            # 최근 window_days 일 조회수 합
    youtube_view_growth: float            # (this_week - last_week) / last_week. 0 안전처리.

    # --- Cultural (spec §9.2 Cultural + §9.5) ---
    cultural_festival_match: float        # 0~N. festival 윈도우 × 태그 매치 점수.
    cultural_bollywood_presence: float    # bollywood_decode 소스 post 수 (or 0/1 시그널).

    # --- Momentum (spec §9.2 Momentum) ---
    momentum_post_growth: float                # (오늘 - 7일평균) / 7일평균. denom 0 이면 0.
    momentum_hashtag_velocity: float           # 주간 증가율.
    # B-2 (M3.G/H 후): IG/YT 신규 entity 시그널 분리 (sub-signal).
    momentum_new_ig_account_ratio: float       # 0~1, IG handle 신규 비율.
    momentum_new_yt_channel_ratio: float       # 0~1, YT channel 신규 비율.

    # --- 라이프사이클 / 방향성 / maturity 판정용 ---
    post_count_total: float               # Phase γ: history schema 마이그 후 float (β2 mixed precision round 제거)
    post_count_today: float               # Phase β2: share-weighted fan-out 으로 fractional
    avg_engagement_rate: float
    # 전일 / 지난 주 같은 비교 기준은 orchestrator 가 별도 dict 로 넘긴다.
