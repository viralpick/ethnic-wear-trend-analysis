"""Backend transport payload (spec §2 — 백엔드 팀 소비).

의도적으로 thin. 별도 view model 을 만들지 않는다 — TrendClusterSummary 를 그대로 감싸고
contract version 만 wrapper 에 올린다.

# OPEN QUESTION: backend team to confirm field names by 2026-04-24
"""
from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict

from contracts.common import CONTRACT_VERSION
from contracts.output import TrendClusterSummary


class BackendScoreboardPayload(BaseModel):
    """
    purpose: 백엔드 API 가 프론트에 내려주는 전송 shape (spec §2 화면 1)
    stage: transport
    ownership: analysis-owned (contract 확정 전까지)
    stability: experimental (4/24 백엔드 팀 리뷰 전까지 placeholder)
    """
    model_config = ConfigDict(frozen=True)

    schema_version: str = CONTRACT_VERSION
    target_date: date
    clusters: list[TrendClusterSummary]


def build_payload(
    target_date: date, summaries: list[TrendClusterSummary]
) -> BackendScoreboardPayload:
    """summaries → BackendScoreboardPayload. 재정렬 / 필드 가공 없음 (pass-through)."""
    return BackendScoreboardPayload(target_date=target_date, clusters=list(summaries))
