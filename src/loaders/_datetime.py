"""Loader 공용 datetime 파싱 helper — TSV/StarRocks 양쪽에서 동일 포맷 처리.

raw DB 또는 TSV 가 발송하는 timestamp 포맷이 reader 간에 동일하므로 한 곳에서 관리.
포맷 / TZ 가정이 달라지면 호출처 모두 영향 — 변경 시 reader 양쪽 통합 테스트 필수.
"""
from __future__ import annotations

from datetime import datetime, timezone


def parse_iso_z(raw: str) -> datetime:
    """ISO8601 with trailing Z (예: 2026-04-19T23:49:20Z) → tz-aware UTC datetime."""
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


def parse_db_timestamp(raw: str) -> datetime:
    """naive timestamp (예: 2026-04-20 23:15:01) → tz-aware UTC datetime.

    DB / TSV 가 wallclock 만 발송 (TZ 정보 없음). UTC 가정은 4/24 sync 합의.
    crawler 가 KST/IST 로 발송하기 시작하면 이 함수가 silent drift 의 시작점.
    """
    return datetime.fromisoformat(raw).replace(tzinfo=timezone.utc)


def parse_yyyymmdd(raw: str) -> datetime:
    """YYYYMMDD (예: 20260304) → tz-aware UTC datetime (00:00:00)."""
    return datetime.strptime(raw, "%Y%m%d").replace(tzinfo=timezone.utc)
