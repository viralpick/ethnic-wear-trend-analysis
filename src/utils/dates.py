"""Date 유틸. rolling window 계산에 쓰인다 (spec §4.2)."""
from __future__ import annotations

from datetime import date, timedelta


def previous_n_calendar_days(reference: date, n: int) -> list[date]:
    """reference 포함 최근 n 일 (오래된 것부터). n=3, ref=4/21 → [4/19, 4/20, 4/21]."""
    if n <= 0:
        return []
    return [reference - timedelta(days=offset) for offset in reversed(range(n))]
