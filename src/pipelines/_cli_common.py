"""Pipeline CLI 공용 helper — entrypoint 간 boilerplate 통합.

`run_daily_pipeline` / `run_scoring_pipeline` 등 여러 main() 가 같은 date 파싱 / settings
fallback 로직을 별도 정의했던 것을 single source 로. 정책 변경 시 한 곳만 수정.
"""
from __future__ import annotations

from datetime import date, datetime


def parse_iso_date(raw: str) -> date:
    """`YYYY-MM-DD` 문자열 → `date`. argparse `--date` / `--start-date` / `--end-date` 공용."""
    return datetime.strptime(raw, "%Y-%m-%d").date()


def resolve_target_date(cli: str | None, settings_target: date | None) -> date:
    """CLI `--date` 우선, 없으면 settings.pipeline.target_date, 그것도 없으면 today.

    여러 entrypoint 공통 — `run_daily_pipeline` / `run_scoring_pipeline` 등.
    """
    if cli:
        return parse_iso_date(cli)
    return settings_target or date.today()


__all__ = ["parse_iso_date", "resolve_target_date"]
