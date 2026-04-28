"""--sink dry_run 출력 헬퍼.

`emit_to_starrocks` 가 FakeStarRocksWriter 로 적재한 결과를 사람이 읽을 수 있는 형태로
요약. row 갯수 + 적재 순서 + 각 table 첫 1 row sample (truncated JSON).

Stream Load 호출 안함. CLI `--sink dry_run` 으로 production row 형태 + null 패턴 검증용.
"""
from __future__ import annotations

import json
from typing import Any

from exporters.starrocks.fake_writer import FakeStarRocksWriter
from utils.logging import get_logger

logger = get_logger(__name__)

_SAMPLE_VALUE_TRUNCATE = 200


def _truncate_value(value: Any) -> Any:
    """긴 list/dict 는 JSON repr 자르기 — 로그 폭발 방지."""
    if isinstance(value, (list, dict)):
        rendered = json.dumps(value, ensure_ascii=False, default=str)
        if len(rendered) <= _SAMPLE_VALUE_TRUNCATE:
            return value
        return rendered[:_SAMPLE_VALUE_TRUNCATE] + "...(truncated)"
    return value


def _format_sample(row: dict[str, Any]) -> str:
    truncated = {k: _truncate_value(v) for k, v in row.items()}
    return json.dumps(truncated, ensure_ascii=False, indent=2, default=str)


def log_dry_run_summary(writer: FakeStarRocksWriter, counts: dict[str, int]) -> None:
    """dry_run sink 결과를 logger 에 INFO 로 출력.

    포맷:
      [dry_run] table=item rows=10
      [dry_run] table=canonical_group rows=10
      ...
      [dry_run] call_order: [item, canonical_group, canonical_object, representative_weekly]
      [dry_run] sample item[0]: { ... pretty json ... }
      [dry_run] sample canonical_group[0]: { ... }
      ...
    """
    for table, n in counts.items():
        logger.info("[dry_run] table=%s rows=%d", table, n)
    logger.info("[dry_run] call_order: %s", writer.call_order)

    for table in counts:
        rows = writer.batches.get(table, [])
        if not rows:
            logger.info("[dry_run] sample %s: (empty)", table)
            continue
        logger.info(
            "[dry_run] sample %s[0]:\n%s", table, _format_sample(rows[0])
        )
