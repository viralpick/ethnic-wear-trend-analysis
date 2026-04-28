"""In-memory `StarRocksWriter` 구현 — 테스트 + dry-run 용.

실 Stream Load 호출 없이 적재된 row 를 메모리에 누적. 테스트에서 `writer.batches[table]`
로 적재 결과 검증, pipeline CLI `--sink starrocks --dry-run` 모드에서도 사용 가능.

원칙:
- write_batch 호출 횟수 / table 별 row dict 그대로 보존 — assertion 친화.
- 같은 table 에 여러 번 호출되면 누적 (extend). spec append-only 시맨틱과 동일.
- 등록 안 된 table 도 자유롭게 받음 (테스트 mock 자유도 우선) — 정합성 검증은 별도.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any


class FakeStarRocksWriter:
    """`StarRocksWriter` Protocol in-memory impl."""

    def __init__(self) -> None:
        self.batches: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.call_count: int = 0
        self.call_order: list[str] = []

    def write_batch(self, table: str, rows: list[dict[str, Any]]) -> int:
        self.call_count += 1
        self.call_order.append(table)
        self.batches[table].extend(rows)
        return len(rows)

    def total_rows(self, table: str | None = None) -> int:
        """assertion 헬퍼. table=None 이면 전체 합산."""
        if table is not None:
            return len(self.batches.get(table, []))
        return sum(len(v) for v in self.batches.values())

    def reset(self) -> None:
        self.batches.clear()
        self.call_count = 0
        self.call_order.clear()
