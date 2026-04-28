"""StarRocks writer Protocol — Stream Load 적재의 추상 인터페이스.

실 구현 (`StarRocksStreamLoadWriter`) 은 후속 step (7.5) 에서 추가. 본 모듈은 Protocol 정의만
포함 — vision extras 격리 패턴과 동일하게 core 코드는 Protocol 만 import, 실 HTTP 의존은
실 구현 내부에 격리.

테스트는 `FakeStarRocksWriter` (별도 모듈) 로 in-memory 적재 검증.

Stream Load 시맨틱:
- 1 HTTP POST = 1 atomic transaction (1 table 단위).
- multi-table 적재는 호출자가 4번 호출 — append-only 라 부분 실패 후 재실행 멱등.
- 응답 row 수 = StarRocks 의 NumberLoadedRows.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class StarRocksWriter(Protocol):
    """4 base 테이블 (`item`/`canonical_group`/`canonical_object`/`representative_weekly`) 적재.

    호출자가 row dict list 를 쌓아 한 번에 write_batch — Stream Load 1 HTTP POST 매핑.
    rows 의 dict key 는 spec §1.x 컬럼명과 동일.
    """

    def write_batch(self, table: str, rows: list[dict[str, Any]]) -> int:
        """`table` 에 rows 적재. 적재 성공 row 수 반환. 부분 실패 → raise."""
        ...
