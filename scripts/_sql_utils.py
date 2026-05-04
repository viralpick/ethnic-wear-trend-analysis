"""ad-hoc SQL DDL 실행 공용 헬퍼.

`init_starrocks_schema.py` ↔ `run_starrocks_migration.py` 가 동일 line-for-line
`_split_statements` 정의 — SQL parse 정책 변경 시 한쪽 누락 위험. single source 통합.
"""
from __future__ import annotations


def split_statements(sql_text: str) -> list[str]:
    """단순 `;` split (StarRocks DDL 은 PROCEDURE/TRIGGER 없어 안전).

    `--` line comment 와 빈 줄은 제거. 주석 안의 `;` 는 안 쓰므로 OK.
    P2 한계 (`feedback`): 문자열 리터럴 안의 `--` / `;` 는 처리 못 함 — DDL 에 등장
    안 함 (Comment '...' 안 ; 등). 향후 등장 시 sqlparse 도입 검토.
    """
    cleaned: list[str] = []
    for line in sql_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("--"):
            continue
        cleaned.append(line)
    joined = "\n".join(cleaned)
    return [s.strip() for s in joined.split(";") if s.strip()]


__all__ = ["split_statements"]
