"""StarRocks ad-hoc migration runner — `migrations/*.sql` 1회 실행용.

`init_starrocks_schema.py` 와 동일한 .env / pymysql 접속 패턴. light schema change
(ALTER TABLE ADD/DROP COLUMN) 같은 1회 마이그레이션을 mysql client 설치 없이 실행한다.

사용법:
    uv run python scripts/run_starrocks_migration.py \\
        src/exporters/starrocks/migrations/001_brand_1_to_n_2026_04_28.sql \\
        --table item

동작:
    1. .env 로드 + STARROCKS_RESULT_DATABASE 접속
    2. (--table 지정 시) 실행 전 DESC <table> 출력
    3. SQL 파일 split + 순차 실행 (autocommit)
    4. (--table 지정 시) 실행 후 DESC <table> 출력 (전후 비교)

idempotent 보장 X — 마이그레이션 SQL 자체에서 책임 (DROP COLUMN 은 not-exists 시 에러).
재실행 안전성을 원하면 --dry-run 으로 먼저 split 만 확인.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import pymysql
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def _connect(database: str) -> pymysql.connections.Connection:
    """DDL 용 — drift 방지 helper 위임 (database 명시 필수)."""
    from loaders.starrocks_connect import connect_ddl
    return connect_ddl(database=database)


def _split_statements(sql_text: str) -> list[str]:
    cleaned: list[str] = []
    for line in sql_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("--"):
            continue
        cleaned.append(line)
    joined = "\n".join(cleaned)
    return [s.strip() for s in joined.split(";") if s.strip()]


def _desc_table(conn: pymysql.connections.Connection, table: str, label: str) -> None:
    with conn.cursor() as cur:
        cur.execute(f"DESC {table}")
        rows = cur.fetchall()
    logger.info("DESC %s [%s] — %d columns:", table, label, len(rows))
    for row in rows:
        logger.info("  %s", row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sql_path", type=Path, help="실행할 .sql 파일 경로")
    parser.add_argument(
        "--table",
        default=None,
        help="실행 전후 DESC 으로 비교할 테이블명 (선택, 안전 검증용)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="SQL split 결과만 출력 (접속 안함)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv()

    sql_path: Path = args.sql_path
    if not sql_path.is_file():
        logger.error("SQL file not found: %s", sql_path)
        return 1

    sql_text = sql_path.read_text(encoding="utf-8")
    statements = _split_statements(sql_text)
    logger.info("loaded %s — %d statements", sql_path.name, len(statements))

    if args.dry_run:
        for i, stmt in enumerate(statements):
            logger.info("[%d] %s", i, stmt.replace("\n", " "))
        return 0

    target_db = os.environ.get("STARROCKS_RESULT_DATABASE", "ethnic_result")
    logger.info("connect db=%s host=%s", target_db, os.environ["STARROCKS_HOST"])

    conn = _connect(database=target_db)
    try:
        if args.table:
            _desc_table(conn, args.table, "BEFORE")
        with conn.cursor() as cur:
            for i, stmt in enumerate(statements):
                logger.info("[%d] EXEC %s", i, stmt.replace("\n", " ")[:160])
                cur.execute(stmt)
        if args.table:
            _desc_table(conn, args.table, "AFTER")
    finally:
        conn.close()

    logger.info("migration applied: %s", sql_path.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
