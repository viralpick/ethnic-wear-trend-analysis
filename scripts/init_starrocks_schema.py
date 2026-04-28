"""ethnic_result DB + 4 base 테이블 + 4 latest view 1회 migration.

실행 전 준비:
  - VPN ON (Pritunl), .env 의 STARROCKS_* 채워짐
  - svc_india_ai_fashion_poc 계정에 ethnic_result DB CREATE/CREATE TABLE 권한 (2026-04-26 infra 응답)

동작:
  1. 대상 host 의 StarRocks 버전 출력 (sanity)
  2. CREATE DATABASE IF NOT EXISTS ethnic_result
  3. USE ethnic_result
  4. ddl/01..05 *.sql 순서대로 실행 (CREATE TABLE / VIEW IF NOT EXISTS 또는 OR REPLACE)
  5. 생성된 테이블/뷰 목록 출력

idempotent: 모두 IF NOT EXISTS / OR REPLACE 라 재실행해도 안전.
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

DDL_DIR = Path(__file__).resolve().parent.parent / "src" / "exporters" / "starrocks" / "ddl"
# write target — env STARROCKS_RESULT_DATABASE 로 명시. raw read DB 는 STARROCKS_RAW_DATABASE.
_TARGET_DB_DEFAULT = "ethnic_result"


def _connect(database: str | None = None) -> pymysql.connections.Connection:
    """database=None 이면 system 접속 (CREATE DATABASE 용)."""
    return pymysql.connect(
        host=os.environ["STARROCKS_HOST"],
        port=int(os.environ["STARROCKS_PORT"]),
        user=os.environ["STARROCKS_USER"],
        password=os.environ["STARROCKS_PASSWORD"],
        database=database or "",
        connect_timeout=10,
        autocommit=True,
    )


def _split_statements(sql_text: str) -> list[str]:
    """단순 ; split (StarRocks DDL 은 PROCEDURE/TRIGGER 없어 안전).

    `--` line comment 와 빈 줄은 제거. 주석 안의 ; 는 안 쓰므로 OK.
    """
    cleaned: list[str] = []
    for line in sql_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("--"):
            continue
        cleaned.append(line)
    joined = "\n".join(cleaned)
    return [s.strip() for s in joined.split(";") if s.strip()]


def _list_ddl_files() -> list[Path]:
    files = sorted(DDL_DIR.glob("*.sql"))
    if not files:
        raise FileNotFoundError(f"No DDL files in {DDL_DIR}")
    return files


def _print_version(conn: pymysql.connections.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute("SELECT current_version()")
        row = cur.fetchone()
        logger.info("StarRocks version: %s", row[0] if row else "?")


def _ensure_database(conn: pymysql.connections.Connection, target_db: str) -> None:
    """DB 가 이미 있으면 skip — infra 가 미리 만들어두는 운영 패턴 대응.

    POC 계정은 CATALOG-level CREATE DATABASE 권한이 없을 수 있어 IF NOT EXISTS 만으론 부족.
    먼저 SHOW DATABASES 로 존재 확인 후, 없을 때만 CREATE 시도.
    """
    with conn.cursor() as cur:
        cur.execute("SHOW DATABASES")
        existing = {r[0] for r in cur.fetchall()}
        if target_db in existing:
            logger.info("database already exists, skip CREATE: %s", target_db)
            return
        cur.execute(f"CREATE DATABASE {target_db}")
        logger.info("created database: %s", target_db)


def _apply_ddl_file(conn: pymysql.connections.Connection, path: Path) -> None:
    sql_text = path.read_text(encoding="utf-8")
    statements = _split_statements(sql_text)
    logger.info("apply %s (%d statements)", path.name, len(statements))
    with conn.cursor() as cur:
        for stmt in statements:
            cur.execute(stmt)


def _list_objects(conn: pymysql.connections.Connection, target_db: str) -> None:
    with conn.cursor() as cur:
        cur.execute(f"SHOW TABLES FROM {target_db}")
        rows = cur.fetchall()
    names = sorted(r[0] for r in rows)
    logger.info("objects in %s: %s", target_db, names)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="DDL 파일 split 결과만 출력 (StarRocks 접속 안함)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv()

    target_db = os.environ.get("STARROCKS_RESULT_DATABASE", _TARGET_DB_DEFAULT)
    raw_db = os.environ.get("STARROCKS_RAW_DATABASE", "png")
    logger.info("write target DB: %s (raw read DB: %s)", target_db, raw_db)

    files = _list_ddl_files()
    logger.info("DDL files: %s", [p.name for p in files])

    if args.dry_run:
        for path in files:
            statements = _split_statements(path.read_text(encoding="utf-8"))
            logger.info("== %s (%d statements) ==", path.name, len(statements))
            for i, stmt in enumerate(statements):
                logger.info("[%d] %s", i, stmt[:120].replace("\n", " "))
        return 0

    sys_conn = _connect(database=None)
    try:
        _print_version(sys_conn)
        _ensure_database(sys_conn, target_db)
    finally:
        sys_conn.close()

    db_conn = _connect(database=target_db)
    try:
        for path in files:
            _apply_ddl_file(db_conn, path)
        _list_objects(db_conn, target_db)
    finally:
        db_conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
