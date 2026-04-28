"""StarRocks INSERT INTO writer — 9030 query 포트 fallback.

Stream Load HTTP (8030) 가 방화벽 차단된 환경에서 적재 path 검증용. multi-row
`INSERT INTO ... VALUES (...), (...)` 한 SQL 문으로 atomic 적재 (autocommit).

JSON 컬럼: `PARSE_JSON('escaped_json_str')` 으로 wrap. datetime/date 는 `'YYYY-MM-DD ...'`
literal. NULL/숫자/bool 는 직접 변환.

대량 적재 시 Stream Load 보다 느림 — 8030 allow-list 후 StarRocksStreamLoadWriter
권장. 본 writer 는 (a) 검증 (b) 8030 차단 임시 우회 용도.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime
from typing import Any

import pymysql
from pymysql.converters import escape_string

logger = logging.getLogger(__name__)


def _format_value(value: Any) -> str:
    """row dict 값 → SQL literal. JSON 은 PARSE_JSON wrap, 문자열은 escape_string.

    row_builder 가 datetime 을 미리 isoformat 문자열로 변환해두므로 datetime 분기는
    안전망 (직접 datetime/date 가 들어와도 처리).
    """
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, datetime):
        return f"'{value.strftime('%Y-%m-%d %H:%M:%S')}'"
    if isinstance(value, date):
        return f"'{value.isoformat()}'"
    if isinstance(value, (dict, list)):
        blob = json.dumps(value, ensure_ascii=False)
        return f"PARSE_JSON('{escape_string(blob)}')"
    if isinstance(value, str):
        return f"'{escape_string(value)}'"
    raise TypeError(f"Unsupported value type for INSERT: {type(value).__name__}")


class StarRocksInsertWriter:
    """`StarRocksWriter` Protocol — pymysql 9030 query 포트 INSERT 적재.

    Stream Load 와 동일하게 row dict key = 컬럼명 (DDL 1:1). atomic 단위는 한 INSERT
    statement = 한 table batch.
    """

    def __init__(
        self,
        host: str,
        port: int,
        database: str,
        user: str,
        password: str,
        *,
        connect_timeout: int = 30,
    ) -> None:
        self._conn_kwargs: dict[str, Any] = {
            "host": host, "port": port, "user": user,
            "password": password, "database": database,
            "connect_timeout": connect_timeout,
            "autocommit": True,
        }
        self._database = database

    @classmethod
    def from_env(cls) -> "StarRocksInsertWriter":
        """env / .env 에서 크리덴셜 로드. caller (CLI) 가 미리 load_dotenv() 가정.

        STARROCKS_HOST / STARROCKS_PORT(default 9030) / STARROCKS_USER /
        STARROCKS_PASSWORD / STARROCKS_RESULT_DATABASE.
        """
        return cls(
            host=os.environ["STARROCKS_HOST"],
            port=int(os.environ.get("STARROCKS_PORT", "9030")),
            database=os.environ["STARROCKS_RESULT_DATABASE"],
            user=os.environ["STARROCKS_USER"],
            password=os.environ["STARROCKS_PASSWORD"],
        )

    def write_batch(self, table: str, rows: list[dict[str, Any]]) -> int:
        if not rows:
            return 0
        cols = list(rows[0].keys())
        col_clause = ", ".join(f"`{c}`" for c in cols)
        tuples = []
        for row in rows:
            values = [_format_value(row.get(c)) for c in cols]
            tuples.append(f"({', '.join(values)})")
        sql = (
            f"INSERT INTO `{self._database}`.`{table}` ({col_clause}) VALUES\n"
            + ",\n".join(tuples)
        )

        logger.info(
            "starrocks_insert_request table=%s rows=%d bytes=%d",
            table, len(rows), len(sql),
        )
        with pymysql.connect(**self._conn_kwargs) as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
        logger.info("starrocks_insert_success table=%s loaded=%d", table, len(rows))
        return len(rows)
