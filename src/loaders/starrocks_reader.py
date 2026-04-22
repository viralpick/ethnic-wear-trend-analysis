"""StarRocks read-only client — `png` 스키마 SELECT 전용.

StarRocks 는 MySQL wire protocol 호환이라 pymysql 로 연결. 우리는 read 권한만 받았고 (agenda
§1), write 경로 (분석 결과 적재) 는 분리 예정이라 이 모듈에 쓰기 메서드를 의도적으로 두지 않는다.

크리덴셜 (.env 또는 환경 변수):
  STARROCKS_HOST        (필수)
  STARROCKS_USER        (필수)
  STARROCKS_PASSWORD    (필수)
  STARROCKS_PORT        (기본 9030 — StarRocks FE query port)
  STARROCKS_DATABASE    (기본 `png`)

`[starrocks]` optional extras 필요: pymysql + python-dotenv.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import pymysql
from dotenv import load_dotenv

from utils.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_PORT = 9030
_DEFAULT_DATABASE = "png"


@dataclass(frozen=True)
class StarRocksConfig:
    host: str
    port: int
    user: str
    password: str
    database: str

    @classmethod
    def from_env(cls) -> "StarRocksConfig":
        load_dotenv()
        host = os.environ.get("STARROCKS_HOST")
        user = os.environ.get("STARROCKS_USER")
        password = os.environ.get("STARROCKS_PASSWORD")
        if not (host and user and password):
            raise RuntimeError(
                "StarRocks credential 없음. .env 에 "
                "STARROCKS_HOST / STARROCKS_USER / STARROCKS_PASSWORD 설정 필요."
            )
        port = int(os.environ.get("STARROCKS_PORT") or _DEFAULT_PORT)
        database = os.environ.get("STARROCKS_DATABASE") or _DEFAULT_DATABASE
        return cls(host=host, port=port, user=user, password=password, database=database)


class StarRocksReader:
    """pymysql 기반 StarRocks SELECT wrapper. 쓰기 메서드 의도적 미제공."""

    def __init__(self, config: StarRocksConfig) -> None:
        self._config = config

    @classmethod
    def from_env(cls) -> "StarRocksReader":
        return cls(StarRocksConfig.from_env())

    @contextmanager
    def connect(self) -> Iterator[pymysql.connections.Connection]:
        conn = pymysql.connect(
            host=self._config.host,
            port=self._config.port,
            user=self._config.user,
            password=self._config.password,
            database=self._config.database,
            cursorclass=pymysql.cursors.DictCursor,
            connect_timeout=10,
            read_timeout=30,
        )
        try:
            yield conn
        finally:
            conn.close()

    def ping(self) -> str:
        """연결 + `SELECT VERSION()` 검증. 성공 시 server version string 반환."""
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT VERSION() AS v")
                row = cur.fetchone()
        return str(row["v"]) if row else ""

    def list_tables(self) -> list[str]:
        """현재 database 의 테이블 이름 리스트 (알파벳 정렬)."""
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SHOW TABLES")
                rows = cur.fetchall()
        return sorted(next(iter(r.values())) for r in rows)

    def count_rows(self, table: str) -> int:
        """`SELECT COUNT(*) FROM {table}`. SQL injection 방지 위해 table 이름 sanity check."""
        self._assert_safe_identifier(table)
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) AS n FROM `{table}`")
                row = cur.fetchone()
        return int(row["n"]) if row else 0

    def sample(self, table: str, limit: int = 3) -> list[dict]:
        """`SELECT * FROM {table} LIMIT N`. 탐색/디버깅용."""
        self._assert_safe_identifier(table)
        if limit <= 0 or limit > 100:
            raise ValueError(f"limit must be 1..100, got {limit}")
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT * FROM `{table}` LIMIT {int(limit)}")
                return list(cur.fetchall())

    @staticmethod
    def _assert_safe_identifier(name: str) -> None:
        """alphanumeric + underscore 만 허용. StarRocks 테이블 이름 convention 에 맞음."""
        if not name.replace("_", "").isalnum():
            raise ValueError(f"unsafe SQL identifier: {name!r}")
