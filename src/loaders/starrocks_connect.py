"""StarRocks 직접 쿼리용 공용 connection helper — `[db]` extras 의존.

scripts/* 가 ad-hoc DDL / verify / smoke 용으로 pymysql.connect 를 직접 호출하던 30+
호출처를 단일 source 로 통합. drift 가 시작된 상태 (timeout 10/15, port string 명시
vs 미명시, RESULT_DATABASE vs RAW_DATABASE, charset/autocommit 일부만) → 환경변수
정책 변경 시 한 곳만 수정.

batch loader (`StarRocksRawLoader`) 는 별도 — page_size / window_mode 같은 batch
의미가 있어 자체 from_env 유지.

크리덴셜 (.env 또는 환경 변수, common):
  STARROCKS_HOST, STARROCKS_USER, STARROCKS_PASSWORD
  STARROCKS_PORT (default "9030") — query 포트. AKS HTTPS ingress 는 stream load 전용

DB 별:
  connect_raw():    STARROCKS_RAW_DATABASE (default "png") — 크롤러 raw 테이블
  connect_result(): STARROCKS_RESULT_DATABASE — 분석 결과 / view 적재 (default 없음)
  connect_ddl():    database 미지정 — CREATE DATABASE / SHOW DATABASES 등 부트스트랩
"""
from __future__ import annotations

import os

import pymysql
from dotenv import load_dotenv

# 일관 default — 모든 scripts/ 호출처가 같은 값을 받도록.
_DEFAULT_PORT = "9030"
_DEFAULT_RAW_DB = "png"
_DEFAULT_CONNECT_TIMEOUT = 15


def _resolve_port() -> int:
    """`STARROCKS_QUERY_PORT` 가 명시되면 그것 — 일부 smoke/build script 가 stream load
    port (8030) 와 구분하기 위해 도입한 변종. 미명시면 `STARROCKS_PORT` 또는 default.
    """
    return int(
        os.environ.get("STARROCKS_QUERY_PORT")
        or os.environ.get("STARROCKS_PORT", _DEFAULT_PORT)
    )


def _common_kwargs(
    *,
    dict_cursor: bool,
    autocommit: bool,
    charset: str | None,
) -> dict:
    kwargs: dict = {
        "host": os.environ["STARROCKS_HOST"],
        "port": _resolve_port(),
        "user": os.environ["STARROCKS_USER"],
        "password": os.environ["STARROCKS_PASSWORD"],
        "connect_timeout": _DEFAULT_CONNECT_TIMEOUT,
        "autocommit": autocommit,
    }
    if dict_cursor:
        kwargs["cursorclass"] = pymysql.cursors.DictCursor
    if charset is not None:
        kwargs["charset"] = charset
    return kwargs


def connect_raw(
    *,
    dict_cursor: bool = True,
    autocommit: bool = False,
    charset: str | None = "utf8mb4",
    load_env: bool = True,
) -> pymysql.connections.Connection:
    """Raw DB (`STARROCKS_RAW_DATABASE`, default "png") 연결.

    `STARROCKS_RAW_DATABASE` 미설정 시 default "png" 사용 — backfill / smoke / count
    공통. AKS HTTPS ingress 는 stream load 전용이므로 query 는 9030 포트.
    """
    if load_env:
        load_dotenv()
    kwargs = _common_kwargs(
        dict_cursor=dict_cursor, autocommit=autocommit, charset=charset,
    )
    kwargs["database"] = os.environ.get("STARROCKS_RAW_DATABASE", _DEFAULT_RAW_DB)
    return pymysql.connect(**kwargs)


def connect_result(
    *,
    dict_cursor: bool = True,
    autocommit: bool = False,
    charset: str | None = "utf8mb4",
    load_env: bool = True,
) -> pymysql.connections.Connection:
    """Result DB (`STARROCKS_RESULT_DATABASE`) 연결 — verify / show / diagnose 용.

    `STARROCKS_RESULT_DATABASE` 필수 — default 없음 (실수로 raw DB 에 verify 쿼리
    날리는 사고 차단).
    """
    if load_env:
        load_dotenv()
    kwargs = _common_kwargs(
        dict_cursor=dict_cursor, autocommit=autocommit, charset=charset,
    )
    kwargs["database"] = os.environ["STARROCKS_RESULT_DATABASE"]
    return pymysql.connect(**kwargs)


def connect_ddl(
    *,
    database: str | None = None,
    autocommit: bool = True,
    load_env: bool = True,
) -> pymysql.connections.Connection:
    """DDL 용 — database 미지정 (CREATE DATABASE / SHOW DATABASES 부트스트랩).

    autocommit default True (DDL 은 autocommit 자연). dict_cursor 미지정 (DDL 은
    cursor.execute 만 사용, fetchall 결과 DictCursor 불필요).
    """
    if load_env:
        load_dotenv()
    kwargs = _common_kwargs(
        dict_cursor=False, autocommit=autocommit, charset=None,
    )
    kwargs["database"] = database or ""
    return pymysql.connect(**kwargs)


__all__ = ["connect_raw", "connect_result", "connect_ddl"]
