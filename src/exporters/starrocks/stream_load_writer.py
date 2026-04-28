"""StarRocks Stream Load HTTP PUT 적재 — `StarRocksWriter` Protocol 실 구현.

StarRocks Stream Load 시맨틱 (4.x docs):
- `PUT http://{fe_host}:{stream_load_port}/api/{db}/{table}/_stream_load`
- Basic Auth (user/password), Expect: 100-continue.
- FE 가 307 redirect → BE 의 동일 path. `requests` 는 redirect 기본 follow + auth 보존.
- 1 HTTP request = 1 atomic transaction. 부분 실패 없음 — 전체 성공/전체 실패.
- `label` 헤더 = idempotency key. 같은 label 재요청 시 "Label Already Exists" 응답으로
  중복 적재 차단. label 충돌 → caller 레이어가 새 label 로 재시도 결정.

응답 JSON 필드 (4.x):
- `Status`: "Success" / "Publish Timeout" / "Label Already Exists" / "Fail" / 기타
  - "Success" → 적재 + publish 완료. NumberLoadedRows = 적재 row 수
  - "Publish Timeout" → 적재 성공, publish 비동기 (조회 가능 시점 지연 가능). 성공으로 간주
  - 그 외 → raise StarRocksLoadError (caller 가 message / ErrorURL 로 진단)
- `NumberLoadedRows`, `NumberFilteredRows`, `Message`, `ErrorURL`

JSON 직렬화:
- format=json + strip_outer_array=true → body = JSON array of row objects.
- row dict 의 datetime/date/time 객체는 caller 가 미리 isoformat 문자열로 변환해 둔다
  (row_builder.py 가 이미 문자열로 채움). 안전망으로 `_default` 가 datetime 도 처리.
- jsonpaths 헤더 미사용 — row dict key 가 곧 컬럼명 (DDL 과 1:1 매핑).
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import date, datetime
from typing import Any

import requests
from requests.auth import HTTPBasicAuth

logger = logging.getLogger(__name__)


_STREAM_LOAD_OK_STATUSES: frozenset[str] = frozenset({
    "Success",
    "Publish Timeout",  # 적재 OK, publish 비동기 — read 지연 가능하지만 데이터 손실 X
})


class StarRocksLoadError(RuntimeError):
    """Stream Load 응답 status != Success/Publish Timeout 시 raise.

    caller 가 `response_payload` 로 ErrorURL/Message 추출해 진단.
    """

    def __init__(
        self,
        table: str,
        label: str,
        status: str,
        response_payload: dict[str, Any],
    ) -> None:
        message = response_payload.get("Message") or "no message"
        error_url = response_payload.get("ErrorURL")
        super().__init__(
            f"StarRocks Stream Load failed: table={table} label={label} "
            f"status={status} message={message!r} error_url={error_url!r}"
        )
        self.table = table
        self.label = label
        self.status = status
        self.response_payload = response_payload


def _json_default(obj: Any) -> Any:
    """row dict 안의 datetime/date 안전망 — row_builder 는 문자열로 채우지만 caller 우회 가능성."""
    if isinstance(obj, datetime):
        return obj.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(obj, date):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _serialize_rows(rows: list[dict[str, Any]]) -> bytes:
    """rows → JSON array bytes (UTF-8). strip_outer_array=true 와 짝."""
    return json.dumps(rows, ensure_ascii=False, default=_json_default).encode("utf-8")


def _make_label(table: str) -> str:
    """label = `{table}_{uuid4hex[:24]}`. StarRocks label 길이 제한 128 — 안전 buffer.

    UUID4 random 이라 동시 적재 충돌 확률 무시 가능. caller 가 명시 label 이 필요하면
    `write_batch(table, rows, label=...)` 으로 override.
    """
    return f"{table}_{uuid.uuid4().hex[:24]}"


class StarRocksStreamLoadWriter:
    """`StarRocksWriter` Protocol HTTP 구현.

    network IO 격리 — `requests` 모듈 호출은 `_put` 안에 한정. 테스트는 monkeypatch 또는
    DI 로 fake response 주입.
    """

    def __init__(
        self,
        host: str,
        port: int | None,
        database: str,
        user: str,
        password: str,
        *,
        request_timeout: float = 300.0,
        scheme: str = "http",
    ) -> None:
        self._host = host
        self._port = port
        self._database = database
        self._auth = HTTPBasicAuth(user, password)
        self._timeout = request_timeout
        self._scheme = scheme

    @classmethod
    def from_env(cls, *, request_timeout: float = 300.0) -> "StarRocksStreamLoadWriter":
        """env / .env 에서 크리덴셜 로드.

        STARROCKS_HOST / STARROCKS_STREAM_LOAD_SCHEME(default http) /
        STARROCKS_STREAM_LOAD_PORT(빈값/미설정 시 default 포트) /
        STARROCKS_USER / STARROCKS_PASSWORD / STARROCKS_RESULT_DATABASE.

        AKS ingress 경유 (`https://starrocks.enhans.ai/api/...`) 의 경우 PORT 를
        비워두면 URL 에 `:port` 가 붙지 않는다 (scheme default 포트 사용).

        `python-dotenv` 는 import 시 자동 로드 안함 — caller (CLI) 가 미리 `load_dotenv()`
        호출했다고 가정. starrocks_raw_loader 와 동일 컨벤션.
        """
        port_raw = os.environ.get("STARROCKS_STREAM_LOAD_PORT", "").strip()
        port = int(port_raw) if port_raw else None
        return cls(
            host=os.environ["STARROCKS_HOST"],
            port=port,
            database=os.environ["STARROCKS_RESULT_DATABASE"],
            user=os.environ["STARROCKS_USER"],
            password=os.environ["STARROCKS_PASSWORD"],
            request_timeout=request_timeout,
            scheme=os.environ.get("STARROCKS_STREAM_LOAD_SCHEME", "http"),
        )

    def _url(self, table: str) -> str:
        netloc = self._host if self._port is None else f"{self._host}:{self._port}"
        return f"{self._scheme}://{netloc}/api/{self._database}/{table}/_stream_load"

    def _headers(self, label: str) -> dict[str, str]:
        return {
            "label": label,
            "Expect": "100-continue",
            "Content-Type": "application/json",
            "format": "json",
            "strip_outer_array": "true",
        }

    def _put(
        self,
        url: str,
        headers: dict[str, str],
        body: bytes,
    ) -> requests.Response:
        """단일 PUT — 테스트에서 monkeypatch 진입점."""
        return requests.put(
            url,
            headers=headers,
            data=body,
            auth=self._auth,
            timeout=self._timeout,
            allow_redirects=True,
        )

    def write_batch(
        self,
        table: str,
        rows: list[dict[str, Any]],
        *,
        label: str | None = None,
    ) -> int:
        """rows 를 `table` 에 atomic 적재. NumberLoadedRows 반환.

        rows 가 비면 0 반환 (HTTP 호출 안함).
        """
        if not rows:
            return 0

        load_label = label or _make_label(table)
        url = self._url(table)
        headers = self._headers(load_label)
        body = _serialize_rows(rows)

        logger.info(
            "stream_load_request table=%s rows=%d label=%s bytes=%d",
            table, len(rows), load_label, len(body),
        )
        response = self._put(url, headers, body)
        response.raise_for_status()
        payload: dict[str, Any] = response.json()

        status = str(payload.get("Status", ""))
        if status not in _STREAM_LOAD_OK_STATUSES:
            raise StarRocksLoadError(
                table=table, label=load_label,
                status=status, response_payload=payload,
            )

        loaded = int(payload.get("NumberLoadedRows", 0))
        filtered = int(payload.get("NumberFilteredRows", 0))
        logger.info(
            "stream_load_success table=%s label=%s loaded=%d filtered=%d status=%s",
            table, load_label, loaded, filtered, status,
        )
        return loaded
