"""StarRocksStreamLoadWriter pinning — HTTP 호출 monkeypatch 로 격리.

검증:
- Protocol 호환 (StarRocksWriter isinstance).
- empty rows → no HTTP, return 0.
- Success status → NumberLoadedRows 반환 + URL/headers/body 정합.
- Publish Timeout → 성공 처리 (publish 비동기, read 지연 가능하지만 데이터 손실 X).
- Fail / Label Already Exists → StarRocksLoadError.
- HTTP 4xx/5xx → raise_for_status 가 raise (Exception 으로 래핑 X — 호출자 진단).
- JSON serialize: datetime safety net + frozenset/set 류 캐치.
- label override 작동.
"""
from __future__ import annotations

import json
from typing import Any

import pytest
import requests

from exporters.starrocks.stream_load_writer import (
    StarRocksLoadError,
    StarRocksStreamLoadWriter,
    _serialize_rows,
)
from exporters.starrocks.writer import StarRocksWriter


class _FakeResponse:
    """requests.Response stand-in — `.raise_for_status()` + `.json()`."""

    def __init__(
        self,
        payload: dict[str, Any],
        *,
        status_code: int = 200,
    ) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            err = requests.HTTPError(f"{self.status_code} error")
            err.response = self  # type: ignore[attr-defined]
            raise err

    def json(self) -> dict[str, Any]:
        return self._payload


@pytest.fixture
def writer() -> StarRocksStreamLoadWriter:
    return StarRocksStreamLoadWriter(
        host="starrocks.test",
        port=8030,
        database="ethnic_result",
        user="svc_test",
        password="pw",
    )


# --------------------------------------------------------------------------- #
# Protocol + edge cases

def test_protocol_compatibility(writer: StarRocksStreamLoadWriter) -> None:
    assert isinstance(writer, StarRocksWriter)


def test_empty_rows_returns_zero_without_http(
    writer: StarRocksStreamLoadWriter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: list[Any] = []

    def _no_http(*args: Any, **kwargs: Any) -> _FakeResponse:
        called.append((args, kwargs))
        return _FakeResponse({"Status": "Success", "NumberLoadedRows": 0})

    monkeypatch.setattr(writer, "_put", _no_http)
    assert writer.write_batch("item", []) == 0
    assert called == []


# --------------------------------------------------------------------------- #
# Success path

def test_success_returns_loaded_rows_and_request_shape(
    writer: StarRocksStreamLoadWriter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def _fake_put(url: str, headers: dict[str, str], body: bytes) -> _FakeResponse:
        captured["url"] = url
        captured["headers"] = headers
        captured["body"] = body
        return _FakeResponse({
            "Status": "Success",
            "NumberLoadedRows": 2,
            "NumberFilteredRows": 0,
        })

    monkeypatch.setattr(writer, "_put", _fake_put)

    rows = [
        {"source": "instagram", "source_post_id": "p1", "schema_version": "pipeline_v1.0"},
        {"source": "instagram", "source_post_id": "p2", "schema_version": "pipeline_v1.0"},
    ]
    n = writer.write_batch("item", rows)
    assert n == 2

    # URL: scheme://host:port/api/{db}/{table}/_stream_load
    assert captured["url"] == "http://starrocks.test:8030/api/ethnic_result/item/_stream_load"

    headers = captured["headers"]
    # Stream Load 필수 헤더.
    assert headers["Content-Type"] == "application/json"
    assert headers["format"] == "json"
    assert headers["strip_outer_array"] == "true"
    assert headers["Expect"] == "100-continue"
    # label 자동 생성 (table prefix).
    assert headers["label"].startswith("item_")
    assert len(headers["label"]) > len("item_")

    # body 는 JSON array.
    parsed = json.loads(captured["body"].decode("utf-8"))
    assert parsed == rows


def test_publish_timeout_treated_as_success(
    writer: StarRocksStreamLoadWriter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        writer,
        "_put",
        lambda *a, **kw: _FakeResponse({
            "Status": "Publish Timeout",
            "NumberLoadedRows": 5,
        }),
    )
    assert writer.write_batch("canonical_group", [{"x": 1}]) == 5


def test_label_override_used_verbatim(
    writer: StarRocksStreamLoadWriter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, str] = {}

    def _fake_put(url: str, headers: dict[str, str], body: bytes) -> _FakeResponse:
        captured["label"] = headers["label"]
        return _FakeResponse({"Status": "Success", "NumberLoadedRows": 1})

    monkeypatch.setattr(writer, "_put", _fake_put)
    writer.write_batch(
        "item",
        [{"id": 1}],
        label="manual_label_2026_04_27_run_42",
    )
    assert captured["label"] == "manual_label_2026_04_27_run_42"


# --------------------------------------------------------------------------- #
# Failure paths

def test_status_fail_raises_load_error(
    writer: StarRocksStreamLoadWriter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "Status": "Fail",
        "Message": "schema mismatch",
        "ErrorURL": "http://starrocks.test:8030/api/_load_error_log?file=fail.log",
    }
    monkeypatch.setattr(writer, "_put", lambda *a, **kw: _FakeResponse(payload))

    with pytest.raises(StarRocksLoadError) as exc:
        writer.write_batch("item", [{"x": 1}])
    err = exc.value
    assert err.table == "item"
    assert err.status == "Fail"
    assert err.response_payload == payload
    assert "schema mismatch" in str(err)
    assert err.label.startswith("item_")


def test_label_already_exists_raises(
    writer: StarRocksStreamLoadWriter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 같은 label 재요청 — caller 가 새 label 로 retry 결정. writer 는 raise.
    monkeypatch.setattr(
        writer,
        "_put",
        lambda *a, **kw: _FakeResponse({
            "Status": "Label Already Exists",
            "Message": "label dup",
        }),
    )
    with pytest.raises(StarRocksLoadError) as exc:
        writer.write_batch("item", [{"x": 1}], label="dup_label")
    assert exc.value.status == "Label Already Exists"


def test_http_error_propagates(
    writer: StarRocksStreamLoadWriter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        writer,
        "_put",
        lambda *a, **kw: _FakeResponse({}, status_code=401),
    )
    with pytest.raises(requests.HTTPError):
        writer.write_batch("item", [{"x": 1}])


# --------------------------------------------------------------------------- #
# Serialization

def test_serialize_rows_handles_datetime_safety_net() -> None:
    from datetime import date, datetime

    body = _serialize_rows([
        {
            "computed_at": datetime(2026, 4, 27, 9, 0, 0),
            "week_start_date": date(2026, 4, 27),
            "value": 1,
        }
    ])
    parsed = json.loads(body.decode("utf-8"))
    assert parsed == [
        {
            "computed_at": "2026-04-27 09:00:00",
            "week_start_date": "2026-04-27",
            "value": 1,
        }
    ]


def test_serialize_rows_unicode_preserved() -> None:
    body = _serialize_rows([{"display_name": "쿠르타 세트"}])
    # ensure_ascii=False 라 raw bytes 에 한글 그대로.
    assert "쿠르타 세트" in body.decode("utf-8")


# --------------------------------------------------------------------------- #
# AKS ingress 경유: scheme=https + port=None → URL 에 :port 미부착

def test_url_omits_port_when_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    w = StarRocksStreamLoadWriter(
        host="starrocks.enhans.ai",
        port=None,
        database="ethnic_result",
        user="svc",
        password="pw",
        scheme="https",
    )

    captured: dict[str, str] = {}

    def _fake_put(url: str, headers: dict[str, str], body: bytes) -> _FakeResponse:
        captured["url"] = url
        return _FakeResponse({"Status": "Success", "NumberLoadedRows": 1})

    monkeypatch.setattr(w, "_put", _fake_put)
    w.write_batch("item", [{"x": 1}])

    # AKS ingress: 443 default, port 미부착.
    assert captured["url"] == "https://starrocks.enhans.ai/api/ethnic_result/item/_stream_load"


def test_from_env_reads_scheme_and_blank_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STARROCKS_HOST", "starrocks.enhans.ai")
    monkeypatch.setenv("STARROCKS_STREAM_LOAD_SCHEME", "https")
    monkeypatch.setenv("STARROCKS_STREAM_LOAD_PORT", "")  # 빈값 → port=None
    monkeypatch.setenv("STARROCKS_RESULT_DATABASE", "ethnic_result")
    monkeypatch.setenv("STARROCKS_USER", "svc")
    monkeypatch.setenv("STARROCKS_PASSWORD", "pw")

    w = StarRocksStreamLoadWriter.from_env()

    assert w._host == "starrocks.enhans.ai"
    assert w._port is None
    assert w._scheme == "https"
    assert w._database == "ethnic_result"
