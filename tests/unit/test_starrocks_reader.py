"""StarRocksReader 단위 — MagicMock 기반 (실 네트워크 X).

StarRocksConfig.from_env credential 분기 / ping / list_tables / count_rows / sample /
_assert_safe_identifier 검증.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("pymysql", reason="starrocks extras required")
pytest.importorskip("dotenv", reason="starrocks extras required")

from loaders.starrocks_reader import StarRocksConfig, StarRocksReader  # noqa: E402

# --------------------------------------------------------------------------- #
# StarRocksConfig.from_env
# --------------------------------------------------------------------------- #

def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "STARROCKS_HOST", "STARROCKS_USER", "STARROCKS_PASSWORD",
        "STARROCKS_PORT", "STARROCKS_DATABASE",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("loaders.starrocks_reader.load_dotenv", lambda *a, **kw: False)


def test_from_env_uses_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("STARROCKS_HOST", "host.example")
    monkeypatch.setenv("STARROCKS_USER", "u")
    monkeypatch.setenv("STARROCKS_PASSWORD", "p")
    cfg = StarRocksConfig.from_env()
    assert cfg.host == "host.example"
    assert cfg.port == 9030
    assert cfg.database == "png"


def test_from_env_respects_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("STARROCKS_HOST", "host.example")
    monkeypatch.setenv("STARROCKS_USER", "u")
    monkeypatch.setenv("STARROCKS_PASSWORD", "p")
    monkeypatch.setenv("STARROCKS_PORT", "9050")
    monkeypatch.setenv("STARROCKS_DATABASE", "other_db")
    cfg = StarRocksConfig.from_env()
    assert cfg.port == 9050
    assert cfg.database == "other_db"


def test_from_env_raises_without_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    with pytest.raises(RuntimeError, match="StarRocks credential"):
        StarRocksConfig.from_env()


# --------------------------------------------------------------------------- #
# StarRocksReader query methods — mocked pymysql.connect
# --------------------------------------------------------------------------- #

def _fake_cursor(fetchone=None, fetchall=None) -> MagicMock:
    cursor = MagicMock()
    cursor.__enter__.return_value = cursor
    cursor.__exit__.return_value = None
    cursor.fetchone.return_value = fetchone
    cursor.fetchall.return_value = fetchall or []
    return cursor


def _fake_conn(cursor: MagicMock) -> MagicMock:
    conn = MagicMock()
    conn.cursor.return_value = cursor
    conn.close = MagicMock()
    return conn


def _make_reader() -> StarRocksReader:
    return StarRocksReader(StarRocksConfig(
        host="h", port=9030, user="u", password="p", database="png",
    ))


def test_ping_returns_version() -> None:
    reader = _make_reader()
    cursor = _fake_cursor(fetchone={"v": "5.5.0-StarRocks-3.1"})
    with patch("pymysql.connect", return_value=_fake_conn(cursor)):
        assert reader.ping() == "5.5.0-StarRocks-3.1"


def test_list_tables_returns_sorted_names() -> None:
    reader = _make_reader()
    cursor = _fake_cursor(fetchall=[{"Tables_in_png": "z_t"}, {"Tables_in_png": "a_t"}])
    with patch("pymysql.connect", return_value=_fake_conn(cursor)):
        assert reader.list_tables() == ["a_t", "z_t"]


def test_count_rows_returns_int() -> None:
    reader = _make_reader()
    cursor = _fake_cursor(fetchone={"n": 42})
    with patch("pymysql.connect", return_value=_fake_conn(cursor)):
        assert reader.count_rows("india_ai_fashion_inatagram_posting") == 42


def test_count_rows_rejects_unsafe_table_name() -> None:
    reader = _make_reader()
    with pytest.raises(ValueError, match="unsafe SQL identifier"):
        reader.count_rows("evil; DROP TABLE x; --")


def test_sample_returns_rows() -> None:
    reader = _make_reader()
    cursor = _fake_cursor(fetchall=[{"x": 1}, {"x": 2}])
    with patch("pymysql.connect", return_value=_fake_conn(cursor)):
        rows = reader.sample("my_table", limit=2)
    assert rows == [{"x": 1}, {"x": 2}]


def test_sample_rejects_out_of_range_limit() -> None:
    reader = _make_reader()
    with pytest.raises(ValueError, match="limit must be 1..100"):
        reader.sample("my_table", limit=0)
    with pytest.raises(ValueError, match="limit must be 1..100"):
        reader.sample("my_table", limit=101)


def test_connection_closed_on_exit() -> None:
    reader = _make_reader()
    cursor = _fake_cursor(fetchone={"v": "x"})
    conn = _fake_conn(cursor)
    with patch("pymysql.connect", return_value=conn):
        reader.ping()
    conn.close.assert_called_once()


# --------------------------------------------------------------------------- #
# _assert_safe_identifier
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", [
    "india_ai_fashion_inatagram_posting",
    "my_table_123",
    "SIMPLE",
])
def test_safe_identifier_accepts_alphanumeric_underscore(name: str) -> None:
    StarRocksReader._assert_safe_identifier(name)  # noqa: SLF001


@pytest.mark.parametrize("name", [
    "table; DROP",
    "table with space",
    "table-dash",
    "table.quoted",
    "",
])
def test_safe_identifier_rejects_dangerous_chars(name: str) -> None:
    with pytest.raises(ValueError, match="unsafe SQL identifier"):
        StarRocksReader._assert_safe_identifier(name)  # noqa: SLF001
