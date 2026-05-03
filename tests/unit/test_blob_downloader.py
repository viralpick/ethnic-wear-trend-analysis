"""BlobDownloader 단위 — Fake Azure client 로 로직 검증 (실 네트워크 X).

검증:
  - container prefix strip
  - 캐시된 파일 idempotent skip
  - blob 없을 때 (`ResourceNotFoundError`) None + no raise
  - 알려지지 않은 예외 (예: 인증 오류) 는 raise (silent drop 금지)
  - from_env credential 누락 시 RuntimeError
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("azure.storage.blob", reason="blob extras required")
pytest.importorskip("dotenv", reason="blob extras required")

from azure.core.exceptions import ClientAuthenticationError, ResourceNotFoundError  # noqa: E402

from loaders.blob_downloader import BlobDownloader  # noqa: E402


def _make_downloader(
    data_by_blob: dict[str, bytes],
    *,
    raise_for_blob: dict[str, Exception] | None = None,
) -> BlobDownloader:
    service = MagicMock()
    raise_for_blob = raise_for_blob or {}

    def get_blob_client(container: str, blob: str):  # noqa: ARG001
        client = MagicMock()
        if blob in raise_for_blob:
            client.download_blob.side_effect = raise_for_blob[blob]
            return client
        payload = data_by_blob.get(blob)
        if payload is None:
            client.download_blob.side_effect = ResourceNotFoundError(f"blob not found: {blob}")
        else:
            stream = MagicMock()
            stream.readall.return_value = payload
            client.download_blob.return_value = stream
        return client

    service.get_blob_client = get_blob_client
    return BlobDownloader(service=service, container="collectify")


def test_download_writes_file(tmp_path: Path) -> None:
    d = _make_downloader({"poc/a.jpg": b"FAKEJPG"})
    out = d.download("collectify/poc/a.jpg", tmp_path)
    assert out is not None
    assert out.read_bytes() == b"FAKEJPG"
    assert out.name == "a.jpg"


def test_download_idempotent_cache(tmp_path: Path) -> None:
    d = _make_downloader({"poc/a.jpg": b"FAKEJPG"})
    first = d.download("collectify/poc/a.jpg", tmp_path)
    # 두 번째 호출 시 disk exists → Azure 호출 없이 동일 path 반환.
    second = d.download("collectify/poc/a.jpg", tmp_path)
    assert first == second
    assert second is not None


def test_download_returns_none_on_resource_not_found(tmp_path: Path) -> None:
    """blob 없음은 알려진 transient/missing — None 반환."""
    d = _make_downloader({})  # 없는 blob → ResourceNotFoundError
    out = d.download("collectify/poc/missing.jpg", tmp_path)
    assert out is None


def test_download_reraises_unknown_exception(tmp_path: Path) -> None:
    """인증 오류 등 알려지지 않은 예외는 silent drop 하지 않고 raise (룰 §4)."""
    d = _make_downloader(
        {},
        raise_for_blob={"poc/auth.jpg": ClientAuthenticationError("invalid sas")},
    )
    with pytest.raises(ClientAuthenticationError):
        d.download("collectify/poc/auth.jpg", tmp_path)


def test_strip_container_prefix_removes_when_present() -> None:
    d = _make_downloader({})
    assert d._strip_container_prefix("collectify/poc/a.jpg") == "poc/a.jpg"


def test_strip_container_prefix_no_op_when_absent() -> None:
    d = _make_downloader({})
    assert d._strip_container_prefix("poc/a.jpg") == "poc/a.jpg"


def test_from_env_requires_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "AZURE_STORAGE_CONNECTION_STRING",
        "AZURE_STORAGE_ACCOUNT_NAME",
        "AZURE_STORAGE_ACCOUNT_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    # load_dotenv 를 no-op 로 mock — 실 .env 가 CWD 에 있어도 테스트 간섭 방지.
    monkeypatch.setattr("loaders.blob_downloader.load_dotenv", lambda *a, **kw: False)
    with pytest.raises(RuntimeError, match="Azure credential"):
        BlobDownloader.from_env()
