"""Azure Blob Storage downloader — posting.tsv 의 image_paths 를 로컬 캐시로 수집.

크리덴셜 우선순위 (.env 또는 환경 변수):
  1) AZURE_STORAGE_CONNECTION_STRING (권장, one-liner)
  2) AZURE_STORAGE_ACCOUNT_NAME + AZURE_STORAGE_ACCOUNT_KEY (fallback)
Container:
  AZURE_STORAGE_CONTAINER 또는 기본값 `collectify`.

idempotent: 같은 blob 을 두 번 요청해도 로컬 파일이 이미 있으면 네트워크 호출 없이 skip.
실패한 blob 은 raise 하지 않고 log 후 None 반환 (batch 처리의 다른 blob 은 계속 진행).

이 모듈은 top-level 로 azure-storage-blob / python-dotenv 를 import. `[blob]` extras 미설치
시 모듈 import 자체가 ImportError — core 는 이 모듈을 절대 top-level import 하지 말 것.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

from utils.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_CONTAINER = "collectify"


@dataclass
class BlobDownloader:
    """Azure Blob SDK 얇은 wrapper — from_env() 로 credential 감지 + idempotent download."""

    service: BlobServiceClient
    container: str

    @classmethod
    def from_env(cls, container: str | None = None) -> "BlobDownloader":
        """환경 변수 / `.env` 에서 credential 읽기. 실패 시 RuntimeError."""
        load_dotenv()
        conn = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
        if conn:
            service = BlobServiceClient.from_connection_string(conn)
        else:
            name = os.environ.get("AZURE_STORAGE_ACCOUNT_NAME")
            key = os.environ.get("AZURE_STORAGE_ACCOUNT_KEY")
            if not (name and key):
                raise RuntimeError(
                    "Azure credential 없음. .env 에 AZURE_STORAGE_CONNECTION_STRING 또는 "
                    "(AZURE_STORAGE_ACCOUNT_NAME + AZURE_STORAGE_ACCOUNT_KEY) 설정 필요."
                )
            service = BlobServiceClient(
                account_url=f"https://{name}.blob.core.windows.net",
                credential=key,
            )
        selected = (
            container
            or os.environ.get("AZURE_STORAGE_CONTAINER")
            or _DEFAULT_CONTAINER
        )
        return cls(service=service, container=selected)

    def download(self, blob_path: str, dest_dir: Path) -> Path | None:
        """blob_path → 로컬 파일 경로. 이미 존재하면 skip. 실패 시 None + log."""
        blob_name = self._strip_container_prefix(blob_path)
        dest = dest_dir / Path(blob_name).name
        if dest.exists():
            return dest
        dest_dir.mkdir(parents=True, exist_ok=True)
        try:
            client = self.service.get_blob_client(container=self.container, blob=blob_name)
            stream = client.download_blob()
            dest.write_bytes(stream.readall())
            return dest
        except Exception as exc:  # noqa: BLE001 — Azure SDK 예외 계보 다양, 로그 후 skip
            logger.info("blob_download_failed path=%s reason=%s", blob_path, exc)
            return None

    def _strip_container_prefix(self, blob_path: str) -> str:
        """`{container}/{blob_name}` 형식이면 container 부분 제거. 아니면 그대로."""
        prefix = f"{self.container}/"
        if blob_path.startswith(prefix):
            return blob_path[len(prefix):]
        return blob_path
