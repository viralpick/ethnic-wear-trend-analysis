"""StarRocks 연결 검증 — `.env` 로드 → ping → list_tables → 주요 테이블 row count.

전제: `uv sync --extra starrocks` + `.env` 에 STARROCKS_* 설정 (VPN 필요할 수 있음).

실행:
  uv run python scripts/test_starrocks_connection.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from loaders.starrocks_reader import StarRocksReader  # noqa: E402

_CORE_TABLES = (
    "india_ai_fashion_inatagram_posting",
    "india_ai_fashion_inatagram_profile",
    "india_ai_fashion_youtube_posting",
    "india_ai_fashionash_tag_search_result",
)


def main() -> None:
    reader = StarRocksReader.from_env()
    cfg = reader._config  # noqa: SLF001 — 디버그 출력용
    print(f"[starrocks] host={cfg.host}:{cfg.port} db={cfg.database} user={cfg.user}")

    version = reader.ping()
    print(f"[starrocks] server version: {version}")

    tables = reader.list_tables()
    print(f"[starrocks] {len(tables)} tables in `{cfg.database}`")

    for t in _CORE_TABLES:
        if t in tables:
            n = reader.count_rows(t)
            print(f"[starrocks] {t}: {n:,} rows")
        else:
            print(f"[starrocks] {t}: (missing)")


if __name__ == "__main__":
    main()
