"""StarRocks posting / hashtag_search / profile / youtube 실 row count 확인.

Step D 파일럿 / 50-color preset / skin LAB 샘플 모수 결정용. memory 상 2026-04-22 기준
posting 1139 / hashtag_search 1274 / profile 46 / youtube 32 였음 — 증가분 측정.

실행:
  uv run python scripts/count_posting_rows.py
  uv run python scripts/count_posting_rows.py --with-ethnic-breakdown

전제: `.env` 에 STARROCKS_* 크리덴셜 설정 + `uv sync --extra db`.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

import pymysql
from dotenv import load_dotenv


# posting 에서 image_paths 가 있는 (= 블롭 접근 가능한) post 만이 Pipeline B 유효 모수.
# download_urls 는 raw URL list, image_paths 는 blob 경로 list. Pipeline B 는 후자 사용.
_QUERIES: dict[str, str] = {
    "ig_posting_total":
        "SELECT COUNT(*) AS c FROM india_ai_fashion_inatagram_posting",
    "ig_posting_with_image":
        "SELECT COUNT(*) AS c FROM india_ai_fashion_inatagram_posting "
        "WHERE download_urls IS NOT NULL AND download_urls != ''",
    "ig_posting_hashtag_entry":
        "SELECT COUNT(*) AS c FROM india_ai_fashion_inatagram_posting "
        "WHERE entry = 'hashtag'",
    "ig_posting_profile_entry":
        "SELECT COUNT(*) AS c FROM india_ai_fashion_inatagram_posting "
        "WHERE entry = 'profile'",
    "ig_hashtag_search_result":
        "SELECT COUNT(*) AS c FROM india_ai_fashion_inatagram_hash_tag_search_result",
    "ig_profile":
        "SELECT COUNT(*) AS c FROM india_ai_fashion_inatagram_profile",
    "ig_profile_posting":
        "SELECT COUNT(*) AS c FROM india_ai_fashion_inatagram_profile_posting",
    "yt_posting_total":
        "SELECT COUNT(*) AS c FROM india_ai_fashion_youtube_posting",
    "yt_posting_with_thumbnail":
        "SELECT COUNT(*) AS c FROM india_ai_fashion_youtube_posting "
        "WHERE thumbnail_url IS NOT NULL AND thumbnail_url != ''",
}


def _connect() -> pymysql.Connection:
    """raw DB 연결 — `loaders.starrocks_connect.connect_raw` 위임 (drift 방지)."""
    from loaders.starrocks_connect import connect_raw
    return connect_raw()


def _run(cur: pymysql.cursors.Cursor, sql: str) -> int:
    cur.execute(sql)
    row = cur.fetchone()
    return int(row["c"]) if row else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-ethnic-breakdown", action="store_true",
        help="entry=hashtag 에서 ethnic 관련 해시태그 포함 비율 샘플 (느림)",
    )
    args = parser.parse_args()

    with _connect() as conn, conn.cursor() as cur:
        results: dict[str, int] = {k: _run(cur, sql) for k, sql in _QUERIES.items()}

        if args.with_ethnic_breakdown:
            # 대표 ethnic 해시태그 5 개 기준 LIKE 매칭 (전체 spec §3 56 개 중 대표만)
            sample_tags = ["kurta", "saree", "lehenga", "anarkali", "salwar"]
            like_clause = " OR ".join(f"LOWER(content) LIKE '%#{t}%'" for t in sample_tags)
            ethnic_sql = (
                "SELECT COUNT(*) AS c FROM india_ai_fashion_inatagram_posting "
                f"WHERE entry = 'hashtag' AND ({like_clause})"
            )
            results["ig_posting_ethnic_hashtag_sample"] = _run(cur, ethnic_sql)

    _print_table(results)
    _print_guidance(results)
    return 0


def _print_table(results: dict[str, int]) -> None:
    width = max(len(k) for k in results)
    print(f"\n{'='*(width + 12)}")
    print(f"  {'metric'.ljust(width)}   count")
    print(f"{'='*(width + 12)}")
    for key, count in results.items():
        print(f"  {key.ljust(width)}   {count:>6,}")
    print(f"{'='*(width + 12)}\n")


def _print_guidance(results: dict[str, int]) -> None:
    """선행 task 들이 사용할 실 모수 판단 가이드."""
    with_image = results.get("ig_posting_with_image", 0)
    print(f"[step-d] Phase 0 파일럿 샘플 풀 (download_urls 有): {with_image:,}")
    print(f"[step-d] 500+ 모수 충족 여부: {'OK' if with_image >= 500 else 'SHORT — 재협의 필요'}")
    print(f"[step-d] 50-color preset k-medoids 입력 상한: {with_image:,}")
    print(f"[step-d] skin LAB 수집 권장 샘플 (100+): {min(with_image, 200):,} (상한 200)")


if __name__ == "__main__":
    sys.exit(main())
