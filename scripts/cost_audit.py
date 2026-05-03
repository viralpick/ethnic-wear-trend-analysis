"""DB 기반 일주일 평균 IG/YT 픽업 + LLM 비용 산출.

실행: uv run python scripts/cost_audit.py
전제: VPN ON + .env STARROCKS_* + uv sync --extra db
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

import pymysql

# production 의 image/video URL 분류 정책과 byte-identical 정렬 (drift 방지)
from loaders.url_parsing import split_image_video_urls  # noqa: E402
from dotenv import load_dotenv

load_dotenv()


def _connect_raw():
    return pymysql.connect(
        host=os.environ["STARROCKS_HOST"],
        port=int(os.environ["STARROCKS_PORT"]),
        user=os.environ["STARROCKS_USER"],
        password=os.environ["STARROCKS_PASSWORD"],
        database=os.environ["STARROCKS_RAW_DATABASE"],
        cursorclass=pymysql.cursors.DictCursor,
        charset="utf8mb4",
    )


def _connect_result():
    return pymysql.connect(
        host=os.environ["STARROCKS_HOST"],
        port=int(os.environ["STARROCKS_PORT"]),
        user=os.environ["STARROCKS_USER"],
        password=os.environ["STARROCKS_PASSWORD"],
        database=os.environ["STARROCKS_RESULT_DATABASE"],
        cursorclass=pymysql.cursors.DictCursor,
        charset="utf8mb4",
    )


_RAW_QUERIES = {
    # 4주 윈도우 (2026-03-30 ~ 2026-04-26): 과거 데이터는 크롤링 미완 — 최근 4주 안정 구간 기준
    # IG raw 4주 픽업 (posting_at)
    "ig_4w": """
        SELECT COUNT(*) AS c
        FROM india_ai_fashion_inatagram_posting
        WHERE DATE(posting_at) >= '2026-03-30'
          AND DATE(posting_at) <= '2026-04-26'
    """,
    # YT raw 4주 픽업 (upload_date YYYYMMDD)
    "yt_4w": """
        SELECT COUNT(*) AS c
        FROM india_ai_fashion_youtube_posting
        WHERE upload_date >= '20260330'
          AND upload_date <= '20260426'
    """,
    # IG download_urls 평균 (4주 윈도우)
    "ig_avg_url_count": """
        SELECT AVG(LENGTH(download_urls) - LENGTH(REPLACE(download_urls, ',', '')) + 1) AS avg_count
        FROM india_ai_fashion_inatagram_posting
        WHERE download_urls IS NOT NULL AND download_urls != ''
          AND DATE(posting_at) >= '2026-03-30' AND DATE(posting_at) <= '2026-04-26'
    """,
    # YT download_urls 평균 (4주 윈도우)
    "yt_avg_url_count": """
        SELECT AVG(LENGTH(download_urls) - LENGTH(REPLACE(download_urls, ',', '')) + 1) AS avg_count
        FROM india_ai_fashion_youtube_posting
        WHERE download_urls IS NOT NULL AND download_urls != ''
          AND upload_date >= '20260330' AND upload_date <= '20260426'
    """,
    # IG video 포함 post 수 (.mp4/.mov/.webm/.m4v 확장자 패턴)
    "ig_posts_with_video": """
        SELECT COUNT(*) AS c
        FROM india_ai_fashion_inatagram_posting
        WHERE DATE(posting_at) >= '2026-03-30' AND DATE(posting_at) <= '2026-04-26'
          AND download_urls IS NOT NULL AND download_urls != ''
          AND (download_urls LIKE '%.mp4%' OR download_urls LIKE '%.mov%'
               OR download_urls LIKE '%.webm%' OR download_urls LIKE '%.m4v%')
    """,
    # IG download_urls 전수 fetch (python 측 분류용)
    "ig_download_urls_fetch": """
        SELECT download_urls
        FROM india_ai_fashion_inatagram_posting
        WHERE DATE(posting_at) >= '2026-03-30' AND DATE(posting_at) <= '2026-04-26'
          AND download_urls IS NOT NULL AND download_urls != ''
    """,
    # IG image-only post (jpg/jpeg 만, video 무) 수
    "ig_posts_image_only": """
        SELECT COUNT(*) AS c
        FROM india_ai_fashion_inatagram_posting
        WHERE DATE(posting_at) >= '2026-03-30' AND DATE(posting_at) <= '2026-04-26'
          AND download_urls IS NOT NULL AND download_urls != ''
          AND download_urls NOT LIKE '%.mp4%' AND download_urls NOT LIKE '%.mov%'
          AND download_urls NOT LIKE '%.webm%' AND download_urls NOT LIKE '%.m4v%'
    """,
    # raw 전체 (cumulative) — 참고
    "ig_total": "SELECT COUNT(*) AS c FROM india_ai_fashion_inatagram_posting",
    "yt_total": "SELECT COUNT(*) AS c FROM india_ai_fashion_youtube_posting",
}

_RESULT_QUERIES = {
    # 최근 7일 적재 (computed_at) item — source 별
    "item_7d_by_source": """
        SELECT source, COUNT(DISTINCT source_post_id) AS c
        FROM item
        WHERE computed_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
        GROUP BY source
    """,
    # 7일 통과 canonical_group (vision 통과 = canonicals 채워진 post)
    "canonical_group_7d_by_source": """
        SELECT item_source AS source, COUNT(DISTINCT CONCAT(item_source_post_id, ':', canonical_index)) AS c
        FROM canonical_group
        WHERE computed_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
        GROUP BY item_source
    """,
    # 7일 canonical_object 수 — 각 canonical 의 frame/image member 수 (Gemini 호출 수와 직결)
    "canonical_object_7d_by_source": """
        SELECT item_source AS source, COUNT(*) AS c
        FROM canonical_object
        WHERE computed_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
        GROUP BY item_source
    """,
    # vision 통과 post 수 (distinct source_post_id with canonical_group)
    "vision_passed_post_7d": """
        SELECT item_source AS source, COUNT(DISTINCT item_source_post_id) AS c
        FROM canonical_group
        WHERE computed_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
        GROUP BY item_source
    """,
}


def _classify_ig_urls(rows: list[dict]) -> dict[str, float]:
    """raw row 의 download_urls 를 production policy 와 동일하게 image/video 분류.

    `loaders.url_parsing.split_image_video_urls` single source 사용 — production daily run
    의 통계와 byte-identical 정렬 보장 (`feedback_ig_carousel_video_share` 45.7% video share).
    """
    total_image = 0
    total_video = 0
    posts_with_video = 0
    posts_image_only = 0
    video_post_video_count = 0
    for row in rows:
        urls = [u.strip() for u in (row.get("download_urls") or "").split(",") if u.strip()]
        images, videos = split_image_video_urls(urls)
        img = len(images)
        vid = len(videos)
        total_image += img
        total_video += vid
        if vid > 0:
            posts_with_video += 1
            video_post_video_count += vid
        else:
            posts_image_only += 1
    n = len(rows)
    return {
        "n_posts": n,
        "total_image_urls": total_image,
        "total_video_urls": total_video,
        "posts_with_video": posts_with_video,
        "posts_image_only": posts_image_only,
        "avg_image_per_post": total_image / n if n else 0,
        "avg_video_per_post": total_video / n if n else 0,
        "avg_video_per_video_post": (
            video_post_video_count / posts_with_video if posts_with_video else 0
        ),
    }


def main() -> None:
    print("=== Raw DB (4-week window 2026-03-30 ~ 2026-04-26) ===")
    with _connect_raw() as conn:
        for name, sql in _RAW_QUERIES.items():
            with conn.cursor() as cur:
                cur.execute(sql)
                rows = cur.fetchall()
            if name == "ig_download_urls_fetch":
                stats = _classify_ig_urls(rows)
                print(f"\n  -- IG url 분류 (python-side):")
                for k, v in stats.items():
                    print(f"    {k}: {v}")
            elif "weekly_breakdown" in name:
                print(f"\n  -- {name}")
                for row in rows:
                    print(f"    {row}")
            else:
                row = rows[0] if rows else {}
                val = row.get("c") if "c" in row else row.get("avg_count")
                print(f"  {name}: {val}")

    print("\n=== Result DB (last 7 days, computed_at) ===")
    with _connect_result() as conn:
        for name, sql in _RESULT_QUERIES.items():
            print(f"\n  -- {name}")
            with conn.cursor() as cur:
                cur.execute(sql)
                rows = cur.fetchall()
            for row in rows:
                print(f"    {row}")


if __name__ == "__main__":
    main()
