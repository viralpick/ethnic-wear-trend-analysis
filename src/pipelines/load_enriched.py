"""enriched.json 파일 로더 + 날짜 필터.

phase=representative 모드에서 여러 batch 의 enriched JSON 파일을 통째로 읽어
posted_at IST 기준으로 [start_date, end_date] 윈도우만 추출.

설계 원칙:
- 같은 source_post_id 가 여러 파일에 있으면 마지막 파일 (glob iteration 순서) 우선.
  enriched.json 은 대부분 unique 라 dedup 은 안전망.
- post_date 는 normalized.post_date — IG: posting_at 파싱, YT: upload_date 파싱 (raw_loader.py).
  둘 다 datetime 으로 정규화되어 있으니 IST 변환 후 date 비교.
"""
from __future__ import annotations

import glob
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from contracts.enriched import EnrichedContentItem
from utils.logging import get_logger

logger = get_logger(__name__)

_IST = timezone(timedelta(hours=5, minutes=30))


def load_enriched_files(glob_pattern: str) -> list[EnrichedContentItem]:
    """glob 패턴으로 enriched.json 파일들 로드 → flat list.

    같은 source_post_id 중복 시 마지막 파일 우선 (glob 정렬 순서).
    """
    paths = sorted(glob.glob(glob_pattern))
    if not paths:
        logger.warning("load_enriched_files no_match pattern=%s", glob_pattern)
        return []

    by_post_id: dict[str, EnrichedContentItem] = {}
    for path in paths:
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("load_enriched_skip path=%s reason=%r", path, exc)
            continue
        for raw in data:
            try:
                item = EnrichedContentItem.model_validate(raw)
            except Exception as exc:
                logger.warning(
                    "load_enriched_validate_skip path=%s reason=%r", path, exc,
                )
                continue
            by_post_id[item.normalized.source_post_id] = item

    logger.info(
        "load_enriched_files loaded=%d files=%d pattern=%s",
        len(by_post_id), len(paths), glob_pattern,
    )
    return list(by_post_id.values())


def _to_ist_date(dt: datetime) -> date:
    """datetime → IST date. tzinfo 없으면 UTC 가정 후 IST 변환."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(_IST).date()


def load_raw_post_urls() -> dict[str, str]:
    """raw DB 의 IG/YT 테이블에서 post_id → 외부 URL 매핑 로드. 실패 시 빈 dict.

    dedup_by_raw_url 의 사전 입력. raw DB 의 동일 url 가진 다른 post_id 들 (재크롤링
    등) 을 중복으로 인식하기 위해 필요. STARROCKS_* 환경변수 미설정 시 빈 dict 반환
    (dedup 미적용).
    """
    import os
    out: dict[str, str] = {}
    try:
        import pymysql  # noqa: I001
        from dotenv import load_dotenv
        from pathlib import Path
        load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")
        if not os.environ.get("STARROCKS_HOST"):
            return out
        conn = pymysql.connect(
            host=os.environ["STARROCKS_HOST"],
            port=int(os.environ.get("STARROCKS_PORT", "9030")),
            user=os.environ["STARROCKS_USER"],
            password=os.environ["STARROCKS_PASSWORD"],
            database=os.environ.get("STARROCKS_RAW_DATABASE", "png"),
            connect_timeout=15,
            cursorclass=pymysql.cursors.DictCursor,
        )
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, url FROM india_ai_fashion_inatagram_posting "
                "WHERE url IS NOT NULL AND url != ''"
            )
            for r in cur.fetchall():
                out[r["id"]] = r["url"]
            cur.execute(
                "SELECT id, url FROM india_ai_fashion_youtube_posting "
                "WHERE url IS NOT NULL AND url != ''"
            )
            for r in cur.fetchall():
                out[r["id"]] = r["url"]
        conn.close()
    except Exception as exc:
        logger.warning("load_raw_post_urls failed: %r — dedup will skip", exc)
    return out


def dedup_by_raw_url(
    items: list[EnrichedContentItem],
    post_urls: dict[str, str],
) -> list[EnrichedContentItem]:
    """raw DB url 기준 dedup — 동일 url 가진 item 중 engagement_raw_count 최대 1건만 keep.

    rep phase 에서 enriched 내 url 중복 (re-crawl 등) 으로 cluster 점수 inflate 방지.
    url 미매핑 (post_id 가 raw DB lookup 실패) item 은 모두 유지 (가짜 dedup 방지).
    """
    if not post_urls:
        return items
    by_url: dict[str, list[EnrichedContentItem]] = {}
    no_url: list[EnrichedContentItem] = []
    for it in items:
        pid = it.normalized.source_post_id
        url = post_urls.get(pid)
        if url:
            by_url.setdefault(url, []).append(it)
        else:
            no_url.append(it)
    out: list[EnrichedContentItem] = list(no_url)
    dropped = 0
    for url, group in by_url.items():
        if len(group) > 1:
            dropped += len(group) - 1
        group.sort(key=lambda it: -(it.normalized.engagement_raw_count or 0))
        out.append(group[0])
    logger.info(
        "dedup_by_raw_url in=%d out=%d dropped=%d no_url=%d",
        len(items), len(out), dropped, len(no_url),
    )
    return out


def filter_by_date_range(
    items: list[EnrichedContentItem],
    *,
    start_date: date,
    end_date: date,
) -> list[EnrichedContentItem]:
    """posted_at IST 기준 [start_date, end_date] 포함 범위 필터.

    start_date / end_date 둘 다 inclusive. start_date <= IST(post_date) <= end_date.
    """
    if start_date > end_date:
        raise ValueError(f"start_date {start_date} > end_date {end_date}")
    out = [
        item for item in items
        if start_date <= _to_ist_date(item.normalized.post_date) <= end_date
    ]
    logger.info(
        "filter_by_date_range start=%s end=%s in=%d out=%d",
        start_date, end_date, len(items), len(out),
    )
    return out
