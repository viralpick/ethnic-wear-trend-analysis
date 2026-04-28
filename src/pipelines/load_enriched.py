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
