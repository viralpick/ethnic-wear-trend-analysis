"""outputs/{date}/summaries.json + enriched.json + payload.json atomic 쓰기.

atomic 의 의미: 같은 디렉토리의 temp 파일에 쓴 뒤 os.rename 으로 교체.
partial 파일을 소비자가 읽는 걸 방지한다 (spec §10.1 Step 5).
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

from contracts.enriched import EnrichedContentItem
from contracts.output import TrendClusterSummary
from utils.io import write_json_atomic


def _date_dir(output_root: Path, target_date: date) -> Path:
    return output_root / target_date.isoformat()


def write_summaries(
    output_root: Path,
    target_date: date,
    summaries: list[TrendClusterSummary],
    filename: str = "summaries.json",
) -> Path:
    """summaries 리스트를 outputs/{date}/summaries.json 로 atomic 쓰기. 경로 반환."""
    path = _date_dir(output_root, target_date) / filename
    write_json_atomic(path, [s.model_dump(mode="json") for s in summaries])
    return path


def write_enriched(
    output_root: Path,
    target_date: date,
    items: list[EnrichedContentItem],
    filename: str = "enriched.json",
) -> Path:
    """enriched items 를 outputs/{date}/enriched.json 로 audit 용 덤프. 경로 반환."""
    path = _date_dir(output_root, target_date) / filename
    write_json_atomic(path, [i.model_dump(mode="json") for i in items])
    return path


def write_payload(
    output_root: Path,
    target_date: date,
    summaries: list[TrendClusterSummary],
    filename: str = "payload.json",
) -> Path:
    """BE 전달용 payload.json — summaries 를 wrapped 포맷으로 atomic 쓰기."""
    path = _date_dir(output_root, target_date) / filename
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "date": target_date.isoformat(),
        "cluster_count": len(summaries),
        "clusters": [s.model_dump(mode="json") for s in summaries],
    }
    write_json_atomic(path, payload)
    return path
