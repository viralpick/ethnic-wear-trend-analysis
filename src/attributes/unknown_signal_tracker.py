"""Unknown hashtag 자동 감지 (spec §4.2).

v1 스코프:
- HASHTAG-ONLY (caption free-text mining X, LLM/VLM X)
- day-bucketed 누적 → 가장 최근 3일만 유지 → count_3day = bucket 합
- threshold ≥ 10 이면 UnknownAttributeSignal 로 surface (reviewed=False)
- outputs/unknown_signals.json 에 atomic rename 으로 영속
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

from attributes.mapping_tables import all_known_hashtags
from contracts.normalized import NormalizedContentItem
from contracts.output import UnknownAttributeSignal
from utils.dates import previous_n_calendar_days
from utils.io import write_json_atomic

_WINDOW_DAYS = 3
_SURFACE_THRESHOLD = 10

# 파일 포맷:
# {
#   "bandhani": {
#     "buckets": {"2026-04-19": 3, "2026-04-21": 8},
#     "likely_category": "technique?",
#     "reviewed": false
#   }
# }


def _normalize_tag(raw: str) -> str:
    return raw.lstrip("#").lower()


def collect_unknown_hashtag_counts(
    items: list[NormalizedContentItem],
) -> Counter[str]:
    """오늘 포스트의 해시태그 중 mapping_tables 의 어느 해시태그에도 안 잡히는 것만 카운트."""
    known = all_known_hashtags()
    counter: Counter[str] = Counter()
    for item in items:
        for raw_tag in item.hashtags:
            tag = _normalize_tag(raw_tag)
            if tag and tag not in known:
                counter[tag] += 1
    return counter


def _load_state(path: Path) -> dict[str, dict[str, Any]]:
    """기존 state 로드. 구버전/오염 엔트리(buckets 키 없음)는 무시."""
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    if not isinstance(raw, dict):
        return {}
    cleaned: dict[str, dict[str, Any]] = {}
    for tag, entry in raw.items():
        if isinstance(entry, dict) and isinstance(entry.get("buckets"), dict):
            cleaned[tag] = entry
    return cleaned


def _prune_outside_window(
    state: dict[str, dict[str, Any]], today: date
) -> None:
    """윈도우 (_WINDOW_DAYS 일) 밖 버킷 제거. 버킷이 비면 태그 자체 제거. In-place."""
    valid_iso = {d.isoformat() for d in previous_n_calendar_days(today, _WINDOW_DAYS)}
    for tag in list(state.keys()):
        buckets = state[tag]["buckets"]
        state[tag]["buckets"] = {d: c for d, c in buckets.items() if d in valid_iso}
        if not state[tag]["buckets"]:
            del state[tag]


def _merge_today(
    state: dict[str, dict[str, Any]], new_counts: Counter[str], today: date
) -> None:
    """오늘 날짜 bucket 에 new_counts 추가. In-place."""
    today_iso = today.isoformat()
    for tag, count in new_counts.items():
        entry = state.setdefault(
            tag,
            {"buckets": {}, "likely_category": None, "reviewed": False},
        )
        entry["buckets"][today_iso] = entry["buckets"].get(today_iso, 0) + count


def _to_signals(state: dict[str, dict[str, Any]]) -> list[UnknownAttributeSignal]:
    """threshold 이상만 surface. count_3day 는 retained bucket 합 (누적 total 아님)."""
    signals: list[UnknownAttributeSignal] = []
    for tag, entry in state.items():
        buckets: dict[str, int] = entry["buckets"]
        count_3day = sum(buckets.values())
        if count_3day < _SURFACE_THRESHOLD:
            continue
        first_seen = min(date.fromisoformat(d) for d in buckets)
        signals.append(
            UnknownAttributeSignal(
                tag=f"#{tag}",
                count_3day=count_3day,
                first_seen=first_seen,
                likely_category=entry.get("likely_category"),
                reviewed=bool(entry.get("reviewed", False)),
            )
        )
    return signals


def run_tracker(
    items: list[NormalizedContentItem],
    path: Path,
    today: date,
) -> list[UnknownAttributeSignal]:
    """엔드-투-엔드: load → prune → merge → persist (atomic) → surface signals."""
    new_counts = collect_unknown_hashtag_counts(items)
    state = _load_state(path)
    _prune_outside_window(state, today)
    _merge_today(state, new_counts, today)
    write_json_atomic(path, state)
    return _to_signals(state)
