"""Unknown hashtag 자동 감지 (spec §4.2).

v2 스코프 (2026-04-30):
- HASHTAG-ONLY (caption free-text mining X, LLM/VLM X)
- bucket key = **post 의 IST post_date** (적재일이 아니라 게시일). 옛 v1 은 적재일을
  bucket 키로 써서 backfill 시 12주 분 hashtag 가 모두 한 bucket 에 모임 → first_seen
  이 적재일로 되어버리는 버그. v2 는 post_date 기준이라 "최초 발견일" 의미 정합.
- 가장 최근 3일 (max post_date 기준) bucket 만 유지 → count_3day = bucket 합
- threshold ≥ 10 이면 UnknownAttributeSignal 로 surface (reviewed=False)
- outputs/unknown_signals.json 에 atomic rename 으로 영속
"""
from __future__ import annotations

import json
from collections import defaultdict
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


def _post_date_ist(item: NormalizedContentItem) -> date | None:
    """post_date (UTC) → IST date. None 가드."""
    if item.post_date is None:
        return None
    # IST 변환 — 단순화: tzinfo 가 있으면 +5:30 offset 적용, 없으면 UTC 가정
    from datetime import timedelta, timezone
    pd = item.post_date
    if pd.tzinfo is None:
        pd = pd.replace(tzinfo=timezone.utc)
    ist = pd.astimezone(timezone(timedelta(hours=5, minutes=30)))
    return ist.date()


def collect_unknown_hashtag_counts(
    items: list[NormalizedContentItem],
) -> dict[tuple[str, str], int]:
    """post 별 hashtag 중 mapping 안 잡히는 것 (tag, post_date_iso) 별 카운트.

    v2: bucket key 가 post_date IST 라서 backfill 시 다양한 post_date 별로 bucket 분리.
    """
    known = all_known_hashtags()
    counter: dict[tuple[str, str], int] = defaultdict(int)
    for item in items:
        pd = _post_date_ist(item)
        if pd is None:
            continue
        pd_iso = pd.isoformat()
        for raw_tag in item.hashtags:
            tag = _normalize_tag(raw_tag)
            if tag and tag not in known:
                counter[(tag, pd_iso)] += 1
    return dict(counter)


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
    state: dict[str, dict[str, Any]], anchor: date
) -> None:
    """윈도우 (_WINDOW_DAYS 일) 밖 버킷 제거. 버킷이 비면 태그 자체 제거. In-place.

    v2: anchor = max(post_date in batch). backfill 시 가장 최근 post_date 기준 last 3 days
    만 keep. 옛 v1 은 today (적재일) 기준이라 backfill 시 옛 post_date 모두 prune 되던 버그.
    """
    valid_iso = {d.isoformat() for d in previous_n_calendar_days(anchor, _WINDOW_DAYS)}
    for tag in list(state.keys()):
        buckets = state[tag]["buckets"]
        state[tag]["buckets"] = {d: c for d, c in buckets.items() if d in valid_iso}
        if not state[tag]["buckets"]:
            del state[tag]


def _merge_buckets(
    state: dict[str, dict[str, Any]], new_counts: dict[tuple[str, str], int]
) -> None:
    """(tag, post_date_iso) 별 카운트를 state.buckets[post_date_iso] 에 더함. In-place.

    v2: bucket key = post_date IST (적재일 X). 옛 v1 의 _merge_today (모든 카운트가 today
    bucket 에 모이던 버그) 대체.
    """
    for (tag, pd_iso), count in new_counts.items():
        entry = state.setdefault(
            tag,
            {"buckets": {}, "likely_category": None, "reviewed": False},
        )
        entry["buckets"][pd_iso] = entry["buckets"].get(pd_iso, 0) + count


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
    """엔드-투-엔드: load → merge → prune → persist → surface.

    v2: bucket key = post_date IST. anchor = max(post_date in batch ∪ today). backfill
    시 12주 분 다양한 post_date 가 bucket 별 분리 → first_seen 정확.
    """
    new_counts = collect_unknown_hashtag_counts(items)
    state = _load_state(path)
    _merge_buckets(state, new_counts)
    # anchor = state ∪ new_counts 의 max post_date IST. today 무시 — backfill 시 적재일이
    # 미래 (모든 post_date 이후) 라도 옛 post_date bucket 들의 last 3 days 가 keep 되게.
    anchor_candidates: list[date] = []
    for (_, pd_iso) in new_counts:
        anchor_candidates.append(date.fromisoformat(pd_iso))
    for entry in state.values():
        for pd_iso in entry["buckets"]:
            anchor_candidates.append(date.fromisoformat(pd_iso))
    anchor = max(anchor_candidates) if anchor_candidates else today
    _prune_outside_window(state, anchor)
    write_json_atomic(path, state)
    return _to_signals(state)
