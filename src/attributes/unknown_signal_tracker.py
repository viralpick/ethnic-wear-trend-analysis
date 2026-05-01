"""Unknown hashtag emergence detector (spec §4.2 / §8.3 v2.2).

v3 (2026-05-01) — emergence rule + co-occurrence + weekly cadence.

Surface 조건 4개 (전부 통과해야 surface):
  1. baseline_window 부재 — 직전 N일 (default 56) 동안 등장 횟수 ≤ floor (default 0)
  2. spike_window 발생  — 최근 M일 (default 14) 동안 등장 횟수 ≥ K (default 3)
  3. ethnic_co_share   — 그 tag 의 post 들 중 매핑된 fashion hashtag 도 같이 가진 비율 ≥ R (default 0.5)
  4. min_posts         — 최소 N posts (default 5) 에서 등장 — measurement stability

옛 v2 (post_date IST bucket + 3일 ≥10건 단순 룰) 폐기. count_3day 의미 변경 →
count_recent_window. anchor 1개 1회 평가 → weekly replay (representative_weekly 와
정합한 cadence).

state file (outputs/unknown_signals_weekly.json):
  {
    "weeks": {
      "2026-04-20": [{tag, count_recent_window, first_seen, week_start_date, ...}, ...],
      ...
    },
    "co_occur": {tag: [n_with_known_fashion, n_total]},
    "buckets": {tag: {post_date_iso: count}}
  }

co_occur / buckets 는 cumulative (prune 안 함 — replay 용). weeks 는 anchor 별 평가 결과.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from attributes.mapping_tables import all_known_hashtags
from contracts.normalized import NormalizedContentItem
from contracts.output import UnknownAttributeSignal
from utils.io import write_json_atomic

# v2.2 default 값. CLI override 가능.
DEFAULT_BASELINE_DAYS = 56      # N — baseline window
DEFAULT_SPIKE_DAYS = 14         # M — spike window
DEFAULT_SPIKE_THRESHOLD = 3     # K — spike count 임계
DEFAULT_BASELINE_FLOOR = 0      # baseline 등장 허용치
DEFAULT_CO_SHARE = 0.5          # R — ethnic_co_share 임계
DEFAULT_MIN_POSTS = 5           # measurement stability 임계


@dataclass(frozen=True)
class EmergenceParams:
    """surface 룰 파라미터 — 호출자가 settings.local 또는 CLI 로 override."""
    baseline_days: int = DEFAULT_BASELINE_DAYS
    spike_days: int = DEFAULT_SPIKE_DAYS
    spike_threshold: int = DEFAULT_SPIKE_THRESHOLD
    baseline_floor: int = DEFAULT_BASELINE_FLOOR
    co_share: float = DEFAULT_CO_SHARE
    min_posts: int = DEFAULT_MIN_POSTS


def _normalize_tag(raw: str) -> str:
    return raw.lstrip("#").lower()


def _post_date_ist(item: NormalizedContentItem) -> date | None:
    """post_date (UTC) → IST date. None 가드."""
    if item.post_date is None:
        return None
    pd = item.post_date
    if pd.tzinfo is None:
        pd = pd.replace(tzinfo=timezone.utc)
    ist = pd.astimezone(timezone(timedelta(hours=5, minutes=30)))
    return ist.date()


def _monday_of(d: date) -> date:
    """주어진 날짜가 속한 주의 월요일 (ISO week start, IST)."""
    return d - timedelta(days=d.weekday())


@dataclass
class EmergenceCounters:
    """append-only counters — replay 용. prune 안 함.

    buckets[tag] = {post_date_iso: count}    # post_date 별 등장 횟수 (post 1개 안 = +N tag instance)
    co_occur[tag] = (n_with_fashion, n_total) # tag 가진 post 중 known fashion hashtag 도 가진 post 수
    known[tag] = bool                          # mapping_tables 매핑된 hashtag 여부
    """
    buckets: dict[str, dict[str, int]]
    co_occur: dict[str, tuple[int, int]]
    known: dict[str, bool]


def build_counters(
    items: list[NormalizedContentItem],
    *,
    base: EmergenceCounters | None = None,
    from_date: date | None = None,
    to_date: date | None = None,
) -> EmergenceCounters:
    """post → counters 누적. base 가 주어지면 그 위에 누적, 없으면 새로 시작.

    from_date/to_date: post_date IST filter (inclusive). None 이면 filter 안 함.

    매핑된 known hashtag 도 buckets / co_occur 에 누적 (hashtag_weekly 적재 source).
    emergence rule 평가 시점에 evaluate_at 가 known 인 것 자동 제외.

    co_occur: 같은 post 안 모든 tag 별 +1 (post-level dedup, 같은 tag 두 번 등장해도 +1).
    buckets:  같은 post 안 모든 tag 별 등장 instance 합 (raw 카운트, post-level dedup X).
    """
    known_set = all_known_hashtags()
    buckets: dict[str, dict[str, int]] = defaultdict(dict)
    co_occur: dict[str, tuple[int, int]] = {}
    known: dict[str, bool] = {}
    if base is not None:
        for tag, b in base.buckets.items():
            buckets[tag] = dict(b)
        co_occur = dict(base.co_occur)
        known = dict(base.known)

    for item in items:
        pd = _post_date_ist(item)
        if pd is None:
            continue
        if from_date is not None and pd < from_date:
            continue
        if to_date is not None and pd > to_date:
            continue
        pd_iso = pd.isoformat()
        normalized = [_normalize_tag(t) for t in item.hashtags]
        normalized = [t for t in normalized if t]
        if not normalized:
            continue
        has_known_fashion = any(t in known_set for t in normalized)
        # buckets — raw instance count (모든 tag, known 포함)
        for t in normalized:
            buckets[t][pd_iso] = buckets[t].get(pd_iso, 0) + 1
            known[t] = t in known_set
        # co_occur — post-level dedup (모든 tag)
        for t in set(normalized):
            n_kf, n_tot = co_occur.get(t, (0, 0))
            co_occur[t] = (n_kf + (1 if has_known_fashion else 0), n_tot + 1)
    return EmergenceCounters(buckets=dict(buckets), co_occur=co_occur, known=known)


def evaluate_at(
    counters: EmergenceCounters,
    anchor: date,
    params: EmergenceParams = EmergenceParams(),
) -> list[UnknownAttributeSignal]:
    """anchor (IST date) 시점에 emergence 룰 평가. surface 통과한 tag 만 list 로 반환.

    spike_window  = [anchor - spike_days + 1, anchor]
    baseline_window = [anchor - spike_days - baseline_days + 1, anchor - spike_days]
    week_start_date = anchor 가 속한 주의 월요일
    """
    week_start = _monday_of(anchor)
    spike_start = anchor - timedelta(days=params.spike_days - 1)
    baseline_end = spike_start - timedelta(days=1)
    baseline_start = baseline_end - timedelta(days=params.baseline_days - 1)
    out: list[UnknownAttributeSignal] = []
    for tag, b in counters.buckets.items():
        # known mapping hashtag 은 surface 대상 아님
        if counters.known.get(tag, False):
            continue
        # window-별 합산
        baseline_sum = sum(
            c for d, c in b.items()
            if baseline_start.isoformat() <= d <= baseline_end.isoformat()
        )
        if baseline_sum > params.baseline_floor:
            continue
        spike_sum = sum(
            c for d, c in b.items()
            if spike_start.isoformat() <= d <= anchor.isoformat()
        )
        if spike_sum < params.spike_threshold:
            continue
        # min_posts + co_share — co_occur 는 post-level dedup count
        n_kf, n_tot = counters.co_occur.get(tag, (0, 0))
        if n_tot < params.min_posts:
            continue
        if n_kf / n_tot < params.co_share:
            continue
        first_seen = min(date.fromisoformat(d) for d in b)
        out.append(UnknownAttributeSignal(
            tag=f"#{tag}",
            week_start_date=week_start,
            count_recent_window=spike_sum,
            first_seen=first_seen,
            likely_category=None,
            reviewed=False,
        ))
    return out


# ---- persistence (outputs/unknown_signals_weekly.json) ----

def _serialize_counters(counters: EmergenceCounters) -> dict[str, Any]:
    return {
        "buckets": counters.buckets,
        "co_occur": {k: list(v) for k, v in counters.co_occur.items()},
        "known": counters.known,
    }


def _deserialize_counters(raw: dict[str, Any]) -> EmergenceCounters:
    if not isinstance(raw, dict):
        return EmergenceCounters(buckets={}, co_occur={}, known={})
    buckets_raw = raw.get("buckets") or {}
    co_raw = raw.get("co_occur") or {}
    known_raw = raw.get("known") or {}
    buckets = {
        tag: {d: int(c) for d, c in (b or {}).items()}
        for tag, b in buckets_raw.items()
        if isinstance(b, dict)
    }
    co_occur = {
        tag: (int(v[0]), int(v[1]))
        for tag, v in co_raw.items()
        if isinstance(v, list) and len(v) == 2
    }
    known = {tag: bool(v) for tag, v in known_raw.items()} if isinstance(known_raw, dict) else {}
    return EmergenceCounters(buckets=buckets, co_occur=co_occur, known=known)


def _serialize_signal(s: UnknownAttributeSignal) -> dict[str, Any]:
    return {
        "tag": s.tag,
        "week_start_date": s.week_start_date.isoformat(),
        "count_recent_window": s.count_recent_window,
        "first_seen": s.first_seen.isoformat(),
        "likely_category": s.likely_category,
        "reviewed": s.reviewed,
    }


def load_state(path: Path) -> tuple[EmergenceCounters, dict[str, list[dict[str, Any]]]]:
    """outputs/unknown_signals_weekly.json — counters + weeks (week_start_iso → signals dict list)."""
    empty = EmergenceCounters(buckets={}, co_occur={}, known={})
    if not path.exists():
        return empty, {}
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError:
        return empty, {}
    if not isinstance(raw, dict):
        return empty, {}
    counters = _deserialize_counters(raw)
    weeks = raw.get("weeks") or {}
    if not isinstance(weeks, dict):
        weeks = {}
    return counters, weeks


def save_state(
    path: Path,
    counters: EmergenceCounters,
    weeks: dict[str, list[dict[str, Any]]],
) -> None:
    payload = _serialize_counters(counters)
    payload["weeks"] = weeks
    write_json_atomic(path, payload)


def compute_weekly_emergence(
    items: list[NormalizedContentItem],
    anchor: date,
    *,
    params: EmergenceParams = EmergenceParams(),
) -> tuple[EmergenceCounters, list[UnknownAttributeSignal]]:
    """매 주 anchor 별 (week_counters, signals) 산출.

    Returns:
      week_counters: 그 주 (week_start ~ anchor) 안 post 만 카운트. hashtag_weekly
        적재 source — 매 주 row = 그 주 안 hashtag 분포 (partition by week).
      signals: emergence rule 통과 surface. evaluate 는 전체 history counters
        (filter 없음) 로 baseline + spike windowed sum + cumulative co_occur 평가.
        co_occur 를 전체 history 로 측정해야 min_posts threshold stability 충분.
    """
    week_start = _monday_of(anchor)
    # week-only counters — emit_hashtag_weekly 에 dump (partition by week)
    week_counters = build_counters(items, from_date=week_start, to_date=anchor)
    # emergence eval — 전체 history counters. evaluate_at 가 windowed sum 으로 평가,
    # co_occur 는 cumulative (stability).
    eval_counters = build_counters(items)
    signals = evaluate_at(eval_counters, anchor, params)
    return week_counters, signals


def run_weekly_replay(
    items: list[NormalizedContentItem],
    path: Path,
    anchor: date,
    *,
    params: EmergenceParams = EmergenceParams(),
    today: date | None = None,
) -> list[UnknownAttributeSignal]:
    """엔드-투-엔드: counters fresh build → anchor 평가 → weeks 누적 저장 → signals 반환.

    매 호출마다 counters 를 items 로 fresh build (옛 state 무시). 같은 batch 를 여러
    anchor 에 호출해도 중복 카운트 risk 없음. caller 가 일반적으로 enriched 전체를 매번
    보내면 안전 — overhead 0.1초 수준.

    weeks (anchor 별 surface 결과 누적) 는 옛 state 에서 read 후 anchor 의 결과로
    덮어쓰기 — weekly orchestrator 가 W1~W12 호출하면 12 entries 누적.

    today: legacy compat. anchor 와 별개. 미사용.
    """
    _, weeks = load_state(path)
    counters = build_counters(items)
    signals = evaluate_at(counters, anchor, params)
    week_start_iso = _monday_of(anchor).isoformat()
    weeks[week_start_iso] = [_serialize_signal(s) for s in signals]
    save_state(path, counters, weeks)
    return signals


# legacy alias — 옛 caller (run_attribute_pipeline / run_daily_pipeline) 호환.
def run_tracker(
    items: list[NormalizedContentItem],
    path: Path,
    today: date,
    *,
    params: EmergenceParams = EmergenceParams(),
) -> list[UnknownAttributeSignal]:
    """legacy compat. anchor = today. weekly replay 미사용 시 1회 평가."""
    anchor_candidates: list[date] = [today]
    for it in items:
        pd = _post_date_ist(it)
        if pd is not None:
            anchor_candidates.append(pd)
    anchor = max(anchor_candidates)
    return run_weekly_replay(items, path, anchor, params=params)
