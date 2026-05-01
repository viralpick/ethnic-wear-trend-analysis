"""unknown_signal_tracker v3 — emergence rule + co-occurrence + weekly replay.

v3 surface 조건 (전부 통과):
1. baseline_window (default 56일) 등장 ≤ baseline_floor (default 0)
2. spike_window (default 14일) 등장 ≥ spike_threshold (default 3)
3. ethnic_co_share ≥ co_share (default 0.5) — tag 의 post 들 중 known fashion hashtag
   같이 가진 비율
4. n_posts ≥ min_posts (default 5)

옛 v2 (≥10 / 3일 단순 룰) 폐기 — count_3day → count_recent_window.
"""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

from attributes.unknown_signal_tracker import (
    EmergenceParams,
    build_counters,
    evaluate_at,
    load_state,
    run_tracker,
    run_weekly_replay,
)
from contracts.common import ContentSource
from contracts.normalized import NormalizedContentItem


def _make(
    hashtags: list[str],
    *,
    post_date: datetime = datetime(2026, 4, 20, 18, 30),  # IST 4/21 00:00
    post_id: str = "p",
) -> NormalizedContentItem:
    return NormalizedContentItem(
        source=ContentSource.INSTAGRAM,
        source_post_id=post_id,
        text_blob="",
        hashtags=hashtags,
        image_urls=[],
        post_date=post_date,
        engagement_raw_count=0,
    )


# ---- positive: 4 조건 모두 통과 ----

def test_emergence_surfaces_when_all_conditions_pass(tmp_path: Path) -> None:
    """baseline 부재 + spike ≥3 + co_share ≥0.5 + min_posts ≥5 → surface."""
    path = tmp_path / "signals.json"
    # 6 posts, 모두 spike window 안 (anchor=4/26 → spike=4/13~4/26).
    # 각 post 가 #handloom (unknown) + #saree (known) 가져 co_share=1.0.
    items = [
        _make(["#handloom", "#saree"], post_date=datetime(2026, 4, 19, 18, 30), post_id=f"p{i}")
        for i in range(6)
    ]
    signals = run_weekly_replay(items, path, anchor=date(2026, 4, 26))

    assert len(signals) == 1
    sig = signals[0]
    assert sig.tag == "#handloom"
    assert sig.count_recent_window == 6  # spike window 안 등장 instance 합
    assert sig.week_start_date == date(2026, 4, 20)  # Mon of 4/26 week
    assert sig.first_seen == date(2026, 4, 20)  # IST


# ---- negative: 각 조건 별 fail ----

def test_baseline_floor_violation(tmp_path: Path) -> None:
    """baseline window 에 1번 등장하면 floor=0 통과 못함."""
    path = tmp_path / "signals.json"
    # baseline window = anchor (4/26) - 14 - 56 일 ~ -15일 = 2026-02-17 ~ 2026-04-12
    # 4/1 에 1번 등장
    items = (
        [_make(["#handloom", "#saree"], post_date=datetime(2026, 3, 31, 18, 30), post_id="old")]
        + [_make(["#handloom", "#saree"], post_date=datetime(2026, 4, 19, 18, 30), post_id=f"p{i}")
           for i in range(6)]
    )
    signals = run_weekly_replay(items, path, anchor=date(2026, 4, 26))

    # baseline 1 > floor 0 → 통과 못함
    assert signals == []


def test_spike_threshold_not_met(tmp_path: Path) -> None:
    """spike < K=3 이면 surface 안 됨."""
    path = tmp_path / "signals.json"
    # 2 posts in spike window — K=3 미달
    items = [
        _make(["#handloom", "#saree"], post_date=datetime(2026, 4, 19, 18, 30), post_id=f"p{i}")
        for i in range(2)
    ]
    signals = run_weekly_replay(items, path, anchor=date(2026, 4, 26))
    assert signals == []


def test_co_share_too_low(tmp_path: Path) -> None:
    """unknown tag 의 post 들이 known fashion hashtag 거의 안 가지면 surface 안 됨."""
    path = tmp_path / "signals.json"
    # 6 posts: 2개만 known fashion (#saree) 같이, 나머지 4개는 unknown only
    items = [
        _make(["#handloom", "#saree"], post_date=datetime(2026, 4, 19, 18, 30), post_id="kf1"),
        _make(["#handloom", "#saree"], post_date=datetime(2026, 4, 20, 18, 30), post_id="kf2"),
        _make(["#handloom"], post_date=datetime(2026, 4, 21, 18, 30), post_id="kf3"),
        _make(["#handloom"], post_date=datetime(2026, 4, 22, 18, 30), post_id="kf4"),
        _make(["#handloom"], post_date=datetime(2026, 4, 23, 18, 30), post_id="kf5"),
        _make(["#handloom"], post_date=datetime(2026, 4, 24, 18, 30), post_id="kf6"),
    ]
    signals = run_weekly_replay(items, path, anchor=date(2026, 4, 26))
    # co_share = 2 / 6 = 0.33 < 0.5 → 통과 못함
    assert signals == []


def test_min_posts_not_met(tmp_path: Path) -> None:
    """post 수 < 5 면 measurement stability 부족 → surface 안 됨."""
    path = tmp_path / "signals.json"
    # 4 posts (min_posts=5 미달)
    items = [
        _make(["#handloom", "#saree"], post_date=datetime(2026, 4, 19, 18, 30), post_id=f"p{i}")
        for i in range(4)
    ]
    signals = run_weekly_replay(items, path, anchor=date(2026, 4, 26))
    assert signals == []


# ---- weekly replay 시나리오 ----

def test_weekly_replay_different_anchors_different_results(tmp_path: Path) -> None:
    """같은 enriched 라도 anchor 별로 spike/baseline 다르게 평가됨."""
    path = tmp_path / "signals.json"
    items = [
        _make(["#handloom", "#saree"], post_date=datetime(2026, 4, 19, 18, 30), post_id=f"p{i}")
        for i in range(6)
    ]
    # anchor=4/26 → spike=4/13~4/26 → 6 surface
    s1 = run_weekly_replay(items, path, anchor=date(2026, 4, 26))
    assert len(s1) == 1

    # anchor=5/12 → spike=4/29~5/12 → 0 (spike 밖)
    s2 = run_weekly_replay([], path, anchor=date(2026, 5, 12))
    assert s2 == []


def test_state_persistence_round_trip(tmp_path: Path) -> None:
    """save → load → 같은 counters 복원."""
    path = tmp_path / "signals.json"
    items = [
        _make(["#handloom", "#saree"], post_date=datetime(2026, 4, 19, 18, 30), post_id=f"p{i}")
        for i in range(6)
    ]
    run_weekly_replay(items, path, anchor=date(2026, 4, 26))

    counters_loaded, weeks = load_state(path)
    assert "handloom" in counters_loaded.buckets
    assert sum(counters_loaded.buckets["handloom"].values()) == 6
    assert counters_loaded.co_occur["handloom"] == (6, 6)
    assert "2026-04-20" in weeks  # Mon of 4/26


def test_caption_text_ignored_hashtag_only(tmp_path: Path) -> None:
    """v3 도 hashtag-only — caption text 안의 단어는 카운트 X."""
    path = tmp_path / "signals.json"
    items = [
        NormalizedContentItem(
            source=ContentSource.INSTAGRAM,
            source_post_id=f"p{i}",
            text_blob="beautiful handloom scarf today",
            hashtags=[],
            image_urls=[],
            post_date=datetime(2026, 4, 19, 18, 30),
            engagement_raw_count=0,
        )
        for i in range(6)
    ]
    signals = run_weekly_replay(items, path, anchor=date(2026, 4, 26))
    assert signals == []


def test_legacy_run_tracker_compat(tmp_path: Path) -> None:
    """옛 caller (today=anchor) 호환 — run_tracker 가 1회 평가."""
    path = tmp_path / "signals.json"
    items = [
        _make(["#handloom", "#saree"], post_date=datetime(2026, 4, 19, 18, 30), post_id=f"p{i}")
        for i in range(6)
    ]
    signals = run_tracker(items, path, today=date(2026, 4, 26))
    assert len(signals) == 1
    assert signals[0].count_recent_window == 6


# ---- params override ----

def test_params_override_relaxes_thresholds(tmp_path: Path) -> None:
    """min_posts=2 / spike_threshold=1 override → 통과."""
    path = tmp_path / "signals.json"
    items = [
        _make(["#handloom", "#saree"], post_date=datetime(2026, 4, 19, 18, 30), post_id=f"p{i}")
        for i in range(2)
    ]
    params = EmergenceParams(min_posts=2, spike_threshold=1)
    signals = run_weekly_replay(items, path, anchor=date(2026, 4, 26), params=params)
    assert len(signals) == 1


# ---- internal: build_counters / evaluate_at 분리 검증 ----

def test_build_counters_separates_buckets_and_co_occur() -> None:
    """post 1개 안의 unknown tag 마다 buckets +N (instance), co_occur post-level dedup +1."""
    items = [
        _make(["#handloom", "#handloom", "#saree"],  # 같은 tag 2번 → buckets +2, co_occur +1
              post_date=datetime(2026, 4, 19, 18, 30), post_id="p1"),
        _make(["#handloom"],
              post_date=datetime(2026, 4, 20, 18, 30), post_id="p2"),
    ]
    c = build_counters(items)
    assert c.buckets["handloom"]["2026-04-20"] == 2  # post p1 IST 4/20
    assert c.buckets["handloom"]["2026-04-21"] == 1  # post p2 IST 4/21
    assert c.co_occur["handloom"] == (1, 2)  # 1 post 가 #saree 같이 / 2 posts total


def test_evaluate_at_window_boundaries() -> None:
    """spike window 경계가 anchor 포함, baseline 경계가 spike 직전."""
    # default: spike_days=14, baseline_days=56
    # anchor=4/26 → spike=[4/13, 4/26], baseline=[2026-02-17, 4/12]
    items = [
        _make(["#handloom", "#saree"], post_date=datetime(2026, 4, 19, 18, 30), post_id=f"p{i}")
        for i in range(6)
    ]
    counters = build_counters(items)
    out = evaluate_at(counters, anchor=date(2026, 4, 26))
    assert len(out) == 1

    # anchor=4/12 → spike=[3/30, 4/12]: 4/20 IST post 들이 spike 밖 → surface 안 됨
    out2 = evaluate_at(counters, anchor=date(2026, 4, 12))
    assert out2 == []
