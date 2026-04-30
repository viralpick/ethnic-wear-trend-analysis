from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

from attributes.unknown_signal_tracker import run_tracker
from contracts.common import ContentSource
from contracts.normalized import NormalizedContentItem


def _make(hashtags: list[str], text: str = "", post_date: datetime | None = None) -> NormalizedContentItem:
    return NormalizedContentItem(
        source=ContentSource.INSTAGRAM,
        source_post_id="t",
        text_blob=text,
        hashtags=hashtags,
        image_urls=[],
        # IST date 가 4/21 가 되도록 UTC 기준 4/20 18:30 (UTC) = 4/21 00:00 IST
        post_date=post_date or datetime(2026, 4, 20, 18, 30),
        engagement_raw_count=0,
    )


def test_empty_state_first_insert_below_threshold(tmp_path: Path) -> None:
    path = tmp_path / "signals.json"
    signals = run_tracker(
        [_make(["#handloom", "#indianwear"])],
        path,
        today=date(2026, 4, 21),
    )

    # threshold(10) 미만이라 surface 되지 않는다.
    assert signals == []
    state = json.loads(path.read_text())
    assert "handloom" in state
    assert state["handloom"]["buckets"]["2026-04-21"] == 1


def test_threshold_crossing_surfaces_signal(tmp_path: Path) -> None:
    path = tmp_path / "signals.json"
    items = [_make(["#handloom"]) for _ in range(10)]

    signals = run_tracker(items, path, today=date(2026, 4, 21))

    assert len(signals) == 1
    assert signals[0].tag == "#handloom"
    assert signals[0].count_3day == 10
    assert signals[0].reviewed is False


def test_three_day_window_drops_old_buckets(tmp_path: Path) -> None:
    """v2: bucket key = post_date IST. anchor = max(post_date) 기준 last 3 days 만 keep.

    Day 1 (post_date=4/18) 5건 → state 에 4/18 bucket 5
    Day 5 (post_date=4/22) 1건 → anchor=4/22, last 3 days = 4/20/21/22 → 4/18 prune
    """
    path = tmp_path / "signals.json"
    # post_date 4/18 IST = UTC 4/17 18:30
    run_tracker(
        [_make(["#handloom"], post_date=datetime(2026, 4, 17, 18, 30)) for _ in range(5)],
        path,
        today=date(2026, 4, 18),
    )
    # post_date 4/22 IST = UTC 4/21 18:30
    signals = run_tracker(
        [_make(["#handloom"], post_date=datetime(2026, 4, 21, 18, 30))],
        path,
        today=date(2026, 4, 22),
    )

    state = json.loads(path.read_text())
    assert "2026-04-18" not in state["handloom"]["buckets"]
    assert state["handloom"]["buckets"]["2026-04-22"] == 1
    # count_3day = 1 이라 surface 되지 않는다.
    assert signals == []


def test_first_seen_uses_post_date_not_load_date_2(tmp_path: Path) -> None:
    """v2 회귀 방지: first_seen 이 적재일이 아닌 post_date IST. 옛 v1 은 today bucket
    이라 backfill 시 모든 first_seen=today 였던 버그."""
    path = tmp_path / "signals.json"
    # 12개 post — post_date 다양 (4/19, 4/20, 4/21 IST 분포). 적재일은 4/29 (오늘).
    items = [
        _make(["#handloom"], post_date=datetime(2026, 4, 18, 18, 30)),  # IST 4/19
        _make(["#handloom"], post_date=datetime(2026, 4, 19, 18, 30)),  # IST 4/20
    ] * 5  # ×5 → 10건
    signals = run_tracker(items, path, today=date(2026, 4, 29))

    assert len(signals) == 1
    # first_seen = 가장 오래된 post_date IST (4/19), 적재일 (4/29) X
    assert signals[0].first_seen == date(2026, 4, 19)
    assert signals[0].count_3day == 10
    state = json.loads(path.read_text())
    # bucket 이 post_date 별로 분리됐는지
    assert "2026-04-19" in state["handloom"]["buckets"]
    assert "2026-04-20" in state["handloom"]["buckets"]
    # 적재일 (4/29) bucket 은 없음 (post 의 post_date 가 4/29 가 아니므로)
    assert "2026-04-29" not in state["handloom"]["buckets"]


def test_caption_text_ignored_hashtag_only(tmp_path: Path) -> None:
    path = tmp_path / "signals.json"
    # 캡션에 "handloom" 단어만 있고 해시태그 없음. v1 은 hashtag-only 이므로 무시.
    item = _make([], text="beautiful handloom scarf today")

    signals = run_tracker([item], path, today=date(2026, 4, 21))

    assert signals == []
    state = json.loads(path.read_text())
    assert state == {}
