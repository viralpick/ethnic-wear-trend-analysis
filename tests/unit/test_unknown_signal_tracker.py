from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

from attributes.unknown_signal_tracker import run_tracker
from contracts.common import ContentSource
from contracts.normalized import NormalizedContentItem


def _make(hashtags: list[str], text: str = "") -> NormalizedContentItem:
    return NormalizedContentItem(
        source=ContentSource.INSTAGRAM,
        source_post_id="t",
        text_blob=text,
        hashtags=hashtags,
        image_urls=[],
        post_date=datetime(2026, 4, 21),
        engagement_raw=0,
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
    path = tmp_path / "signals.json"

    # Day 1: 5 occurrences
    run_tracker(
        [_make(["#handloom"]) for _ in range(5)],
        path,
        today=date(2026, 4, 18),
    )
    # Day 5: 1 occurrence — day 1 은 3일 윈도우 밖이라 drop 되어야 한다.
    signals = run_tracker(
        [_make(["#handloom"])],
        path,
        today=date(2026, 4, 22),
    )

    state = json.loads(path.read_text())
    assert "2026-04-18" not in state["handloom"]["buckets"]
    assert state["handloom"]["buckets"]["2026-04-22"] == 1
    # count_3day = 1 이라 surface 되지 않는다.
    assert signals == []


def test_caption_text_ignored_hashtag_only(tmp_path: Path) -> None:
    path = tmp_path / "signals.json"
    # 캡션에 "handloom" 단어만 있고 해시태그 없음. v1 은 hashtag-only 이므로 무시.
    item = _make([], text="beautiful handloom scarf today")

    signals = run_tracker([item], path, today=date(2026, 4, 21))

    assert signals == []
    state = json.loads(path.read_text())
    assert state == {}
