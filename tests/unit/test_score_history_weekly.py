"""WeeklyScoreHistory pinning — pipeline_spec_v1.0 §3.2 / §3.4 (I) weekly bucket.

검증 대상:
- ISO week 키 포맷 (`YYYY-Www`).
- update_weekly → save → reload round-trip (JSON shape 안정).
- get_baseline = 지난 주 score (없으면 None).
- get_trajectory_12w 길이 12 + 부족분 0 패딩 (oldest → newest).
- 같은 주 재호출 시 덮어쓰기 (재처리 의도).
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from scoring.score_history_weekly import (
    WeeklyScoreHistory,
    iso_week_key,
    week_start_monday,
)


def test_iso_week_key_format() -> None:
    # 2026-04-27 (월) 은 ISO 2026-W18 의 시작.
    assert iso_week_key(date(2026, 4, 27)) == "2026-W18"
    # 2026-W01 은 2025-12-29 (월) 부터.
    assert iso_week_key(date(2025, 12, 29)) == "2026-W01"


def test_week_start_monday() -> None:
    # 2026-04-30 (목) → 2026-04-27 (월).
    assert week_start_monday(date(2026, 4, 30)) == date(2026, 4, 27)


def test_update_save_reload_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "score_history_weekly.json"
    h1 = WeeklyScoreHistory(path)
    h1.update_weekly(
        "kurta__chikankari__cotton",
        target_date=date(2026, 4, 27),
        score=42.5,
        post_count=12,
        youtube_views_total=15000.0,
        hashtag_counts={"#kurta": 3},
        accounts=["acc_a", "acc_b"],
    )
    h1.save()

    raw = json.loads(path.read_text())
    bucket = raw["kurta__chikankari__cotton"]["2026-W18"]
    assert bucket["score"] == 42.5
    assert bucket["post_count"] == 12
    assert bucket["week_start_date"] == "2026-04-27"

    h2 = WeeklyScoreHistory(path)
    assert h2.get_weekly_score("kurta__chikankari__cotton", date(2026, 5, 1)) == 42.5


def test_baseline_returns_last_week_score(tmp_path: Path) -> None:
    path = tmp_path / "score_history_weekly.json"
    h = WeeklyScoreHistory(path)
    h.update_weekly("k", target_date=date(2026, 4, 20), score=30.0)  # W17
    h.update_weekly("k", target_date=date(2026, 4, 27), score=40.0)  # W18

    assert h.get_baseline("k", date(2026, 4, 27)) == 30.0
    # 지난 주 데이터 없으면 None.
    assert h.get_baseline("k", date(2026, 4, 20)) is None


def test_trajectory_12w_pads_with_zeros(tmp_path: Path) -> None:
    path = tmp_path / "score_history_weekly.json"
    h = WeeklyScoreHistory(path)
    # 최근 3주만 채움. 9주는 0 패딩.
    h.update_weekly("k", target_date=date(2026, 4, 13), score=10.0)  # W16
    h.update_weekly("k", target_date=date(2026, 4, 20), score=20.0)  # W17
    h.update_weekly("k", target_date=date(2026, 4, 27), score=30.0)  # W18

    traj = h.get_trajectory_12w("k", target_date=date(2026, 4, 27))
    assert len(traj) == 12
    # oldest (11주 전) → newest (이번 주)
    assert traj[-3:] == [10.0, 20.0, 30.0]
    assert traj[:9] == [0.0] * 9


def test_same_week_update_overwrites(tmp_path: Path) -> None:
    path = tmp_path / "score_history_weekly.json"
    h = WeeklyScoreHistory(path)
    h.update_weekly("k", target_date=date(2026, 4, 27), score=10.0)
    h.update_weekly("k", target_date=date(2026, 4, 30), score=20.0)  # 같은 W18

    assert h.get_weekly_score("k", date(2026, 4, 27)) == 20.0
