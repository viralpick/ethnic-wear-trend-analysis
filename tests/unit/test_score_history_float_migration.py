"""Phase γ (2026-04-28) — ScoreHistory schema int → float read-cast pinning.

검증 포인트:
- post_count: int 으로 저장된 기존 json → `_read_count` 가 float 로 자연 cast
- post_count: float (1.5 등) 도 그대로 read 후 sum 정확
- update(post_count: float) 직접 적재 + 동일 file round-trip 으로 float 복귀
- get_total_post_count 가 float 합 반환
- get_post_count_history list[float] 반환

backwards compat: 기존 outputs/score_history.json (int post_count) 을 변경 없이 read.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from scoring.score_history import ScoreHistory
from scoring.score_history_weekly import WeeklyScoreHistory


# --------------------------------------------------------------------------- #
# read-cast: int 저장된 기존 json
# --------------------------------------------------------------------------- #


def test_legacy_int_post_count_read_as_float(tmp_path: Path) -> None:
    """기존 int post_count 저장 파일이 read 시 float 로 자연 cast (γ migration)."""
    history_file = tmp_path / "score_history.json"
    history_file.write_text(json.dumps({
        "kurta_set__chikankari__cotton": {
            "2026-04-25": {"score": 50.0, "post_count": 3},  # int
            "2026-04-26": {"score": 55.0, "post_count": 5},  # int
        }
    }), encoding="utf-8")
    history = ScoreHistory(history_file)
    total = history.get_total_post_count("kurta_set__chikankari__cotton")
    assert total == 8.0
    assert isinstance(total, float)
    counts = history.get_post_count_history(
        "kurta_set__chikankari__cotton", date(2026, 4, 27), days=2
    )
    assert counts == [5.0, 3.0]
    assert all(isinstance(c, float) for c in counts)


def test_float_post_count_round_trip(tmp_path: Path) -> None:
    """update(post_count=float) → save → reload → get_total_post_count 정확."""
    history_file = tmp_path / "score_history.json"
    history = ScoreHistory(history_file)
    history.update(
        "casual_saree__block_print__cotton", date(2026, 4, 27),
        score=42.0, post_count=2.5,  # share-weighted fractional
    )
    history.update(
        "casual_saree__block_print__cotton", date(2026, 4, 26),
        score=40.0, post_count=1.25,
    )
    history.save()

    reloaded = ScoreHistory(history_file)
    total = reloaded.get_total_post_count("casual_saree__block_print__cotton")
    assert total == pytest.approx(3.75)


def test_total_post_count_empty_returns_zero_float(tmp_path: Path) -> None:
    history = ScoreHistory(tmp_path / "score_history.json")
    total = history.get_total_post_count("nonexistent_cluster")
    assert total == 0.0
    assert isinstance(total, float)


# --------------------------------------------------------------------------- #
# WeeklyScoreHistory — post_count default 0.0 + float 호환
# --------------------------------------------------------------------------- #


def test_weekly_update_float_post_count(tmp_path: Path) -> None:
    weekly_file = tmp_path / "score_history_weekly.json"
    weekly = WeeklyScoreHistory(weekly_file)
    weekly.update_weekly(
        "kurta_set__chikankari__cotton", date(2026, 4, 27),
        score=80.0, post_count=10.5,
    )
    weekly.save()

    raw = json.loads(weekly_file.read_text(encoding="utf-8"))
    bucket = raw["kurta_set__chikankari__cotton"]
    week_entry = next(iter(bucket.values()))
    assert week_entry["post_count"] == 10.5
