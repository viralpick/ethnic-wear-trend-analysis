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


# --------------------------------------------------------------------------- #
# B-2 (M3.G/H 후) — accounts (IG handle) / channels (YT) 분리 누적 + sub-signal ratio
# --------------------------------------------------------------------------- #


def test_update_separates_accounts_and_channels(tmp_path: Path) -> None:
    """B-2: bucket 에 accounts (IG) / channels (YT) 분리 적재."""
    history_file = tmp_path / "score_history.json"
    history = ScoreHistory(history_file)
    history.update(
        "kurta_set__solid__cotton", date(2026, 4, 28),
        score=50.0, post_count=2.0,
        accounts=["@sridevi", "@masoom"],
        channels=["Khushi Malhotra"],
    )
    history.save()

    raw = json.loads(history_file.read_text(encoding="utf-8"))
    entry = raw["kurta_set__solid__cotton"]["2026-04-28"]
    assert entry["accounts"] == ["@sridevi", "@masoom"]
    assert entry["channels"] == ["Khushi Malhotra"]


def test_legacy_entry_without_channels_reads_as_empty(tmp_path: Path) -> None:
    """기존 entry (channels 키 없음) → get_new_yt_channel_ratio 가 자연 처리 ([] default)."""
    history_file = tmp_path / "score_history.json"
    history_file.write_text(json.dumps({
        "kurta_set__solid__cotton": {
            "2026-04-27": {
                "score": 50.0,
                "post_count": 2.0,
                "accounts": ["@old_handle"],
                # channels 키 없음 (legacy)
            }
        }
    }))
    history = ScoreHistory(history_file)
    # today_channels=["new_yt"] → seen 에 channels 없으니 모두 신규 = ratio 1.0.
    ratio = history.get_new_yt_channel_ratio(
        "kurta_set__solid__cotton", date(2026, 4, 28),
        window_days=7, today_channels=["new_yt"],
    )
    assert ratio == 1.0


def test_new_ig_yt_ratios_are_independent(tmp_path: Path) -> None:
    """IG handle 누적이 YT channel 신규 비율에 영향 없고, 그 반대도 동일."""
    history_file = tmp_path / "score_history.json"
    history = ScoreHistory(history_file)
    history.update(
        "X", date(2026, 4, 27),
        score=10.0, post_count=1.0,
        accounts=["@same_handle"],
        channels=["YT_X"],
    )
    # IG: today=["@same_handle"] → 0/1 신규 (있음).
    ig_ratio = history.get_new_ig_account_ratio(
        "X", date(2026, 4, 28), window_days=7, today_accounts=["@same_handle"]
    )
    assert ig_ratio == 0.0
    # YT: today=["YT_NEW"] → 1/1 신규 (히스토리 channels 와 무관). IG handle 누적은 영향 X.
    yt_ratio = history.get_new_yt_channel_ratio(
        "X", date(2026, 4, 28), window_days=7, today_channels=["YT_NEW"]
    )
    assert yt_ratio == 1.0


def test_score_breakdown_exposes_momentum_components() -> None:
    """B-2: ScoreBreakdown.momentum_components 가 raw sub-signal 노출."""
    from contracts.output import MomentumComponents, ScoreBreakdown
    bd = ScoreBreakdown(
        social=10.0, youtube=5.0, cultural=3.0, momentum=8.0,
        momentum_components=MomentumComponents(
            post_growth=0.5, hashtag_velocity=0.2,
            new_ig_account_ratio=0.3, new_yt_channel_ratio=0.7,
        ),
    )
    # momentum 합산 (8.0) 은 그대로, sub-signal raw 별도 노출.
    assert bd.momentum == 8.0
    assert bd.momentum_components.new_ig_account_ratio == 0.3
    assert bd.momentum_components.new_yt_channel_ratio == 0.7
