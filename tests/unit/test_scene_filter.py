"""scene_filter Protocol + Fake/Noop 구현 단위 테스트.

Phase 1 (2026-04-24) 확장: 2-stage 구조 도입. FilterVerdict.stage, PersonVerdict,
classify_persons 를 새로 테스트.
"""
from __future__ import annotations

import numpy as np
import pytest

from vision.scene_filter import (
    FakeSceneFilter,
    FilterVerdict,
    NoopSceneFilter,
    PersonVerdict,
    SceneFilter,
)


def _rgb() -> np.ndarray:
    return np.zeros((10, 10, 3), dtype=np.uint8)


# --------------------------------------------------------------------------- #
# NoopSceneFilter
# --------------------------------------------------------------------------- #

def test_noop_always_passes() -> None:
    f = NoopSceneFilter()
    v = f.accept(_rgb(), "frame-0")
    assert v.passed is True
    assert v.reason == "disabled"
    assert v.scene_scores == {}
    assert v.gender_scores == {}
    assert v.age_scores == {}


# --------------------------------------------------------------------------- #
# FakeSceneFilter — default (no drop_rule)
# --------------------------------------------------------------------------- #

def test_fake_default_passes_with_confident_scores() -> None:
    f = FakeSceneFilter()
    v = f.accept(_rgb(), "frame-0")
    assert v.passed is True
    assert v.reason == "ok"
    # 대표 prompt 들이 scores 에 들어있고 softmax 합이 1 에 가까움
    assert pytest.approx(sum(v.scene_scores.values()), rel=1e-3) == 1.0
    assert pytest.approx(sum(v.gender_scores.values()), rel=1e-3) == 1.0
    assert pytest.approx(sum(v.age_scores.values()), rel=1e-3) == 1.0


# --------------------------------------------------------------------------- #
# FakeSceneFilter — drop_rule
# --------------------------------------------------------------------------- #

def test_fake_drop_rule_triggers_drop() -> None:
    f = FakeSceneFilter(
        drop_rule=lambda fid: "scene_reject" if fid.startswith("statue") else None,
    )
    dropped = f.accept(_rgb(), "statue-01")
    assert dropped.passed is False
    assert dropped.reason == "scene_reject"
    passed = f.accept(_rgb(), "portrait-01")
    assert passed.passed is True


def test_fake_drop_rule_can_return_any_reason() -> None:
    f = FakeSceneFilter(drop_rule=lambda fid: "gender_reject")
    v = f.accept(_rgb(), "x")
    assert v.reason == "gender_reject"


# --------------------------------------------------------------------------- #
# Protocol 런타임 적합성
# --------------------------------------------------------------------------- #

def test_noop_and_fake_both_satisfy_protocol() -> None:
    assert isinstance(NoopSceneFilter(), SceneFilter)
    assert isinstance(FakeSceneFilter(), SceneFilter)


# --------------------------------------------------------------------------- #
# FilterVerdict dataclass 속성
# --------------------------------------------------------------------------- #

def test_verdict_is_frozen() -> None:
    v = FilterVerdict(passed=True, reason="ok")
    with pytest.raises((AttributeError, Exception)):
        v.passed = False  # type: ignore[misc]


def test_verdict_default_empty_scores() -> None:
    v = FilterVerdict(passed=False, reason="scene_reject")
    assert v.scene_scores == {}
    assert v.gender_scores == {}
    assert v.age_scores == {}


# --------------------------------------------------------------------------- #
# FilterVerdict.stage (Phase 1)
# --------------------------------------------------------------------------- #

def test_verdict_default_stage_is_stage1_pass() -> None:
    v = FilterVerdict(passed=True, reason="ok")
    assert v.stage == "stage1_pass"


def test_noop_verdict_stage_is_disabled() -> None:
    v = NoopSceneFilter().accept(_rgb(), "x")
    assert v.stage == "disabled"


def test_fake_drop_rule_sets_stage1_reject() -> None:
    f = FakeSceneFilter(drop_rule=lambda _fid: "scene_reject")
    v = f.accept(_rgb(), "x")
    assert v.passed is False
    assert v.stage == "stage1_reject"


def test_fake_forced_stage_propagates() -> None:
    f = FakeSceneFilter(forced_stage="stage1_mix_needs_stage2")
    v = f.accept(_rgb(), "x")
    assert v.passed is True
    assert v.stage == "stage1_mix_needs_stage2"


# --------------------------------------------------------------------------- #
# PersonVerdict + classify_persons (Phase 1)
# --------------------------------------------------------------------------- #

def test_person_verdict_is_frozen() -> None:
    pv = PersonVerdict(passed=True, reason="ok", bbox=(0, 0, 10, 10))
    with pytest.raises((AttributeError, Exception)):
        pv.passed = False  # type: ignore[misc]


def test_noop_classify_persons_passes_all_with_disabled_reason() -> None:
    bboxes = [(0, 0, 10, 10), (20, 20, 40, 40)]
    out = NoopSceneFilter().classify_persons(_rgb(), bboxes)
    assert len(out) == 2
    assert all(pv.passed for pv in out)
    assert all(pv.reason == "disabled" for pv in out)
    assert [pv.bbox for pv in out] == bboxes


def test_fake_classify_persons_default_all_pass() -> None:
    bboxes = [(0, 0, 10, 10), (5, 5, 50, 50)]
    f = FakeSceneFilter()
    # accept 를 먼저 호출하지 않아도 classify_persons 는 독립 동작해야 함.
    out = f.classify_persons(_rgb(), bboxes)
    assert [pv.passed for pv in out] == [True, True]
    assert all(pv.reason == "ok" for pv in out)
    assert all(pv.gender_scores and pv.age_scores for pv in out)


def test_fake_bbox_drop_rule_selectively_drops() -> None:
    bboxes = [(0, 0, 10, 10), (20, 20, 40, 40), (50, 50, 80, 80)]
    f = FakeSceneFilter(
        bbox_drop_rule=lambda fid, idx, _b: "stage2_female_low" if idx == 1 else None,
    )
    f.accept(_rgb(), "frame-A")  # frame_id 저장
    out = f.classify_persons(_rgb(), bboxes)
    assert [pv.passed for pv in out] == [True, False, True]
    assert out[1].reason == "stage2_female_low"
    assert out[1].bbox == (20, 20, 40, 40)


def test_fake_bbox_drop_rule_receives_frame_id() -> None:
    seen_fids: list[str] = []
    f = FakeSceneFilter(
        bbox_drop_rule=lambda fid, _idx, _b: seen_fids.append(fid) or None,  # type: ignore[func-returns-value]
    )
    f.accept(_rgb(), "frame-B")
    f.classify_persons(_rgb(), [(0, 0, 10, 10), (2, 2, 5, 5)])
    assert seen_fids == ["frame-B", "frame-B"]


# --------------------------------------------------------------------------- #
# Protocol 런타임 적합성 (Phase 1 classify_persons 포함)
# --------------------------------------------------------------------------- #

def test_noop_and_fake_still_satisfy_protocol_with_classify_persons() -> None:
    # classify_persons 도 Protocol 에 포함된 이후에도 isinstance 통과해야 함.
    assert isinstance(NoopSceneFilter(), SceneFilter)
    assert isinstance(FakeSceneFilter(), SceneFilter)
