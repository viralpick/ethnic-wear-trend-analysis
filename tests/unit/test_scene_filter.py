"""scene_filter Protocol + Fake/Noop 구현 단위 테스트."""
from __future__ import annotations

import numpy as np
import pytest

from vision.scene_filter import (
    FakeSceneFilter,
    FilterVerdict,
    NoopSceneFilter,
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
