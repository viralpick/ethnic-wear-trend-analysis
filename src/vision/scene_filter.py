"""이미지 단위 scene + demographic pre-filter (Protocol + Fake/Noop 구현).

Pipeline B 의 YOLO + segformer 돌기 **전** 에 frame 1장당 3가지 zero-shot 분류를 돌려
타겟 이미지 (person + clothing / female / adult) 가 아니면 drop.

이 모듈은 numpy 외 의존 없음 — core 코드 / 테스트가 transformers 없이 import 가능.
실 구현 `CLIPSceneFilter` 는 `src/vision/scene_filter_clip.py` 에 분리 (vision extras).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class FilterVerdict:
    """1 frame 에 대한 filter 판정 결과.

    passed: True → Pipeline B 진행. False → drop.
    reason: "ok" (passed) / "scene_reject" / "scene_low_confidence" / "gender_reject" /
            "age_reject" / "disabled" (Noop 사용 시).
    *_scores: prompt → softmax probability. UI / bias 감사 용 — 왜 그 판정이 나왔는지
              투명 하게 기록. Noop / 사용자 미관심 필드는 빈 dict.
    """
    passed: bool
    reason: str
    scene_scores: dict[str, float] = field(default_factory=dict)
    gender_scores: dict[str, float] = field(default_factory=dict)
    age_scores: dict[str, float] = field(default_factory=dict)


@runtime_checkable
class SceneFilter(Protocol):
    """frame 1장 → FilterVerdict. stateful (모델 무거움) 이라 frame loop 밖에서 1회 생성."""
    def accept(self, rgb: np.ndarray, frame_id: str) -> FilterVerdict: ...


class NoopSceneFilter:
    """disabled — 항상 pass. SceneFilter 비활성 시 기본 구현."""

    def accept(self, rgb: np.ndarray, frame_id: str) -> FilterVerdict:  # noqa: ARG002
        return FilterVerdict(passed=True, reason="disabled")


class FakeSceneFilter:
    """결정론 — 테스트 용. drop_rule 로 frame_id 당 판정 사용자 지정.

    drop_rule(frame_id) 가 truthy reason 리턴하면 drop, None 이면 pass.
    없으면 항상 pass (확정 scores 포함해 HTML 렌더러 smoke 도 가능).
    """

    def __init__(self, drop_rule: Callable[[str], str | None] | None = None) -> None:
        self._drop_rule = drop_rule

    def accept(self, rgb: np.ndarray, frame_id: str) -> FilterVerdict:  # noqa: ARG002
        if self._drop_rule is not None:
            reason = self._drop_rule(frame_id)
            if reason:
                return FilterVerdict(passed=False, reason=reason)
        return FilterVerdict(
            passed=True, reason="ok",
            scene_scores={"clothing": 0.9, "statue": 0.05, "product": 0.03, "landscape": 0.02},
            gender_scores={"woman": 0.82, "man": 0.18},
            age_scores={"child": 0.12, "adult": 0.88},
        )
