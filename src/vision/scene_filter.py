"""이미지 단위 scene + demographic pre-filter (2-stage, Protocol + Fake/Noop 구현).

Phase 1 2-stage 재구성 (2026-04-24):
- Stage 1 (image-level, `accept`): scene fashion 통과 + woman softmax ≥ stage1_female_min +
  adult softmax ≥ stage1_adult_min. 둘 중 하나라도 미달 → stage1_reject (frame skip).
  통과 시 v2 (2026-04-25, adult-woman-only 통합): man AND child softmax 가 모두
  stage2_mix_threshold 이상일 때만 stage=stage1_mix_needs_stage2 — YOLO person BBOX 별
  Stage 2 CLIP 판정 필요. 그 외는 stage1_pass — Gemini v0.6 프롬프트가 비-adult-female
  검출 제외 방어.
- Stage 2 (per-BBOX, `classify_persons`): BBOX crop → CLIP gender/age softmax → adult+female
  조건 충족 BBOX 만 keep.

이 모듈은 numpy 외 의존 없음 — core 코드 / 테스트가 transformers 없이 import 가능.
실 구현 `CLIPSceneFilter` 는 `src/vision/scene_filter_clip.py` 에 분리 (vision extras).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

import numpy as np

# Stage 1 판정 결과 라벨. disabled 는 NoopSceneFilter 전용 sentinel.
Stage = Literal[
    "stage1_pass",             # pure female+adult — Stage 2 불필요
    "stage1_mix_needs_stage2", # female+adult signal 있지만 man/child mix — per-bbox 재판정
    "stage1_reject",           # 여성 또는 성인 signal 부재 또는 scene off-fashion
    "disabled",                # SceneFilter 비활성 (NoopSceneFilter)
]


@dataclass(frozen=True)
class FilterVerdict:
    """1 frame 에 대한 Stage 1 판정 결과.

    passed: True → Pipeline B 진행. False → frame 통째로 drop.
    stage: 다음 단계 결정용 — stage1_pass 면 bbox 전부 진행, stage1_mix_needs_stage2 면
           pipeline_b_extractor 가 classify_persons 호출해 bbox 단위 재판정.
    reason: "ok" (passed) / "scene_reject" / "stage1_female_low" / "stage1_adult_low" /
            "disabled" (Noop). Fake 는 drop_rule 반환값 그대로.
    *_scores: prompt → softmax probability. UI / bias 감사 용. Noop / 사용자 미관심 필드는 빈 dict.
    """
    passed: bool
    reason: str
    stage: Stage = "stage1_pass"
    scene_scores: dict[str, float] = field(default_factory=dict)
    gender_scores: dict[str, float] = field(default_factory=dict)
    age_scores: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class PersonVerdict:
    """1 person BBOX 에 대한 Stage 2 판정 결과.

    passed=True 면 해당 BBOX 만 segformer / palette 파이프라인 진행.
    bbox 는 frame 좌표계 (x1, y1, x2, y2) — pipeline_b_extractor 가 BBoxDropRecord 에 기록.
    reason: "ok" / "stage2_female_low" / "stage2_adult_low" / "too_small" / "disabled".
    """
    passed: bool
    reason: str
    bbox: tuple[int, int, int, int]
    gender_scores: dict[str, float] = field(default_factory=dict)
    age_scores: dict[str, float] = field(default_factory=dict)


@runtime_checkable
class SceneFilter(Protocol):
    """frame 1장 → FilterVerdict. stateful (모델 무거움) 이라 frame loop 밖에서 1회 생성.

    classify_persons 는 stage1_mix_needs_stage2 일 때만 pipeline_b_extractor 가 호출.
    다른 stage 에서는 호출하지 않음 (비용 절감). Noop/Fake 도 Protocol 충족을 위해 구현.
    """
    def accept(self, rgb: np.ndarray, frame_id: str) -> FilterVerdict: ...
    def classify_persons(
        self,
        rgb: np.ndarray,
        bboxes: list[tuple[int, int, int, int]],
    ) -> list[PersonVerdict]: ...


class NoopSceneFilter:
    """disabled — 항상 pass. SceneFilter 비활성 시 기본 구현."""

    def accept(self, rgb: np.ndarray, frame_id: str) -> FilterVerdict:  # noqa: ARG002
        return FilterVerdict(passed=True, reason="disabled", stage="disabled")

    def classify_persons(
        self,
        rgb: np.ndarray,  # noqa: ARG002
        bboxes: list[tuple[int, int, int, int]],
    ) -> list[PersonVerdict]:
        # Noop 에서 Stage 2 호출은 정상 흐름에선 발생하지 않음 (pipeline_b_extractor 가
        # stage=disabled 면 bbox 필터링 skip). 방어적으로 전부 pass 반환.
        return [PersonVerdict(passed=True, reason="disabled", bbox=b) for b in bboxes]


class FakeSceneFilter:
    """결정론 — 테스트 용. frame 판정은 drop_rule, bbox 판정은 bbox_drop_rule 로 지정.

    drop_rule(frame_id) 가 truthy reason 리턴하면 frame drop (stage=stage1_reject),
    None 이면 pass. 기본 pass 시 stage 는 forced_stage (default stage1_pass).
    bbox_drop_rule(frame_id, bbox_idx, bbox) 가 truthy reason 리턴하면 해당 bbox drop.
    """

    def __init__(
        self,
        drop_rule: Callable[[str], str | None] | None = None,
        bbox_drop_rule: Callable[
            [str, int, tuple[int, int, int, int]], str | None,
        ] | None = None,
        forced_stage: Stage = "stage1_pass",
    ) -> None:
        self._drop_rule = drop_rule
        self._bbox_drop_rule = bbox_drop_rule
        self._forced_stage = forced_stage
        self._last_frame_id: str = ""

    def accept(self, rgb: np.ndarray, frame_id: str) -> FilterVerdict:  # noqa: ARG002
        # classify_persons 는 frame_id 를 받지 않으므로 마지막 accept 된 frame_id 를
        # bbox_drop_rule 에 넘기려 보관. 단일 스레드 테스트 전제.
        self._last_frame_id = frame_id
        if self._drop_rule is not None:
            reason = self._drop_rule(frame_id)
            if reason:
                return FilterVerdict(passed=False, reason=reason, stage="stage1_reject")
        return FilterVerdict(
            passed=True, reason="ok", stage=self._forced_stage,
            scene_scores={"clothing": 0.9, "statue": 0.05, "product": 0.03, "landscape": 0.02},
            gender_scores={"woman": 0.82, "man": 0.18},
            age_scores={"child": 0.12, "adult": 0.88},
        )

    def classify_persons(
        self,
        rgb: np.ndarray,  # noqa: ARG002
        bboxes: list[tuple[int, int, int, int]],
    ) -> list[PersonVerdict]:
        out: list[PersonVerdict] = []
        for idx, bbox in enumerate(bboxes):
            if self._bbox_drop_rule is not None:
                reason = self._bbox_drop_rule(self._last_frame_id, idx, bbox)
                if reason:
                    out.append(PersonVerdict(passed=False, reason=reason, bbox=bbox))
                    continue
            out.append(PersonVerdict(
                passed=True, reason="ok", bbox=bbox,
                gender_scores={"woman": 0.80, "man": 0.20},
                age_scores={"child": 0.10, "adult": 0.90},
            ))
        return out
