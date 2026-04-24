"""pipeline_b_extractor 2-stage scene filter 분기 단위 테스트 (Phase 1).

extract_instances 의 stage 분기 (stage1_reject / stage1_pass / stage1_mix_needs_stage2
/ disabled + stage2_enabled 토글) 를 fake SceneFilter + monkeypatched detect_people /
_instances_from_bbox 로 검증. 실 YOLO / segformer 호출 없음.

vision extras 필요 (pipeline_b_extractor 가 top-level torch/transformers/ultralytics 를
import) — 없으면 module skip.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch", reason="vision extras required")
pytest.importorskip("transformers", reason="vision extras required")
pytest.importorskip("ultralytics", reason="vision extras required")
pytest.importorskip("PIL.Image", reason="vision extras required")


from settings import load_settings  # noqa: E402
from vision.frame_source import Frame  # noqa: E402
from vision.pipeline_b_extractor import SegBundle, extract_instances  # noqa: E402
from vision.scene_filter import FakeSceneFilter, NoopSceneFilter  # noqa: E402


def _frame(fid: str = "f0", h: int = 200, w: int = 200) -> Frame:
    return Frame(id=fid, rgb=np.zeros((h, w, 3), dtype=np.uint8), source_type="image")


class _StaticFrameSource:
    def __init__(self, frames: list[Frame]) -> None:
        self._frames = frames

    def iter_frames(self):
        yield from self._frames


def _make_bundle(scene_filter) -> SegBundle:
    # yolo / seg_* 은 테스트에서 호출되지 않도록 detect_people / _instances_from_bbox 를
    # monkeypatch. dummy object 로 충분.
    return SegBundle(
        yolo=object(),
        seg_processor=object(),
        seg_model=object(),
        device="cpu",
        scene_filter=scene_filter,
    )


def _patch_detect_and_instances(monkeypatch, bboxes, on_instance=None):
    """detect_people → 고정 bbox 리스트. _instances_from_bbox → on_instance 콜백 (빈 list 반환)."""
    calls: list[tuple[int, tuple[int, int, int, int]]] = []

    def fake_detect(_yolo, _rgb):
        return list(bboxes)

    def fake_instances(_frame, idx, bbox, _ctx):
        calls.append((idx, bbox))
        if on_instance is not None:
            on_instance(idx, bbox)
        return []

    monkeypatch.setattr(
        "vision.pipeline_b_extractor.detect_people", fake_detect,
    )
    monkeypatch.setattr(
        "vision.pipeline_b_extractor._instances_from_bbox", fake_instances,
    )
    return calls


# --------------------------------------------------------------------------- #
# stage1_reject: frame 통째로 drop
# --------------------------------------------------------------------------- #

def test_stage1_reject_drops_frame(monkeypatch) -> None:
    calls = _patch_detect_and_instances(monkeypatch, bboxes=[(0, 0, 100, 100)])
    cfg = load_settings().vision
    bundle = _make_bundle(
        FakeSceneFilter(drop_rule=lambda _fid: "stage1_female_low")
    )
    src = _StaticFrameSource([_frame("f0")])

    instances, stats = extract_instances(src, bundle, cfg)

    assert instances == []
    assert calls == []  # detect_people / _instances_from_bbox 호출 X
    assert len(stats.filtered_out) == 1
    assert stats.filtered_out[0].frame_id == "f0"
    assert stats.filtered_out[0].verdict.stage == "stage1_reject"
    assert stats.bbox_filtered_out == []


# --------------------------------------------------------------------------- #
# stage1_pass: 모든 bbox 진행 (기존 동작)
# --------------------------------------------------------------------------- #

def test_stage1_pass_runs_all_bboxes(monkeypatch) -> None:
    calls = _patch_detect_and_instances(
        monkeypatch,
        bboxes=[(0, 0, 100, 100), (110, 0, 200, 100)],
    )
    cfg = load_settings().vision
    # FakeSceneFilter default forced_stage=stage1_pass.
    bundle = _make_bundle(FakeSceneFilter())
    src = _StaticFrameSource([_frame("f0")])

    _, stats = extract_instances(src, bundle, cfg)

    assert [idx for idx, _ in calls] == [0, 1]
    assert stats.filtered_out == []
    assert stats.bbox_filtered_out == []


# --------------------------------------------------------------------------- #
# stage1_mix_needs_stage2 + stage2_enabled: 선택적 bbox drop
# --------------------------------------------------------------------------- #

def test_stage1_mix_filters_bboxes_via_classify_persons(monkeypatch) -> None:
    bboxes = [(0, 0, 100, 100), (110, 0, 200, 100), (0, 110, 100, 200)]
    calls = _patch_detect_and_instances(monkeypatch, bboxes=bboxes)

    cfg = load_settings().vision
    cfg.scene_filter.stage2_enabled = True
    sf = FakeSceneFilter(
        forced_stage="stage1_mix_needs_stage2",
        bbox_drop_rule=lambda _fid, idx, _b: (
            "stage2_female_low" if idx == 1 else None
        ),
    )
    bundle = _make_bundle(sf)
    src = _StaticFrameSource([_frame("f0")])

    _, stats = extract_instances(src, bundle, cfg)

    # passed bboxes: idx 0, 2 — bbox_idx 는 원래 yolo 순서 유지
    assert [idx for idx, _ in calls] == [0, 2]
    assert stats.filtered_out == []
    assert len(stats.bbox_filtered_out) == 1
    dropped = stats.bbox_filtered_out[0]
    assert dropped.frame_id == "f0"
    assert dropped.bbox_idx == 1
    assert dropped.verdict.reason == "stage2_female_low"
    assert dropped.verdict.bbox == bboxes[1]


# --------------------------------------------------------------------------- #
# stage2_enabled=False: mix 여도 classify_persons 호출 X
# --------------------------------------------------------------------------- #

def test_stage2_disabled_toggle_runs_all_bboxes(monkeypatch) -> None:
    bboxes = [(0, 0, 100, 100), (110, 0, 200, 100)]
    calls = _patch_detect_and_instances(monkeypatch, bboxes=bboxes)
    classify_calls: list[int] = []

    class TrackingFake(FakeSceneFilter):
        def classify_persons(self, rgb, bboxes):  # noqa: D401
            classify_calls.append(len(bboxes))
            return super().classify_persons(rgb, bboxes)

    cfg = load_settings().vision
    cfg.scene_filter.stage2_enabled = False
    bundle = _make_bundle(TrackingFake(forced_stage="stage1_mix_needs_stage2"))
    src = _StaticFrameSource([_frame("f0")])

    _, stats = extract_instances(src, bundle, cfg)

    assert classify_calls == []  # Stage 2 호출 안 됨
    assert [idx for idx, _ in calls] == [0, 1]
    assert stats.bbox_filtered_out == []


# --------------------------------------------------------------------------- #
# disabled (Noop) stage: stage2 분기 안 탐 (기존 동작)
# --------------------------------------------------------------------------- #

def test_disabled_stage_runs_all_bboxes(monkeypatch) -> None:
    calls = _patch_detect_and_instances(monkeypatch, bboxes=[(0, 0, 100, 100)])
    cfg = load_settings().vision
    bundle = _make_bundle(NoopSceneFilter())
    src = _StaticFrameSource([_frame("f0")])

    _, stats = extract_instances(src, bundle, cfg)

    assert [idx for idx, _ in calls] == [0]
    assert stats.filtered_out == []
    assert stats.bbox_filtered_out == []


# --------------------------------------------------------------------------- #
# mix + 빈 bbox: classify_persons 호출 skip (empty list 가드)
# --------------------------------------------------------------------------- #

def test_stage1_mix_with_no_bboxes_does_not_call_classify(monkeypatch) -> None:
    classify_calls: list[int] = []

    class TrackingFake(FakeSceneFilter):
        def classify_persons(self, rgb, bboxes):
            classify_calls.append(len(bboxes))
            return super().classify_persons(rgb, bboxes)

    # fallback_full_image_on_no_person 이 True 면 빈 detect → [(0,0,w,h)] 하나 주입됨.
    # 이 경우에도 mix 분기가 동작해야 함. 테스트는 fallback_full_image_on_no_person=False 로 분리.
    calls = _patch_detect_and_instances(monkeypatch, bboxes=[])
    cfg = load_settings().vision
    cfg.fallback_full_image_on_no_person = False
    cfg.scene_filter.stage2_enabled = True
    bundle = _make_bundle(TrackingFake(forced_stage="stage1_mix_needs_stage2"))
    src = _StaticFrameSource([_frame("f0")])

    _, stats = extract_instances(src, bundle, cfg)

    assert classify_calls == []  # bbox 가 없으면 classify_persons 호출 skip
    assert calls == []
    assert stats.bbox_filtered_out == []


# --------------------------------------------------------------------------- #
# fallback + mix 엣지 케이스: YOLO 미탐 → fallback bbox (전체 이미지) 도 stage 2 대상.
# advisor 결정: "애매한 frame 은 보수적 drop" 수용. stage2 가 pass 하면 1 instance,
# drop 하면 bbox_filtered_out 에 기록되고 instance 없음. 이 동작을 pinning.
# --------------------------------------------------------------------------- #

def test_fallback_bbox_in_mix_goes_through_stage2_and_can_pass(monkeypatch) -> None:
    calls = _patch_detect_and_instances(monkeypatch, bboxes=[])  # YOLO 미탐
    cfg = load_settings().vision
    cfg.fallback_full_image_on_no_person = True
    cfg.scene_filter.stage2_enabled = True
    # default forced_stage + bbox_drop_rule=None → stage 2 에서 pass
    bundle = _make_bundle(FakeSceneFilter(forced_stage="stage1_mix_needs_stage2"))
    src = _StaticFrameSource([_frame("f0", h=100, w=100)])

    _, stats = extract_instances(src, bundle, cfg)

    # fallback bbox (0, 0, 100, 100) 가 stage 2 pass → instance 처리 호출됨
    assert [idx for idx, _ in calls] == [0]
    assert calls[0][1] == (0, 0, 100, 100)
    assert stats.bbox_filtered_out == []
    assert stats.fallback_triggered is True


def test_fallback_bbox_in_mix_can_be_dropped_by_stage2(monkeypatch) -> None:
    calls = _patch_detect_and_instances(monkeypatch, bboxes=[])  # YOLO 미탐
    cfg = load_settings().vision
    cfg.fallback_full_image_on_no_person = True
    cfg.scene_filter.stage2_enabled = True
    sf = FakeSceneFilter(
        forced_stage="stage1_mix_needs_stage2",
        bbox_drop_rule=lambda _fid, _idx, _b: "stage2_female_low",
    )
    bundle = _make_bundle(sf)
    src = _StaticFrameSource([_frame("f0", h=100, w=100)])

    _, stats = extract_instances(src, bundle, cfg)

    # fallback bbox 가 stage 2 에서 drop → instance 호출 0, bbox_filtered_out 1
    assert calls == []
    assert len(stats.bbox_filtered_out) == 1
    assert stats.bbox_filtered_out[0].verdict.reason == "stage2_female_low"
    assert stats.fallback_triggered is True
