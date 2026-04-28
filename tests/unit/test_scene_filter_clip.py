"""CLIPSceneFilter Stage 1/2 threshold 분기 단위 테스트 (Phase 1).

실 CLIP 모델 로드 없이, __new__ 로 인스턴스 만든 뒤 `_image_embedding` 과
`_softmax_scores` 를 instance attribute 로 override 해 softmax 결과를 고정. 이러면
threshold / stage 분기 로직만 격리 검증 가능.

vision extras 필요 (scene_filter_clip 이 top-level 로 torch / transformers / PIL import).
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch", reason="vision extras required")
pytest.importorskip("transformers", reason="vision extras required")
pytest.importorskip("PIL.Image", reason="vision extras required")


from settings import SceneFilterConfig  # noqa: E402
from vision.scene_filter_clip import CLIPSceneFilter  # noqa: E402


def _make_stub_filter(cfg: SceneFilterConfig) -> CLIPSceneFilter:
    """CLIPModel 로드 없이 CLIPSceneFilter 인스턴스 생성. embedding/softmax 는 테스트에서 주입."""
    f = CLIPSceneFilter.__new__(CLIPSceneFilter)
    f._cfg = cfg  # type: ignore[attr-defined]
    f._device = "cpu"  # type: ignore[attr-defined]
    f._scene_emb = None  # type: ignore[attr-defined]
    f._gender_emb = None  # type: ignore[attr-defined]
    f._age_emb = None  # type: ignore[attr-defined]
    return f


def _inject_stage1_scores(
    f: CLIPSceneFilter,
    scene: list[float],
    gender: list[float],
    age: list[float],
) -> None:
    """accept() 가 호출될 때 반환할 softmax 값 고정. prompts 길이와 일치해야."""
    cfg = f._cfg  # type: ignore[attr-defined]

    def fake_image_embedding(_rgb: np.ndarray):
        return "img_emb"

    def fake_softmax_scores(_img_emb, _text_emb, prompts):
        if prompts == cfg.scene_prompts:
            values = scene
        elif prompts == cfg.gender_prompts:
            values = gender
        elif prompts == cfg.age_prompts:
            values = age
        else:
            raise AssertionError(f"unexpected prompts: {prompts}")
        return {p: v for p, v in zip(prompts, values)}

    f._image_embedding = fake_image_embedding  # type: ignore[attr-defined]
    f._softmax_scores = fake_softmax_scores  # type: ignore[attr-defined]


def _inject_stage2_scores(
    f: CLIPSceneFilter,
    per_bbox: list[dict[str, list[float]]],
) -> None:
    """classify_persons 가 bbox 루프에서 소비할 softmax 값 큐 주입.

    per_bbox: [{"gender": [...], "age": [...]}, ...]
    """
    cfg = f._cfg  # type: ignore[attr-defined]
    queue = iter(per_bbox)
    current: dict[str, list[float]] = {}

    def fake_image_embedding(_rgb: np.ndarray):
        nonlocal current
        current = next(queue)
        return "bbox_emb"

    def fake_softmax_scores(_img_emb, _text_emb, prompts):
        if prompts == cfg.gender_prompts:
            values = current["gender"]
        elif prompts == cfg.age_prompts:
            values = current["age"]
        else:
            raise AssertionError(f"unexpected prompts in stage2: {prompts}")
        return {p: v for p, v in zip(prompts, values)}

    f._image_embedding = fake_image_embedding  # type: ignore[attr-defined]
    f._softmax_scores = fake_softmax_scores  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# Stage 1 — accept()
# --------------------------------------------------------------------------- #

def test_stage1_pure_female_adult_passes_stage1_pass() -> None:
    cfg = SceneFilterConfig()
    f = _make_stub_filter(cfg)
    _inject_stage1_scores(
        f,
        scene=[0.90, 0.05, 0.05],
        gender=[0.85, 0.15],  # woman high, man low
        age=[0.10, 0.90],     # adult high, child low
    )
    v = f.accept(np.zeros((10, 10, 3), dtype=np.uint8), "x")
    assert v.passed is True
    assert v.reason == "ok"
    assert v.stage == "stage1_pass"


def test_stage1_scene_reject() -> None:
    cfg = SceneFilterConfig()
    f = _make_stub_filter(cfg)
    _inject_stage1_scores(
        f,
        scene=[0.10, 0.70, 0.20],  # fashion < 0.20
        gender=[0.85, 0.15],
        age=[0.05, 0.95],
    )
    v = f.accept(np.zeros((10, 10, 3), dtype=np.uint8), "x")
    assert v.passed is False
    assert v.reason == "scene_reject"
    assert v.stage == "stage1_reject"


def test_stage1_female_low_rejects() -> None:
    cfg = SceneFilterConfig()
    f = _make_stub_filter(cfg)
    _inject_stage1_scores(
        f,
        scene=[0.90, 0.05, 0.05],
        gender=[0.20, 0.80],  # woman < stage1_female_min (0.30)
        age=[0.05, 0.95],
    )
    v = f.accept(np.zeros((10, 10, 3), dtype=np.uint8), "x")
    assert v.passed is False
    assert v.reason == "stage1_female_low"
    assert v.stage == "stage1_reject"


def test_stage1_adult_low_rejects() -> None:
    cfg = SceneFilterConfig()
    f = _make_stub_filter(cfg)
    _inject_stage1_scores(
        f,
        scene=[0.90, 0.05, 0.05],
        gender=[0.85, 0.15],
        age=[0.80, 0.20],  # adult < stage1_adult_min
    )
    v = f.accept(np.zeros((10, 10, 3), dtype=np.uint8), "x")
    assert v.passed is False
    assert v.reason == "stage1_adult_low"
    assert v.stage == "stage1_reject"


def test_stage1_mix_triggers_stage2_only_when_4way() -> None:
    """v2 (2026-04-25): male AND child 둘 다 mix_threshold 이상일 때만 stage2 지시.

    adult-woman-only 통합 후 BBOX 게이트는 '성인 여성 + 성인 남성 + 아동' 4-way 혼합
    프레임에서만 트리거. 그 외는 Gemini v0.6 프롬프트가 비-adult-female 제외 방어.
    """
    cfg = SceneFilterConfig()
    f = _make_stub_filter(cfg)
    _inject_stage1_scores(
        f,
        scene=[0.90, 0.05, 0.05],
        gender=[0.55, 0.45],  # woman ≥ 0.30 AND man ≥ 0.30
        age=[0.40, 0.60],     # adult ≥ 0.30 AND child ≥ 0.30
    )
    v = f.accept(np.zeros((10, 10, 3), dtype=np.uint8), "x")
    assert v.passed is True
    assert v.reason == "ok"
    assert v.stage == "stage1_mix_needs_stage2"


def test_stage1_woman_plus_man_only_passes_without_stage2() -> None:
    """성인 여성 + 성인 남성 (아동 X) — v2 AND 로직에서는 stage2 우회 → stage1_pass.

    이 케이스의 비-adult-female 방어선은 Gemini v0.6 프롬프트.
    """
    cfg = SceneFilterConfig()
    f = _make_stub_filter(cfg)
    _inject_stage1_scores(
        f,
        scene=[0.90, 0.05, 0.05],
        gender=[0.55, 0.45],  # woman + man 둘 다 mix threshold 이상
        age=[0.05, 0.95],     # adult only, child < threshold
    )
    v = f.accept(np.zeros((10, 10, 3), dtype=np.uint8), "x")
    assert v.passed is True
    assert v.stage == "stage1_pass"


def test_stage1_woman_plus_child_only_passes_without_stage2() -> None:
    """성인 여성 + 아동 (성인 남성 X) — v2 AND 로직에서는 stage2 우회 → stage1_pass."""
    cfg = SceneFilterConfig()
    f = _make_stub_filter(cfg)
    _inject_stage1_scores(
        f,
        scene=[0.90, 0.05, 0.05],
        gender=[0.85, 0.15],  # man < threshold
        age=[0.40, 0.60],     # adult + child 둘 다 threshold 이상
    )
    v = f.accept(np.zeros((10, 10, 3), dtype=np.uint8), "x")
    assert v.passed is True
    assert v.stage == "stage1_pass"


# --------------------------------------------------------------------------- #
# Stage 2 — classify_persons()
# --------------------------------------------------------------------------- #

def test_classify_persons_too_small_bbox_drops() -> None:
    cfg = SceneFilterConfig()  # default min_person_bbox_side_px=80
    f = _make_stub_filter(cfg)
    # embedding / softmax 는 호출되지 않아야 함 (짧은 변 < 80)
    _inject_stage2_scores(f, per_bbox=[])

    rgb = np.zeros((300, 300, 3), dtype=np.uint8)
    out = f.classify_persons(rgb, [(0, 0, 50, 200)])  # 짧은 변 50 < 80
    assert len(out) == 1
    assert out[0].passed is False
    assert out[0].reason == "too_small"
    assert out[0].bbox == (0, 0, 50, 200)


def test_classify_persons_pass_on_female_adult() -> None:
    cfg = SceneFilterConfig()
    f = _make_stub_filter(cfg)
    _inject_stage2_scores(f, per_bbox=[
        {"gender": [0.75, 0.25], "age": [0.10, 0.90]},
    ])
    rgb = np.zeros((300, 300, 3), dtype=np.uint8)
    out = f.classify_persons(rgb, [(0, 0, 200, 280)])
    assert len(out) == 1
    assert out[0].passed is True
    assert out[0].reason == "ok"
    assert out[0].gender_scores  # softmax 기록 확인
    assert out[0].age_scores


def test_classify_persons_female_low_drops() -> None:
    cfg = SceneFilterConfig()  # default stage2_female_min=0.50
    f = _make_stub_filter(cfg)
    _inject_stage2_scores(f, per_bbox=[
        {"gender": [0.40, 0.60], "age": [0.10, 0.90]},
    ])
    rgb = np.zeros((300, 300, 3), dtype=np.uint8)
    out = f.classify_persons(rgb, [(0, 0, 200, 280)])
    assert out[0].passed is False
    assert out[0].reason == "stage2_female_low"


def test_classify_persons_adult_low_drops() -> None:
    cfg = SceneFilterConfig()  # default stage2_adult_min=0.50
    f = _make_stub_filter(cfg)
    _inject_stage2_scores(f, per_bbox=[
        {"gender": [0.75, 0.25], "age": [0.55, 0.45]},  # adult < 0.50
    ])
    rgb = np.zeros((300, 300, 3), dtype=np.uint8)
    out = f.classify_persons(rgb, [(0, 0, 200, 280)])
    assert out[0].passed is False
    assert out[0].reason == "stage2_adult_low"


def test_classify_persons_mixed_bboxes_per_bbox_verdict() -> None:
    """3개 BBOX: pass / too_small / stage2_female_low."""
    cfg = SceneFilterConfig()
    f = _make_stub_filter(cfg)
    # too_small 은 embedding 호출 안 하므로 queue 는 2개만.
    _inject_stage2_scores(f, per_bbox=[
        {"gender": [0.80, 0.20], "age": [0.10, 0.90]},   # bbox 0 pass
        {"gender": [0.30, 0.70], "age": [0.10, 0.90]},   # bbox 2 female_low
    ])
    rgb = np.zeros((300, 300, 3), dtype=np.uint8)
    out = f.classify_persons(rgb, [
        (0, 0, 200, 280),     # pass (>= 80 short side)
        (0, 0, 30, 150),      # too_small (short=30)
        (0, 0, 150, 200),     # female_low
    ])
    assert [pv.passed for pv in out] == [True, False, False]
    assert [pv.reason for pv in out] == ["ok", "too_small", "stage2_female_low"]
