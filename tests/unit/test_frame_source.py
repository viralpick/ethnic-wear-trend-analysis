"""FrameSource Protocol + ImageFrameSource + VideoFrameSource 단위 테스트.

Protocol / dataclass / stub 동작만 단위. 실 JPG 로딩 (PIL) 테스트는 vision extras 필요하므로
importorskip 로 가드. VideoFrameSource 는 cv2.VideoCapture monkeypatch fake 로 candidate
seek + selector 호출 + Frame yield 흐름 검증 (실 영상 코덱 의존 회피).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vision.frame_source import (
    Frame,
    FrameSource,
    ImageFrameSource,
    VideoFrameSource,
)
from vision.video_frame_selector import VideoFrameSelectorConfig

# --------------------------------------------------------------------------- #
# Frame dataclass / Protocol
# --------------------------------------------------------------------------- #

def test_frame_dataclass_fields() -> None:
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    frame = Frame(id="t1", rgb=rgb, source_type="image")
    assert frame.id == "t1"
    assert frame.source_type == "image"
    assert frame.rgb.shape == (4, 4, 3)


def test_frame_is_frozen() -> None:
    frame = Frame(id="t1", rgb=np.zeros((2, 2, 3), dtype=np.uint8), source_type="image")
    with pytest.raises(Exception):
        frame.id = "t2"  # type: ignore[misc]


def test_frame_source_protocol_runtime_checkable() -> None:
    class DummySource:
        def iter_frames(self):
            yield Frame(id="x", rgb=np.zeros((2, 2, 3), np.uint8), source_type="image")

    assert isinstance(DummySource(), FrameSource)


def test_frame_source_protocol_rejects_missing_iter_frames() -> None:
    class NotASource:
        def other_method(self) -> None: ...

    assert not isinstance(NotASource(), FrameSource)


# --------------------------------------------------------------------------- #
# ImageFrameSource
# --------------------------------------------------------------------------- #

def test_image_frame_source_init_accepts_paths() -> None:
    src = ImageFrameSource([Path("/tmp/a.jpg"), Path("/tmp/b.jpg")])
    # Protocol 준수 (iter_frames 존재) — 실 파일 로딩은 다음 테스트에서.
    assert isinstance(src, FrameSource)


def test_image_frame_source_init_accepts_str_paths() -> None:
    # 사용 편의: str 도 받아 내부에서 Path 로 정규화.
    src = ImageFrameSource([Path("/tmp/a.jpg")])
    assert isinstance(src, FrameSource)


def test_image_frame_source_iter_loads_jpg(tmp_path: Path) -> None:
    """실 JPG 로드 — PIL 필요. vision extras 미설치 시 skip."""
    pil = pytest.importorskip("PIL.Image", reason="vision extras required for ImageFrameSource")
    # 10x10 단색 이미지 파일 두 개 생성.
    red = pil.new("RGB", (10, 10), (255, 0, 0))
    green = pil.new("RGB", (10, 10), (0, 255, 0))
    red_path = tmp_path / "red.jpg"
    green_path = tmp_path / "green.jpg"
    red.save(red_path)
    green.save(green_path)

    src = ImageFrameSource([red_path, green_path])
    frames = list(src.iter_frames())
    assert len(frames) == 2
    assert all(f.source_type == "image" for f in frames)
    # ID 는 파일 stem.
    assert {f.id for f in frames} == {"red", "green"}
    # RGB 차원 유지.
    assert all(f.rgb.shape == (10, 10, 3) for f in frames)
    # JPEG 압축이라 정확히 (255,0,0) 은 아닐 수 있지만 red dominant 확인.
    red_frame = next(f for f in frames if f.id == "red")
    assert red_frame.rgb[..., 0].mean() > red_frame.rgb[..., 1].mean()


# --------------------------------------------------------------------------- #
# VideoFrameSource (cv2.VideoCapture monkeypatch fake)
# --------------------------------------------------------------------------- #

cv2 = pytest.importorskip("cv2", reason="vision extras required for VideoFrameSource")


def _checkerboard_bgr(rgb: tuple[int, int, int], h: int = 64, w: int = 64, square: int = 8) -> np.ndarray:
    """sharp edge 패턴 (Laplacian variance 큼). cv2.VideoCapture.read 가 BGR 반환하므로 BGR."""
    arr = np.full((h, w, 3), 255, dtype=np.uint8)  # 흰색 배경
    bgr = (rgb[2], rgb[1], rgb[0])
    for y in range(0, h, square):
        for x in range(0, w, square):
            if ((y // square) + (x // square)) % 2 == 0:
                arr[y:y + square, x:x + square] = bgr
    return arr


class _FakeCapture:
    """cv2.VideoCapture stand-in. 미리 frame list 받아 set/read 로 응답."""

    def __init__(self, frames_bgr: list[np.ndarray] | None, opened: bool = True) -> None:
        self._frames = frames_bgr or []
        self._opened = opened
        self._cursor = 0
        self.released = False

    def isOpened(self) -> bool:
        return self._opened

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(len(self._frames))
        return 0.0

    def set(self, prop: int, value: Any) -> bool:
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self._cursor = int(value)
            return True
        return False

    def read(self) -> tuple[bool, np.ndarray | None]:
        if 0 <= self._cursor < len(self._frames):
            return True, self._frames[self._cursor]
        return False, None

    def release(self) -> None:
        self.released = True


def _patch_capture(monkeypatch: pytest.MonkeyPatch, frames_bgr: list[np.ndarray] | None, *, opened: bool = True) -> _FakeCapture:
    fake = _FakeCapture(frames_bgr, opened=opened)
    monkeypatch.setattr(cv2, "VideoCapture", lambda _path: fake)
    return fake


def _default_video_cfg(**overrides: Any) -> VideoFrameSelectorConfig:
    base = {
        "n_candidate": 4,
        "n_final": 4,
        "blur_min": 100.0,
        "brightness_range": (30.0, 225.0),
        "scene_corr_max": 0.85,
        "histogram_bins": 32,
    }
    base.update(overrides)
    return VideoFrameSelectorConfig(**base)


def test_video_frame_source_protocol_compliance() -> None:
    # init 만으로 Protocol 준수.
    assert isinstance(VideoFrameSource(Path("/tmp/x.mp4"), _default_video_cfg()), FrameSource)


def test_video_frame_source_yields_frames_in_time_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """4 candidate (다른 색조) → 모두 quality+diversity pass → 4 Frame, 시간순 + global idx 보존."""
    frames_bgr = [
        _checkerboard_bgr((255, 0, 0)),    # red, gray=76
        _checkerboard_bgr((0, 255, 0)),    # green, gray=150
        _checkerboard_bgr((255, 0, 255)),  # magenta, gray=105
        _checkerboard_bgr((0, 255, 255)),  # cyan, gray=179
    ]
    fake = _patch_capture(monkeypatch, frames_bgr)

    src = VideoFrameSource(Path("/tmp/clip.mp4"), _default_video_cfg(n_candidate=4))
    frames = list(src.iter_frames())

    assert [f.source_type for f in frames] == ["video"] * 4
    # candidate indices = [0, 1, 2, 3] (total=4, n_cand=4 → 균등 = identity)
    assert [f.id for f in frames] == ["clip_f0", "clip_f1", "clip_f2", "clip_f3"]
    assert fake.released is True


def test_video_frame_source_uniform_sampling_across_total(monkeypatch: pytest.MonkeyPatch) -> None:
    """total=8, n_candidate=4 → indices [0, 2, 4, 6] 균등 sampling. fade-out 끝(7) 제외."""
    frames_bgr = [_checkerboard_bgr((255, 0, 0))] * 8
    # 단, 같은 frame 4번 보내면 NMS 로 1개만 keep — sampling 검증 목적이라 다른 색조로 채움.
    palette = [(255, 0, 0), (0, 255, 0), (255, 0, 255), (0, 255, 255),
               (200, 100, 50), (50, 200, 100), (100, 50, 200), (180, 180, 80)]
    frames_bgr = [_checkerboard_bgr(c) for c in palette]
    _patch_capture(monkeypatch, frames_bgr)

    src = VideoFrameSource(Path("/tmp/clip.mp4"), _default_video_cfg(n_candidate=4, n_final=4))
    frames = list(src.iter_frames())
    # candidate global idx 는 0/2/4/6. yield 도 그 안에서만.
    yielded_idx = [int(f.id.split("_f")[1]) for f in frames]
    assert set(yielded_idx).issubset({0, 2, 4, 6})


def test_video_frame_source_short_video_caps_n_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    """total=2 < n_candidate=4 → 2개만 후보로 들어감 (강제 패딩 X)."""
    frames_bgr = [
        _checkerboard_bgr((255, 0, 0)),
        _checkerboard_bgr((0, 255, 0)),
    ]
    _patch_capture(monkeypatch, frames_bgr)

    src = VideoFrameSource(Path("/tmp/short.mp4"), _default_video_cfg(n_candidate=4, n_final=4))
    frames = list(src.iter_frames())
    assert len(frames) == 2
    assert {f.id for f in frames} == {"short_f0", "short_f1"}


def test_video_frame_source_empty_video_yields_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """total=0 → 빈 iterator (예외 X). cap.release 는 finally 에서 보장."""
    fake = _patch_capture(monkeypatch, frames_bgr=[])
    src = VideoFrameSource(Path("/tmp/empty.mp4"), _default_video_cfg())
    assert list(src.iter_frames()) == []
    assert fake.released is True


def test_video_frame_source_open_failure_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _patch_capture(monkeypatch, frames_bgr=None, opened=False)
    src = VideoFrameSource(Path("/tmp/missing.mp4"), _default_video_cfg())
    with pytest.raises(RuntimeError, match="VideoCapture failed"):
        list(src.iter_frames())
    # isOpened 실패 → release 도 호출 안 됨 (예외 후 caller 가 처리).
    assert fake.released is False


def test_video_frame_source_quality_filter_drops_solid_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    """단색 frame (Laplacian≈0) → quality fail → drop. 체커보드만 yield."""
    solid = np.full((64, 64, 3), 128, dtype=np.uint8)  # 회색 단색
    frames_bgr = [
        solid,
        _checkerboard_bgr((255, 0, 0)),
        solid.copy(),
        _checkerboard_bgr((0, 255, 0)),
    ]
    _patch_capture(monkeypatch, frames_bgr)

    src = VideoFrameSource(Path("/tmp/clip.mp4"), _default_video_cfg(n_candidate=4, n_final=4))
    frames = list(src.iter_frames())
    yielded_idx = [int(f.id.split("_f")[1]) for f in frames]
    assert yielded_idx == [1, 3]


def test_video_frame_source_release_called_on_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """selector 가 어떤 이유로든 raise 해도 cap.release 보장."""
    frames_bgr = [_checkerboard_bgr((255, 0, 0))]
    fake = _patch_capture(monkeypatch, frames_bgr)

    # select_top_frames 를 raise 로 monkeypatch — finally 블록 검증 목적.
    import vision.frame_source as fs_module
    monkeypatch.setattr(
        fs_module,
        "select_top_frames",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    src = VideoFrameSource(Path("/tmp/clip.mp4"), _default_video_cfg(n_candidate=1))
    with pytest.raises(RuntimeError, match="boom"):
        list(src.iter_frames())
    assert fake.released is True
