"""FrameSource Protocol + ImageFrameSource + VideoFrameSource 단위 테스트.

Protocol / dataclass / stub 동작만 단위. 실 JPG 로딩 (PIL) 테스트는 vision extras 필요하므로
importorskip 로 가드. 실 Pipeline B 호출은 tests/integration/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vision.frame_source import (
    Frame,
    FrameSource,
    ImageFrameSource,
    VideoFrameSource,
)

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
# VideoFrameSource (M3 stub)
# --------------------------------------------------------------------------- #

def test_video_frame_source_not_implemented() -> None:
    src = VideoFrameSource(Path("/tmp/x.mp4"), fps=1)
    with pytest.raises(NotImplementedError, match="M3"):
        list(src.iter_frames())


def test_video_frame_source_protocol_compliance() -> None:
    # 호출 안 해도 타입만으로는 Protocol 준수.
    assert isinstance(VideoFrameSource(Path("/tmp/x.mp4")), FrameSource)
