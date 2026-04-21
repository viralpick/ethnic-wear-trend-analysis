"""Frame source 추상화 — 이미지/영상을 공통 Frame 단위로 공급 (spec §7.2 경계 반영).

Step C (2026-04-21): IG 정적 이미지 포스트 → ImageFrameSource (캐러셀 N장).
                     IG Reel → VideoFrameSource (M3 에서 ffmpeg 연결).
                     YT 영상 → 이 모듈 호출 대상 아님 (spec §7.2).

PipelineBExtractor 는 FrameSource 를 받아 `iter_frames()` 로 돌림 → garment segmentation.
PIL 은 ImageFrameSource 내부에서만 lazy import — Protocol/dataclass 자체는 numpy 만 필요.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal, Protocol, runtime_checkable

import numpy as np

FrameSourceKind = Literal["image", "video"]


@dataclass(frozen=True)
class Frame:
    """frame 1개. 정적 이미지 1장 또는 영상의 1 프레임."""
    id: str
    rgb: np.ndarray       # (H, W, 3) uint8
    source_type: FrameSourceKind


@runtime_checkable
class FrameSource(Protocol):
    def iter_frames(self) -> Iterator[Frame]: ...


class ImageFrameSource:
    """JPG/PNG 파일 N장 → N Frame. IG 캐러셀 전형적 사용처.

    PIL 은 lazy import (iter_frames 안). frame_source 모듈 자체는 vision extras 없이도 import
    가능 — Protocol/dataclass 기반 코드 (예: Fake 구현) 를 core 에서 쓸 수 있게.
    """

    def __init__(self, paths: list[Path]) -> None:
        self._paths = [Path(p) for p in paths]

    def iter_frames(self) -> Iterator[Frame]:
        from PIL import Image  # lazy — `uv sync --extra vision` 필요

        for path in self._paths:
            img = Image.open(path).convert("RGB")
            yield Frame(
                id=path.stem,
                rgb=np.asarray(img, dtype=np.uint8),
                source_type="image",
            )


class VideoFrameSource:
    """영상 1건 → fps 에 따른 N Frame. M3 에서 ffmpeg subprocess 연결 예정.

    현재는 import 는 가능하되 iter_frames 호출 시 NotImplementedError.
    spec §7.2: YT 영상은 이 클래스 호출 대상 아님 (IG Reel 용). 호출부에서 ContentSource
    check 해서 YT 는 reject.
    """

    def __init__(self, video_path: Path, fps: int = 1) -> None:
        self._video_path = Path(video_path)
        self._fps = fps

    def iter_frames(self) -> Iterator[Frame]:
        raise NotImplementedError(
            "VideoFrameSource: M3 에서 ffmpeg subprocess 연결 예정. "
            "YT 영상은 spec §7.2 에 따라 호출 대상 아님 — IG Reel 에만 사용할 것."
        )
