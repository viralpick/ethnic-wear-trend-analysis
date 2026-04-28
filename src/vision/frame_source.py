"""Frame source 추상화 — 이미지/영상을 공통 Frame 단위로 공급.

- IG 정적 이미지 / IG 캐러셀 이미지 → `ImageFrameSource` (PIL).
- IG Reel / YT 영상 / IG 캐러셀 video → `VideoFrameSource` (cv2 + selector).

VideoFrameSource: 영상 전체 균등 N candidate sampling → quality score (Laplacian
variance + brightness 게이트) → scene diversity NMS (HSV H+S histogram corr) → top
n_final Frame yield. cv2.VideoCapture wrapper. selector cfg 만 외부 입력.

PipelineBExtractor 는 FrameSource 를 받아 `iter_frames()` 로 돌림 → garment segmentation.
PIL/cv2 는 각 구현 내부에서만 lazy import — Protocol/dataclass 자체는 numpy 만 필요.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal, Protocol, runtime_checkable

import numpy as np

from vision.video_frame_selector import VideoFrameSelectorConfig, select_top_frames

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
    """영상 1건 → quality + diversity 통과한 top n_final Frame.

    Two-pass: cv2.VideoCapture 로 균등 cfg.n_candidate frame 추출 → select_top_frames 로
    quality+diversity 필터 → 시간순 Frame yield. Frame.id = "{video_stem}_f{global_idx}"
    (영상 전체 frame index, candidate index 아님 — drilldown 추적용).

    매우 짧은 영상 (total_frame < n_candidate) 도 자연스럽게 동작 — n_candidate 가 total
    로 cap 되어 모든 frame 이 후보에 들어감. 영상 열기 실패 / FRAME_COUNT 0 이면 빈 iterator.
    """

    def __init__(self, video_path: Path, cfg: VideoFrameSelectorConfig) -> None:
        self._video_path = Path(video_path)
        self._cfg = cfg

    def iter_frames(self) -> Iterator[Frame]:
        import cv2  # lazy — `uv sync --extra vision` 필요

        cap = cv2.VideoCapture(str(self._video_path))
        if not cap.isOpened():
            raise RuntimeError(f"cv2.VideoCapture failed to open: {self._video_path}")
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                return
            n_cand = min(self._cfg.n_candidate, total)
            # 균등 sampling: 0..total-1 에서 n_cand 개. 끝 frame 은 보통 fade-out 이라 제외.
            cand_indices = [int(i * total / n_cand) for i in range(n_cand)]
            candidates: list[tuple[int, np.ndarray]] = []
            for idx in cand_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, bgr = cap.read()
                if not ok or bgr is None:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                candidates.append((idx, rgb))
            if not candidates:
                return
            rgb_only = [rgb for _, rgb in candidates]
            kept_local = select_top_frames(rgb_only, self._cfg)
            stem = self._video_path.stem
            for local_idx in kept_local:
                global_idx, rgb = candidates[local_idx]
                yield Frame(
                    id=f"{stem}_f{global_idx}",
                    rgb=rgb,
                    source_type="video",
                )
        finally:
            cap.release()
