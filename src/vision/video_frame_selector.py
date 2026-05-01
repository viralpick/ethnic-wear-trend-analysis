"""영상 frame selection — 균등 sampling + quality score + scene diversity NMS.

Two-pass:
  Pass 1: 영상 전체 균등 N candidate (interval = duration / n_candidate)
  Pass 2: 각 candidate quality_score 계산
  Pass 3: score desc 정렬 → NMS-style scene diversity 가드 → top n_final

선결 결정 (2026-04-28 user):
- candidate=50 / final=20 → 영상 길이 무관 cost 일정 (decode 50 × ~10ms = 500ms)
- 영상 짧아 candidate < 50 이면 자연스럽게 줄어듦 (강제 패딩 X)
- 길이 dependent interval: 30초 → 0.6초 / 5분 → 6초. 영상 전체 균등 커버 보장
- diversity NMS: score desc 순회, 직전 keep 과 hist_corr ≥ scene_corr_max 면 skip
  → 같은 장면 best frame N장 중복 차단. 다양한 컷 보장
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vision.frame_quality import compute_quality_score, histogram_correlation


@dataclass(frozen=True)
class VideoFrameSelectorConfig:
    """frame selection 파라미터. 모든 magic value 단일 source."""
    n_candidate: int = 30
    n_final: int = 15
    blur_min: float = 100.0
    brightness_range: tuple[float, float] = (30.0, 225.0)
    # scene change 임계값 (cv2 HISTCMP_CORREL). 이 값 이상이면 "같은 장면" 으로 보고 skip.
    # 0.85 = 강한 컷 차단 + 같은 의상 다른 각도는 통과 (fashion 영상 typical).
    scene_corr_max: float = 0.85
    histogram_bins: int = 32


def select_top_frames(
    candidates: list[np.ndarray],
    cfg: VideoFrameSelectorConfig,
) -> list[int]:
    """`candidates` 중 quality + diversity 통과한 top frame indices (시간순 정렬).

    candidates: 영상 전체에서 균등 sampling 된 RGB frame list (시간순).
    return: selected indices into candidates, 시간순 (input order 보존).

    빈 input 또는 quality 통과 frame 0개면 빈 list. n_final 못 채워도 강제 패딩 X.
    """
    if not candidates or cfg.n_final <= 0:
        return []

    scores = [
        compute_quality_score(
            c,
            blur_min=cfg.blur_min,
            brightness_range=cfg.brightness_range,
        )
        for c in candidates
    ]

    # quality fail (score=0) 제외.
    eligible = [(i, s) for i, s in enumerate(scores) if s > 0.0]
    if not eligible:
        return []

    # score desc 정렬 (동률 시 시간 asc — 결정론).
    eligible.sort(key=lambda item: (-item[1], item[0]))

    kept_indices: list[int] = []
    kept_frames: list[np.ndarray] = []
    for idx, _score in eligible:
        cand = candidates[idx]
        is_diverse = all(
            histogram_correlation(cand, kept, bins=cfg.histogram_bins) < cfg.scene_corr_max
            for kept in kept_frames
        )
        if not is_diverse:
            continue
        kept_indices.append(idx)
        kept_frames.append(cand)
        if len(kept_indices) >= cfg.n_final:
            break

    return sorted(kept_indices)
