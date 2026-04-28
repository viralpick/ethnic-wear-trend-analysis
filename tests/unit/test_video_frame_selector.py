"""video_frame_selector 알고리즘 pinning — 50 candidate → score → top N + diversity NMS.

검증 axes:
- empty input → empty output
- 모든 frame quality fail (단색/너무 어두움) → empty
- 시간순 정렬 보존 (input order 0,1,2... → output indices ascending)
- diversity NMS: 동일 frame 두 번 후보 → 하나만 keep
- n_final cap: 후보 30 + 모두 quality pass + 모두 diverse → cap=20 만 keep
- 다양한 색조 + 단색 mix → 단색 (quality fail) 자동 제외 + 색조 frame 만 keep
"""
from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from vision.video_frame_selector import VideoFrameSelectorConfig, select_top_frames


def _solid_color(rgb: tuple[int, int, int], h: int = 64, w: int = 64) -> np.ndarray:
    return np.tile(np.array(rgb, dtype=np.uint8), (h, w, 1))


def _checkerboard_color(rgb: tuple[int, int, int], h: int = 64, w: int = 64, square: int = 8) -> np.ndarray:
    """RGB + 흰색 alternating sharp edge.

    inverse(255-rgb) 대신 흰색 사용 — yellow vs blue 처럼 inverse 로 hist 분포가 같아져
    diversity 검증이 깨지는 함정 방지. 흰색은 모든 frame 공통 (S=0 bin) 이라 H+S 색조
    bin 만 frame 간 차별화.
    """
    arr = np.full((h, w, 3), 255, dtype=np.uint8)   # 기본 흰색.
    for y in range(0, h, square):
        for x in range(0, w, square):
            if ((y // square) + (x // square)) % 2 == 0:
                arr[y:y + square, x:x + square] = rgb
    return arr


def _default_cfg(**overrides) -> VideoFrameSelectorConfig:
    base = {
        "n_candidate": 50,
        "n_final": 20,
        "blur_min": 100.0,
        "brightness_range": (30.0, 225.0),
        "scene_corr_max": 0.85,
        "histogram_bins": 32,
    }
    base.update(overrides)
    return VideoFrameSelectorConfig(**base)


def test_empty_candidates_returns_empty() -> None:
    assert select_top_frames([], _default_cfg()) == []


def test_all_quality_fail_returns_empty() -> None:
    """단색 frame = blur 0 → 전부 quality fail → 빈 결과."""
    candidates = [_solid_color((128, 128, 128)) for _ in range(10)]
    assert select_top_frames(candidates, _default_cfg()) == []


def test_indices_returned_in_time_order() -> None:
    """input 순서 (시간 asc) 가 output 에 보존."""
    # 색조 4개 — H 다름 + brightness 모두 30~225 안.
    # blue (gray=29) / yellow (gray=226) 는 exposure cutoff 걸려서 magenta/cyan 사용.
    candidates = [
        _checkerboard_color((255, 0, 0)),     # red, gray=76
        _checkerboard_color((0, 255, 0)),     # green, gray=150
        _checkerboard_color((255, 0, 255)),   # magenta, gray=105
        _checkerboard_color((0, 255, 255)),   # cyan, gray=179
    ]
    selected = select_top_frames(candidates, _default_cfg(n_final=4))
    assert selected == sorted(selected)
    assert selected == [0, 1, 2, 3]


def test_diversity_nms_dedups_duplicates() -> None:
    """동일 frame 3번 + 다른 색조 1개 → diversity NMS 가 dup 차단, 2개만 keep."""
    same = _checkerboard_color((255, 0, 0))
    other = _checkerboard_color((0, 0, 255))
    candidates = [same, same.copy(), same.copy(), other]
    selected = select_top_frames(candidates, _default_cfg(n_final=10))
    assert len(selected) == 2
    # 시간 순서: 첫 same (0) + other (3).
    assert selected == [0, 3]


def test_n_final_cap_enforced() -> None:
    """30 candidate 가 모두 diverse + quality pass 일 때 cap=5 까지만."""
    # 30개 색조 (H 채널 각도 다름) — 모두 diverse 해야 cap 검증.
    candidates = []
    for i in range(30):
        # H 0, 6, 12, ... 174 (30 step). HSV.H = 0~180 in cv2.
        # H 를 RGB 로 변환하기 어려우니 다양한 RGB 직접:
        r = (i * 37 + 80) % 200 + 30
        g = (i * 53 + 40) % 200 + 30
        b = (i * 71 + 120) % 200 + 30
        candidates.append(_checkerboard_color((r, g, b)))

    selected = select_top_frames(candidates, _default_cfg(n_final=5))
    assert len(selected) <= 5
    # 시간 순서 보존.
    assert selected == sorted(selected)


def test_quality_fails_filtered_before_nms() -> None:
    """단색 (blur=0) 와 체커보드 (blur 큼) mix → 체커보드만 keep."""
    candidates = [
        _solid_color((128, 128, 128)),         # quality fail
        _checkerboard_color((255, 100, 50)),   # quality pass
        _solid_color((10, 10, 10)),            # quality fail (under-exposure)
        _checkerboard_color((50, 200, 100)),   # quality pass
        _solid_color((250, 250, 250)),         # quality fail (over-exposure)
    ]
    selected = select_top_frames(candidates, _default_cfg(n_final=10))
    assert selected == [1, 3]


def test_n_final_zero_returns_empty() -> None:
    candidates = [_checkerboard_color((255, 0, 0))]
    assert select_top_frames(candidates, _default_cfg(n_final=0)) == []
