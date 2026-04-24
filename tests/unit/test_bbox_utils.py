"""bbox_utils — normalized (x,y,w,h) → pixel (x1,y1,x2,y2) 변환 pinning."""
from __future__ import annotations

from vision.bbox_utils import normalized_xywh_to_pixel_xyxy


def test_normalized_to_pixel_basic() -> None:
    # 중앙 절반 bbox, 800×1000 image (H×W). x*W, y*H.
    result = normalized_xywh_to_pixel_xyxy((0.25, 0.25, 0.5, 0.5), 800, 1000)
    assert result == (250, 200, 750, 600)


def test_boundary_overshoot_is_clipped() -> None:
    # LLM 이 1.0001 내는 케이스. image 경계로 clip.
    result = normalized_xywh_to_pixel_xyxy((0.0, 0.0, 1.0001, 1.0001), 100, 100)
    assert result == (0, 0, 100, 100)


def test_too_small_crop_returns_none() -> None:
    # w = 0.10 × 100 = 10 px → MIN_CROP_PX(32) 미만.
    assert normalized_xywh_to_pixel_xyxy((0.0, 0.0, 0.10, 0.50), 100, 100) is None
    # h 만 작아도 None.
    assert normalized_xywh_to_pixel_xyxy((0.0, 0.0, 0.50, 0.10), 100, 100) is None
    # 둘 다 충분히 크면 반환.
    assert normalized_xywh_to_pixel_xyxy((0.0, 0.0, 0.50, 0.50), 100, 100) == (0, 0, 50, 50)
