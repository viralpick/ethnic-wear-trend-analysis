"""BBOX 좌표 변환 — LLM normalized (x, y, w, h) → crop pixel (x1, y1, x2, y2).

Phase 3 canonical_extractor 전용 유틸. `EthnicOutfit.person_bbox` 는 Phase 2 LLM 이 내는
[x, y, w, h] normalized [0..1] 좌표인 반면 `run_segformer` / crop_rgb 는 pixel int
(x1, y1, x2, y2) 를 기대한다. 변환 + 경계 clip + MIN_CROP_PX 가드.

MIN_CROP_PX 는 `pipeline_b_extractor.MIN_CROP_PX` 와 동일 값으로 유지 (segformer 입력이
너무 작으면 upsample 이 의미 없음). 순환 import 회피 위해 별도 상수로 복제.
"""
from __future__ import annotations

MIN_CROP_PX = 32


def normalized_xywh_to_pixel_xyxy(
    bbox_01: tuple[float, float, float, float],
    image_height: int,
    image_width: int,
) -> tuple[int, int, int, int] | None:
    """(x, y, w, h) ∈ [0..1] → (x1, y1, x2, y2) int pixel. 너무 작으면 None.

    - LLM 이 경계 overshoot (w=1.0001 등) 를 낼 수 있어 image_width/height 로 clip.
    - MIN_CROP_PX 미만 변이면 None — 호출부에서 skip signal 로 사용.
    """
    x, y, w, h = bbox_01
    x1 = int(round(x * image_width))
    y1 = int(round(y * image_height))
    x2 = int(round((x + w) * image_width))
    y2 = int(round((y + h) * image_height))
    x1c = max(0, min(image_width, x1))
    y1c = max(0, min(image_height, y1))
    x2c = max(0, min(image_width, x2))
    y2c = max(0, min(image_height, y2))
    if x2c - x1c < MIN_CROP_PX or y2c - y1c < MIN_CROP_PX:
        return None
    return x1c, y1c, x2c, y2c
