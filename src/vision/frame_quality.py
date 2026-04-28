"""영상 frame quality scoring pure function — Laplacian + brightness + HSV histogram.

cv2 module-level import. vision/ 안 모듈이라 `[vision]` extras 의존. core (contracts/
attributes/ scoring/ aggregation/ 등) 에서 import 금지 (`.claude/CLAUDE.md` 격리 규칙).

알고리즘 근거:
- Laplacian variance: 사진 quality 의 학계 표준 motion-blur 메트릭. variance 낮을수록
  edge 가 흐릿 = blur. 100 cutoff 는 fashion/IG 콘텐츠 경험치
- HSV (H+S) 2D histogram correlation: 같은 장면의 노출 변화를 "다른 장면" 으로 오인
  방지 위해 V 채널 제외. cv2.HISTCMP_CORREL ∈ [-1, 1], 1 = 동일
"""
from __future__ import annotations

import cv2
import numpy as np


def compute_blur_score(rgb: np.ndarray) -> float:
    """Laplacian variance — 높을수록 선명. fashion 영상 컷 기준 100 미만이면 blur 판정."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_brightness(rgb: np.ndarray) -> float:
    """평균 밝기 (0~255). 30 미만 under-exposure / 225 초과 over-exposure."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.mean(gray)[0])


def compute_quality_score(
    rgb: np.ndarray,
    *,
    blur_min: float,
    brightness_range: tuple[float, float],
) -> float:
    """exposure fail 또는 blur fail 이면 0, 둘 다 통과 시 blur_score 그대로 반환.

    blur 의 절대값을 score 로 사용 — 더 선명한 frame 이 항상 우선. 가중치 / 정규화 X
    (영상 마다 blur scale 이 달라 정규화하면 cross-video 비교만 가능, 영상 내부 ranking
    에는 raw value 가 직접적).
    """
    brightness = compute_brightness(rgb)
    if not (brightness_range[0] <= brightness <= brightness_range[1]):
        return 0.0
    blur = compute_blur_score(rgb)
    if blur < blur_min:
        return 0.0
    return blur


def histogram_correlation(
    rgb_a: np.ndarray,
    rgb_b: np.ndarray,
    *,
    bins: int = 32,
) -> float:
    """HSV (H+S) 2D color histogram correlation. 1.0 = 동일 장면, < 0.85 ≈ scene change.

    V 채널 제외 이유: 노출 변화 (실내/실외 같은 장면에서 빛 변화) 가 컷으로 오인되는
    false positive 차단. fashion 영상은 색조 (H+S) 가 의상 식별 핵심 신호.
    """
    hsv_a = cv2.cvtColor(rgb_a, cv2.COLOR_RGB2HSV)
    hsv_b = cv2.cvtColor(rgb_b, cv2.COLOR_RGB2HSV)
    hist_a = cv2.calcHist([hsv_a], [0, 1], None, [bins, bins], [0, 180, 0, 256])
    hist_b = cv2.calcHist([hsv_b], [0, 1], None, [bins, bins], [0, 180, 0, 256])
    cv2.normalize(hist_a, hist_a)
    cv2.normalize(hist_b, hist_b)
    return float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL))
