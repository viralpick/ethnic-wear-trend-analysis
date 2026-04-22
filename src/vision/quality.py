"""Pipeline B palette quality metrics — post 별 퀄리티 진단.

목적:
  - comparison.html 에 post 별 🟢 OK / 🟡 warning / 🔴 danger badge 표시
  - quality.json 에 기계 판독용 dump
  - noise 가 dominant 한 post 를 cluster-level aggregation 전에 flag/drop 근거

현재 metrics (Phase 1):
  skin_leak_count            — palette 중 skin LAB box 안에 드는 chip 수
  chip_similarity_warning    — palette chip 간 ΔE76 < threshold 인 쌍 수 (중복 chip)
  post_total_garment_pixels  — 모든 frame 의 cleaned garment pixel 합 (신뢰도 힌트)
  yolo_detected_persons      — YOLO 가 감지한 person bbox 수 (전체 frame 합산)
  fallback_triggered         — 어느 frame 이라도 fallback_full_image 발동했는가

Phase 2 후보 (frame-level / product-only 판정):
  skin_pixels_in_crop_ratio  — fallback 발동 시 skin class 비율 (사람 없으면 소품 추정)
  frame_palette_consistency  — 캐러셀 frame 간 top-1 ΔE76 편차
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from contracts.common import ColorPaletteItem
from vision.color_space import delta_e76, hex_skin_leak, rgb_to_lab

_DEFAULT_CHIP_SIM_DE = 10.0
_MIN_SAFE_TOTAL_PIXELS = 5000


@dataclass(frozen=True)
class PostQuality:
    """post 단위 Pipeline B 퀄리티 지표."""
    skin_leak_count: int
    chip_similarity_warning: int
    post_total_garment_pixels: int
    yolo_detected_persons: int
    fallback_triggered: bool
    garment_class_counts: dict[str, int] = field(default_factory=dict)

    @property
    def level(self) -> str:
        """badge 레벨 — ok / warning / danger."""
        if self.post_total_garment_pixels < _MIN_SAFE_TOTAL_PIXELS:
            return "danger"
        if self.skin_leak_count > 0 or self.chip_similarity_warning > 0:
            return "warning"
        if self.fallback_triggered:
            return "warning"
        return "ok"

    @property
    def badge(self) -> str:
        return {"ok": "🟢", "warning": "🟡", "danger": "🔴"}[self.level]


def count_skin_leaks(palette: list[ColorPaletteItem]) -> int:
    """palette chip 중 skin LAB box 안에 드는 것 수 (drop_skin 사후 누수 검증)."""
    return sum(1 for c in palette if hex_skin_leak(c.hex_display))


def count_chip_similarity(
    palette: list[ColorPaletteItem],
    threshold_de: float = _DEFAULT_CHIP_SIM_DE,
) -> int:
    """palette chip 쌍 중 ΔE76 < threshold 인 쌍 수 (중복 chip — k 낮추기 신호)."""
    if len(palette) < 2:
        return 0
    labs = [rgb_to_lab(np.array([c.r, c.g, c.b])) for c in palette]
    warnings = 0
    for i in range(len(labs)):
        for j in range(i + 1, len(labs)):
            if delta_e76(labs[i], labs[j]) < threshold_de:
                warnings += 1
    return warnings
