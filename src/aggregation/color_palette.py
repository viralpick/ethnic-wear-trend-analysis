"""Post-level ColorInfo 리스트 → drill-down palette chip (spec §4.1 ④, §5.4).

Step B (2026-04-21): bucket quantize 방식 → LAB KMeans 로 교체.
- 각 ColorInfo = post 1건의 대표 RGB (VLM/Pipeline B 출력)
- 이 리스트를 LAB 공간으로 변환 → KMeans(top_k, random_state=0) → cluster 별 대표색 + pct
- family 는 각 cluster 에 속한 ColorInfo 들의 family 최빈값

frame-level (pixel sample) aggregation 은 Step C 의 pipeline_b_extractor 에서 별도. 이 모듈은
오직 "post-level ColorInfo N개 → palette chip K개" 를 담당.

결정론성: random_state=0 고정. 같은 입력 → 같은 palette (snapshot 테스트 호환).
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.cluster import KMeans

from contracts.common import ColorFamily, ColorPaletteItem
from contracts.enriched import ColorInfo
from settings import PaletteConfig
from vision.color_space import lab_to_rgb, rgb_to_hex, rgb_to_lab


def _family_of(members: list[ColorInfo]) -> ColorFamily:
    """cluster 에 속한 ColorInfo 들의 family 최빈값. 없으면 NEUTRAL."""
    families = [c.family for c in members if c.family is not None]
    if not families:
        return ColorFamily.NEUTRAL
    return Counter(families).most_common(1)[0][0]


def _round_rgb(rgb_float: np.ndarray) -> tuple[int, int, int]:
    """LAB centroid → RGB float → 0~255 int clamp + round."""
    clipped = np.clip(rgb_float, 0, 255).round().astype(int)
    return int(clipped[0]), int(clipped[1]), int(clipped[2])


def build_palette(
    colors: list[ColorInfo], cfg: PaletteConfig
) -> list[ColorPaletteItem]:
    """ColorInfo 리스트 → top_k ColorPaletteItem. weight desc sort."""
    if not colors:
        return []

    rgb = np.array([[c.r, c.g, c.b] for c in colors], dtype=np.float32)
    lab = rgb_to_lab(rgb)

    k_eff = min(cfg.top_k, len(colors))
    km = KMeans(n_clusters=k_eff, n_init=4, random_state=0).fit(lab)
    labels = km.labels_
    centers_lab = km.cluster_centers_
    centers_rgb = lab_to_rgb(centers_lab)

    total = len(colors)
    palette: list[ColorPaletteItem] = []
    for cluster_idx in range(k_eff):
        members = [colors[i] for i, label in enumerate(labels) if label == cluster_idx]
        if not members:
            continue
        r, g, b = _round_rgb(centers_rgb[cluster_idx])
        palette.append(
            ColorPaletteItem(
                r=r, g=g, b=b,
                hex_display=rgb_to_hex(np.array([r, g, b])),
                name=f"cluster_{cluster_idx}_{r:02x}{g:02x}{b:02x}",
                family=_family_of(members),
                pct=len(members) / total,
            )
        )
    palette.sort(key=lambda item: -item.pct)
    return palette
