"""Phase B3b — post-level palette 생산.

`CanonicalOutfit.palette` 들을 area_ratio × within-share 가중으로 flatten 한 뒤,
ΔE76 ≤ 10 greedy pair-merge → share<0.05 drop → top N cap (3) → sum=1.0 재정규화.
결과는 post 대표 색 최대 3개 (`list[PaletteCluster]`).

설계 원칙:
- pure function — canonical pydantic 입력, pydantic PaletteCluster list 출력.
- weight 은 pixel 증거만: `weight = representative.person_bbox_area_ratio *
  palette_cluster.share`. "post 전체에서 이 색이 차지하는 물리적 지분" 해석 — 작은
  canonical 의 색이 큰 canonical 의 색과 동등해지는 왜곡 차단.
- merge / drop / hex 변환은 `vision.palette_merge_utils` 공용 헬퍼 (single source).
- PRESET_MATCH_THRESHOLD (15.0, canonical_palette) 와 다른 값 (10.0) — preset 매칭용
  이 아니라 merge 용이므로 혼용 금지.
"""
from __future__ import annotations

from contracts.common import PaletteCluster
from contracts.vision import CanonicalOutfit
from vision.color_space import hex_to_rgb, rgb_to_lab
from vision.palette_merge_utils import (
    WeightedColorEntry,
    drop_small_and_cap,
    merge_greedy_lab,
    to_pydantic_palette,
)

MAX_POST_PALETTE_CLUSTERS: int = 3
POST_PALETTE_MERGE_DELTA_E: float = 10.0
MIN_POST_PALETTE_SHARE: float = 0.05


def _flatten(canonicals: list[CanonicalOutfit]) -> list[WeightedColorEntry]:
    """canonical 들의 palette 를 area_ratio × within-share 가중으로 펼친다.

    palette 가 비거나 area_ratio=0 인 canonical 은 자연스럽게 skip. hex → RGB → LAB 는
    `color_space` 의 정확한 D65 경로를 그대로 사용.
    """
    out: list[WeightedColorEntry] = []
    for canonical in canonicals:
        area = canonical.representative.person_bbox_area_ratio
        if area <= 0.0 or not canonical.palette:
            continue
        for cluster in canonical.palette:
            weight = float(area) * float(cluster.share)
            if weight <= 0.0:
                continue
            rgb = hex_to_rgb(cluster.hex)
            lab = rgb_to_lab(rgb)
            out.append(
                WeightedColorEntry(
                    lab=(float(lab[0]), float(lab[1]), float(lab[2])),
                    weight=weight,
                    family=cluster.family,
                )
            )
    return out


def build_post_palette(
    canonicals: list[CanonicalOutfit],
    *,
    merge_deltae76_threshold: float = POST_PALETTE_MERGE_DELTA_E,
    min_cluster_share: float = MIN_POST_PALETTE_SHARE,
    max_clusters: int = MAX_POST_PALETTE_CLUSTERS,
) -> list[PaletteCluster]:
    """canonical 들의 palette 를 area_ratio × within-share 가중으로 merge → top N.

    빈 입력 / 모든 canonical 이 palette=[] → `[]`. Fake path (canonicals=[]) 도 자연
    빈 반환이 되어 snapshot/smoke 에서 별도 분기 불필요.
    """
    flat = _flatten(canonicals)
    if not flat:
        return []
    merged = merge_greedy_lab(flat, merge_deltae76_threshold)
    capped = drop_small_and_cap(
        merged, min_share=min_cluster_share, max_clusters=max_clusters,
    )
    return to_pydantic_palette(capped)


__all__ = [
    "MAX_POST_PALETTE_CLUSTERS",
    "MIN_POST_PALETTE_SHARE",
    "POST_PALETTE_MERGE_DELTA_E",
    "build_post_palette",
]
