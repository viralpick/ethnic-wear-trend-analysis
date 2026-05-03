"""Phase B3c + β4 — cluster-level palette 생산 (β4 share-weighted vote).

한 trend cluster 에 속한 여러 post 의 `post_palette` (각 sum=1.0) 을 평탄화해
ΔE76 ≤ 10 greedy pair-merge → share<0.05 drop → top 5 cap → sum=1.0 재정규화.
결과는 `DrilldownPayload.color_palette` 에 그대로 실린다.

가중치 공식 (β4 변경 전 옵션 A, 2026-04-24 advisor 합의):
- `cluster_weight = post_palette_cluster.share` — post 간 동등 (one-post-one-vote)
- 3층 (canonical → post → cluster) 의 각 층이 전 층의 정규화된 결과만 소비하는 흐름.

β4 (2026-04-28) 부터 per-post weight 추가 — multi-fan-out item 이 cluster 안에서
가지는 share 가 cluster_weight 에 곱해진다. 즉 `cluster_weight = post_palette_cluster.share
× item_share_in_cluster`. multi-cluster 가 같은 post_palette 를 다른 가중으로 소비.

merge / drop / hex 변환은 `vision.palette_merge_utils` 공용 헬퍼 (single source) —
post_palette 와 동일 알고리즘.
"""
from __future__ import annotations

from contracts.common import PaletteCluster
from vision.color_space import hex_to_rgb, rgb_to_lab
from vision.palette_merge_utils import (
    WeightedColorEntry,
    drop_small_and_cap,
    merge_greedy_lab,
    to_pydantic_palette,
)

MAX_CLUSTER_PALETTE_CLUSTERS: int = 5
CLUSTER_PALETTE_MERGE_DELTA_E: float = 10.0
MIN_CLUSTER_PALETTE_SHARE: float = 0.05


def _flatten(
    post_palettes_with_weight: list[tuple[list[PaletteCluster], float]],
) -> list[WeightedColorEntry]:
    """각 post 의 post_palette 를 평탄화 (β4 share-weighted).

    weight = cluster.share × post_weight (β4: post_weight = item_share_in_cluster).
    post_weight ≤ 0 이거나 cluster.share ≤ 0 이면 skip. hex → RGB → LAB 는 color_space
    의 D65 경로를 사용해 증거 체인을 유지.
    """
    out: list[WeightedColorEntry] = []
    for palette, post_weight in post_palettes_with_weight:
        if not palette or post_weight <= 0.0:
            continue
        for cluster in palette:
            weight = float(cluster.share) * float(post_weight)
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


def build_cluster_palette(
    post_palettes_with_weight: list[tuple[list[PaletteCluster], float]],
    *,
    merge_deltae76_threshold: float = CLUSTER_PALETTE_MERGE_DELTA_E,
    min_cluster_share: float = MIN_CLUSTER_PALETTE_SHARE,
    max_clusters: int = MAX_CLUSTER_PALETTE_CLUSTERS,
) -> list[PaletteCluster]:
    """여러 post 의 post_palette 들을 cluster 대표 색 최대 `max_clusters` 로 병합 (β4).

    각 entry = (post_palette, post_weight). post_weight = β3 이전 1.0, β4 부터 cluster
    안 item share. 빈 입력 / 모든 post_palette 가 빈 경우 → `[]`. 입력은 사전에 post_palette
    를 거친 결과 (각 sum=1.0) 라 가정하지만, 본 함수는 그 invariant 를 자체 검증하지
    않는다 — 호출 체인에서 보장 (build_cluster_summary → build_cluster_palette).
    """
    flat = _flatten(post_palettes_with_weight)
    if not flat:
        return []
    merged = merge_greedy_lab(flat, merge_deltae76_threshold)
    capped = drop_small_and_cap(
        merged, min_share=min_cluster_share, max_clusters=max_clusters,
    )
    return to_pydantic_palette(capped)


__all__ = [
    "CLUSTER_PALETTE_MERGE_DELTA_E",
    "MAX_CLUSTER_PALETTE_CLUSTERS",
    "MIN_CLUSTER_PALETTE_SHARE",
    "build_cluster_palette",
]
