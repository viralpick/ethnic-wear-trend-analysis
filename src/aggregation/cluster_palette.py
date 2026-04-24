"""Phase B3c — cluster-level palette 생산.

한 trend cluster 에 속한 여러 post 의 `post_palette` (각 sum=1.0) 을 평탄화해
ΔE76 ≤ 10 greedy pair-merge → share<0.05 drop → top 5 cap → sum=1.0 재정규화.
결과는 `DrilldownPayload.color_palette` 에 그대로 실린다.

가중치 공식 (옵션 A, 2026-04-24 advisor 합의):
- `cluster_weight = post_palette_cluster.share` — post 간 동등 (one-post-one-vote)
- 3층 (canonical → post → cluster) 의 각 층이 전 층의 정규화된 결과만 소비하는 흐름.
- 작은 post 의 튀는 색이 과대표현되면 configs/local.yaml knob 으로 옵션 B (× post
  물리 질량) 도입 여지 남김.

duplication note: `_Weighted` / `_merge_greedy` / `_drop_small_and_cap` / `_to_pydantic`
/ `_pick_family` 는 vision.post_palette 와 거의 동일. B3c 완료 후 shared util 로 추출
예정 (safe-refactor — 한 번에 여러 모듈 대규모 변경 금지).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from contracts.common import ColorFamily, PaletteCluster
from vision.color_space import hex_to_rgb, lab_to_rgb, rgb_to_hex, rgb_to_lab

MAX_CLUSTER_PALETTE_CLUSTERS: int = 5
CLUSTER_PALETTE_MERGE_DELTA_E: float = 10.0
MIN_CLUSTER_PALETTE_SHARE: float = 0.05


@dataclass(frozen=True)
class _Weighted:
    lab: tuple[float, float, float]
    weight: float
    family: ColorFamily | None


def _family_order(family: ColorFamily | None) -> tuple[int, int]:
    """merge tiebreak 키 — (None 후순위, enum 선언 순서)."""
    if family is None:
        return (1, 0)
    members: list[Enum] = list(ColorFamily)
    return (0, members.index(family))


def _pick_family(
    a_weight: float, a_family: ColorFamily | None,
    b_weight: float, b_family: ColorFamily | None,
) -> ColorFamily | None:
    a_key = (-a_weight, *_family_order(a_family))
    b_key = (-b_weight, *_family_order(b_family))
    return a_family if a_key <= b_key else b_family


def _flatten(post_palettes: list[list[PaletteCluster]]) -> list[_Weighted]:
    """각 post 의 post_palette 를 one-post-one-vote 로 평탄화.

    weight = cluster.share 그대로 — post 간 동등 가중. hex → RGB → LAB 는 color_space
    의 D65 경로를 사용해 증거 체인을 유지.
    """
    out: list[_Weighted] = []
    for palette in post_palettes:
        if not palette:
            continue
        for cluster in palette:
            weight = float(cluster.share)
            if weight <= 0.0:
                continue
            rgb = hex_to_rgb(cluster.hex)
            lab = rgb_to_lab(rgb)
            out.append(
                _Weighted(
                    lab=(float(lab[0]), float(lab[1]), float(lab[2])),
                    weight=weight,
                    family=cluster.family,
                )
            )
    return out


def _merge_greedy(items: list[_Weighted], threshold: float) -> list[_Weighted]:
    """post_palette._merge_greedy 와 동일 규칙. 가장 가까운 pair 부터 단일 merge.

    tiebreak: (distance asc, i asc, j asc). 입력 크기 <= ~post수 × 3 이라 O(k^3) 무관.
    """
    working = list(items)
    while len(working) > 1:
        best: tuple[float, int, int] | None = None
        for i in range(len(working)):
            for j in range(i + 1, len(working)):
                li = np.array(working[i].lab, dtype=np.float32)
                lj = np.array(working[j].lab, dtype=np.float32)
                d = float(np.linalg.norm(li - lj))
                if d >= threshold:
                    continue
                if best is None or (d, i, j) < best:
                    best = (d, i, j)
        if best is None:
            break
        _d, i, j = best
        a, b = working[i], working[j]
        total = a.weight + b.weight
        merged_lab = tuple(
            (a.weight * ai + b.weight * bi) / total
            for ai, bi in zip(a.lab, b.lab)
        )
        merged = _Weighted(
            lab=(float(merged_lab[0]), float(merged_lab[1]), float(merged_lab[2])),
            weight=total,
            family=_pick_family(a.weight, a.family, b.weight, b.family),
        )
        working.pop(j)
        working[i] = merged
    return working


def _drop_small_and_cap(
    items: list[_Weighted], min_share: float, max_clusters: int,
) -> list[_Weighted]:
    total = sum(it.weight for it in items)
    if total <= 0.0:
        return []
    kept = [it for it in items if (it.weight / total) >= min_share]
    if not kept:
        return []
    kept.sort(key=lambda it: -it.weight)
    return kept[:max_clusters]


def _to_pydantic(items: list[_Weighted]) -> list[PaletteCluster]:
    if not items:
        return []
    total = sum(it.weight for it in items)
    if total <= 0.0:
        return []
    labs = np.array([it.lab for it in items], dtype=np.float32)
    rgbs = lab_to_rgb(labs)
    out: list[PaletteCluster] = []
    for it, rgb in zip(items, rgbs):
        out.append(
            PaletteCluster(
                hex=rgb_to_hex(rgb),
                share=float(it.weight / total),
                family=it.family,
            )
        )
    return out


def build_cluster_palette(
    post_palettes: list[list[PaletteCluster]],
    *,
    merge_deltae76_threshold: float = CLUSTER_PALETTE_MERGE_DELTA_E,
    min_cluster_share: float = MIN_CLUSTER_PALETTE_SHARE,
    max_clusters: int = MAX_CLUSTER_PALETTE_CLUSTERS,
) -> list[PaletteCluster]:
    """여러 post 의 post_palette 들을 cluster 대표 색 최대 `max_clusters` 로 병합.

    빈 입력 / 모든 post_palette 가 빈 경우 → `[]`. 입력은 사전에 post_palette 를
    거친 결과 (각 sum=1.0) 라 가정하지만, 본 함수는 그 invariant 를 자체 검증하지
    않는다 — 호출 체인에서 보장 (build_cluster_summary → build_cluster_palette).
    """
    flat = _flatten(post_palettes)
    if not flat:
        return []
    merged = _merge_greedy(flat, merge_deltae76_threshold)
    capped = _drop_small_and_cap(merged, min_cluster_share, max_clusters)
    return _to_pydantic(capped)


__all__ = [
    "CLUSTER_PALETTE_MERGE_DELTA_E",
    "MAX_CLUSTER_PALETTE_CLUSTERS",
    "MIN_CLUSTER_PALETTE_SHARE",
    "build_cluster_palette",
]
