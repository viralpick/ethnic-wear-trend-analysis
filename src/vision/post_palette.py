"""Phase B3b — post-level palette 생산.

`CanonicalOutfit.palette` 들을 area_ratio × within-share 가중으로 flatten 한 뒤,
ΔE76 ≤ 10 greedy pair-merge → share<0.05 drop → top N cap (3) → sum=1.0 재정규화.
결과는 post 대표 색 최대 3개 (`list[PaletteCluster]`).

설계 원칙:
- pure function — canonical pydantic 입력, pydantic PaletteCluster list 출력.
- weight 은 pixel 증거만: `weight = representative.person_bbox_area_ratio *
  palette_cluster.share`. "post 전체에서 이 색이 차지하는 물리적 지분" 해석 — 작은
  canonical 의 색이 큰 canonical 의 색과 동등해지는 왜곡 차단.
- family 는 merge 시 더 큰 weight 쪽 family 를 계승. 동점이면 `ColorFamily` 멤버 순서
  (Counter.most_common 금지 원칙, feedback_counter_most_common_order).
- hex 는 merge 후 LAB centroid → rgb → hex 재계산 — pixel 증거 원칙 유지.
- PRESET_MATCH_THRESHOLD (15.0, canonical_palette) 와 다른 값 (10.0) — preset 매칭용
  이 아니라 merge 용이므로 혼용 금지.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from contracts.common import ColorFamily, PaletteCluster
from contracts.vision import CanonicalOutfit
from vision.color_space import hex_to_rgb, lab_to_rgb, rgb_to_hex, rgb_to_lab

MAX_POST_PALETTE_CLUSTERS: int = 3
POST_PALETTE_MERGE_DELTA_E: float = 10.0
MIN_POST_PALETTE_SHARE: float = 0.05


@dataclass(frozen=True)
class _Weighted:
    """merge 중간 표현 — LAB 좌표 + weight + family (None 허용)."""
    lab: tuple[float, float, float]
    weight: float
    family: ColorFamily | None


def _family_order(family: ColorFamily | None) -> tuple[int, int]:
    """merge tiebreak 키 — (None 후순위, enum 선언 순서).

    `ColorFamily` 는 `StrEnum`. `list(ColorFamily).index` 는 선언 순서대로 번호 부여.
    None 은 enum 보다 뒤로 보냄 (family 확정된 쪽 우선).
    """
    if family is None:
        return (1, 0)
    members: list[Enum] = list(ColorFamily)
    return (0, members.index(family))


def _pick_family(a_weight: float, a_family: ColorFamily | None,
                 b_weight: float, b_family: ColorFamily | None) -> ColorFamily | None:
    """merge 두 클러스터 family 결정 — weight 큰 쪽, 동점은 enum 순서."""
    a_key = (-a_weight, *_family_order(a_family))
    b_key = (-b_weight, *_family_order(b_family))
    return a_family if a_key <= b_key else b_family


def _flatten(canonicals: list[CanonicalOutfit]) -> list[_Weighted]:
    """canonical 들의 palette 를 area_ratio × within-share 가중으로 펼친다.

    palette 가 비거나 area_ratio=0 인 canonical 은 자연스럽게 skip. hex → RGB → LAB 는
    `color_space` 의 정확한 D65 경로를 그대로 사용.
    """
    out: list[_Weighted] = []
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
                _Weighted(
                    lab=(float(lab[0]), float(lab[1]), float(lab[2])),
                    weight=weight,
                    family=cluster.family,
                )
            )
    return out


def _merge_greedy(items: list[_Weighted], threshold: float) -> list[_Weighted]:
    """dynamic_palette `_merge_by_deltae76` 와 동일한 greedy pair merge.

    매 iteration 가장 가까운 pair 찾아 (distance asc, i asc, j asc) tiebreak 로 단일
    merge. centroid 는 weight 가중 평균, family 는 `_pick_family` 규칙. O(k^3) 지만
    flatten 결과 k <= ~15 (canonical 5 × palette 3) 라 무관.
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
        # j 먼저 제거 (i < j 라 인덱스 무효화 없음)
        working.pop(j)
        working[i] = merged
    return working


def _drop_small_and_cap(
    items: list[_Weighted], min_share: float, max_clusters: int,
) -> list[_Weighted]:
    """weight 기준 share < min_share drop → weight desc 정렬 → top N cap.

    drop 전 total 로 share 를 평가 (drop 후 재정규화는 호출부). top-N 에서 자른 뒤
    호출부가 다시 normalize 해 sum=1.0 을 만든다.
    """
    total = sum(it.weight for it in items)
    if total <= 0.0:
        return []
    kept = [it for it in items if (it.weight / total) >= min_share]
    if not kept:
        return []
    kept.sort(key=lambda it: -it.weight)
    return kept[:max_clusters]


def _to_pydantic(items: list[_Weighted]) -> list[PaletteCluster]:
    """LAB centroid → RGB → hex 재계산, share 재정규화 해 pydantic 경계로 넘긴다."""
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
    merged = _merge_greedy(flat, merge_deltae76_threshold)
    capped = _drop_small_and_cap(merged, min_cluster_share, max_clusters)
    return _to_pydantic(capped)


__all__ = [
    "MAX_POST_PALETTE_CLUSTERS",
    "MIN_POST_PALETTE_SHARE",
    "POST_PALETTE_MERGE_DELTA_E",
    "build_post_palette",
]
