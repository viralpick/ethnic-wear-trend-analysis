"""Palette merge 공용 유틸 — post_palette / cluster_palette / 기타 weighted LAB 병합.

이 모듈이 single source — `vision.post_palette` 와 `aggregation.cluster_palette` 가
이전에 동일 코드 (`_Weighted`/`_merge_greedy`/`_drop_small_and_cap`/`_to_pydantic`/
`_pick_family`) 를 별도 정의했던 것을 통합. 알고리즘 변경은 여기서만, drift 차단
(roadmap: post_palette ↔ cluster_palette 공통 유틸 P3 TODO).

설계:
- weight 평균 LAB centroid + family tiebreak (`pick_family`) 는 두 호출 path 동일
- ΔE 거리는 `np.linalg.norm` 경유 (BLAS 잠재 비결정성, hex 칩 시각용 — score path 가
  아닌 시각 결과물). score path 의 ΔE76 은 `vision.color_space.delta_e76_tuple` 사용.
- private symbol import 금지 룰 (`feedback_private_symbol_import`) 준수 — 외부 호출
  대상은 underscore 없는 public 이름.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from contracts.common import ColorFamily, PaletteCluster
from vision.color_space import lab_to_rgb, rgb_to_hex


@dataclass(frozen=True)
class WeightedColorEntry:
    """merge 중간 표현 — LAB 좌표 + weight + family (None 허용)."""
    lab: tuple[float, float, float]
    weight: float
    family: ColorFamily | None


def _family_order(family: ColorFamily | None) -> tuple[int, int]:
    """merge tiebreak 키 — (None 후순위, enum 선언 순서).

    `ColorFamily` 는 `StrEnum`. None 은 enum 보다 뒤로 보내 family 확정 쪽 우선.
    """
    if family is None:
        return (1, 0)
    members: list[Enum] = list(ColorFamily)
    return (0, members.index(family))


def pick_family(
    a_weight: float, a_family: ColorFamily | None,
    b_weight: float, b_family: ColorFamily | None,
) -> ColorFamily | None:
    """merge 두 cluster family 결정 — weight 큰 쪽, 동점은 enum 선언 순서.

    `Counter.most_common` 회피 (`feedback_counter_most_common_order`) — 결정론 핵심.
    """
    a_key = (-a_weight, *_family_order(a_family))
    b_key = (-b_weight, *_family_order(b_family))
    return a_family if a_key <= b_key else b_family


def merge_greedy_lab(
    items: list[WeightedColorEntry], threshold: float,
) -> list[WeightedColorEntry]:
    """ΔE76 ≤ threshold 인 가장 가까운 pair 부터 단일 merge — 반복.

    tiebreak: (distance asc, i asc, j asc). centroid 는 weight 가중 평균, family 는
    `pick_family` 규칙. O(k^3) 지만 호출처 k <= ~15~수십 작아 무관.
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
        merged = WeightedColorEntry(
            lab=(float(merged_lab[0]), float(merged_lab[1]), float(merged_lab[2])),
            weight=total,
            family=pick_family(a.weight, a.family, b.weight, b.family),
        )
        # j 먼저 제거 (i < j 라 인덱스 무효화 없음)
        working.pop(j)
        working[i] = merged
    return working


def drop_small_and_cap(
    items: list[WeightedColorEntry],
    *,
    min_share: float,
    max_clusters: int,
) -> list[WeightedColorEntry]:
    """weight 기준 share < min_share drop → weight desc 정렬 → top N cap.

    drop 전 total 로 share 평가 (drop 후 재정규화는 호출부 / `to_pydantic_palette`).
    """
    total = sum(it.weight for it in items)
    if total <= 0.0:
        return []
    kept = [it for it in items if (it.weight / total) >= min_share]
    if not kept:
        return []
    kept.sort(key=lambda it: -it.weight)
    return kept[:max_clusters]


def to_pydantic_palette(items: list[WeightedColorEntry]) -> list[PaletteCluster]:
    """LAB centroid → RGB → hex 재계산, share 재정규화 후 pydantic 경계로 넘김."""
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


__all__ = [
    "WeightedColorEntry",
    "pick_family",
    "merge_greedy_lab",
    "drop_small_and_cap",
    "to_pydantic_palette",
]
