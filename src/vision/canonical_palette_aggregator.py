"""Phase 3 — canonical 단위 통합 weighted KMeans aggregator.

per-object β-hybrid (Phase 1+2, `vision/hybrid_palette.build_object_palette`) 가 출력한
`(list[WeightedCluster], etc_weight)` 를 canonical 단위로 합쳐 통합 KMeans 로 묶고,
top_n cap + cut_off_share + family resolve 까지 한 번에 수행한다.

설계 원칙:
- pure function — per_object_results + matcher_entries + cfg → (palette, cut_off_share).
- sample_weight 는 Phase 1 의 frame-area normalize float weight.
- k = max(per-object Phase 1 final k). 모든 오브젝트 빈 결과면 단락.
- KMeans random_state=42 / n_init=10 — 결정성 우선. dynamic_palette 의 (0, 4) 와 의도적
  으로 다름. Phase 3 는 통합 단계라 local minima 회피가 더 중요.
- distinct LAB point < k 일 때 sklearn 이 빈 cluster 를 반환할 수 있어 weight_sum=0 필터.
  (`feedback_kmeans_empty_centroid` — Phase 4 동일 방어.)
- cut_off_share 의미 (2026-04-26 재정의):
    cut_off_share = (top_n cap 으로 잘린 share) + (per-object R2 etc weight 합산 share).
    palette 에 표시 안 된 모든 잔여 비중. 사용자 결정 — etc 를 별도 필드 X 로 단순화.
- family resolve 는 통합 centroid LAB 으로 1회 — `canonical_palette.resolve_family` 재사용
  (preset ΔE76 ≤ 15 → MatcherEntry.family, 그 외 lab_to_family rule fallback).
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from contracts.common import PaletteCluster
from settings import HybridPaletteConfig
from vision.canonical_palette import resolve_family
from vision.color_family_preset import MatcherEntry
from vision.color_space import lab_to_rgb, rgb_to_hex
from vision.dynamic_palette import PaletteCluster as PixelCluster
from vision.hybrid_palette import WeightedCluster


def _flatten_objects(
    per_object_results: list[tuple[list[WeightedCluster], float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """list[(WC list, etc_weight)] → (LAB N×3, weight N, anchor_flags N, etc_weight 합).

    빈 오브젝트는 자연 skip. etc_weight 합은 cut_off_share 에 흡수.
    `anchor_flags` (F-13): 각 입력 WC 의 is_anchor 를 ndarray (bool) 로. Phase 3 KMeans
    label propagation 에 활용.
    """
    lab_rows: list[tuple[float, float, float]] = []
    weights: list[float] = []
    anchor_flags: list[bool] = []
    etc_total: float = 0.0
    for clusters, etc_w in per_object_results:
        etc_total += float(etc_w)
        for c in clusters:
            lab_rows.append(c.lab)
            weights.append(float(c.weight))
            anchor_flags.append(bool(c.is_anchor))
    if not lab_rows:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=bool),
            etc_total,
        )
    return (
        np.asarray(lab_rows, dtype=np.float32),
        np.asarray(weights, dtype=np.float32),
        np.asarray(anchor_flags, dtype=bool),
        etc_total,
    )


def _resolve_k(
    per_object_results: list[tuple[list[WeightedCluster], float]],
) -> int:
    """k = max(len(obj_i.clusters)). 모두 빈 결과면 0."""
    max_k = 0
    for clusters, _etc in per_object_results:
        if len(clusters) > max_k:
            max_k = len(clusters)
    return max_k


def _kmeans_weighted(
    lab_pixels: np.ndarray, weights: np.ndarray, anchor_flags: np.ndarray, k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """LAB N×3 + sample_weight N + anchor_flags N → (centers k'×3, weight_sums k',
    centroid_anchor_flags k'). 빈 centroid 제거.

    `random_state=42` / `n_init=10` 고정 — Phase 3 결정성 pin. distinct LAB point < k 일
    때 sklearn 이 빈 centroid 를 반환할 수 있어 weight_sum 0 필터로 방어.

    F-13: anchor_flags 는 각 입력 WC 의 is_anchor. centroid 는 한 입력이라도 anchor 면
    anchor 로 마킹 (permissive — Gemini ∩ KMeans 신호가 한 번이라도 들어가면 anchor).
    """
    k_eff = min(k, lab_pixels.shape[0])
    if k_eff <= 0:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=bool),
        )
    km = KMeans(n_clusters=k_eff, n_init=10, random_state=42).fit(
        lab_pixels, sample_weight=weights,
    )
    weight_sums = np.zeros(k_eff, dtype=np.float32)
    centroid_anchor = np.zeros(k_eff, dtype=bool)
    for label, w, flag in zip(km.labels_, weights, anchor_flags):
        idx = int(label)
        weight_sums[idx] += float(w)
        if flag:
            centroid_anchor[idx] = True
    nonempty = weight_sums > 0
    return (
        km.cluster_centers_[nonempty].astype(np.float32),
        weight_sums[nonempty],
        centroid_anchor[nonempty],
    )


def _palette_from_capped(
    kept: list[tuple[tuple[float, float, float], tuple[int, int, int], str, float]],
    matcher_entries: list[MatcherEntry],
) -> list[PaletteCluster]:
    """공통 출력 — (lab, rgb, hex, renormed_share) → PaletteCluster.

    family resolve 는 PixelCluster 어댑터로 1회 — `aggregate_canonical_palette` /
    `finalize_object_palette` 가 같은 룰 공유.
    """
    out: list[PaletteCluster] = []
    for lab, rgb, hex_, share in kept:
        pixel = PixelCluster(hex=hex_, rgb=rgb, lab=lab, share=share)
        out.append(
            PaletteCluster(
                hex=hex_, share=share,
                family=resolve_family(pixel, matcher_entries),
            ),
        )
    return out


def finalize_object_palette(
    weighted: list[WeightedCluster],
    etc_weight: float,
    matcher_entries: list[MatcherEntry],
    cfg: HybridPaletteConfig,
) -> tuple[list[PaletteCluster], float]:
    """β-hybrid Phase 1 출력 → 멤버 단위 PaletteCluster (spec §6.5 OutfitMember.palette).

    canonical 통합 KMeans 와 다른 점: 1 object 의 cluster 는 이미 분리돼 있어 KMeans 재실행
    불필요. 입력 좌표 (hex/rgb/lab) 그대로 사용. 절차:
      1. grand_total = Σweight + etc_weight.
      2. share = weight / grand_total per cluster (작은 raw share).
      3. anchor priority + share desc top_n cap (canonical aggregator 와 동일 룰):
         anchor 먼저 share desc 로 채우고 잔여 슬롯에 non-anchor share desc.
      4. cut_off_share = 1.0 − Σ(capped raw shares).
      5. capped shares 재정규화 → sum=1.0 (contracts.PaletteCluster 합 invariant).
      6. family resolve via `resolve_family` (PRESET_MATCH_THRESHOLD=15).
      7. 출력 순서 = share desc.

    edge:
      - weighted=[], etc_weight=0 → ([], 0.0).
      - weighted=[], etc_weight>0 → ([], 1.0). 전부 잔여 = 머지 못한 색.
      - grand_total <= 0 → ([], 0.0). 방어 (실전 weight>0 invariant).
    """
    if not weighted:
        return [], (1.0 if etc_weight > 0.0 else 0.0)
    grand_total = sum(c.weight for c in weighted) + float(etc_weight)
    if grand_total <= 0.0:
        return [], 0.0
    enriched: list[tuple[WeightedCluster, float]] = sorted(
        ((c, c.weight / grand_total) for c in weighted),
        key=lambda x: -x[1],
    )
    top_n = cfg.top_n
    anchors = [(c, s) for c, s in enriched if c.is_anchor]
    non_anchors = [(c, s) for c, s in enriched if not c.is_anchor]
    kept_anchors = anchors[:top_n]
    remaining = max(0, top_n - len(kept_anchors))
    kept = sorted(
        kept_anchors + non_anchors[:remaining], key=lambda x: -x[1],
    )
    capped_share_sum = sum(s for _, s in kept)
    cut_off = max(0.0, 1.0 - capped_share_sum)
    if capped_share_sum <= 0.0:
        return [], (1.0 if etc_weight > 0.0 else 0.0)
    capped: list[tuple[tuple[float, float, float], tuple[int, int, int], str, float]] = [
        (c.lab, c.rgb, c.hex, s / capped_share_sum) for c, s in kept
    ]
    return _palette_from_capped(capped, matcher_entries), cut_off


def aggregate_canonical_palette(
    per_object_results: list[tuple[list[WeightedCluster], float]],
    matcher_entries: list[MatcherEntry],
    cfg: HybridPaletteConfig,
) -> tuple[list[PaletteCluster], float]:
    """canonical 통합 palette 생성.

    pipeline:
      1. flatten — (LAB, weight, anchor_flags, etc_weight 합) 평탄화.
      2. k = max(per-obj k). 0 이면 etc 만 있더라도 ([], 1.0) 반환 (전부 잔여).
      3. weighted KMeans (sample_weight=weight, random_state=42, n_init=10).
      4. share = weight / (Σweight + Σetc_weight). centroid anchor flag propagate.
      5. top_n cap with anchor priority (F-13):
         - anchor centroid 는 share desc 로 최대 top_n 까지 우선 보존.
         - 잔여 슬롯이 있으면 non-anchor centroid 를 share desc 로 채움.
         - 출력 순서: 모든 cap 통과 centroid 를 share desc 로 재정렬.
         - cut_off_share = 1.0 − Σ(cap 통과 shares).
      6. cap 통과 share 재정규화 → contracts PaletteCluster (sum=1.0 invariant).
      7. family resolve (canonical_palette.resolve_family).

    edge case:
      - per_object_results 빈 list / 모든 inner 빈 + etc=0 → ([], 0.0).
      - 모든 obj clusters 빈데 etc>0 → ([], 1.0) (전부 잔여 = 머지 못한 색).
      - 모든 obj k=1 → max k=1 → KMeans 단순 weighted mean → 단일 cluster.
        cut_off 는 etc share 만큼.
      - anchor 개수가 top_n 초과면 share desc 로 top_n 만 보존 (non-anchor 0개).
    """
    lab_pixels, weights, anchor_flags, etc_total = _flatten_objects(per_object_results)
    grand_total = float(weights.sum()) + etc_total
    if lab_pixels.shape[0] == 0:
        if etc_total > 0.0 and grand_total > 0.0:
            return [], 1.0
        return [], 0.0
    k = _resolve_k(per_object_results)
    if k <= 0:
        return [], (1.0 if etc_total > 0.0 else 0.0)
    centers, weight_sums, centroid_anchor = _kmeans_weighted(
        lab_pixels, weights, anchor_flags, k,
    )
    if centers.shape[0] == 0:
        return [], (1.0 if etc_total > 0.0 else 0.0)

    if grand_total <= 0.0:
        return [], 0.0
    shares = (weight_sums / grand_total).astype(np.float64)
    order = np.argsort(-shares, kind="stable")
    centers = centers[order]
    shares = shares[order]
    centroid_anchor = centroid_anchor[order]

    top_n = cfg.top_n
    # F-13: anchor priority — anchor centroid 우선, 잔여를 non-anchor 로 채움.
    anchor_idx = np.where(centroid_anchor)[0]
    non_anchor_idx = np.where(~centroid_anchor)[0]
    kept_anchor = anchor_idx[:top_n]
    remaining = max(0, top_n - len(kept_anchor))
    kept_non_anchor = non_anchor_idx[:remaining]
    kept = np.sort(np.concatenate([kept_anchor, kept_non_anchor]))
    capped_centers = centers[kept]
    capped_shares = shares[kept]
    cut_off = float(1.0 - capped_shares.sum())
    if cut_off < 0.0:
        cut_off = 0.0

    renorm_total = float(capped_shares.sum())
    if renorm_total <= 0.0:
        return [], (1.0 if etc_total > 0.0 else 0.0)
    renormed = capped_shares / renorm_total

    rgb_centers = lab_to_rgb(capped_centers)
    out: list[PaletteCluster] = []
    for lab_c, rgb_c, s in zip(capped_centers, rgb_centers, renormed):
        r, g, b = np.clip(rgb_c, 0, 255).astype(int).tolist()
        pixel = PixelCluster(
            hex=rgb_to_hex(rgb_c),
            rgb=(int(r), int(g), int(b)),
            lab=(float(lab_c[0]), float(lab_c[1]), float(lab_c[2])),
            share=float(s),
        )
        out.append(
            PaletteCluster(
                hex=pixel.hex,
                share=float(s),
                family=resolve_family(pixel, matcher_entries),
            ),
        )
    return out, cut_off


__all__ = ["aggregate_canonical_palette", "finalize_object_palette"]
