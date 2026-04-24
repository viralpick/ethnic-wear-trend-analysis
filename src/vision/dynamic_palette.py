"""Phase 4 canonical outfit 동적 k palette — M3.A Step D Phase 3 → 4 경로.

roadmap.md §80 canonical: `KMeans k=5 → ΔE76 merge (default 10) → cluster share < 0.05 drop`.
최종 k ∈ {0..initial_k} (단/2/3/4/5색 옷 자연 대응, pool 이 비정상적으로 작으면 0).

본 모듈은 pure CV — pixel in, `PaletteCluster` out. family 분류는 Phase 5 adapter 책임
(color_preset_picks_top3 우선 → LAB rule fallback). contracts 에 LAB/numpy 노출 X, 대신
numpy-free dataclass 로만 경계를 넘김.

알고리즘 결정:
- **greedy pair-merge** (union-find 대신): 매 iteration 최단 pair 만 merge, centroid 재계산
  후 다시 거리 평가. transitive chain (A~B~C 체인이 A~C 는 멀어도 union-find 로 묶이는) 문제
  차단. gradient 색 팔레트 (연한 살구↔오렌지↔빨강) 에서 중요.
- **결정성**: tiebreak (distance asc, i asc, j asc) 로 고정. 같은 입력 → 같은 출력.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans

from settings import DynamicPaletteConfig
from vision.color_space import lab_to_rgb, rgb_to_hex, rgb_to_lab


@dataclass(frozen=True)
class PaletteCluster:
    """동적 k palette 의 cluster 1 개. numpy-free — contracts 경계 안전.

    hex: '#RRGGBB' 대문자. rgb: (r,g,b) 0~255 int. lab: (L,a,b) float. share: 0..1 (합=1.0).
    """
    hex: str
    rgb: tuple[int, int, int]
    lab: tuple[float, float, float]
    share: float


def _kmeans_lab(lab_pixels: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """LAB pixel (N,3) → (centers (k',3), counts (k',)). sklearn KMeans random_state=0 고정.

    k_eff = min(k, max(1, N // 50)) — extract_colors 와 동일 규칙. distinct LAB point 가
    k_eff 보다 적으면 sklearn 이 빈 cluster (count=0) 를 반환해 downstream merge 에서 NaN
    발생 — count=0 cluster 를 제거하고 유효한 것만 반환.
    """
    k_eff = min(k, max(1, lab_pixels.shape[0] // 50))
    km = KMeans(n_clusters=k_eff, n_init=4, random_state=0).fit(lab_pixels)
    counts = np.bincount(km.labels_, minlength=k_eff).astype(np.float32)
    nonempty = counts > 0
    return km.cluster_centers_[nonempty].astype(np.float32), counts[nonempty]


def _merge_by_deltae76(
    centers: np.ndarray, weights: np.ndarray, threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """greedy ΔE76 pair-merge — threshold 이하 최단 pair 반복 합침.

    merge 수식: new_center = (w_i * c_i + w_j * c_j) / (w_i + w_j), new_weight = w_i + w_j.
    각 iteration 은 (distance asc, i asc, j asc) tiebreak 로 단일 pair 만 선택. 전체 O(k^3)
    이지만 k <= 5 라 무관.
    """
    centers = centers.copy()
    weights = weights.copy()
    while centers.shape[0] > 1:
        best: tuple[float, int, int] | None = None
        for i in range(centers.shape[0]):
            for j in range(i + 1, centers.shape[0]):
                d = float(np.linalg.norm(centers[i] - centers[j]))
                if d >= threshold:
                    continue
                if best is None or (d, i, j) < best:
                    best = (d, i, j)
        if best is None:
            break
        _d, i, j = best
        wi, wj = float(weights[i]), float(weights[j])
        merged = (wi * centers[i] + wj * centers[j]) / (wi + wj)
        # i 슬롯 갱신, j 제거 (i < j 항상 유지)
        centers[i] = merged
        weights[i] = wi + wj
        centers = np.delete(centers, j, axis=0)
        weights = np.delete(weights, j, axis=0)
    return centers, weights


def _drop_small_shares(
    centers: np.ndarray, weights: np.ndarray, min_share: float,
) -> tuple[np.ndarray, np.ndarray]:
    """share < min_share cluster drop + 나머지 재정규화.

    total=0 또는 전부 drop 이면 빈 배열 반환. 재정규화는 drop 후 남은 weight 합으로 나눔
    (drop 된 share 를 남은 cluster 에 비례 배분).
    """
    total = float(weights.sum())
    if total <= 0.0:
        return centers[:0], weights[:0]
    shares = weights / total
    keep = shares >= min_share
    if not keep.any():
        return centers[:0], weights[:0]
    kept_centers = centers[keep]
    kept_weights = weights[keep]
    return kept_centers, kept_weights / float(kept_weights.sum())


def extract_dynamic_palette(
    rgb_pixels: np.ndarray, cfg: DynamicPaletteConfig,
) -> list[PaletteCluster]:
    """canonical pool (RGB N,3) → 동적 k palette. share desc 정렬.

    파이프라인: min_pixels 가드 → RGB→LAB → KMeans(initial_k) → greedy ΔE76 merge →
    small share drop. 빈 반환 = pool 불충분 or 전부 drop.

    반환 PaletteCluster.share 는 재정규화된 값으로 sum ≈ 1.0 (float 오차). 호출부 (Phase 5
    adapter) 가 preset/family 매핑을 붙여 ColorPaletteItem 으로 변환.
    """
    if rgb_pixels.shape[0] < cfg.min_pixels:
        return []
    lab = rgb_to_lab(rgb_pixels)
    centers, weights = _kmeans_lab(lab, cfg.initial_k)
    centers, weights = _merge_by_deltae76(centers, weights, cfg.merge_deltae76_threshold)
    centers, shares = _drop_small_shares(centers, weights, cfg.min_cluster_share)
    if centers.shape[0] == 0:
        return []
    order = np.argsort(-shares, kind="stable")
    centers = centers[order]
    shares = shares[order]
    rgb_centers = lab_to_rgb(centers)
    out: list[PaletteCluster] = []
    for lab_c, rgb_c, s in zip(centers, rgb_centers, shares):
        r, g, b = np.clip(rgb_c, 0, 255).astype(int).tolist()
        out.append(
            PaletteCluster(
                hex=rgb_to_hex(rgb_c),
                rgb=(int(r), int(g), int(b)),
                lab=(float(lab_c[0]), float(lab_c[1]), float(lab_c[2])),
                share=float(s),
            )
        )
    return out
