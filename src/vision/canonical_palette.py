"""Phase B2 — canonical 단위 palette 생산.

`extract_dynamic_palette` (vision dataclass PaletteCluster) → top N 재정규화 →
family 매핑 (preset ΔE76 → `lab_to_family` fallback) → `contracts.common.PaletteCluster`
로 변환. `CanonicalOutfit.palette` (max 3) 를 채우는 B3 adapter 가 소비한다.

설계 원칙:
- pure function — pixels + config 가 입력, pydantic PaletteCluster list 가 출력.
- B1 segformer class 분기로 생긴 `CanonicalOutfitPixels.pooled_pixels` 를 그대로 받음.
- palette merge / share 계산은 오직 pixel 기반 (`extract_dynamic_palette`). Gemini
  color_preset_picks_top3 은 미참여 (dedup 전용 — feedback_gemini_color_dedup_only).
- family 매핑만 preset 이름이 아닌 preset LAB 좌표와 비교해 이용. hex/share 자체는
  pixel 증거만.
"""
from __future__ import annotations

import numpy as np

from contracts.common import ColorFamily, PaletteCluster
from settings import DynamicPaletteConfig
from vision.color_family_preset import MatcherEntry, lab_to_family
from vision.dynamic_palette import PaletteCluster as PixelCluster
from vision.dynamic_palette import extract_dynamic_palette

MAX_CANONICAL_PALETTE_CLUSTERS: int = 3
PRESET_MATCH_THRESHOLD: float = 15.0


def _top_n_renormalize(
    clusters: list[PixelCluster], max_clusters: int,
) -> list[PixelCluster]:
    """share desc 정렬된 PixelCluster list → top N 자르고 share 재정규화.

    `extract_dynamic_palette` 는 이미 share desc 로 정렬해 반환. 상위 N 만 남기고
    재정규화해 sum=1.0 invariant (PaletteCluster.share) 를 유지. total=0 방어는
    extract_dynamic_palette 가 이미 수행 (빈 리스트 반환).
    """
    if max_clusters <= 0 or not clusters:
        return []
    top = clusters[:max_clusters]
    total = sum(c.share for c in top)
    if total <= 0.0:
        return []
    return [
        PixelCluster(
            hex=c.hex, rgb=c.rgb, lab=c.lab, share=float(c.share / total),
        )
        for c in top
    ]


def _resolve_family(
    cluster: PixelCluster, matcher_entries: list[MatcherEntry],
    threshold: float = PRESET_MATCH_THRESHOLD,
) -> ColorFamily:
    """cluster.lab 에 가장 가까운 preset entry 의 family 를 반환. ΔE76 > threshold
    이거나 entries 가 비면 `lab_to_family` rule fallback.

    preset name 은 버림 — contracts PaletteCluster 에 name 없음 (hex/share/family).
    Gemini pick 과 무관한 pixel 증거 기반 매핑.
    """
    cL, ca, cb = cluster.lab
    best: MatcherEntry | None = None
    best_de = float("inf")
    for entry in matcher_entries:
        eL, ea, eb = entry.lab
        de = ((cL - eL) ** 2 + (ca - ea) ** 2 + (cb - eb) ** 2) ** 0.5
        if de < best_de:
            best_de = de
            best = entry
    if best is not None and best_de <= threshold:
        return best.family
    return lab_to_family(cL, ca, cb)


def build_canonical_palette(
    rgb_pixels: np.ndarray,
    dyn_cfg: DynamicPaletteConfig,
    matcher_entries: list[MatcherEntry],
    max_clusters: int = MAX_CANONICAL_PALETTE_CLUSTERS,
) -> list[PaletteCluster]:
    """canonical pool (RGB N,3 uint8) → contracts PaletteCluster list (max N).

    pipeline:
      1. extract_dynamic_palette — KMeans(LAB) + ΔE76 merge + small-share drop
         (k ∈ {0..5}, share desc 정렬).
      2. top N 자르고 재정규화 (sum=1.0).
      3. 각 cluster 당 family 매핑 (preset matcher 우선, lab_to_family fallback).
      4. dataclass → pydantic PaletteCluster (hex+share+family).

    빈 반환 = pool 불충분 / 전부 share<min_cluster_share drop. 이 경우 B3 adapter 는
    `CanonicalOutfit.palette=[]` 로 둬야 함 (canonical 자체는 enriched 에 유지 —
    라벨 보존 invariant, project_color_pipeline_redesign advisor 피드백).
    """
    if rgb_pixels.size == 0:
        return []
    pixel_clusters = extract_dynamic_palette(rgb_pixels, dyn_cfg)
    top = _top_n_renormalize(pixel_clusters, max_clusters)
    return [
        PaletteCluster(
            hex=c.hex,
            share=c.share,
            family=_resolve_family(c, matcher_entries),
        )
        for c in top
    ]


__all__ = [
    "MAX_CANONICAL_PALETTE_CLUSTERS",
    "PRESET_MATCH_THRESHOLD",
    "build_canonical_palette",
]
