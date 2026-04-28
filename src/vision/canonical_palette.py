"""β-hybrid 파이프라인용 cluster → family 매핑 헬퍼.

이전 union-pool 시절의 `build_canonical_palette` / `MAX_CANONICAL_PALETTE_CLUSTERS` 는
β-hybrid 재설계 (per-object → 통합 KMeans) 단계 D-6 에서 제거됨. 남은 것:

- `resolve_family(cluster, matcher_entries, threshold)` — preset ΔE76 매칭 → family.
  `vision.hybrid_palette` (Phase 1 R1 anchor 재해석) 와
  `vision.canonical_palette_aggregator` (Phase 3 통합 centroid family) 가 import.
- `PRESET_MATCH_THRESHOLD = 15.0` — `resolve_family` default. canonical/post/cluster
  단계 어디서나 같은 임계 사용 (R1 anchor 의 `pick_match_deltae76=25.0` 과는 다른 의미).

설계 원칙:
- family 매핑만 preset 이름이 아닌 preset LAB 좌표와 비교. hex/share 자체는 pixel 증거만.
"""
from __future__ import annotations

from contracts.common import ColorFamily
from vision.color_family_preset import MatcherEntry, lab_to_family
from vision.dynamic_palette import PaletteCluster as PixelCluster

PRESET_MATCH_THRESHOLD: float = 15.0


def resolve_family(
    cluster: PixelCluster, matcher_entries: list[MatcherEntry],
    threshold: float = PRESET_MATCH_THRESHOLD,
) -> ColorFamily:
    """cluster.lab 에 가장 가까운 preset entry 의 family 를 반환. ΔE76 > threshold
    이거나 entries 가 비면 `lab_to_family` rule fallback.

    preset name 은 버림 — contracts PaletteCluster 에 name 없음 (hex/share/family).
    Gemini pick 과 무관한 pixel 증거 기반 매핑.

    public — `vision.hybrid_palette` 가 R1 merge 후 family 재해석 용으로 import 한다
    (private import 금지 규칙, feedback_private_symbol_import).
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


__all__ = [
    "PRESET_MATCH_THRESHOLD",
    "resolve_family",
]
