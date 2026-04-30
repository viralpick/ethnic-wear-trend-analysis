"""Cluster 배정 (spec §5.1, 2026-04-30 sync 변경: G__F 2축).

cluster_key = `garment_type__fabric` (G + F). technique 은 cluster 내 distribution 으로
표시 (silhouette / occasion 처럼). 정확 매칭 (G + F 둘 다 resolved) 이 STRONGLY PREFERRED.

금지:
- 누락 속성을 짐작으로 채우기
- 정확 매칭을 강제하기 위해 값을 coerce 하기
- contract 에 assignment_confidence 류 필드 추가하기
"""
from __future__ import annotations

from contracts.common import Fabric, GarmentType

UNCLASSIFIED = "unclassified"
_UNKNOWN = "unknown"


# --------------------------------------------------------------------------- #
# Key 생성 helpers
# --------------------------------------------------------------------------- #

def build_exact_key(garment_type: GarmentType, fabric: Fabric) -> str:
    """spec §5.1 (2026-04-30) — '{garment_type}__{fabric}' 2축 키."""
    return build_exact_key_strs(garment_type.value, fabric.value)


def build_exact_key_strs(g: str, f: str) -> str:
    """문자열 버전 — 분포 dict 의 key 가 이미 enum.value 인 경우 (assign_shares) 용."""
    return f"{g}__{f}"


def _build_partial_key(
    garment_type: GarmentType | None,
    fabric: Fabric | None,
) -> str:
    g = garment_type.value if garment_type else _UNKNOWN
    f = fabric.value if fabric else _UNKNOWN
    return f"{g}__{f}"


def _matches_partial(
    cluster_key: str,
    garment_type: GarmentType | None,
    fabric: Fabric | None,
) -> bool:
    parts = cluster_key.split("__")
    if len(parts) != 2:
        return False
    g_part, f_part = parts
    if g_part == _UNKNOWN or f_part == _UNKNOWN:
        # partial-key 끼리 매칭은 의미 없음. 정확 2축 키만 후보로.
        return False
    if garment_type and g_part != garment_type.value:
        return False
    if fabric and f_part != fabric.value:
        return False
    return True


# --------------------------------------------------------------------------- #
# EXACT-match path (preferred)
# --------------------------------------------------------------------------- #

def assign_exact(garment_type: GarmentType, fabric: Fabric) -> str:
    """G + F 모두 resolved. 정확 키 그대로 반환."""
    return build_exact_key(garment_type, fabric)


# --------------------------------------------------------------------------- #
# PARTIAL-match path (weak signal)
# --------------------------------------------------------------------------- #

def assign_partial(
    garment_type: GarmentType | None,
    fabric: Fabric | None,
    cluster_totals: dict[str, int],
) -> str:
    """기존 정확 키 클러스터 중 부분 매칭되는 것 → 카운트 max → 없으면 partial 키 생성."""
    candidates = [
        key for key in cluster_totals
        if _matches_partial(key, garment_type, fabric)
    ]
    if candidates:
        return min(candidates, key=lambda k: (-cluster_totals[k], k))
    return _build_partial_key(garment_type, fabric)


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #

def assign_cluster(
    garment_type: GarmentType | None,
    fabric: Fabric | None,
    cluster_totals: dict[str, int],
) -> str:
    """G + F 2축 dispatch. EXACT / UNCLASSIFIED / PARTIAL."""
    if garment_type is not None and fabric is not None:
        return assign_exact(garment_type, fabric)
    if garment_type is None and fabric is None:
        return UNCLASSIFIED
    return assign_partial(garment_type, fabric, cluster_totals)


# --------------------------------------------------------------------------- #
# share-weighted assign — N:N path (2축 G__F)
# --------------------------------------------------------------------------- #

# multiplier_for_n(N) / multiplier_for_n(2). G__F 2축 기준:
# N=2 → 1.0 (full), N=1 → 0.5 (partial), N=0 → 0
_PARTIAL_MASS_BY_N: dict[int, float] = {1: 0.5, 2: 1.0}


def assign_shares(
    garment_dist: dict[str, float],
    fabric_dist: dict[str, float],
    *,
    min_share: float = 0.0,
) -> dict[str, float]:
    """G/F 분포 dict → cluster_key 별 share dict (cross-product, multiplier_ratio 가중).

    - N (resolved axis 수) = 0 → 빈 dict
    - N≥1 → cross-product. 비어있는 axis 는 `_UNKNOWN` placeholder (1.0 share).
      share × multiplier_ratio — N=2 → ×1.0 / N=1 → ×0.5.
    - per-item mass: N=2 = 1.0 / N=1 = 0.5 / N=0 = 0
    - cluster_key 포맷 = `g__f`.
    - min_share 이하는 drop (default 0).
    """
    n_axes = sum(1 for d in (garment_dist, fabric_dist) if d)
    if n_axes == 0:
        return {}

    g_eff = garment_dist or {_UNKNOWN: 1.0}
    f_eff = fabric_dist or {_UNKNOWN: 1.0}
    mult_ratio = _PARTIAL_MASS_BY_N[n_axes]

    out: dict[str, float] = {}
    for g, gp in g_eff.items():
        if gp <= 0.0:
            continue
        for f, fp in f_eff.items():
            if fp <= 0.0:
                continue
            share = gp * fp * mult_ratio
            if share <= min_share:
                continue
            out[build_exact_key_strs(g, f)] = share
    return out
