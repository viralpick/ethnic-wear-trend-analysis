"""Cluster 배정 (spec §5.2).

정확 매칭 (3축 전부) 이 STRONGLY PREFERRED. 부분 매칭은 파이프라인을 굴리기 위한 최소한의
fallback 일 뿐, 품질 시그널이 아니다. 아래 두 경로는 의도적으로 분리되어 있어 읽는 사람이
weak branch 를 한눈에 식별할 수 있다.

금지:
- 누락 속성을 짐작으로 채우기
- 정확 매칭을 강제하기 위해 값을 coerce 하기
- contract 에 assignment_confidence 류 필드 추가하기
"""
from __future__ import annotations

from contracts.common import Fabric, GarmentType, Technique

UNCLASSIFIED = "unclassified"
_UNKNOWN = "unknown"


# --------------------------------------------------------------------------- #
# Key 생성 helpers
# --------------------------------------------------------------------------- #

def build_exact_key(
    garment_type: GarmentType, technique: Technique, fabric: Fabric
) -> str:
    """spec §5.1 — '{garment_type}__{technique}__{fabric}' 완전 키."""
    return build_exact_key_strs(garment_type.value, technique.value, fabric.value)


def build_exact_key_strs(g: str, t: str, f: str) -> str:
    """문자열 버전 — 분포 dict 의 key 가 이미 enum.value 인 경우 (assign_shares) 용.

    `build_exact_key` 와 같은 포맷이지만 enum 변환을 거치지 않음. cluster_key 포맷
    변경 시 두 함수가 동시에 따라간다.
    """
    return f"{g}__{t}__{f}"


def _build_partial_key(
    garment_type: GarmentType | None,
    technique: Technique | None,
    fabric: Fabric | None,
) -> str:
    g = garment_type.value if garment_type else _UNKNOWN
    t = technique.value if technique else _UNKNOWN
    f = fabric.value if fabric else _UNKNOWN
    return f"{g}__{t}__{f}"


def _matches_partial(
    cluster_key: str,
    garment_type: GarmentType | None,
    technique: Technique | None,
    fabric: Fabric | None,
) -> bool:
    parts = cluster_key.split("__")
    if len(parts) != 3:
        return False
    g_part, t_part, f_part = parts
    if g_part == _UNKNOWN or t_part == _UNKNOWN or f_part == _UNKNOWN:
        # partial-key 끼리 매칭은 의미 없음. 정확 3축 키만 후보로.
        return False
    if garment_type and g_part != garment_type.value:
        return False
    if technique and t_part != technique.value:
        return False
    if fabric and f_part != fabric.value:
        return False
    return True


# --------------------------------------------------------------------------- #
# EXACT-match path (preferred)
# --------------------------------------------------------------------------- #

def assign_exact(
    garment_type: GarmentType, technique: Technique, fabric: Fabric
) -> str:
    """3축 전부 resolved 된 경우. 정확 키를 그대로 반환."""
    return build_exact_key(garment_type, technique, fabric)


# --------------------------------------------------------------------------- #
# PARTIAL-match path (weak signal)
# --------------------------------------------------------------------------- #

# TODO(§5.2): partial match is a weak signal; consider downgrading in aggregation
# weighting later. 현재는 단순히 파이프라인을 굴리기 위해 분류만 한다.
def assign_partial(
    garment_type: GarmentType | None,
    technique: Technique | None,
    fabric: Fabric | None,
    cluster_totals: dict[str, int],
) -> str:
    """기존 정확 키 클러스터 중 부분 매칭되는 것 → 카운트 max → 없으면 partial 키 생성."""
    candidates = [
        key for key in cluster_totals
        if _matches_partial(key, garment_type, technique, fabric)
    ]
    if candidates:
        # tie-break: count DESC, key ASC. 첫 런(cluster_totals 비어있음)엔 이 branch 미도달.
        return min(candidates, key=lambda k: (-cluster_totals[k], k))
    return _build_partial_key(garment_type, technique, fabric)


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #

def assign_cluster(
    garment_type: GarmentType | None,
    technique: Technique | None,
    fabric: Fabric | None,
    cluster_totals: dict[str, int],
) -> str:
    """spec §5.2 전체 분기. 세 경로 중 하나로 라우팅."""
    # EXACT path — strongly preferred
    if garment_type is not None and technique is not None and fabric is not None:
        return assign_exact(garment_type, technique, fabric)

    # ALL-NULL path — unclassified bucket (spec §5.2 case 4)
    if garment_type is None and technique is None and fabric is None:
        return UNCLASSIFIED

    # PARTIAL path — weak signal, keep pipeline flowing
    return assign_partial(garment_type, technique, fabric, cluster_totals)


# --------------------------------------------------------------------------- #
# Phase α (2026-04-28): share-weighted assign — N:N path
# --------------------------------------------------------------------------- #
# pipeline_spec §2.4 line 252-255: representative score 입력은 contribution-
# weighted. 1 item 의 G/T/F 분포 dict 가 cross-product 으로 여러 cluster_key 에
# 부분 share 로 fan-out. winner-takes-all (assign_cluster) 와 평행 — Phase β
# (build_cluster_summary share-weighted fan-out) 가 wire 할 때까지 호출자 없음.

# multiplier_for_n(N) / multiplier_for_n(3) — partial 활성화 후 per-item mass.
# representative_builder._MULTIPLIER_BY_N 와 단일 source 의도지만, 순환 import 회피
# 위해 인라인 매핑 (후속 cleanup 으로 단일 모듈 분리 예정).
_PARTIAL_MASS_BY_N: dict[int, float] = {1: 0.2, 2: 0.5, 3: 1.0}


def assign_shares(
    garment_dist: dict[str, float],
    technique_dist: dict[str, float],
    fabric_dist: dict[str, float],
    *,
    min_share: float = 0.0,
) -> dict[str, float]:
    """G/T/F 분포 dict → cluster_key 별 share dict (cross-product, multiplier_ratio 가중).

    Phase β3 + partial(g) 활성화 (2026-04-28):
    - N (resolved axis 수) = 0 → 빈 dict
    - N≥1 → cross-product. 비어있는 axis 는 `_UNKNOWN` placeholder (1.0 share).
      share × multiplier_ratio 가중 — N=3 → ×1.0 / N=2 → ×0.5 / N=1 → ×0.2.
      (multiplier_for_n(N) / multiplier_for_n(3) — β1 effective_item_count 와 단위 정합)
    - per-item mass: N=3 = 1.0 / N=2 = 0.5 / N=1 = 0.2 / N=0 = 0
    - cluster_key 포맷 = `g__t__f` (enum value 또는 unknown placeholder).
      `_build_partial_key` / `assign_exact` 와 같은 포맷 — winner-keyed 와 fan-out cluster
      space 가 동일.
    - min_share 이하는 drop (default 0 → 0 share 만 drop).
    """
    n_axes = sum(1 for d in (garment_dist, technique_dist, fabric_dist) if d)
    if n_axes == 0:
        return {}

    g_eff = garment_dist or {_UNKNOWN: 1.0}
    t_eff = technique_dist or {_UNKNOWN: 1.0}
    f_eff = fabric_dist or {_UNKNOWN: 1.0}
    mult_ratio = _PARTIAL_MASS_BY_N[n_axes]

    out: dict[str, float] = {}
    for g, gp in g_eff.items():
        if gp <= 0.0:
            continue
        for t, tp in t_eff.items():
            if tp <= 0.0:
                continue
            for f, fp in f_eff.items():
                if fp <= 0.0:
                    continue
                share = gp * tp * fp * mult_ratio
                if share <= min_share:
                    continue
                out[build_exact_key_strs(g, t, f)] = share
    return out
