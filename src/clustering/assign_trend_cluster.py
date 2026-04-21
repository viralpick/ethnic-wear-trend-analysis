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
    return f"{garment_type.value}__{technique.value}__{fabric.value}"


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
