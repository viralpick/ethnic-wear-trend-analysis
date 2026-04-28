"""canonical_palette pinning — `resolve_family` preset/fallback + threshold 상수.

D-6 (2026-04-25): `build_canonical_palette` / `MAX_CANONICAL_PALETTE_CLUSTERS` /
`_top_n_renormalize` 는 β-hybrid 재설계로 제거됨. 본 모듈에 남은 검증은 family
매핑 헬퍼와 `PRESET_MATCH_THRESHOLD` 정의값뿐.
"""
from __future__ import annotations

import pytest

pytest.importorskip("sklearn", reason="sklearn required")

from contracts.common import ColorFamily  # noqa: E402
from vision.canonical_palette import (  # noqa: E402
    PRESET_MATCH_THRESHOLD,
    resolve_family,
)
from vision.color_family_preset import MatcherEntry  # noqa: E402
from vision.dynamic_palette import PaletteCluster as PixelCluster  # noqa: E402


def _px_cluster(
    hex_: str = "#AA0000", rgb=(170, 0, 0), lab=(30.0, 60.0, 50.0), share: float = 1.0,
) -> PixelCluster:
    return PixelCluster(hex=hex_, rgb=rgb, lab=lab, share=share)


# --------------------------------------------------------------------------- #
# resolve_family — preset ΔE76 매칭 vs lab_to_family fallback
# --------------------------------------------------------------------------- #

def testresolve_family_uses_preset_when_within_threshold() -> None:
    # cluster LAB=(50, 10, 10), preset entry LAB=(50, 10, 10) → ΔE=0 → PASTEL 반환
    entries = [
        MatcherEntry(name="preset_pastel", lab=(50.0, 10.0, 10.0), family=ColorFamily.PASTEL),
        MatcherEntry(name="preset_jewel", lab=(30.0, 60.0, 50.0), family=ColorFamily.JEWEL),
    ]
    cluster = _px_cluster(lab=(50.0, 10.0, 10.0))
    assert resolve_family(cluster, entries) == ColorFamily.PASTEL


def testresolve_family_falls_back_when_no_entry_within_threshold() -> None:
    # preset entries 전부 멀면 lab_to_family rule 호출
    # cluster LAB=(90, 2, 2) → chroma ≈ 2.8, L>85 → WHITE_ON_WHITE (lab_to_family rule)
    entries = [
        MatcherEntry(name="far", lab=(0.0, 70.0, 70.0), family=ColorFamily.JEWEL),
    ]
    cluster = _px_cluster(lab=(90.0, 2.0, 2.0))
    assert resolve_family(cluster, entries) == ColorFamily.WHITE_ON_WHITE


def testresolve_family_empty_entries_uses_lab_to_family() -> None:
    # L=20, chroma=3 → NEUTRAL rule
    cluster = _px_cluster(lab=(20.0, 2.0, 2.0))
    assert resolve_family(cluster, []) == ColorFamily.NEUTRAL


def testresolve_family_picks_nearest_among_multiple() -> None:
    entries = [
        MatcherEntry(name="far", lab=(10.0, 10.0, 10.0), family=ColorFamily.JEWEL),
        MatcherEntry(name="near", lab=(51.0, 10.0, 10.0), family=ColorFamily.PASTEL),
        MatcherEntry(name="mid", lab=(55.0, 15.0, 15.0), family=ColorFamily.EARTH),
    ]
    cluster = _px_cluster(lab=(50.0, 10.0, 10.0))
    assert resolve_family(cluster, entries) == ColorFamily.PASTEL


def test_preset_match_threshold_default_value_pinned() -> None:
    # canonical source for preset ΔE76 threshold — 변경 시 pool_02 resmoke + full smoke 필요
    # (50-color preset spacing 과 연동).
    assert PRESET_MATCH_THRESHOLD == 15.0
