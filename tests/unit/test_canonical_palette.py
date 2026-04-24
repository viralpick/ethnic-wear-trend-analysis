"""Phase B2 canonical_palette pinning — top-N 재정규화 / family 매핑 / e2e.

sklearn (numpy) + dynamic_palette 필요. color_preset MatcherEntry 는 테스트 내 inline.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn", reason="sklearn required")

from contracts.common import ColorFamily, PaletteCluster  # noqa: E402
from settings import DynamicPaletteConfig  # noqa: E402
from vision.canonical_palette import (  # noqa: E402
    MAX_CANONICAL_PALETTE_CLUSTERS,
    PRESET_MATCH_THRESHOLD,
    _resolve_family,
    _top_n_renormalize,
    build_canonical_palette,
)
from vision.color_family_preset import MatcherEntry  # noqa: E402
from vision.dynamic_palette import PaletteCluster as PixelCluster  # noqa: E402


def _fill(color: tuple[int, int, int], n: int) -> np.ndarray:
    arr = np.empty((n, 3), dtype=np.uint8)
    arr[:] = color
    return arr


def _px_cluster(
    hex_: str = "#AA0000", rgb=(170, 0, 0), lab=(30.0, 60.0, 50.0), share: float = 1.0,
) -> PixelCluster:
    return PixelCluster(hex=hex_, rgb=rgb, lab=lab, share=share)


# --------------------------------------------------------------------------- #
# _top_n_renormalize
# --------------------------------------------------------------------------- #

def test_top_n_renormalize_cuts_and_rebalances() -> None:
    # share 0.5 / 0.3 / 0.15 / 0.05 → top 3 = 0.5/0.3/0.15 (sum 0.95) → 재정규화 후 sum=1.0
    clusters = [
        _px_cluster(share=0.5), _px_cluster(share=0.3),
        _px_cluster(share=0.15), _px_cluster(share=0.05),
    ]
    result = _top_n_renormalize(clusters, 3)
    assert len(result) == 3
    assert sum(c.share for c in result) == pytest.approx(1.0, abs=1e-6)
    # 비율 유지 확인 (0.5 / 0.95 ≈ 0.5263)
    assert result[0].share == pytest.approx(0.5 / 0.95, abs=1e-6)


def test_top_n_renormalize_keeps_all_when_already_small() -> None:
    clusters = [_px_cluster(share=0.6), _px_cluster(share=0.4)]
    result = _top_n_renormalize(clusters, 3)
    assert len(result) == 2
    assert sum(c.share for c in result) == pytest.approx(1.0, abs=1e-6)


def test_top_n_renormalize_empty_returns_empty() -> None:
    assert _top_n_renormalize([], 3) == []


def test_top_n_renormalize_zero_max_returns_empty() -> None:
    clusters = [_px_cluster(share=1.0)]
    assert _top_n_renormalize(clusters, 0) == []


# --------------------------------------------------------------------------- #
# _resolve_family — preset ΔE76 매칭 vs lab_to_family fallback
# --------------------------------------------------------------------------- #

def test_resolve_family_uses_preset_when_within_threshold() -> None:
    # cluster LAB=(50, 10, 10), preset entry LAB=(50, 10, 10) → ΔE=0 → PASTEL 반환
    entries = [
        MatcherEntry(name="preset_pastel", lab=(50.0, 10.0, 10.0), family=ColorFamily.PASTEL),
        MatcherEntry(name="preset_jewel", lab=(30.0, 60.0, 50.0), family=ColorFamily.JEWEL),
    ]
    cluster = _px_cluster(lab=(50.0, 10.0, 10.0))
    assert _resolve_family(cluster, entries) == ColorFamily.PASTEL


def test_resolve_family_falls_back_when_no_entry_within_threshold() -> None:
    # preset entries 전부 멀면 lab_to_family rule 호출
    # cluster LAB=(90, 2, 2) → chroma ≈ 2.8, L>85 → WHITE_ON_WHITE (lab_to_family rule)
    entries = [
        MatcherEntry(name="far", lab=(0.0, 70.0, 70.0), family=ColorFamily.JEWEL),
    ]
    cluster = _px_cluster(lab=(90.0, 2.0, 2.0))
    assert _resolve_family(cluster, entries) == ColorFamily.WHITE_ON_WHITE


def test_resolve_family_empty_entries_uses_lab_to_family() -> None:
    # L=20, chroma=3 → NEUTRAL rule
    cluster = _px_cluster(lab=(20.0, 2.0, 2.0))
    assert _resolve_family(cluster, []) == ColorFamily.NEUTRAL


def test_resolve_family_picks_nearest_among_multiple() -> None:
    entries = [
        MatcherEntry(name="far", lab=(10.0, 10.0, 10.0), family=ColorFamily.JEWEL),
        MatcherEntry(name="near", lab=(51.0, 10.0, 10.0), family=ColorFamily.PASTEL),
        MatcherEntry(name="mid", lab=(55.0, 15.0, 15.0), family=ColorFamily.EARTH),
    ]
    cluster = _px_cluster(lab=(50.0, 10.0, 10.0))
    assert _resolve_family(cluster, entries) == ColorFamily.PASTEL


# --------------------------------------------------------------------------- #
# build_canonical_palette — e2e
# --------------------------------------------------------------------------- #

def test_build_canonical_palette_single_color_pool_yields_one_cluster() -> None:
    pixels = _fill((220, 20, 20), 1000)
    result = build_canonical_palette(pixels, DynamicPaletteConfig(), [])
    assert len(result) == 1
    assert result[0].share == pytest.approx(1.0, abs=1e-6)
    assert result[0].hex.startswith("#")
    assert isinstance(result[0], PaletteCluster)  # pydantic 경계


def test_build_canonical_palette_caps_at_max_clusters() -> None:
    # 5개 뚜렷한 색 → kmeans k=5, ΔE76 서로 충분히 멀어 merge X → top 3 자름
    pixels = np.concatenate([
        _fill((220, 20, 20), 500),    # red
        _fill((20, 20, 220), 500),    # blue
        _fill((20, 180, 20), 500),    # green
        _fill((230, 230, 30), 500),   # yellow
        _fill((180, 30, 180), 500),   # magenta
    ])
    result = build_canonical_palette(pixels, DynamicPaletteConfig(), [])
    assert len(result) <= MAX_CANONICAL_PALETTE_CLUSTERS
    assert sum(c.share for c in result) == pytest.approx(1.0, abs=1e-6)


def test_build_canonical_palette_empty_pool_returns_empty() -> None:
    empty = np.empty((0, 3), dtype=np.uint8)
    result = build_canonical_palette(empty, DynamicPaletteConfig(), [])
    assert result == []


def test_build_canonical_palette_below_min_pixels_returns_empty() -> None:
    # min_pixels=150 default — 50 pixel pool 은 버려짐
    pixels = _fill((200, 200, 200), 50)
    result = build_canonical_palette(pixels, DynamicPaletteConfig(), [])
    assert result == []


def test_build_canonical_palette_assigns_family_from_preset() -> None:
    # 단색 pool, preset 매칭으로 family 확정. preset LAB 은 실제 cluster LAB 과
    # 동일하게 계산해 ΔE=0 으로 매칭 보장 — 실제 cluster LAB 값에 의존하지 않게.
    from vision.color_space import rgb_to_lab  # noqa: PLC0415
    rgb = np.array([[220, 20, 20]], dtype=np.uint8)
    L, a, b = rgb_to_lab(rgb)[0]
    pixels = _fill((220, 20, 20), 1000)
    entries = [
        MatcherEntry(
            name="red_jewel", lab=(float(L), float(a), float(b)),
            family=ColorFamily.JEWEL,
        ),
    ]
    result = build_canonical_palette(pixels, DynamicPaletteConfig(), entries)
    assert len(result) == 1
    assert result[0].family == ColorFamily.JEWEL


def test_build_canonical_palette_family_fallback_without_entries() -> None:
    # 흰색 pool → lab_to_family 만 사용 → WHITE_ON_WHITE
    pixels = _fill((245, 245, 245), 1000)
    result = build_canonical_palette(pixels, DynamicPaletteConfig(), [])
    assert len(result) == 1
    assert result[0].family == ColorFamily.WHITE_ON_WHITE


def test_build_canonical_palette_is_deterministic() -> None:
    pixels = np.concatenate([
        _fill((220, 20, 20), 500),
        _fill((20, 20, 220), 500),
        _fill((20, 180, 20), 500),
    ])
    cfg = DynamicPaletteConfig()
    r1 = build_canonical_palette(pixels, cfg, [])
    r2 = build_canonical_palette(pixels, cfg, [])
    assert [(c.hex, c.family) for c in r1] == [(c.hex, c.family) for c in r2]
    assert [c.share for c in r1] == pytest.approx([c.share for c in r2])


def test_preset_match_threshold_default_value_pinned() -> None:
    # canonical source for preset ΔE76 threshold — B3a 에서 adapter 쪽 중복 제거 후 이 값
    # 이 유일. 변경 시 pool_02 resmoke + full smoke 필요 (50-color preset spacing 과 연동).
    assert PRESET_MATCH_THRESHOLD == 15.0
