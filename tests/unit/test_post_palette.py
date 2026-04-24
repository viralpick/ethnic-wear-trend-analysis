"""Phase B3b post_palette pinning — area_ratio × within-share 가중 ΔE76 merge.

PaletteCluster → hex/share/family 만 pydantic 경계에 노출. 내부 LAB 연산은 color_space
의 D65 경로를 그대로 사용 (dynamic_palette 와 같은 pixel 증거 원칙).
"""
from __future__ import annotations

import pytest

pytest.importorskip("sklearn", reason="sklearn required (color_space deps)")

from contracts.common import ColorFamily, PaletteCluster  # noqa: E402
from contracts.vision import CanonicalOutfit, EthnicOutfit, OutfitMember  # noqa: E402
from vision.post_palette import (  # noqa: E402
    MAX_POST_PALETTE_CLUSTERS,
    MIN_POST_PALETTE_SHARE,
    POST_PALETTE_MERGE_DELTA_E,
    build_post_palette,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _outfit(area: float = 0.5) -> EthnicOutfit:
    return EthnicOutfit(
        person_bbox=(0.1, 0.1, 0.5, 0.7),
        person_bbox_area_ratio=area,
        upper_garment_type="kurta",
        upper_is_ethnic=True,
        lower_garment_type="palazzo",
        lower_is_ethnic=True,
        dress_as_single=False,
    )


def _member(image_id: str = "img0", outfit_index: int = 0) -> OutfitMember:
    return OutfitMember(
        image_id=image_id,
        outfit_index=outfit_index,
        person_bbox=(0.1, 0.1, 0.5, 0.7),
    )


def _canonical(
    *,
    area: float = 0.5,
    palette: list[PaletteCluster] | None = None,
    canonical_index: int = 0,
) -> CanonicalOutfit:
    return CanonicalOutfit(
        canonical_index=canonical_index,
        representative=_outfit(area=area),
        members=[_member()],
        palette=palette or [],
    )


def _pc(hex_: str, share: float, family: ColorFamily | None = ColorFamily.JEWEL) -> PaletteCluster:
    return PaletteCluster(hex=hex_, share=share, family=family)


# --------------------------------------------------------------------------- #
# constants pinned (prevent accidental threshold drift)
# --------------------------------------------------------------------------- #

def test_constants_pinned() -> None:
    # merge threshold 는 dynamic_palette 와 같은 10.0 (preset 15 와 별개)
    assert POST_PALETTE_MERGE_DELTA_E == 10.0
    assert MIN_POST_PALETTE_SHARE == 0.05
    assert MAX_POST_PALETTE_CLUSTERS == 3


# --------------------------------------------------------------------------- #
# empty / trivial paths
# --------------------------------------------------------------------------- #

def test_empty_canonicals_returns_empty() -> None:
    assert build_post_palette([]) == []


def test_all_palettes_empty_returns_empty() -> None:
    canonicals = [_canonical(area=0.5, palette=[]), _canonical(area=0.3, palette=[])]
    assert build_post_palette(canonicals) == []


def test_zero_area_canonical_skipped() -> None:
    # area=0 은 weight 가 0 이라 flatten 에서 skip.
    canonicals = [_canonical(area=0.0, palette=[_pc("#AA0000", 1.0)])]
    assert build_post_palette(canonicals) == []


# --------------------------------------------------------------------------- #
# single canonical
# --------------------------------------------------------------------------- #

def test_single_canonical_single_color() -> None:
    canonicals = [
        _canonical(area=0.4, palette=[_pc("#AA0000", 1.0, ColorFamily.JEWEL)]),
    ]
    result = build_post_palette(canonicals)
    assert len(result) == 1
    assert result[0].share == pytest.approx(1.0, abs=1e-6)
    assert result[0].family == ColorFamily.JEWEL


# --------------------------------------------------------------------------- #
# merge near-identical LAB from two canonicals
# --------------------------------------------------------------------------- #

def test_two_canonicals_same_color_merge_family_by_weight() -> None:
    # 같은 hex → ΔE=0 → 1 cluster. area × share weight 로 family 결정.
    # c1: area 0.6 × share 1.0 = 0.6  (PASTEL)
    # c2: area 0.3 × share 1.0 = 0.3  (JEWEL)  → PASTEL 승리
    canonicals = [
        _canonical(
            area=0.6, palette=[_pc("#AABBCC", 1.0, ColorFamily.PASTEL)],
            canonical_index=0,
        ),
        _canonical(
            area=0.3, palette=[_pc("#AABBCC", 1.0, ColorFamily.JEWEL)],
            canonical_index=1,
        ),
    ]
    result = build_post_palette(canonicals)
    assert len(result) == 1
    assert result[0].share == pytest.approx(1.0, abs=1e-6)
    assert result[0].family == ColorFamily.PASTEL


# --------------------------------------------------------------------------- #
# distinct colors preserved
# --------------------------------------------------------------------------- #

def test_far_colors_do_not_merge() -> None:
    # red / blue / green — ΔE76 훨씬 > 10 → 3 clusters 유지.
    canonicals = [
        _canonical(area=0.5, palette=[_pc("#CC0000", 1.0, ColorFamily.JEWEL)]),
        _canonical(area=0.3, palette=[_pc("#0000CC", 1.0, ColorFamily.JEWEL)]),
        _canonical(area=0.2, palette=[_pc("#00AA00", 1.0, ColorFamily.JEWEL)]),
    ]
    result = build_post_palette(canonicals)
    assert len(result) == 3
    assert sum(c.share for c in result) == pytest.approx(1.0, abs=1e-6)
    # weight desc: 0.5 / 0.3 / 0.2 → share 0.5 / 0.3 / 0.2
    shares = [c.share for c in result]
    assert shares == sorted(shares, reverse=True)
    assert shares[0] == pytest.approx(0.5, abs=1e-6)


# --------------------------------------------------------------------------- #
# cap at MAX_POST_PALETTE_CLUSTERS
# --------------------------------------------------------------------------- #

def test_caps_at_max_clusters() -> None:
    # 서로 멀리 떨어진 5색 → merge 없음 → drop<0.05 없음 → top 3 cap.
    canonicals = [
        _canonical(area=0.25, palette=[_pc("#CC0000", 1.0)], canonical_index=0),
        _canonical(area=0.20, palette=[_pc("#0000CC", 1.0)], canonical_index=1),
        _canonical(area=0.20, palette=[_pc("#00AA00", 1.0)], canonical_index=2),
        _canonical(area=0.20, palette=[_pc("#E6E600", 1.0)], canonical_index=3),
        _canonical(area=0.15, palette=[_pc("#AA00AA", 1.0)], canonical_index=4),
    ]
    result = build_post_palette(canonicals)
    assert len(result) == MAX_POST_PALETTE_CLUSTERS
    assert sum(c.share for c in result) == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# small-share drop
# --------------------------------------------------------------------------- #

def test_small_share_dropped_before_cap() -> None:
    # weight: 0.50 / 0.40 / 0.02  (0.02 → share 0.022 < 0.05 drop)
    # 3색 서로 멀어서 merge 없음. drop 후 2 cluster 재정규화.
    canonicals = [
        _canonical(area=0.50, palette=[_pc("#CC0000", 1.0)], canonical_index=0),
        _canonical(area=0.40, palette=[_pc("#0000CC", 1.0)], canonical_index=1),
        _canonical(area=0.02, palette=[_pc("#00AA00", 1.0)], canonical_index=2),
    ]
    result = build_post_palette(canonicals)
    assert len(result) == 2
    assert sum(c.share for c in result) == pytest.approx(1.0, abs=1e-6)
    # 0.50 / 0.90 ≈ 0.5556, 0.40 / 0.90 ≈ 0.4444
    assert result[0].share == pytest.approx(0.50 / 0.90, abs=1e-6)


# --------------------------------------------------------------------------- #
# determinism
# --------------------------------------------------------------------------- #

def test_is_deterministic() -> None:
    canonicals = [
        _canonical(area=0.5, palette=[
            _pc("#AA0000", 0.7, ColorFamily.JEWEL),
            _pc("#0000AA", 0.3, ColorFamily.JEWEL),
        ], canonical_index=0),
        _canonical(area=0.3, palette=[
            _pc("#00AA00", 1.0, ColorFamily.EARTH),
        ], canonical_index=1),
    ]
    r1 = build_post_palette(canonicals)
    r2 = build_post_palette(canonicals)
    assert [(c.hex, c.family) for c in r1] == [(c.hex, c.family) for c in r2]
    assert [c.share for c in r1] == pytest.approx([c.share for c in r2])


# --------------------------------------------------------------------------- #
# family tiebreak — equal weight, enum order decides
# --------------------------------------------------------------------------- #

def test_family_tiebreak_by_enum_order_on_equal_weight() -> None:
    # 같은 hex (ΔE=0), 같은 area × share → weight 동점. ColorFamily 선언 순서:
    # PASTEL < EARTH < NEUTRAL < WHITE_ON_WHITE < JEWEL < BRIGHT ...
    # → JEWEL(4) 보다 PASTEL(0) 이 선택되어야 함.
    canonicals = [
        _canonical(
            area=0.4, palette=[_pc("#AABBCC", 1.0, ColorFamily.JEWEL)],
            canonical_index=0,
        ),
        _canonical(
            area=0.4, palette=[_pc("#AABBCC", 1.0, ColorFamily.PASTEL)],
            canonical_index=1,
        ),
    ]
    result = build_post_palette(canonicals)
    assert len(result) == 1
    assert result[0].family == ColorFamily.PASTEL
