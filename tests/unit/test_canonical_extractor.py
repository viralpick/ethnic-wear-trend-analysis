"""Phase 3 canonical_extractor pinning — size drop / member extraction / pool / e2e.

vision extras (torch / transformers / ultralytics) 필수 — pipeline_b_extractor 가
top-level 로 import 하므로. importorskip 으로 extras 없는 환경에선 skip.

실 segformer 호출은 monkeypatch 로 stub (Phase 4.5 의 dedup_post 와 결합해 e2e 검증).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("torch", reason="vision extras required")
pytest.importorskip("transformers", reason="vision extras required")
pytest.importorskip("ultralytics", reason="vision extras required")


from contracts.common import ColorFamily, Silhouette  # noqa: E402
from contracts.vision import (  # noqa: E402
    CanonicalOutfit,
    EthnicOutfit,
    GarmentAnalysis,
    OutfitMember,
)
from settings import OutfitDedupConfig, load_settings  # noqa: E402
from vision.canonical_extractor import (  # noqa: E402
    _extract_member_pixels,
    _pool_canonical,
    drop_small_outfits,
    extract_canonical_pixels,
)
from vision.pipeline_b_extractor import SegBundle  # noqa: E402
from vision.scene_filter import NoopSceneFilter  # noqa: E402

# LAB box 밖 (a=71) — segformer garment pixel 이 전부 이 색이면 skin drop 0.
VIBRANT_RED: tuple[int, int, int] = (230, 30, 40)

_FAMILY_MAP: dict[str, ColorFamily] = {
    "ivory": ColorFamily.WHITE_ON_WHITE,
    "saffron": ColorFamily.BRIGHT,
    "pool_05": ColorFamily.NEUTRAL,
}


def _outfit(area_ratio: float, **overrides) -> EthnicOutfit:
    side = math.sqrt(area_ratio)
    base = dict(
        person_bbox=(0.0, 0.0, side, side),
        person_bbox_area_ratio=area_ratio,
        upper_garment_type="kurta",
        lower_garment_type="palazzo",
        dress_as_single=False,
        silhouette=Silhouette.A_LINE,
        fabric="cotton",
        technique="chikankari",
        color_preset_picks_top3=["ivory", "saffron", "pool_05"],
    )
    base.update(overrides)
    return EthnicOutfit(**base)


def _make_bundle() -> SegBundle:
    # run_segformer 는 monkeypatch 대상 — yolo/seg_* 는 dummy 로 충분.
    return SegBundle(
        yolo=object(), seg_processor=object(), seg_model=object(),
        device="cpu", scene_filter=NoopSceneFilter(),
    )


def _seg_stub_top_half_upper(_bundle, crop_rgb):
    """stub: crop 상단 절반을 upper-clothes(4), 나머지 background(0)."""
    h, w = crop_rgb.shape[:2]
    seg = np.zeros((h, w), dtype=np.int32)
    seg[: h // 2] = 4
    return seg


def _seg_stub_all_background(_bundle, crop_rgb):
    """stub: 전 crop background — garment pixel 0."""
    h, w = crop_rgb.shape[:2]
    return np.zeros((h, w), dtype=np.int32)


def _make_rgb(h: int = 200, w: int = 200, color: tuple[int, int, int] = VIBRANT_RED):
    rgb = np.empty((h, w, 3), dtype=np.uint8)
    rgb[:, :] = color
    return rgb


# --------------------------------------------------------------------------- #
# drop_small_outfits — size drop
# --------------------------------------------------------------------------- #

def test_drop_small_outfits_filters_below_threshold() -> None:
    small = _outfit(0.05)
    big = _outfit(0.30)
    analysis = GarmentAnalysis(is_india_ethnic_wear=True, outfits=[small, big])
    result = drop_small_outfits(analysis, min_area_ratio=0.10)
    assert len(result.outfits) == 1
    assert result.outfits[0].person_bbox_area_ratio == 0.30
    assert result.is_india_ethnic_wear is True


def test_drop_small_outfits_preserves_ethnic_flag_when_all_dropped() -> None:
    tiny1 = _outfit(0.03)
    tiny2 = _outfit(0.07)
    analysis = GarmentAnalysis(is_india_ethnic_wear=True, outfits=[tiny1, tiny2])
    result = drop_small_outfits(analysis, min_area_ratio=0.10)
    assert result.outfits == []
    assert result.is_india_ethnic_wear is True


def test_drop_small_outfits_no_change_returns_same_object() -> None:
    a = _outfit(0.50)
    b = _outfit(0.20)
    analysis = GarmentAnalysis(is_india_ethnic_wear=True, outfits=[a, b])
    result = drop_small_outfits(analysis, min_area_ratio=0.10)
    assert result is analysis


# --------------------------------------------------------------------------- #
# _extract_member_pixels — monkeypatched run_segformer
# --------------------------------------------------------------------------- #

def test_extract_member_pixels_returns_cleaned_pool(monkeypatch) -> None:
    monkeypatch.setattr(
        "vision.canonical_extractor.run_segformer", _seg_stub_top_half_upper,
    )
    rgb = _make_rgb(200, 200)
    cfg = load_settings().vision
    # bbox (x=0.1, y=0.1, w=0.5, h=0.5) → crop 100x100 (x:20..120, y:20..120).
    cleaned, primary, secondary = _extract_member_pixels(
        rgb, (0.1, 0.1, 0.5, 0.5), _make_bundle(), cfg,
    )
    # 상단 50 row × 100 col = 5000 garment pixel, 전부 VIBRANT (LAB 밖) → skin drop 0.
    assert cleaned.shape == (5000, 3)
    assert primary == 0
    assert secondary == 0


def test_extract_member_pixels_skips_too_small_bbox(monkeypatch) -> None:
    called = []

    def counting_stub(_bundle, _crop):
        called.append(1)
        return np.zeros(_crop.shape[:2], dtype=np.int32)

    monkeypatch.setattr(
        "vision.canonical_extractor.run_segformer", counting_stub,
    )
    rgb = _make_rgb(200, 200)
    cfg = load_settings().vision
    # bbox 10×10 px < MIN_CROP_PX(32).
    cleaned, primary, secondary = _extract_member_pixels(
        rgb, (0.0, 0.0, 0.05, 0.05), _make_bundle(), cfg,
    )
    assert cleaned.shape == (0, 3)
    assert (primary, secondary) == (0, 0)
    assert called == []   # run_segformer 호출 X


# --------------------------------------------------------------------------- #
# _pool_canonical — members 합산 / empty pool None
# --------------------------------------------------------------------------- #

def test_pool_canonical_concats_members(monkeypatch) -> None:
    monkeypatch.setattr(
        "vision.canonical_extractor.run_segformer", _seg_stub_top_half_upper,
    )
    cfg = load_settings().vision
    rep = _outfit(0.25)
    canonical = CanonicalOutfit(
        canonical_index=0,
        representative=rep,
        members=[
            OutfitMember(
                image_id="img_0", outfit_index=0,
                person_bbox=(0.0, 0.0, 0.5, 0.5),
            ),
            OutfitMember(
                image_id="img_1", outfit_index=0,
                person_bbox=(0.0, 0.0, 0.5, 0.5),
            ),
        ],
    )
    frame_map = {"img_0": _make_rgb(200, 200), "img_1": _make_rgb(200, 200)}
    result = _pool_canonical(canonical, frame_map, _make_bundle(), cfg)
    assert result is not None
    # crop 100x100 × 상단 절반 garment = 5000 pixel × 2 member = 10000.
    assert result.pooled_pixels.shape == (10000, 3)
    assert result.per_image_pixel_counts == {"img_0": 5000, "img_1": 5000}
    assert (result.skin_drop_primary_total, result.skin_drop_secondary_total) == (0, 0)


def test_pool_canonical_returns_none_when_all_empty(monkeypatch) -> None:
    # 모든 member 가 background-only → garment mask 빈 → empty pool.
    monkeypatch.setattr(
        "vision.canonical_extractor.run_segformer", _seg_stub_all_background,
    )
    cfg = load_settings().vision
    rep = _outfit(0.25)
    canonical = CanonicalOutfit(
        canonical_index=0, representative=rep,
        members=[OutfitMember(
            image_id="img_0", outfit_index=0,
            person_bbox=(0.0, 0.0, 0.5, 0.5),
        )],
    )
    frame_map = {"img_0": _make_rgb(200, 200)}
    assert _pool_canonical(canonical, frame_map, _make_bundle(), cfg) is None


# --------------------------------------------------------------------------- #
# extract_canonical_pixels — end-to-end (size drop + dedup + pool)
# --------------------------------------------------------------------------- #

def test_extract_canonical_pixels_e2e_merges_same_outfit(monkeypatch) -> None:
    monkeypatch.setattr(
        "vision.canonical_extractor.run_segformer", _seg_stub_top_half_upper,
    )
    rgb = _make_rgb(200, 200)
    outfit = _outfit(0.25)
    # 2-image post 에 같은 outfit 속성 → Phase 4.5 가 1 canonical 로 병합.
    post_items = [
        ("img_0", rgb, GarmentAnalysis(is_india_ethnic_wear=True, outfits=[outfit])),
        ("img_1", rgb, GarmentAnalysis(is_india_ethnic_wear=True, outfits=[outfit])),
    ]
    result = extract_canonical_pixels(
        post_items, _make_bundle(), load_settings().vision,
        OutfitDedupConfig(), _FAMILY_MAP,
    )
    assert len(result) == 1
    assert {m.image_id for m in result[0].members_meta} == {"img_0", "img_1"}
    assert result[0].pooled_pixels.shape == (10000, 3)


def test_extract_canonical_pixels_drops_small_outfits(monkeypatch) -> None:
    monkeypatch.setattr(
        "vision.canonical_extractor.run_segformer", _seg_stub_top_half_upper,
    )
    rgb = _make_rgb(200, 200)
    # area_ratio 0.05 < min 0.10 — dedup 진입 전 drop.
    tiny = _outfit(0.05)
    post_items = [
        ("img_0", rgb, GarmentAnalysis(is_india_ethnic_wear=True, outfits=[tiny])),
    ]
    result = extract_canonical_pixels(
        post_items, _make_bundle(), load_settings().vision,
        OutfitDedupConfig(), _FAMILY_MAP,
    )
    assert result == []


def test_extract_canonical_pixels_skips_empty_pool(monkeypatch) -> None:
    # outfit keep, 하지만 segformer 가 전부 background → pool 빈 → 결과에서 skip.
    monkeypatch.setattr(
        "vision.canonical_extractor.run_segformer", _seg_stub_all_background,
    )
    rgb = _make_rgb(200, 200)
    outfit = _outfit(0.25)
    post_items = [
        ("img_0", rgb, GarmentAnalysis(is_india_ethnic_wear=True, outfits=[outfit])),
    ]
    result = extract_canonical_pixels(
        post_items, _make_bundle(), load_settings().vision,
        OutfitDedupConfig(), _FAMILY_MAP,
    )
    assert result == []
