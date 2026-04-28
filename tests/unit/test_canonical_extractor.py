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
    ObjectPool,
    _build_picks_lookup,
    _build_skin_drop_config,
    _extract_member_pixels,
    _lookup_member_picks,
    _select_wear_class_ids,
    drop_small_outfits,
    extract_canonical_pixels_per_object,
)
from vision.pipeline_b_extractor import SegBundle  # noqa: E402
from vision.scene_filter import NoopSceneFilter  # noqa: E402
from vision.segformer_constants import (  # noqa: E402
    DRESS_CLASS_IDS,
    LOWER_CLASS_IDS,
    UPPER_CLASS_IDS,
    WEAR_CLASS_IDS,
)

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
        upper_is_ethnic=True,
        lower_garment_type="palazzo",
        lower_is_ethnic=True,
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
    skin_drop_cfg = _build_skin_drop_config(load_settings().vision)
    # bbox (x=0.1, y=0.1, w=0.5, h=0.5) → crop 100x100 (x:20..120, y:20..120).
    cleaned, primary, secondary = _extract_member_pixels(
        rgb, (0.1, 0.1, 0.5, 0.5), _make_bundle(), skin_drop_cfg, WEAR_CLASS_IDS,
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
    skin_drop_cfg = _build_skin_drop_config(load_settings().vision)
    # bbox 10×10 px < MIN_CROP_PX(32).
    cleaned, primary, secondary = _extract_member_pixels(
        rgb, (0.0, 0.0, 0.05, 0.05), _make_bundle(), skin_drop_cfg, WEAR_CLASS_IDS,
    )
    assert cleaned.shape == (0, 3)
    assert (primary, secondary) == (0, 0)
    assert called == []   # run_segformer 호출 X


# --------------------------------------------------------------------------- #
# B1 — _select_wear_class_ids ethnic-aware pool selection
# --------------------------------------------------------------------------- #

def test_select_wear_class_ids_upper_only() -> None:
    # F-10 (2026-04-26): segformer 가 saree top 같은 ethnic 상의를 dress 로 분류해도
    # 픽셀이 빠지지 않도록 DRESS_CLASS_IDS union.
    o = _outfit(0.25, upper_is_ethnic=True, lower_is_ethnic=False)
    assert _select_wear_class_ids(o) == UPPER_CLASS_IDS | DRESS_CLASS_IDS


def test_select_wear_class_ids_lower_only() -> None:
    o = _outfit(0.25, upper_is_ethnic=False, lower_is_ethnic=True)
    assert _select_wear_class_ids(o) == LOWER_CLASS_IDS | DRESS_CLASS_IDS


def test_select_wear_class_ids_two_piece_both_ethnic() -> None:
    # 양쪽 ethnic 시 DRESS_CLASS_IDS 도 union — Gemini 2-piece ↔ segformer dress
    # disagreement 케이스 방어 (F-7 algorithmic gap fix). 어느 쪽이 옳든 dress
    # 픽셀을 ethnic pool 에 포함하는 게 정합.
    o = _outfit(0.25, upper_is_ethnic=True, lower_is_ethnic=True)
    assert _select_wear_class_ids(o) == (
        UPPER_CLASS_IDS | LOWER_CLASS_IDS | DRESS_CLASS_IDS
    )


def test_select_wear_class_ids_dress_as_single_ethnic() -> None:
    # F-10 (2026-04-26): dress_as_single + ethnic → segformer 가 어떻게 쪼개도 모두
    # keep (UPPER + LOWER + DRESS). Sridevi saree 의 maroon 상의가 upper-clothes 로
    # 분류돼 dress-only pool 에서 96% 손실되던 케이스 방어.
    o = _outfit(
        0.25, dress_as_single=True, upper_is_ethnic=True,
        lower_garment_type=None, lower_is_ethnic=None,
    )
    assert _select_wear_class_ids(o) == (
        UPPER_CLASS_IDS | LOWER_CLASS_IDS | DRESS_CLASS_IDS
    )


def test_select_wear_class_ids_dress_as_single_non_ethnic() -> None:
    o = _outfit(
        0.25, dress_as_single=True, upper_is_ethnic=False,
        lower_garment_type=None, lower_is_ethnic=None,
    )
    assert _select_wear_class_ids(o) == frozenset()


def test_select_wear_class_ids_neither_ethnic_returns_empty() -> None:
    o = _outfit(0.25, upper_is_ethnic=False, lower_is_ethnic=False)
    assert _select_wear_class_ids(o) == frozenset()


def test_select_wear_class_ids_none_treated_as_false() -> None:
    # Gemini 누락 방어 — None 은 보수적으로 False 취급.
    o = _outfit(0.25, upper_is_ethnic=None, lower_is_ethnic=None)
    assert _select_wear_class_ids(o) == frozenset()


def test_select_wear_class_ids_none_emits_warning(caplog) -> None:
    # 실패 숨김 금지 (CLAUDE.md #4) — None 은 WARNING 로 가시화.
    o = _outfit(0.25, upper_is_ethnic=None, lower_is_ethnic=True)
    import logging  # noqa: PLC0415
    with caplog.at_level(logging.WARNING, logger="vision.canonical_extractor"):
        _select_wear_class_ids(o)
    assert any(
        "canonical_ethnic_flag_missing" in r.message for r in caplog.records
    )


def test_select_wear_class_ids_dress_as_single_none_warns(caplog) -> None:
    o = _outfit(
        0.25, dress_as_single=True, upper_is_ethnic=None,
        lower_garment_type=None, lower_is_ethnic=None,
    )
    import logging  # noqa: PLC0415
    with caplog.at_level(logging.WARNING, logger="vision.canonical_extractor"):
        result = _select_wear_class_ids(o)
    assert result == frozenset()
    assert any(
        "canonical_ethnic_flag_missing" in r.message
        and "dress_as_single=True" in r.message
        for r in caplog.records
    )


def test_select_wear_class_ids_bool_false_no_warning(caplog) -> None:
    # False 는 정상 판정 — WARNING 없이 조용히 frozenset() 반환.
    o = _outfit(0.25, upper_is_ethnic=False, lower_is_ethnic=False)
    import logging  # noqa: PLC0415
    with caplog.at_level(logging.WARNING, logger="vision.canonical_extractor"):
        _select_wear_class_ids(o)
    assert not any(
        "canonical_ethnic_flag_missing" in r.message for r in caplog.records
    )


# --------------------------------------------------------------------------- #
# β-hybrid 재설계 (per-object) — picks lookup util
# --------------------------------------------------------------------------- #

def test_build_picks_lookup_indexes_filtered_outfits() -> None:
    a = _outfit(0.30, color_preset_picks_top3=["ivory"])
    b = _outfit(0.20, color_preset_picks_top3=["saffron", "pool_05"])
    items = [
        ("img_0", GarmentAnalysis(is_india_ethnic_wear=True, outfits=[a, b])),
        ("img_1", GarmentAnalysis(is_india_ethnic_wear=True, outfits=[a])),
    ]
    lookup = _build_picks_lookup(items)
    assert lookup[("img_0", 0)] == ["ivory"]
    assert lookup[("img_0", 1)] == ["saffron", "pool_05"]
    assert lookup[("img_1", 0)] == ["ivory"]
    assert len(lookup) == 3


def test_lookup_member_picks_raises_on_miss() -> None:
    member = OutfitMember(
        image_id="img_missing", outfit_index=0,
        person_bbox=(0.0, 0.0, 0.5, 0.5),
    )
    with pytest.raises(KeyError, match="picks lookup miss"):
        _lookup_member_picks({}, member)


# --------------------------------------------------------------------------- #
# extract_canonical_pixels_per_object — Phase 1 entry
# --------------------------------------------------------------------------- #

def test_extract_canonical_pixels_per_object_returns_one_pool_per_member(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "vision.canonical_extractor.run_segformer", _seg_stub_top_half_upper,
    )
    rgb = _make_rgb(200, 200)
    outfit = _outfit(0.25, color_preset_picks_top3=["ivory", "saffron"])
    post_items = [
        ("img_0", rgb, GarmentAnalysis(is_india_ethnic_wear=True, outfits=[outfit])),
        ("img_1", rgb, GarmentAnalysis(is_india_ethnic_wear=True, outfits=[outfit])),
    ]
    result = extract_canonical_pixels_per_object(
        post_items, _make_bundle(), load_settings().vision,
        OutfitDedupConfig(), _FAMILY_MAP,
    )
    assert len(result) == 1
    canonical, pools = result[0]
    assert canonical.canonical_index == 0
    assert len(pools) == 2
    # 2 members → 2 ObjectPool, 각 5000 px (상단 절반 garment).
    assert {p.member.image_id for p in pools} == {"img_0", "img_1"}
    for p in pools:
        assert isinstance(p, ObjectPool)
        assert p.rgb_pixels.shape == (5000, 3)
        assert p.picks == ["ivory", "saffron"]
        assert (p.skin_drop_primary, p.skin_drop_secondary) == (0, 0)
        # _make_rgb(200, 200) → H × W = 40_000. frame_area normalize 의 분모.
        assert p.frame_area == 200 * 200


def test_extract_canonical_pixels_per_object_picks_per_member_independent(
    monkeypatch,
) -> None:
    # 같은 dedup component 의 멤버가 각자 부분적으로 다른 picks 를 갖고 있을 때 ObjectPool
    # 별로 자기 picks 회수. dedup 가 묶었다고 picks 평균/대표 사용 X.
    # 일부 겹치게 (≥2) 만들어 dedup merge 를 통과시키되 정확히 같지는 않게.
    monkeypatch.setattr(
        "vision.canonical_extractor.run_segformer", _seg_stub_top_half_upper,
    )
    rgb = _make_rgb(200, 200)
    a = _outfit(0.25, color_preset_picks_top3=["ivory", "saffron", "pool_05"])
    b = _outfit(0.25, color_preset_picks_top3=["ivory", "saffron"])
    post_items = [
        ("img_0", rgb, GarmentAnalysis(is_india_ethnic_wear=True, outfits=[a])),
        ("img_1", rgb, GarmentAnalysis(is_india_ethnic_wear=True, outfits=[b])),
    ]
    result = extract_canonical_pixels_per_object(
        post_items, _make_bundle(), load_settings().vision,
        OutfitDedupConfig(), _FAMILY_MAP,
    )
    # garment_type/family 동일 + |top3 ∩ top3| = 2 → dedup 1 canonical.
    assert len(result) == 1
    _, pools = result[0]
    assert len(pools) == 2
    by_image = {p.member.image_id: p.picks for p in pools}
    assert by_image["img_0"] == ["ivory", "saffron", "pool_05"]
    assert by_image["img_1"] == ["ivory", "saffron"]


def test_extract_canonical_pixels_per_object_drops_small_outfits(monkeypatch) -> None:
    monkeypatch.setattr(
        "vision.canonical_extractor.run_segformer", _seg_stub_top_half_upper,
    )
    rgb = _make_rgb(200, 200)
    tiny = _outfit(0.05)  # < min 0.10
    post_items = [
        ("img_0", rgb, GarmentAnalysis(is_india_ethnic_wear=True, outfits=[tiny])),
    ]
    result = extract_canonical_pixels_per_object(
        post_items, _make_bundle(), load_settings().vision,
        OutfitDedupConfig(), _FAMILY_MAP,
    )
    assert result == []


def test_extract_canonical_pixels_per_object_preserves_canonical_when_pool_empty(
    monkeypatch,
) -> None:
    # 라벨 보존 invariant — non-ethnic 또는 background-only 면 pools=[] 로 함께 반환.
    monkeypatch.setattr(
        "vision.canonical_extractor.run_segformer", _seg_stub_all_background,
    )
    rgb = _make_rgb(200, 200)
    outfit = _outfit(0.25)
    post_items = [
        ("img_0", rgb, GarmentAnalysis(is_india_ethnic_wear=True, outfits=[outfit])),
    ]
    result = extract_canonical_pixels_per_object(
        post_items, _make_bundle(), load_settings().vision,
        OutfitDedupConfig(), _FAMILY_MAP,
    )
    assert len(result) == 1
    canonical, pools = result[0]
    assert canonical.canonical_index == 0
    assert pools == []


def test_extract_canonical_pixels_per_object_non_ethnic_skips_segformer(
    monkeypatch,
) -> None:
    # representative.upper/lower 모두 False → wear_class_ids 빈 → segformer 호출 0.
    called = []

    def counting_stub(_bundle, crop_rgb):
        called.append(1)
        h, w = crop_rgb.shape[:2]
        return np.zeros((h, w), dtype=np.int32)

    monkeypatch.setattr("vision.canonical_extractor.run_segformer", counting_stub)
    rgb = _make_rgb(200, 200)
    outfit = _outfit(0.25, upper_is_ethnic=False, lower_is_ethnic=False)
    post_items = [
        ("img_0", rgb, GarmentAnalysis(is_india_ethnic_wear=True, outfits=[outfit])),
    ]
    result = extract_canonical_pixels_per_object(
        post_items, _make_bundle(), load_settings().vision,
        OutfitDedupConfig(), _FAMILY_MAP,
    )
    assert len(result) == 1
    _, pools = result[0]
    assert pools == []
    assert called == []
