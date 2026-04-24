"""vision.traditional_filter — whitelist 기반 post-filter 단위 테스트.

`is_outfit_traditional` (원시 인자) 와 `apply_to_analysis` (GarmentAnalysis) 양쪽 검증.
flip 발생 시 logger.info 기록되는지 caplog 로 확인.
"""
from __future__ import annotations

import logging

import pytest

# vision.traditional_filter 는 json_repair / google.genai 에 의존 안 하지만,
# contracts.vision 을 쓰므로 core extras 만 필요 → importorskip 불필요.
from contracts.common import Silhouette
from contracts.vision import EthnicOutfit, GarmentAnalysis
from vision.traditional_filter import apply_to_analysis, is_outfit_traditional


def _outfit(
    *,
    upper: str | None,
    lower: str | None,
    single: bool = False,
    silhouette: Silhouette | None = Silhouette.STRAIGHT,
) -> EthnicOutfit:
    return EthnicOutfit(
        person_bbox=(0.1, 0.1, 0.5, 0.5),
        person_bbox_area_ratio=0.25,
        upper_garment_type=upper,
        lower_garment_type=lower,
        dress_as_single=single,
        silhouette=silhouette,
        color_preset_picks_top3=[],
    )


# --------------------------------------------------------------------------- #
# is_outfit_traditional — 원시 인자 API
# --------------------------------------------------------------------------- #

def test_two_piece_traditional_upper_and_lower():
    assert is_outfit_traditional("kurta", "churidar", dress_as_single=False) is True


def test_two_piece_only_upper_traditional():
    # 예: kurta + jeans (ethnic 으로 판정)
    assert is_outfit_traditional("kurta", "jeans", dress_as_single=False) is True


def test_two_piece_only_lower_traditional():
    # T-shirt + palazzo → indo-fusion, TRUE
    assert is_outfit_traditional("t_shirt", "palazzo", dress_as_single=False) is True


def test_two_piece_neither_traditional_is_false():
    # crop top + shorts → FALSE
    assert is_outfit_traditional("crop_top", "shorts", dress_as_single=False) is False


def test_single_piece_traditional_upper_only():
    # saree drape → TRUE (single, lower=None)
    assert is_outfit_traditional("saree", None, dress_as_single=True) is True


def test_single_piece_non_traditional_is_false():
    # bodycon dress (single) — whitelist 밖이라 FALSE
    assert is_outfit_traditional("bodycon_dress", None, dress_as_single=True) is False


def test_single_piece_ignores_lower_field():
    # single=True 면 lower 값은 무시. LLM 이 lower 를 채워도 upper 만 본다.
    assert is_outfit_traditional("saree", "palazzo", dress_as_single=True) is True
    assert is_outfit_traditional("bodycon_dress", "palazzo", dress_as_single=True) is False


def test_case_insensitive():
    assert is_outfit_traditional("KURTA", "CHURIDAR", dress_as_single=False) is True


def test_none_upper_lower_is_false():
    assert is_outfit_traditional(None, None, dress_as_single=False) is False


# --------------------------------------------------------------------------- #
# apply_to_analysis — GarmentAnalysis 레벨
# --------------------------------------------------------------------------- #

def test_apply_passes_through_when_already_false():
    analysis = GarmentAnalysis(is_india_ethnic_wear=False, outfits=[])
    result = apply_to_analysis(analysis)
    assert result.is_india_ethnic_wear is False
    assert result is analysis  # no-op 시 동일 인스턴스


def test_apply_flips_true_with_no_outfits(caplog):
    analysis = GarmentAnalysis(is_india_ethnic_wear=True, outfits=[])
    with caplog.at_level(logging.INFO, logger="vision.traditional_filter"):
        result = apply_to_analysis(analysis)
    assert result.is_india_ethnic_wear is False
    assert any(
        "gemini_filter_flipped" in r.message and "no_outfits" in r.message
        for r in caplog.records
    )


def test_apply_keeps_true_when_any_outfit_traditional():
    analysis = GarmentAnalysis(
        is_india_ethnic_wear=True,
        outfits=[_outfit(upper="kurta", lower="churidar")],
    )
    result = apply_to_analysis(analysis)
    assert result.is_india_ethnic_wear is True


def test_apply_flips_when_no_outfit_traditional(caplog):
    analysis = GarmentAnalysis(
        is_india_ethnic_wear=True,
        outfits=[_outfit(upper="crop_top", lower="shorts")],
    )
    with caplog.at_level(logging.INFO, logger="vision.traditional_filter"):
        result = apply_to_analysis(analysis)
    assert result.is_india_ethnic_wear is False
    assert any(
        "gemini_filter_flipped" in r.message and "no_traditional_silhouette" in r.message
        for r in caplog.records
    )


def test_apply_keeps_true_when_mixed_outfits_one_traditional():
    # 배경 인물이 Western 이어도 주인공이 kurta 면 TRUE 유지
    analysis = GarmentAnalysis(
        is_india_ethnic_wear=True,
        outfits=[
            _outfit(upper="kurta", lower="churidar"),
            _outfit(upper="t_shirt", lower="jeans"),
        ],
    )
    result = apply_to_analysis(analysis)
    assert result.is_india_ethnic_wear is True


def test_apply_single_piece_saree_stays_true():
    analysis = GarmentAnalysis(
        is_india_ethnic_wear=True,
        outfits=[_outfit(upper="saree", lower=None, single=True)],
    )
    result = apply_to_analysis(analysis)
    assert result.is_india_ethnic_wear is True


def test_apply_preserves_outfits_on_flip():
    """flip 은 binary 만 바꾸고 outfits 원본 유지 — 분석 정보 보존."""
    outfits = [_outfit(upper="crop_top", lower="shorts")]
    analysis = GarmentAnalysis(is_india_ethnic_wear=True, outfits=outfits)
    result = apply_to_analysis(analysis)
    assert result.is_india_ethnic_wear is False
    assert len(result.outfits) == 1
    assert result.outfits[0].upper_garment_type == "crop_top"


# sanity — pytest 가 warnings 로 쓰지 않게 명시 사용
_ = pytest  # noqa: F841
