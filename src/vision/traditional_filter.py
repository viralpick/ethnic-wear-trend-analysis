"""Traditional silhouette whitelist 기반 binary override — Phase 2 post-filter.

Phase 0 v0.2 재파일럿에서 확인: 프롬프트만으로 fusion wear (block print crop top
등) 오분류는 대부분 방어되지만, 드물게 LLM 이 `is_india_ethnic_wear=True` 를 내면서
garment_type 이 whitelist 에 없는 edge case 가 남을 수 있어 안전망 추가.

원칙:
- 안전망 (safety net). 주 정화는 프롬프트가 담당, whitelist 는 LLM 이 정의를 흘렸을 때만 동작.
- flip (True → False) 만 일어난다. False → True 로 강제 승격은 하지 않는다 (LLM 판단 존중).
- 적용 시 logger.info("gemini_filter_flipped reason=..." 로만 기록 — Pydantic schema 는
  분석 결과만 담고 처리 이력은 로그로.

`is_outfit_traditional(upper, lower, dress_as_single)` 은 원시 인자 기반이라 pilot dict /
GarmentAnalysis 양쪽에서 재사용 가능 (pilot 은 dict key 에서, production 은 model field
에서 읽어 호출).
"""
from __future__ import annotations

import logging

from contracts.vision import EthnicOutfit, GarmentAnalysis

logger = logging.getLogger(__name__)

# 프롬프트 정의: "indo-fusion only if upper OR lower has traditional silhouette".
# LLM 이 is_india_ethnic_wear=True 로 내도, garment_type 이 아래 whitelist 에
# 어느 쪽도 없으면 (예: upper=top/t-shirt, lower=pants/jeans/leggings) override False.

TRAD_UPPER: frozenset[str] = frozenset({
    "kurta", "kurti", "anarkali", "saree", "sari", "blouse", "choli",
    "sherwani", "tunic", "ethnic_shirt", "kurta_dress", "angrakha",
    "lehenga",  # single piece 시 upper 에 들어올 수 있음
})
TRAD_LOWER: frozenset[str] = frozenset({
    "palazzo", "churidar", "salwar", "sharara", "dhoti",
    "pajama", "pyjama", "lehenga", "ghagra", "dupatta",
})


def is_outfit_traditional(
    upper: str | None,
    lower: str | None,
    dress_as_single: bool,
) -> bool:
    """outfit 한 개가 traditional silhouette 을 갖는지 판정.

    원시 인자 기반 — pilot dict (scripts/pilot_llm_bbox.py) 와 production
    EthnicOutfit (src/vision/gemini_client.py) 양쪽에서 호출한다.

    - single piece (saree/lehenga-as-single/ethnic_dress): upper 만 본다
    - two-piece: upper OR lower 중 하나라도 whitelist 면 traditional
    """
    upper_norm = (upper or "").lower()
    if dress_as_single:
        return upper_norm in TRAD_UPPER
    lower_norm = (lower or "").lower()
    return upper_norm in TRAD_UPPER or lower_norm in TRAD_LOWER


def apply_to_analysis(analysis: GarmentAnalysis) -> GarmentAnalysis:
    """GarmentAnalysis 에 whitelist post-filter 적용 후 반환.

    원본 `is_india_ethnic_wear=False` 는 그대로 통과.
    True 이지만 outfits 가 비었거나 traditional outfit 이 없으면 False 로 flip.
    flip 발생 시 logger.info 로 reason 기록 (schema 에는 추가 필드 없음).
    """
    if not analysis.is_india_ethnic_wear:
        return analysis
    if not analysis.outfits:
        logger.info(
            "gemini_filter_flipped reason=binary_true_but_no_outfits"
        )
        return analysis.model_copy(update={"is_india_ethnic_wear": False})
    if any(_outfit_is_traditional_model(o) for o in analysis.outfits):
        return analysis
    logger.info(
        "gemini_filter_flipped reason=no_traditional_silhouette outfits=%s",
        [
            {
                "upper": o.upper_garment_type,
                "lower": o.lower_garment_type,
                "single": o.dress_as_single,
            }
            for o in analysis.outfits
        ],
    )
    return analysis.model_copy(update={"is_india_ethnic_wear": False})


def _outfit_is_traditional_model(outfit: EthnicOutfit) -> bool:
    return is_outfit_traditional(
        outfit.upper_garment_type,
        outfit.lower_garment_type,
        outfit.dress_as_single,
    )
