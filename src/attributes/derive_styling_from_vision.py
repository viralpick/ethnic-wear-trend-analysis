"""EthnicOutfit → StylingCombo 파생 — M3.I P0 + P1 매핑.

Gemini vision LLM 이 채우는 `EthnicOutfit` 의 `dress_as_single` / `lower_garment_type` /
`is_co_ord_set` / `outer_layer` 를 StylingCombo enum 으로 파생. 추가 LLM call 0 —
기존 추출 결과 재해석만.

우선순위 (위에서부터 first match):
1. `dress_as_single=True` → STANDALONE (saree drape / lehenga-as-single / ethnic_dress).
2. `is_co_ord_set=True` → CO_ORD_SET (matched upper+lower set 이 specific bottom 보다
   더 유의미한 styling 시그널이라 우선).
3. `lower_garment_type` 매핑 (mutually exclusive bottoms):
   - palazzo → WITH_PALAZZO
   - churidar / salwar → WITH_CHURIDAR
   - pants / trousers / pyjama → WITH_PANTS
   - jeans / denim → WITH_JEANS
4. `outer_layer` 매핑 (bottom 이 매칭 안되거나 unknown 일 때만):
   - dupatta / shawl / stole → WITH_DUPATTA
   - jacket / cardigan / nehru / shrug → WITH_JACKET

순서 의도: bottom 이 specific 하면 그게 dominant styling info. dupatta 는 60-80% 의
ethnic outfit 에 default 라 outer_layer 만 보고 WITH_DUPATTA 를 주면 카테고리가 무의미해짐.
specific bottom 매칭이 우선이고, outer_layer 는 bottom 미상일 때의 fallback.

No-match (모든 시그널 None / 매핑 표 외) → None. caller 가 NULL 적재 — fuzzy 추정 금지.
"""
from __future__ import annotations

from contracts.common import StylingCombo
from contracts.vision import EthnicOutfit


_LOWER_TO_STYLING: dict[str, StylingCombo] = {
    "palazzo": StylingCombo.WITH_PALAZZO,
    "churidar": StylingCombo.WITH_CHURIDAR,
    "salwar": StylingCombo.WITH_CHURIDAR,
    "pants": StylingCombo.WITH_PANTS,
    "trousers": StylingCombo.WITH_PANTS,
    "pyjama": StylingCombo.WITH_PANTS,
    "jeans": StylingCombo.WITH_JEANS,
    "denim": StylingCombo.WITH_JEANS,
}

_OUTER_TO_STYLING: dict[str, StylingCombo] = {
    "dupatta": StylingCombo.WITH_DUPATTA,
    "shawl": StylingCombo.WITH_DUPATTA,
    "stole": StylingCombo.WITH_DUPATTA,
    "jacket": StylingCombo.WITH_JACKET,
    "cardigan": StylingCombo.WITH_JACKET,
    "nehru": StylingCombo.WITH_JACKET,
    "shrug": StylingCombo.WITH_JACKET,
}


def derive_styling_from_outfit(outfit: EthnicOutfit) -> StylingCombo | None:
    """단일 outfit → StylingCombo 매핑. 매칭 안되면 None."""
    if outfit.dress_as_single:
        return StylingCombo.STANDALONE

    if outfit.is_co_ord_set:
        return StylingCombo.CO_ORD_SET

    lower = outfit.lower_garment_type
    if lower is not None:
        match = _LOWER_TO_STYLING.get(lower.lower())
        if match is not None:
            return match

    outer = outfit.outer_layer
    if outer is not None:
        return _OUTER_TO_STYLING.get(outer.lower())

    return None
