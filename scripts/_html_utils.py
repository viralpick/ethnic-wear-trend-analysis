"""scripts/* HTML 빌더 공용 헬퍼 — escape + luminance + chip.

이전에는 `build_review_html._esc` / `pilot_llm_bbox._escape` / `render_color_preset_html._luma`
/ `build_review_html._hex_text_color` 가 별도 정의 → drift (`_luma` threshold 0.55,
`_hex_text_color` threshold 130/255≈0.51 달랐음). single source 통합 (drift 차단,
`feedback_visual_with_color_swatch` 일관 적용).
"""
from __future__ import annotations

import html as _html_mod
from typing import Any


def escape(s: Any) -> str:
    """`html.escape` 얇은 wrapper — None safe + str 변환.

    HTML 출력 시 user-content (caption / handle / brand 등) 모두 이 함수 거치도록.
    """
    if s is None:
        return ""
    return _html_mod.escape(str(s))


def text_color_for_bg(
    hex_v: str | None,
    *,
    dark: str = "#000",
    light: str = "#fff",
) -> str:
    """배경 hex 의 ITU-R BT.601 luminance 기준 contrast 색상.

    luminance >= 130 (out of 255, ≈ 0.51) → 어두운 텍스트 (`dark`),
    아니면 밝은 텍스트 (`light`). hex 형식 유효성 검증 — 잘못된 입력은 dark fallback.

    threshold 130 은 build_review_html 의 기존 `_hex_text_color` 와 일치 (review HTML
    canonical). render_color_preset_html 도 이 임계로 통일.
    """
    if not hex_v or not hex_v.startswith("#") or len(hex_v) < 7:
        return dark
    try:
        r = int(hex_v[1:3], 16)
        g = int(hex_v[3:5], 16)
        b = int(hex_v[5:7], 16)
    except ValueError:
        return dark
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return light if lum < 130 else dark


__all__ = ["escape", "text_color_for_bg"]
