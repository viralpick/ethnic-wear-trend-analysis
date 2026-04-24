"""50-color preset name → ColorFamily 매핑 (Phase 4.5 dedup 전용).

`outputs/color_preset/color_preset.json` 의 각 entry 는 single hex + lab 을 가지지만
family 라벨이 없다. pipeline_b_extractor.py:197 의 per-pixel LAB→family classifier 는
M4 예정이라 아직 없고, 본 모듈은 **preset 범위 한정** 으로 단순 rule-based 매핑만
제공한다 — dedup 용 coarse signal 목적.

scope 제한:
- 입력은 single hex / single LAB 좌표만. DUAL_TONE / MULTICOLOR 는 여러 pixel 에서만
  판정 가능하므로 이 모듈은 5가지 (WHITE_ON_WHITE / NEUTRAL / PASTEL / JEWEL / BRIGHT /
  EARTH) 중 하나만 반환.
- numpy 의존 없음 — core/vision-extras-free 환경에서도 import 가능.
- pipeline_b_extractor.py:197 의 per-pixel TODO 는 이 rule 과 분리. M4 에서 pixel-level
  classifier 가 도입되면 이 모듈과 공유 여부 재검토.

Rule 의 경험적 threshold 는 `outputs/color_preset/color_preset.json` 50 color 에 대해
육안 점검 후 tune. full LAB 기반 학습 classifier 대체 아님.
"""
from __future__ import annotations

import json
from pathlib import Path

from contracts.common import ColorFamily


def lab_to_family(L: float, a: float, b: float) -> ColorFamily:
    """LAB 좌표 1개 → ColorFamily. dedup 용 coarse rule.

    thresholds:
      - chroma = sqrt(a^2 + b^2)
      - L > 85 AND chroma < 12  → WHITE_ON_WHITE (ivory/white/off-white)
      - chroma < 12              → NEUTRAL (gray/black/무채색)
      - L > 70 AND chroma < 30   → PASTEL (baby pink, powder blue 등)
      - L < 45 AND chroma > 30   → JEWEL (sapphire, emerald, ruby 등)
      - chroma > 55              → BRIGHT (saffron, rani pink 등)
      - b > 10 AND a > 0 AND 25 < L < 65 → EARTH (rust, terracotta, mustard 등)
      - 나머지 → NEUTRAL
    """
    chroma = (a * a + b * b) ** 0.5
    if L > 85 and chroma < 12:
        return ColorFamily.WHITE_ON_WHITE
    if chroma < 12:
        return ColorFamily.NEUTRAL
    if L > 70 and chroma < 30:
        return ColorFamily.PASTEL
    if L < 45 and chroma > 30:
        return ColorFamily.JEWEL
    if chroma > 55:
        return ColorFamily.BRIGHT
    if b > 10 and a > 0 and 25 < L < 65:
        return ColorFamily.EARTH
    return ColorFamily.NEUTRAL


def load_preset_family_map(preset_path: Path) -> dict[str, ColorFamily]:
    """color_preset.json → {name: ColorFamily} dict.

    preset_path 포맷: `[{"name": "pool_00", "hex": "#...", "lab": [L, a, b], "origin": ...}, ...]`
    lab 필드 없는 entry 는 RuntimeError (preset 파일이 deterministic 해야 함).
    """
    raw = json.loads(preset_path.read_text(encoding="utf-8"))
    mapping: dict[str, ColorFamily] = {}
    for entry in raw:
        name = entry["name"]
        lab = entry.get("lab")
        if lab is None or len(lab) != 3:
            raise RuntimeError(
                f"preset entry name={name!r} 에 lab 필드 없음 또는 shape 불일치. "
                "outputs/color_preset/color_preset.json 재생성 필요."
            )
        mapping[name] = lab_to_family(float(lab[0]), float(lab[1]), float(lab[2]))
    return mapping
