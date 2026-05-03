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
from dataclasses import dataclass
from pathlib import Path

from contracts.common import ColorFamily


# `lab_to_family` 6-단계 cascade 의 임계값 — F-8 (mid-L × mid-chroma 갭 수정) 도입
# 후 회귀 위험. 모듈 상수 승격으로 변경 추적 + 단위 테스트가 임계 직접 참조 가능.
_ACHROMATIC_CHROMA = 12.0  # chroma < 12 → WHITE/NEUTRAL
_WHITE_LIGHTNESS = 85.0    # L > 85 + achromatic → WHITE_ON_WHITE
_PASTEL_LIGHTNESS = 70.0   # L > 70 AND chroma ≤ 55 → PASTEL
_PASTEL_CHROMA_MAX = 55.0  # PASTEL / EARTH 상한 chroma
_EARTH_LIGHTNESS_LO = 25.0
_EARTH_LIGHTNESS_HI = 65.0
_EARTH_B_MIN = 10.0        # warm-side b 최소
_JEWEL_LIGHTNESS_MAX = 45.0
_JEWEL_CHROMA_MIN = 20.0
_BRIGHT_CHROMA_MIN = 55.0


def lab_to_family(L: float, a: float, b: float) -> ColorFamily:
    """LAB 좌표 1개 → ColorFamily. dedup 용 coarse rule.

    chroma-first 분기 — 기존 6-line cascade 가 mid-L × mid-chroma 영역을 NEUTRAL 로 흡수
    (pool_13 indigo / pool_29 fuchsia 등 6 색) 하던 갭 수정 (F-8).

    thresholds (순서 의존, 모듈 상수 `_ACHROMATIC_CHROMA` 등 참조):
      1. chroma < 12 (achromatic) → L > 85 면 WHITE_ON_WHITE, else NEUTRAL
      2. L > 70 AND chroma <= 55 → PASTEL (소프트 톤)
      3. EARTH 우선: 25 < L < 65 AND a > 0 AND b > 10 AND chroma <= 55 → EARTH
         (rust/terracotta/mustard — warm-side mid-L. JEWEL/BRIGHT 보다 먼저 평가해
          unit test lab(45,28,30) chroma=41 케이스 보존)
      4. JEWEL: L <= 45 AND chroma >= 20 → JEWEL (deep saturated — sapphire, ruby 등)
      5. BRIGHT: chroma > 55 → BRIGHT (vivid saturated — saffron, rani pink 등)
      6. fallback JEWEL — mid-L × mid-chroma 영역 (cool 계열 + 중간 명도 + 중간 채도)
         은 NEUTRAL 보다 JEWEL 이 dedup 정합. teal/mid-blue/lavender 등.
    """
    chroma = (a * a + b * b) ** 0.5
    if chroma < _ACHROMATIC_CHROMA:
        return ColorFamily.WHITE_ON_WHITE if L > _WHITE_LIGHTNESS else ColorFamily.NEUTRAL
    if L > _PASTEL_LIGHTNESS and chroma <= _PASTEL_CHROMA_MAX:
        return ColorFamily.PASTEL
    if (
        _EARTH_LIGHTNESS_LO < L < _EARTH_LIGHTNESS_HI
        and a > 0 and b > _EARTH_B_MIN
        and chroma <= _PASTEL_CHROMA_MAX
    ):
        return ColorFamily.EARTH
    if L <= _JEWEL_LIGHTNESS_MAX and chroma >= _JEWEL_CHROMA_MIN:
        return ColorFamily.JEWEL
    if chroma > _BRIGHT_CHROMA_MIN:
        return ColorFamily.BRIGHT
    return ColorFamily.JEWEL


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


@dataclass(frozen=True)
class MatcherEntry:
    """Phase 5 adapter ΔE76 매칭용 preset entry — name + LAB + 사전계산 family."""
    name: str
    lab: tuple[float, float, float]
    family: ColorFamily


@dataclass(frozen=True)
class PresetViews:
    """color_preset.json 1회 로드에서 파생된 3 view — Phase 5 adapter 용.

    llm_preset: VisionLLMClient.extract_garment(preset=...) 입력 형식 ([{"name","hex"}]).
    matcher_entries: PaletteCluster.lab → preset.lab 최단 ΔE76 매칭 후보.
    family_map: Phase 4.5 dedup_post 입력 (preset name → ColorFamily).
    """
    llm_preset: list[dict[str, str]]
    matcher_entries: list[MatcherEntry]
    family_map: dict[str, ColorFamily]


def load_preset_views(preset_path: Path) -> PresetViews:
    """color_preset.json → PresetViews (3 view). 파일 1회 로드.

    entry 당 lab 필수 — 없으면 RuntimeError (load_preset_family_map 과 동일 gate).
    """
    raw = json.loads(preset_path.read_text(encoding="utf-8"))
    llm_preset: list[dict[str, str]] = []
    matcher_entries: list[MatcherEntry] = []
    family_map: dict[str, ColorFamily] = {}
    for entry in raw:
        name = entry["name"]
        lab = entry.get("lab")
        if lab is None or len(lab) != 3:
            raise RuntimeError(
                f"preset entry name={name!r} 에 lab 필드 없음 또는 shape 불일치. "
                "outputs/color_preset/color_preset.json 재생성 필요."
            )
        lab_tuple = (float(lab[0]), float(lab[1]), float(lab[2]))
        family = lab_to_family(*lab_tuple)
        llm_preset.append({"name": name, "hex": entry["hex"]})
        matcher_entries.append(MatcherEntry(name=name, lab=lab_tuple, family=family))
        family_map[name] = family
    return PresetViews(
        llm_preset=llm_preset,
        matcher_entries=matcher_entries,
        family_map=family_map,
    )
