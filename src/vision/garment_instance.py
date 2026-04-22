"""GarmentInstance — frame × person × garment_class 단위 색상 instance (phase 3).

기존 post-level aggregate (모든 pixel concat → 1회 KMeans) 는 "서로 다른 옷이 섞이는"
구조 문제가 있었음 (agenda §5 #5). Phase 2 에서 frame-level 분리까지 했고, 이제 phase 3
에서 **옷 객체 단위** 로 완전 분리.

원칙 (사용자 지시 2026-04-22):
- post / video 1개 당 서로 다른 "옷" 에 대해서만 palette 추출
- 1 instance = (frame × person bbox × garment_class) 교집합
- 단색 옷 → hex 1개 / 다색 옷 → hex set
- 같은 옷 중복 등장 시 sub-linear 가중치 (log 기반)
- hex set 은 최종적으로 (garment_type × fabric) 매핑됨

post aggregate:
- post 내 모든 instance 수집 → find_duplicate_groups 로 같은 옷 묶음
- 그룹당 weight = 1 + log(count) (단일이면 1.0, 2개면 1.69, 3개면 2.10 …)
- 대표 instance 의 chip × weight × chip.pct 정렬 → top-k

판정 기준 (모두 config 로 tunable):
- single_color: 같은 instance 내 top-2 chip ΔE76 < single_color_max_delta_e
- duplicate: 두 instance 가 같은 garment_class + top-1 chip ΔE76 < duplicate_max_delta_e
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from contracts.common import ColorPaletteItem
from vision.color_space import delta_e76, rgb_to_lab

WeightFormula = Literal["log", "linear", "sqrt"]


@dataclass(frozen=True)
class GarmentInstance:
    """1 post 내 한 person × 한 garment_class 의 독립 palette 단위."""
    instance_id: str                    # f"{frame_id}:p{bbox_idx}:{class}"
    frame_id: str
    bbox: tuple[int, int, int, int]
    garment_class: str                  # upper-clothes / pants / dress / skirt / hat / shoes
    palette: list[ColorPaletteItem]     # 단색 instance → 1, 다색 → 2~k
    is_single_color: bool
    pixel_count: int
    skin_drop_ratio: float              # drop_skin_adaptive 의 drop_ratio (판정 근거)
    duplicate_group_id: int | None = field(default=None)


def classify_single_color(
    palette: list[ColorPaletteItem], max_delta_e: float = 8.0,
) -> bool:
    """palette chip 들이 모두 top-1 에서 ΔE < max_delta_e 이면 단색 판정.

    예: chip 1=#B8D4C3, chip 2=#BCD6C6 → ΔE 작음 → 단색.
    chip 1=#B8D4C3, chip 2=#2B211C → ΔE 크게 → 다색.
    """
    if len(palette) < 2:
        return True
    lab_top = _palette_lab(palette[0])
    for chip in palette[1:]:
        if delta_e76(lab_top, _palette_lab(chip)) >= max_delta_e:
            return False
    return True


def _palette_lab(chip: ColorPaletteItem) -> np.ndarray:
    return rgb_to_lab(np.array([chip.r, chip.g, chip.b], dtype=np.float32))


def find_duplicate_groups(
    instances: list[GarmentInstance],
    max_delta_e: float = 15.0,
) -> list[list[GarmentInstance]]:
    """같은 garment_class + top-1 chip ΔE < threshold 인 instance 들을 그룹화.

    간단한 greedy clustering: 첫 instance 를 기준으로 이후 instance 들 중 매칭되는 것 흡수.
    중복 없음 (각 instance 는 한 group 에만). 그룹 순서는 instance 입력 순서 따라감.
    """
    groups: list[list[GarmentInstance]] = []
    assigned = [False] * len(instances)
    for i, inst_a in enumerate(instances):
        if assigned[i]:
            continue
        if not inst_a.palette:
            assigned[i] = True
            groups.append([inst_a])
            continue
        group = [inst_a]
        assigned[i] = True
        lab_a = _palette_lab(inst_a.palette[0])
        for j in range(i + 1, len(instances)):
            if assigned[j]:
                continue
            inst_b = instances[j]
            if inst_b.garment_class != inst_a.garment_class or not inst_b.palette:
                continue
            lab_b = _palette_lab(inst_b.palette[0])
            if delta_e76(lab_a, lab_b) < max_delta_e:
                group.append(inst_b)
                assigned[j] = True
        groups.append(group)
    return groups


def _weight_for(count: int, formula: WeightFormula) -> float:
    """sub-linear 가중치 — count 에 따른 영향력. 기본 log(1 + count) 형태."""
    if count <= 1:
        return 1.0
    if formula == "log":
        return 1.0 + math.log(count)
    if formula == "sqrt":
        return math.sqrt(count)
    if formula == "linear":
        return float(count)
    raise ValueError(f"unknown weight_formula: {formula}")


def aggregate_post_palette(
    instances: list[GarmentInstance],
    top_k: int = 5,
    duplicate_max_delta_e: float = 15.0,
    weight_formula: WeightFormula = "log",
) -> tuple[list[ColorPaletteItem], list[list[GarmentInstance]]]:
    """instance 들을 duplicate 묶음 + 가중치 적용 → post-level top-k palette.

    흐름:
    1. find_duplicate_groups 로 같은 옷 묶음
    2. 그룹당 weight (sub-linear in count)
    3. 대표 instance (첫 멤버) 의 palette chip 들을 weight × chip.pct 로 스코어링
    4. 전 그룹의 chip 을 flat list 로 모아 score 내림차순 top-k
    5. pct 재정규화

    그룹 정보도 함께 반환 (HTML 표시 / 디버깅용).
    """
    if not instances:
        return [], []
    groups = find_duplicate_groups(instances, max_delta_e=duplicate_max_delta_e)
    scored: list[tuple[ColorPaletteItem, float]] = []
    for group in groups:
        rep = group[0]
        if not rep.palette:
            continue
        weight = _weight_for(len(group), weight_formula)
        for chip in rep.palette:
            scored.append((chip, chip.pct * weight))
    if not scored:
        return [], groups
    scored.sort(key=lambda x: -x[1])
    top = scored[: top_k]
    total = sum(s for _, s in top)
    post_palette = [
        ColorPaletteItem(
            r=c.r, g=c.g, b=c.b, hex_display=c.hex_display,
            name=c.name, family=c.family,
            pct=s / total if total > 0 else 0.0,
        )
        for c, s in top
    ]
    return post_palette, groups
