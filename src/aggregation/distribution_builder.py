"""Item-level attribute distribution — pipeline_spec_v1.0 §2.1 / §2.2 / §2.7.

post (= 1 item) 안에서 한 attribute 의 distribution {value: pct} 를 계산한다.
text(=rule/gpt) 와 vision(=canonical groups) 두 source 를 가중 합산.

설계 원칙:
- text 가중치: rule=6.0 / gpt=3.0 / 그 외=0.0 (B1 user 결정).
- vision 가중치: G = log2(Σ n_objects + 1), group 비례 분배는 spec §2.7 의
  group_to_item_contrib (= log2(n_objects+1) × log2(area×100+1)) 비율 사용.
  spec §2.1 본문의 "n_objects 비례 분배" 는 §2.7 와 모순 — §2.7 (Q6 user 결정 = 곱셈) 우선.
- vision-only 속성 (silhouette): text 부분 없이 group 분배만 (로직 B).
- value=None 인 group/text 는 distribution 에 포함 안됨 (단, 가중치 합이 0 이면 빈 dict).

이 모듈은 source-agnostic (instagram/youtube 구분 없음). 호출자가 representative
매칭 단계에서 source 별로 다시 분해 (factor_contribution 계산).
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from contracts.common import ClassificationMethod


@dataclass(frozen=True)
class GroupSnapshot:
    """canonical group 1개의 distribution 입력 — representative 의 attr 단일값 + 멤버 통계.

    `value` 는 group 단일값 (Phase 4.5 dedup 시 다수결 + tie-break 으로 이미 결정됨).
    None 이면 distribution 에서 제외.
    """
    value: str | None
    n_objects: int            # 그룹 내 멤버 수 (= len(CanonicalOutfit.members))
    mean_area_ratio: float    # 멤버들의 person_bbox_area_ratio 평균


_TEXT_WEIGHT_BY_METHOD: dict[ClassificationMethod, float] = {
    ClassificationMethod.RULE: 6.0,
    ClassificationMethod.LLM: 3.0,  # spec §2.1 의 "gpt" = ClassificationMethod.LLM
}


def text_contribution_weight(method: ClassificationMethod | None) -> float:
    """spec §2.1 — text 가중치 매핑. None / 미등록 method → 0.0.

    VLM 은 vision 경로 (canonical_groups) 로 들어오므로 여기선 0.0 가중치 (= text 채널
    이 아닌데 우연히 method 가 VLM 으로 찍힌 경우는 무가중치 처리).
    """
    if method is None:
        return 0.0
    return _TEXT_WEIGHT_BY_METHOD.get(method, 0.0)


def group_to_item_contrib(n_objects: int, mean_area_ratio: float) -> float:
    """spec §2.7 — log2(n_objects+1) × log2(area×100+1). 두 축 모두 0 → 0."""
    if n_objects <= 0:
        return 0.0
    return math.log2(n_objects + 1) * math.log2(max(0.0, mean_area_ratio) * 100 + 1)


def _g_total(groups: list[GroupSnapshot]) -> float:
    """spec §2.7 — G = log2(Σ n_objects + 1). 모든 group 의 객체 수 합 기준."""
    total = sum(g.n_objects for g in groups)
    if total <= 0:
        return 0.0
    return math.log2(total + 1)


def _group_shares_of_g(groups: list[GroupSnapshot]) -> list[float]:
    """G 를 group_to_item_contrib 비례로 분배 (spec §2.7 명시). value=None 도 포함된 채 반환.

    분모 = Σ contrib. 0 이면 모두 0 반환 (G 가 양수여도 분배 불가).
    """
    g = _g_total(groups)
    if g <= 0:
        return [0.0] * len(groups)
    contribs = [group_to_item_contrib(g.n_objects, g.mean_area_ratio) for g in groups]
    denom = sum(contribs)
    if denom <= 0:
        return [0.0] * len(groups)
    return [g * c / denom for c in contribs]


def build_distribution(
    text_value: str | None,
    text_method: ClassificationMethod | None,
    canonical_groups: list[GroupSnapshot],
    *,
    vision_only: bool = False,
) -> dict[str, float]:
    """post 단위 한 attribute 의 distribution {value: pct}, 합=1.0.

    spec §2.1 (text+vision 합산) 또는 §2.2 (vision_only=True, silhouette).

    합 0 (text/vision 둘 다 비었거나 None) → 빈 dict.
    """
    totals: dict[str, float] = defaultdict(float)

    if not vision_only and text_value is not None:
        weight = text_contribution_weight(text_method)
        if weight > 0.0:
            totals[text_value] += weight

    shares = _group_shares_of_g(canonical_groups)
    for snap, share in zip(canonical_groups, shares, strict=True):
        if snap.value is None or share <= 0.0:
            continue
        totals[snap.value] += share

    total_sum = sum(totals.values())
    if total_sum <= 0.0:
        return {}
    return {k: v / total_sum for k, v in totals.items()}
