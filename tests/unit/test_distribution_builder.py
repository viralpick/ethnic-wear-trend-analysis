"""distribution_builder pinning — pipeline_spec_v1.0 §2.1 / §2.2 / §2.7.

검증 대상:
- text 가중치: RULE=6.0 / LLM=3.0 / 그 외=0.0.
- group_to_item_contrib = log2(n+1) × log2(area×100+1).
- vision-only (silhouette) → text 무시.
- value=None 은 distribution 에서 제외.
- 빈 입력 → {} (정규화 발산 방지).
- distribution 합 = 1.0.
"""
from __future__ import annotations

import math

import pytest

from aggregation.distribution_builder import (
    GroupSnapshot,
    build_distribution,
    group_to_item_contrib,
    text_contribution_weight,
)
from contracts.common import ClassificationMethod


def test_text_contribution_weight_table() -> None:
    assert text_contribution_weight(ClassificationMethod.RULE) == 6.0
    assert text_contribution_weight(ClassificationMethod.LLM) == 3.0
    assert text_contribution_weight(ClassificationMethod.VLM) == 0.0
    assert text_contribution_weight(None) == 0.0


def test_group_to_item_contrib_formula() -> None:
    # n=3, area=0.5 → log2(4) × log2(51) = 2.0 × ~5.672 ≈ 11.344
    expected = math.log2(4) * math.log2(51)
    assert group_to_item_contrib(3, 0.5) == pytest.approx(expected)
    # zero objects → 0
    assert group_to_item_contrib(0, 0.5) == 0.0
    # zero area → log2(1)=0 (offset+1 → log2(0+1)=0)
    assert group_to_item_contrib(2, 0.0) == 0.0


def test_build_distribution_text_only() -> None:
    dist = build_distribution(
        text_value="kurta",
        text_method=ClassificationMethod.RULE,
        canonical_groups=[],
    )
    assert dist == {"kurta": 1.0}


def test_build_distribution_vision_only_silhouette() -> None:
    # vision_only=True 면 text 무시.
    groups = [
        GroupSnapshot(value="a_line", n_objects=2, mean_area_ratio=0.4),
        GroupSnapshot(value="straight", n_objects=1, mean_area_ratio=0.2),
    ]
    dist = build_distribution(
        text_value="kurta",  # ignored
        text_method=ClassificationMethod.RULE,  # ignored
        canonical_groups=groups,
        vision_only=True,
    )
    assert sum(dist.values()) == pytest.approx(1.0)
    assert "kurta" not in dist
    assert dist["a_line"] > dist["straight"]  # 더 많은 objects + 큰 area


def test_build_distribution_text_plus_vision_same_value() -> None:
    # 같은 value 는 합산: text(rule, +6) + group share.
    groups = [GroupSnapshot(value="kurta", n_objects=2, mean_area_ratio=0.5)]
    dist = build_distribution(
        text_value="kurta",
        text_method=ClassificationMethod.RULE,
        canonical_groups=groups,
    )
    assert dist == {"kurta": 1.0}


def test_build_distribution_text_plus_vision_different_values() -> None:
    # text=kurta(rule, 6.0) + group=saree share. 분배 비율 검증.
    groups = [GroupSnapshot(value="saree", n_objects=2, mean_area_ratio=0.5)]
    dist = build_distribution(
        text_value="kurta",
        text_method=ClassificationMethod.RULE,
        canonical_groups=groups,
    )
    assert set(dist.keys()) == {"kurta", "saree"}
    assert sum(dist.values()) == pytest.approx(1.0)
    # text=6.0, vision share = G = log2(3) ≈ 1.585 → kurta 가 더 큰 비중
    assert dist["kurta"] > dist["saree"]


def test_build_distribution_value_none_excluded() -> None:
    # group value=None 은 distribution 에서 빠짐.
    groups = [
        GroupSnapshot(value=None, n_objects=5, mean_area_ratio=0.9),
        GroupSnapshot(value="a_line", n_objects=1, mean_area_ratio=0.2),
    ]
    dist = build_distribution(
        text_value=None,
        text_method=None,
        canonical_groups=groups,
        vision_only=True,
    )
    assert dist == {"a_line": pytest.approx(1.0)}


def test_build_distribution_empty_returns_empty_dict() -> None:
    # text/vision 둘 다 비었으면 정규화 발산 대신 {}.
    dist = build_distribution(
        text_value=None,
        text_method=None,
        canonical_groups=[],
    )
    assert dist == {}
