"""minmax_same_run normalize 단위 테스트 (spec §9 mandate)."""
from __future__ import annotations

import pytest

from scoring.normalize import apply_normalization, minmax_same_run


def test_empty_returns_empty() -> None:
    assert minmax_same_run([]) == []


def test_single_element_maps_to_zero() -> None:
    # 하나짜리면 max == min 이라 전부 0.0.
    assert minmax_same_run([3.0]) == [0.0]


def test_all_equal_values_collapse_to_zero() -> None:
    # 전부 같은 값이면 분모 0 → 정의상 0.0.
    assert minmax_same_run([5.0, 5.0, 5.0]) == [0.0, 0.0, 0.0]


def test_two_clusters_with_known_spread() -> None:
    # 2 and 10 → (2-2)/(10-2)=0, (10-2)/(10-2)=1
    result = minmax_same_run([2.0, 10.0])
    assert result == [0.0, 1.0]


def test_three_values_spread_maps_midpoint() -> None:
    result = minmax_same_run([1.0, 5.0, 9.0])
    assert result == [0.0, 0.5, 1.0]


def test_negative_values_normalize_same_way() -> None:
    # min=-10, max=10 → 0.5, 0.0, 1.0
    result = minmax_same_run([0.0, -10.0, 10.0])
    assert result == [0.5, 0.0, 1.0]


def test_apply_normalization_dispatches_minmax() -> None:
    assert apply_normalization([1.0, 2.0, 3.0], "minmax_same_run") == [0.0, 0.5, 1.0]


def test_apply_normalization_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="Unsupported normalization_method"):
        apply_normalization([1.0, 2.0], "zscore")
