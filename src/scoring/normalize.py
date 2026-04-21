"""Same-run min-max 정규화 (spec §9 mandate).

Formula:
    normalized = (x - min) / (max - min)     if max > min else 0.0

모든 sub-score 에 동일하게 적용한다. override 는 configs/local.yaml `scoring.normalization_method`.
다른 방식 (z-score / log / percentile / softmax) 은 해당 override 가 명시되기 전에는 절대 도입 금지.
"""
from __future__ import annotations


def minmax_same_run(values: list[float]) -> list[float]:
    """빈 리스트 → 빈 리스트. 모든 값이 같으면 전부 0.0. 그 외는 (x-min)/(max-min)."""
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi <= lo:
        return [0.0 for _ in values]
    span = hi - lo
    return [(v - lo) / span for v in values]


def apply_normalization(
    values: list[float], method: str = "minmax_same_run"
) -> list[float]:
    """configs/local.yaml 의 normalization_method 문자열로 dispatch.

    `minmax_same_run` 외의 값은 아직 구현되지 않았다 (의도적). 추가하려면 명시적 커밋.
    """
    if method == "minmax_same_run":
        return minmax_same_run(values)
    raise ValueError(f"Unsupported normalization_method={method!r}. See spec §9 mandate.")
