"""Silhouette 분포 추출 (spec §4.1 ⑤, §7).

Case 1: 미분류 포스트의 silhouette 를 VLM 으로 추출.
Case 2: 클러스터 palette 보강과 함께 silhouette 분포 업데이트.

이 모듈은 extract_color_features.VLMClient 를 재사용한다 (별도 client 만들지 않는다).
"""
from __future__ import annotations

from collections import Counter

from contracts.common import Silhouette
from contracts.normalized import NormalizedContentItem
from vision.extract_color_features import (
    VLMClient,
    VLMVisualResult,
    extract_color_batch,
)


def extract_silhouettes(
    items: list[NormalizedContentItem],
    client: VLMClient,
    cap: int | None = None,
) -> dict[str, Silhouette | None]:
    """source_post_id → 추출된 Silhouette (None 포함) 매핑."""
    results: list[VLMVisualResult] = extract_color_batch(items, client, cap=cap)
    return {r.source_post_id: r.silhouette for r in results}


def silhouette_distribution(
    assignments: dict[str, Silhouette | None],
) -> dict[str, float]:
    """None 을 제외한 정규화 분포. 모두 None 이면 빈 dict."""
    valid = [s for s in assignments.values() if s is not None]
    if not valid:
        return {}
    counter: Counter[str] = Counter(s.value for s in valid)
    total = sum(counter.values())
    return {key: count / total for key, count in counter.items()}
