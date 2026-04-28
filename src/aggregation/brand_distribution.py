"""로직 C (2026-04-29) — cluster brand_distribution.

cluster 안 (item, share) 페어들의 `brands` list 를 log-scale 균등 분배 + share-weighted
합산 → top N + threshold drop → 정규화.

규칙 (사용자 결정 2026-04-29):
- post 1건이 N 개 brand 를 동시에 언급하면 영향력은 1 / log2(N+1) 로 감쇠.
- 그 영향력은 N 개 brand 에 균등 분할 (각 1/N).
- 위에 cluster fan-out share 를 곱한 값이 한 (post, cluster, brand) 의 raw 기여.
- cluster 에서 모든 brand 기여를 합산 → 정규화 → share desc 정렬.
- threshold (default 0.05) 미만 drop, top_n (default 5) cut.
- drop/cut 후 살아남은 brand 들에 대해 한 번 더 정규화 (sum=1.0 유지).

브랜드 dedup: post 의 brands list 안에서 같은 name 은 한 번만 카운트 (account_handle 추출
+ caption mention 합치기 시 자연 발생). N 은 dedup 후 길이.

mass preservation 비대상: brand 는 categorical 시그널이라 cluster 단위 mass 합 보존
의무가 없음 (다른 distribution 들은 mass-preserving). 그래서 1/log2(N+1) × 1/N 형태
의 sub-1 weight 사용 가능.
"""
from __future__ import annotations

import math
from collections import defaultdict

from contracts.common import DistributionMap
from contracts.enriched import EnrichedContentItem

# 로직 C default. 변경하려면 호출부에서 명시 override.
_DEFAULT_TOP_N = 5
_DEFAULT_MIN_SHARE = 0.05


def _unique_brand_names(item: EnrichedContentItem) -> list[str]:
    """post 의 brands list → dedup 된 name list. 등장 순서 유지 (deterministic)."""
    seen: set[str] = set()
    out: list[str] = []
    for b in item.brands:
        name = b.name
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def _post_log_weight(n_brands: int) -> float:
    """N 개 brand 가 동시 언급된 post 1건의 영향력 = 1 / log2(N+1).
    N=1 → 1.0, N=2 → ≈0.631, N=3 → 0.5, N=5 → ≈0.387.
    """
    if n_brands <= 0:
        return 0.0
    return 1.0 / math.log2(n_brands + 1)


def compute_brand_distribution(
    items_with_share: list[tuple[EnrichedContentItem, float]],
    *,
    top_n: int = _DEFAULT_TOP_N,
    min_share: float = _DEFAULT_MIN_SHARE,
) -> DistributionMap:
    """cluster 안 (item, share) → 정규화된 brand_distribution {name: pct}.

    빈 결과 (모든 post 가 brands=[]) → 빈 dict.
    threshold/top_n 후 살아남은 entry 가 없으면 빈 dict.
    살아남은 entry 들은 한 번 더 정규화돼 합=1.0.

    반환 dict 의 insertion order 는 share desc 정렬. JSON 직렬화 시 순서 보존.
    """
    raw: defaultdict[str, float] = defaultdict(float)
    for item, share in items_with_share:
        if share <= 0.0:
            continue
        names = _unique_brand_names(item)
        if not names:
            continue
        n = len(names)
        per_brand = share * _post_log_weight(n) * (1.0 / n)
        if per_brand <= 0.0:
            continue
        for name in names:
            raw[name] += per_brand

    if not raw:
        return {}

    raw_total = sum(raw.values())
    if raw_total <= 0.0:
        return {}

    # 1차 정규화 (전체 mass 기준 share).
    normalized = [(name, weight / raw_total) for name, weight in raw.items()]
    # share desc + name asc tiebreak (deterministic).
    normalized.sort(key=lambda kv: (-kv[1], kv[0]))

    # threshold drop + top_n cut.
    survivors = [
        (name, share) for name, share in normalized
        if share >= min_share
    ][:top_n]
    if not survivors:
        return {}

    # 2차 정규화 — 살아남은 entry 합=1.0.
    survivor_sum = sum(share for _, share in survivors)
    if survivor_sum <= 0.0:
        return {}
    return {name: share / survivor_sum for name, share in survivors}
