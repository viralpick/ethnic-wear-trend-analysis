"""Item → Representative 매칭 + 기여도 합성 — pipeline_spec_v1.0 §2.4.

cross-product 후 (g, t, f) 조합별 매칭 share 계산, multiplier 곱셈, source 별 누적.

설계 원칙:
- L4 (multiplier): N=1 → 1.0x / N=2 → 2.5x / N=3 → 5.0x. spec §C.2 결정으로
  representative 적재는 N=3 (G/T/F 모두 결정) 만 — 즉 multiplier 항상 5.0.
  단, 함수는 future-proof 하게 N=1~3 모두 지원 (partial 적재 정책 변경 시).
- L5 (sparse filter): `total_item_contribution > 0` 인 representative 만 emit.
- L6 (factor_contribution): source 별 contribution 합 / 전체 합. instagram + youtube = 1.0.

이 모듈은 source-agnostic — 호출자가 ItemDistribution.source 만 채워주면 된다.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import product

from clustering.assign_trend_cluster import assign_shares
from contracts.common import ContentSource


_MULTIPLIER_BY_N: dict[int, float] = {
    1: 1.0,
    2: 2.5,
    3: 5.0,
}


@dataclass(frozen=True)
class ItemDistribution:
    """1 item (post/video) 의 G/T/F distribution + source.

    빈 dict 는 "결정 안됨" 의미. spec §C.2 (representative 적재 = G/T/F 모두 결정) 에
    따르면 한 attr 이라도 비면 N<3 이므로 함수 내부에서 자연 제외.

    `item_base_unit` 는 spec §2.4 마지막 줄의 향후 engagement 가중 자리 — 현재 1.0 default.
    """
    item_id: str
    source: ContentSource
    garment_type: dict[str, float] = field(default_factory=dict)
    technique: dict[str, float] = field(default_factory=dict)
    fabric: dict[str, float] = field(default_factory=dict)
    item_base_unit: float = 1.0


@dataclass(frozen=True)
class RepresentativeContribution:
    """1 item 이 1 representative 에 보태는 기여도 + multiplier."""
    representative_key: str  # "g__t__f" 포맷
    item_id: str
    source: ContentSource
    match_share: float       # cross-product 곱
    multiplier: float        # 1.0 / 2.5 / 5.0
    contribution: float      # match_share × multiplier × item_base_unit


@dataclass(frozen=True)
class RepresentativeAggregate:
    """1 representative 의 적재 단위 — sparse filter 통과 결과.

    factor_contribution[source] = Σ contribution(source) / Σ contribution(all). 합=1.0.
    """
    representative_key: str
    total_item_contribution: float
    factor_contribution: dict[ContentSource, float]
    member_count: int  # 기여한 item 수 (debug / sparse 진단용)


def representative_key(g: str, t: str, f: str) -> str:
    """spec §C.2 representative_key 포맷 — `garment__technique__fabric`."""
    return f"{g}__{t}__{f}"


def multiplier_for_n(n: int) -> float:
    """spec §2.4 매칭 multiplier. N=0 → 0.0 (representative 후보 아님)."""
    return _MULTIPLIER_BY_N.get(n, 0.0)


def _item_contributions(item: ItemDistribution) -> list[RepresentativeContribution]:
    """1 item → cross-product 후 representative 별 contribution 목록.

    G/T/F 한 distribution 이라도 비면 N<3 이므로 빈 list. (현 phase 정책 = N=3 만 emit.)
    """
    if not item.garment_type or not item.technique or not item.fabric:
        return []

    out: list[RepresentativeContribution] = []
    mult = multiplier_for_n(3)
    for (g, gp), (t, tp), (f, fp) in product(
        item.garment_type.items(),
        item.technique.items(),
        item.fabric.items(),
    ):
        share = gp * tp * fp
        if share <= 0.0:
            continue
        contrib = share * mult * item.item_base_unit
        out.append(RepresentativeContribution(
            representative_key=representative_key(g, t, f),
            item_id=item.item_id,
            source=item.source,
            match_share=share,
            multiplier=mult,
            contribution=contrib,
        ))
    return out


def build_contributions(
    items: list[ItemDistribution],
) -> list[RepresentativeContribution]:
    """모든 item 을 펼쳐 representative-level contribution flat list 로 반환.

    호출자가 sparse filter / factor_contribution 단계에 사용 (별도 함수에서).
    """
    out: list[RepresentativeContribution] = []
    for item in items:
        out.extend(_item_contributions(item))
    return out


def aggregate_representatives(
    contributions: list[RepresentativeContribution],
    *,
    sources: tuple[ContentSource, ...] = (
        ContentSource.INSTAGRAM,
        ContentSource.YOUTUBE,
    ),
) -> list[RepresentativeAggregate]:
    """spec §L5 sparse + §L6 factor_contribution.

    - `total_item_contribution > 0` 인 representative 만 반환.
    - factor_contribution 은 모든 등록 source 키를 0.0 으로 초기화 (FE 분기 단순화).
    - 관측된 source 가 `sources` 튜플 밖이어도 누락 안되도록 자동 포함 (합=1.0 invariant 방어).
    - 결과는 representative_key asc sort (deterministic).
    """
    observed: set[ContentSource] = {c.source for c in contributions}
    all_sources: tuple[ContentSource, ...] = tuple(sources) + tuple(
        s for s in observed if s not in sources
    )

    by_key: dict[str, list[RepresentativeContribution]] = defaultdict(list)
    for c in contributions:
        by_key[c.representative_key].append(c)

    out: list[RepresentativeAggregate] = []
    for key in sorted(by_key.keys()):
        rows = by_key[key]
        total = sum(r.contribution for r in rows)
        if total <= 0.0:
            continue
        per_source: dict[ContentSource, float] = {s: 0.0 for s in all_sources}
        for r in rows:
            per_source[r.source] = per_source.get(r.source, 0.0) + r.contribution
        factor = {s: per_source[s] / total for s in all_sources}
        # 합=1.0 invariant defensive check — 부동소수 epsilon 만 허용.
        assert abs(sum(factor.values()) - 1.0) < 1e-9, (
            f"factor_contribution sum != 1.0 for {key}: {factor}"
        )
        out.append(RepresentativeAggregate(
            representative_key=key,
            total_item_contribution=total,
            factor_contribution=factor,
            member_count=len({r.item_id for r in rows}),
        ))
    return out


def item_cluster_shares(item: ItemDistribution) -> dict[str, float]:
    """Phase α (2026-04-28): item → cluster_key 별 raw share dict (multiplier 없음).

    pipeline_spec §2.4 line 252-255 의 contribution-weighted score 입력용 — 1 item
    의 G/T/F 분포가 cross-product 으로 여러 cluster_key 에 share 로 fan-out.
    representative_key 와 cluster_key 는 동일 포맷 (`g__t__f`) 이므로
    `assign_shares` 위임 — cross-product 로직 single source.

    contribution = share × multiplier × item_base_unit 인데, score 합산은 share
    만 보면 됨 (multiplier 는 representative 적재 단위, score 는 item 의 representative
    참여 확률 그대로 가중). 따라서 raw share 만 반환.

    G/T/F 한 distribution 이라도 비면 빈 dict (N<3 정책 — `_item_contributions` /
    `assign_shares` 와 동일).

    Phase β 에서 build_cluster_summary 가 이 함수로 fan-out 호출 예정.
    """
    return assign_shares(item.garment_type, item.technique, item.fabric)


def top_evidence_per_source(
    contributions: list[RepresentativeContribution],
    *,
    k: int = 4,
) -> dict[ContentSource, list[RepresentativeContribution]]:
    """spec §4 (12 화면 항목) #12 — IG/YT evidence top k.

    한 representative 안에서 contribution desc top k. tie-break = item_id asc (deterministic).
    호출자가 representative_key 별로 contributions 를 사전에 그룹핑해서 전달.
    """
    by_source: dict[ContentSource, list[RepresentativeContribution]] = defaultdict(list)
    for c in contributions:
        by_source[c.source].append(c)
    out: dict[ContentSource, list[RepresentativeContribution]] = {}
    for source, rows in by_source.items():
        rows_sorted = sorted(rows, key=lambda r: (-r.contribution, r.item_id))
        out[source] = rows_sorted[:k]
    return out
