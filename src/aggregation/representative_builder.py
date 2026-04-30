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

# G__F 2축 (2026-04-30 sync): N=2 (G+F) → 2.5x, N=1 (G or F) → 1.0x.
# representative 적재 권장은 N=2 (둘 다 resolved). N=1 partial 도 emit 시 multiplier 작아 자연 weight ↓.
_MULTIPLIER_BY_N: dict[int, float] = {
    1: 1.0,
    2: 2.5,
}


@dataclass(frozen=True)
class ItemDistribution:
    """1 item (post/video) 의 G/F distribution + source. technique 은 cluster
    drilldown 의 distribution 으로 별도 노출 — cluster_key 에서는 빠짐 (2026-04-30 sync).

    `item_base_unit` 는 향후 engagement 가중 자리 — 현재 1.0 default.
    """
    item_id: str
    source: ContentSource
    garment_type: dict[str, float] = field(default_factory=dict)
    fabric: dict[str, float] = field(default_factory=dict)
    technique: dict[str, float] = field(default_factory=dict)  # drilldown 표시용
    item_base_unit: float = 1.0


@dataclass(frozen=True)
class RepresentativeContribution:
    """1 item 이 1 representative 에 보태는 기여도 + multiplier (G__F 2축)."""
    representative_key: str  # "g__f" 포맷
    item_id: str
    source: ContentSource
    match_share: float       # cross-product 곱 (G × F)
    multiplier: float        # 1.0 / 2.5
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


def representative_key(g: str, f: str) -> str:
    """representative_key 포맷 (2026-04-30) — `garment__fabric` (2축)."""
    return f"{g}__{f}"


def multiplier_for_n(n: int) -> float:
    """매칭 multiplier (G__F 2축). N=0 → 0.0 (후보 아님)."""
    return _MULTIPLIER_BY_N.get(n, 0.0)


def _resolved_axis_count(item: ItemDistribution) -> int:
    """G/F 중 비어있지 않은 축 수 (0~2). technique 은 cluster_key 에서 빠짐."""
    return sum(1 for d in (item.garment_type, item.fabric) if d)


def effective_item_count(items: list[ItemDistribution]) -> float:
    """multiplier-scaled batch denominator. N=2 기준 정규화 — N=0:0 / N=1:0.4 / N=2:1.0.
    분자 multiplier scale 과 단위 정합.
    """
    full = multiplier_for_n(2)
    return sum(multiplier_for_n(_resolved_axis_count(item)) / full for item in items)


# `clustering.assign_trend_cluster._UNKNOWN` 와 같은 placeholder ("unknown" 하드코드 — partial
# key 포맷 단일 source). 후속 cleanup 으로 두 모듈 공유 상수로 분리 예정.
_UNKNOWN_AXIS = "unknown"


def _item_contributions(item: ItemDistribution) -> list[RepresentativeContribution]:
    """1 item → G__F cross-product representative contribution 목록 (2026-04-30 sync).

    - N=0 → 빈 list (후보 아님)
    - N≥1 → cross-product (G × F). 비어있는 axis 는 `_UNKNOWN_AXIS` placeholder.
      multiplier = multiplier_for_n(N) (1.0 / 2.5).
    - contribution = share × multiplier × item_base_unit.
    """
    n_axes = _resolved_axis_count(item)
    if n_axes == 0:
        return []

    g_eff = item.garment_type or {_UNKNOWN_AXIS: 1.0}
    f_eff = item.fabric or {_UNKNOWN_AXIS: 1.0}
    mult = multiplier_for_n(n_axes)

    out: list[RepresentativeContribution] = []
    for (g, gp), (f, fp) in product(g_eff.items(), f_eff.items()):
        share = gp * fp
        if share <= 0.0:
            continue
        contrib = share * mult * item.item_base_unit
        out.append(RepresentativeContribution(
            representative_key=representative_key(g, f),
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
    """item → cluster_key 별 raw share dict. G__F 2축 (2026-04-30 sync).

    1 item 의 G/F 분포가 cross-product 으로 여러 cluster_key 에 share 로 fan-out.
    representative_key 와 cluster_key 는 동일 포맷 (`g__f`).

    technique 은 cluster fan-out 키에서 빠짐 — cluster drilldown 의 technique_distribution
    으로만 노출. share 합 = N=2 → 1.0, N=1 → 0.5, N=0 → 0.
    """
    return assign_shares(item.garment_type, item.fabric)


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
