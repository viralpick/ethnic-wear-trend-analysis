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
    """1 item (post/video) 의 G/F distribution + source.

    `item_base_unit` 는 growth_rate 가중치 — 현재 1.0 default.

    Phase v2.1 (2026-04-30 (A)): `cluster_shares` 는 canonical 단위 fan-out 결과 —
    각 canonical 의 (g_i, f_i) 가 자기 cluster_key 에 group_to_item_contrib 비례 mass.
    옛 cross-product (G × F distribution mix) 가 multi-canonical post 에서 가짜 cluster
    매칭 (예: canonical 0=kurta+cotton, canonical 1=saree+silk → cross-product `kurta__silk`
    가짜 매칭) 을 만들던 갭 해소. distribution 들 (garment_type/fabric/technique) 은 화면
    표시 용도 그대로 유지 (item_dist drilldown / cluster summary aggregation).
    """
    item_id: str
    source: ContentSource
    garment_type: dict[str, float] = field(default_factory=dict)
    fabric: dict[str, float] = field(default_factory=dict)
    technique: dict[str, float] = field(default_factory=dict)  # drilldown 표시용
    cluster_shares: dict[str, float] = field(default_factory=dict)
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
    """G/F 중 비어있지 않은 축 수 (0~2). technique 은 cluster_key 에서 빠짐.

    Phase v2.1 호환: cluster_shares 가 비어있으면 0, 모든 cluster_key 가 unknown axis 만
    포함하면 0, 그 외엔 cluster_key 의 unknown 갯수 분포로 추정 (max resolved axis count).
    """
    if item.cluster_shares:
        max_resolved = 0
        for ck in item.cluster_shares:
            n_resolved = sum(1 for ax in ck.split("__") if ax != _UNKNOWN_AXIS)
            max_resolved = max(max_resolved, n_resolved)
        return max_resolved
    return sum(1 for d in (item.garment_type, item.fabric) if d)


def effective_item_count(items: list[ItemDistribution]) -> float:
    """multiplier-scaled batch denominator. N=2 기준 정규화 — N=0:0 / N=1:0.4 / N=2:1.0.

    Phase v2.1 (A): cluster_shares 의 multiplier-scaled 합으로 직접 산출 (canonical 단위
    fan-out 결과 정확 반영). cluster_shares 비어있으면 _resolved_axis_count fallback.
    """
    full = multiplier_for_n(2)
    if full <= 0:
        return 0.0
    total = 0.0
    for item in items:
        if item.cluster_shares:
            for ck, share in item.cluster_shares.items():
                mult = _multiplier_from_cluster_key(ck)
                total += share * mult / full
        else:
            n = _resolved_axis_count(item)
            total += multiplier_for_n(n) / full
    return total


# `clustering.assign_trend_cluster._UNKNOWN` 와 같은 placeholder ("unknown" 하드코드 — partial
# key 포맷 단일 source). 후속 cleanup 으로 두 모듈 공유 상수로 분리 예정.
_UNKNOWN_AXIS = "unknown"


def _multiplier_from_cluster_key(cluster_key: str) -> float:
    """cluster_key 의 unknown axis 갯수로 multiplier 결정 (Phase v2.1 (A))."""
    n_resolved = sum(1 for ax in cluster_key.split("__") if ax != _UNKNOWN_AXIS)
    return multiplier_for_n(n_resolved)


def _item_contributions(item: ItemDistribution) -> list[RepresentativeContribution]:
    """1 item → representative contribution 목록.

    Phase v2.1 (2026-04-30 (A)): `item.cluster_shares` 가 canonical 단위 fan-out 결과
    (caller `enriched_to_item_distribution` 에서 산출). cross-product 폐기.

    Backwards-compat: cluster_shares 빈 dict 면 옛 G/F cross-product fallback. 옛 test
    fixture / 직접 ItemDistribution 만드는 caller 호환용.

    multiplier 는 cluster_key 의 unknown axis 갯수로 결정 (N=2 → 2.5, N=1 → 1.0).
    contribution = share × multiplier × item_base_unit.
    """
    if item.cluster_shares:
        out: list[RepresentativeContribution] = []
        for ck, share in item.cluster_shares.items():
            if share <= 0.0:
                continue
            mult = _multiplier_from_cluster_key(ck)
            if mult <= 0.0:
                continue
            contrib = share * mult * item.item_base_unit
            out.append(RepresentativeContribution(
                representative_key=ck,
                item_id=item.item_id,
                source=item.source,
                match_share=share,
                multiplier=mult,
                contribution=contrib,
            ))
        return out

    # Fallback (옛 cross-product) — distribution 직접 만든 caller / fixture 용.
    n_axes = sum(1 for d in (item.garment_type, item.fabric) if d)
    if n_axes == 0:
        return []
    g_eff = item.garment_type or {_UNKNOWN_AXIS: 1.0}
    f_eff = item.fabric or {_UNKNOWN_AXIS: 1.0}
    mult = multiplier_for_n(n_axes)
    out_legacy: list[RepresentativeContribution] = []
    for (g, gp), (f, fp) in product(g_eff.items(), f_eff.items()):
        share = gp * fp
        if share <= 0.0:
            continue
        contrib = share * mult * item.item_base_unit
        out_legacy.append(RepresentativeContribution(
            representative_key=representative_key(g, f),
            item_id=item.item_id,
            source=item.source,
            match_share=share,
            multiplier=mult,
            contribution=contrib,
        ))
    return out_legacy


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
    """item → cluster_key 별 raw share dict. Phase v2.1 (A) (2026-04-30).

    canonical 단위 fan-out 결과를 그대로 반환 — 옛 cross-product (G × F) 폐기.
    `enriched_to_item_distribution` 가 산출 후 ItemDistribution.cluster_shares 에 주입.

    Backwards-compat: cluster_shares 빈 dict 면 옛 cross-product fallback.
    """
    if item.cluster_shares:
        return dict(item.cluster_shares)
    return assign_shares(item.garment_type, item.fabric)


def canonical_cluster_shares(canonicals: list, *, base_unit: float = 1.0) -> dict[str, float]:
    """canonical 단위 cluster fan-out (Phase v2.1 (A), 2026-04-30).

    각 canonical 의 (g, f) 가 자기 cluster_key 에 mass 분배. mass = canonical 의
    `group_to_item_contrib` (= log2(n_objects+1) × log2(area×100+1)) / Σ. multi-canonical
    같은 cluster_key 면 합산.

    partial canonical (g O / f X 또는 그 반대) → unknown axis placeholder cluster_key
    (예: `straight_kurta__unknown`). g X + f X 인 canonical 은 drop.

    base_unit: growth_rate factor 등 caller 가중치 (default 1.0).

    `build_cluster_summary._canonical_cluster_entries` 와 drop 정책 + 키 형식 +
    contrib 공식 동일해야 함 (`feedback_score_contributor_path_symmetry`). 공용 helper
    (`canonical_mean_area_ratio`, `group_to_item_contrib`, `build_exact_key_strs`) 사용 —
    한쪽만 inline 으로 바꾸면 silent drift. mass 스케일링만 의도적으로 다름 (여기는
    base_unit 정규화, summary path 는 G × base_unit 곱).

    Returns:
      cluster_key → share (per item, sum ≤ base_unit). multiplier 미적용 (caller 가
      `_item_contributions` 에서 cluster_key 의 unknown axis 갯수로 결정).
    """
    from contracts.vision import is_canonical_ethnic
    from aggregation.distribution_builder import group_to_item_contrib
    from aggregation.item_distribution_builder import canonical_mean_area_ratio
    from aggregation.vision_normalize import (
        normalize_garment_for_cluster, normalize_fabric,
    )
    from clustering.assign_trend_cluster import build_exact_key_strs

    eth = [c for c in canonicals if is_canonical_ethnic(c)]
    if not eth:
        return {}

    weighted: list[tuple[float, str | None, str | None]] = []
    for c in eth:
        n = len(c.members) if c.members else 0
        if n <= 0:
            continue
        contrib = group_to_item_contrib(n, canonical_mean_area_ratio(c))
        if contrib <= 0:
            continue
        g_enum = normalize_garment_for_cluster(c.representative)
        f_enum = normalize_fabric(c.representative)
        g = g_enum.value if g_enum else None
        f = f_enum.value if f_enum else None
        if g is None and f is None:
            continue
        weighted.append((contrib, g, f))

    if not weighted:
        return {}
    total_contrib = sum(w[0] for w in weighted)
    if total_contrib <= 0:
        return {}

    out: dict[str, float] = {}
    for contrib, g, f in weighted:
        ck = build_exact_key_strs(g or _UNKNOWN_AXIS, f or _UNKNOWN_AXIS)
        out[ck] = out.get(ck, 0.0) + (contrib / total_contrib) * base_unit
    return out


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
