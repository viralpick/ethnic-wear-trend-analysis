"""Phase β3 + partial(g) (2026-04-28) — group_by_cluster cross-product fan-out pinning.

검증 포인트 (spec §2.4 share-weighted summary path):
- N=3 item: share > 0 인 모든 cluster_key 에 등록 (winner-keyed 동치 X)
- N<3 item (1 또는 2 축만 채워짐): partial 활성화로 multiplier_ratio (0.5/0.2) 가중
  share 로 placeholder (`unknown`) cluster 에 등록
- N=0 item (G/T/F 모두 빔): 어느 cluster 에도 등록 X (assign_shares 빈 dict)
- contract `trend_cluster_key` read X — 다른 winner key 가 set 되어 있어도 무시
- 다른 cluster 매칭 item 은 별도 cluster 로 분리 (mass preservation)
- score_and_export 정합 — summary path ↔ score path 같은 cluster space
"""
from __future__ import annotations

from datetime import datetime

import pytest

from aggregation.build_cluster_summary import group_by_cluster
from contracts.common import (
    ClassificationMethod,
    ContentSource,
    Fabric,
    GarmentType,
    Technique,
)
from contracts.enriched import EnrichedContentItem
from contracts.normalized import NormalizedContentItem
from contracts.vision import CanonicalOutfit, EthnicOutfit, OutfitMember


def _normalized(post_id: str) -> NormalizedContentItem:
    return NormalizedContentItem(
        source=ContentSource.INSTAGRAM,
        source_post_id=post_id,
        text_blob="",
        hashtags=[],
        image_urls=[],
        post_date=datetime(2026, 4, 27),
        engagement_raw_count=100,
        account_handle=None,
        account_followers=1000,
    )


def _enriched(
    post_id: str,
    *,
    g: GarmentType | None,
    t: Technique | None,
    f: Fabric | None,
    cluster_key: str | None,
) -> EnrichedContentItem:
    """text RULE method 로 결정성 확보. cluster_key 는 winner contract 값 (read X 검증용)."""
    methods: dict[str, ClassificationMethod] = {}
    if g is not None:
        methods["garment_type"] = ClassificationMethod.RULE
    if t is not None:
        methods["technique"] = ClassificationMethod.RULE
    if f is not None:
        methods["fabric"] = ClassificationMethod.RULE
    # 2026-05-02: canonical=0 → fan-out 미참여 정책. 옛 fixture (canonicals=[]) 의도
    # 보존 위해 g/f 채워진 케이스는 자동 canonical 1개 주입 (vision 통과 시뮬).
    return EnrichedContentItem(
        normalized=_normalized(post_id),
        garment_type=g,
        fabric=f,
        technique=t,
        canonicals=_auto_canonical(g, t, f),
        classification_method_per_attribute=methods,
        trend_cluster_key=cluster_key,
    )


def _auto_canonical(
    g: GarmentType | None, t: Technique | None, f: Fabric | None,
) -> list[CanonicalOutfit]:
    """g/f 둘 다 None 이면 빈 list. 하나라도 있으면 enum value 기반 canonical 1개."""
    if g is None and f is None:
        return []
    bbox = (0.1, 0.1, 0.5, 0.7)
    outfit = EthnicOutfit(
        person_bbox=bbox,
        person_bbox_area_ratio=0.35,
        upper_garment_type=g.value if g else "kurta",
        upper_is_ethnic=True,
        lower_garment_type="palazzo",
        lower_is_ethnic=True,
        dress_as_single=False,
        fabric=f.value if f else "cotton",
        technique=t.value if t else "chikankari",
        color_preset_picks_top3=[],
    )
    return [CanonicalOutfit(
        canonical_index=0,
        representative=outfit,
        members=[OutfitMember(image_id="img_0", outfit_index=0, person_bbox=bbox)],
    )]


# --------------------------------------------------------------------------- #
# β3 fan-out invariants
# --------------------------------------------------------------------------- #


def test_n3_single_distribution_registers_in_one_cluster() -> None:
    """G/T/F 모두 단일값 → cross-product 1 cluster, share=1.0 → 1 cluster grouping (β4 tuple)."""
    item = _enriched(
        "p1",
        g=GarmentType.KURTA_SET, t=Technique.CHIKANKARI, f=Fabric.COTTON,
        cluster_key="kurta_set__cotton",
    )
    grouped = group_by_cluster([item])
    assert set(grouped.keys()) == {"kurta_set__cotton"}
    entries = grouped["kurta_set__cotton"]
    assert len(entries) == 1
    pair_item, pair_share = entries[0]
    assert pair_item is item
    assert pair_share == pytest.approx(1.0)  # N=3 multiplier_ratio


def test_n_lt_3_item_registers_in_partial_cluster() -> None:
    """technique 누락은 cluster_key 에 영향 없음 → G/F exact cluster 에 full share 등록.
    """
    item = _enriched(
        "p1",
        g=GarmentType.KURTA_SET, t=None, f=Fabric.COTTON,
        cluster_key="kurta_set__cotton",  # winner contract 값 — 무시되고 있어도 동치
    )
    grouped = group_by_cluster([item])
    assert set(grouped.keys()) == {"kurta_set__cotton"}
    entries = grouped["kurta_set__cotton"]
    assert len(entries) == 1
    pair_item, pair_share = entries[0]
    assert pair_item is item
    assert pair_share == pytest.approx(1.0)


def test_n_zero_item_registers_in_no_cluster() -> None:
    """G/T/F 모두 빔 (N=0) → assign_shares 빈 dict → grouping X."""
    item = _enriched(
        "p1",
        g=None, t=None, f=None,
        cluster_key="unclassified",
    )
    grouped = group_by_cluster([item])
    assert grouped == {}


def test_two_different_n3_items_split_into_two_clusters() -> None:
    """다른 G/T/F 두 item → 두 cluster 분리. mass preservation (각 share=1.0, β4 tuple)."""
    item_a = _enriched(
        "p1",
        g=GarmentType.KURTA_SET, t=Technique.CHIKANKARI, f=Fabric.COTTON,
        cluster_key="kurta_set__cotton",
    )
    item_b = _enriched(
        "p2",
        g=GarmentType.CASUAL_SAREE, t=Technique.BLOCK_PRINT, f=Fabric.CHANDERI,
        cluster_key="casual_saree__chanderi",
    )
    grouped = group_by_cluster([item_a, item_b])
    assert set(grouped.keys()) == {
        "kurta_set__cotton",
        "casual_saree__chanderi",
    }
    assert grouped["kurta_set__cotton"] == [(item_a, pytest.approx(1.0))]
    assert grouped["casual_saree__chanderi"] == [(item_b, pytest.approx(1.0))]


def test_winner_contract_key_is_not_read() -> None:
    """β3 후 group_by_cluster 가 trend_cluster_key (winner) 를 무시하고
    ItemDistribution 기반 assign_shares 만 따름 — winner 값과 다른 cluster 가 결과여도
    distribution 결과 cluster 에만 등록."""
    item = _enriched(
        "p1",
        g=GarmentType.KURTA_SET, t=Technique.CHIKANKARI, f=Fabric.COTTON,
        # winner 가 일부러 다른 값 (drift 시뮬) — 무시되어야 정상
        cluster_key="totally_unrelated__winner__key",
    )
    grouped = group_by_cluster([item])
    # distribution 기반 cluster_key 만 등장. winner key 는 grouped 에 없음.
    assert "totally_unrelated__winner__key" not in grouped
    assert "kurta_set__cotton" in grouped


def test_same_cluster_two_items_aggregated() -> None:
    """동일 G/T/F 두 item → 같은 cluster 의 list 에 두 (item, share) 등록 (β4 tuple)."""
    item_a = _enriched(
        "p1",
        g=GarmentType.KURTA_SET, t=Technique.CHIKANKARI, f=Fabric.COTTON,
        cluster_key="kurta_set__cotton",
    )
    item_b = _enriched(
        "p2",
        g=GarmentType.KURTA_SET, t=Technique.CHIKANKARI, f=Fabric.COTTON,
        cluster_key="kurta_set__cotton",
    )
    grouped = group_by_cluster([item_a, item_b])
    assert list(grouped.keys()) == ["kurta_set__cotton"]
    entries = grouped["kurta_set__cotton"]
    assert [pair[0] for pair in entries] == [item_a, item_b]
    assert all(pair[1] == pytest.approx(1.0) for pair in entries)


def test_empty_input_returns_empty_dict() -> None:
    assert group_by_cluster([]) == {}
