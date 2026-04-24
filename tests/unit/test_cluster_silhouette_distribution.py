"""B3d pinning — silhouette_distribution 은 canonicals[0].representative.silhouette 에서
one-post-one-vote 로 계산된다 (post-level item.silhouette 제거).

`outfit_dedup._assign_canonical_index` 는 component 를 rep.area_ratio desc 로 정렬해 0 부터
부여하므로 `canonicals[0]` 은 post 내에서 rep bbox 가 가장 큰 outfit 이다. 따라서 post 당
1표 = 대표 canonical 의 silhouette. 같은 post 의 작은 canonical 은 silhouette 분포에 기여
하지 않는다 (multi-outfit 이어도 post 단위 1표, B3c cluster_palette 원칙과 대칭).
"""
from __future__ import annotations

from datetime import datetime

from aggregation.build_cluster_summary import make_drilldown
from contracts.common import ContentSource, Silhouette
from contracts.enriched import EnrichedContentItem
from contracts.normalized import NormalizedContentItem
from contracts.vision import CanonicalOutfit, EthnicOutfit, OutfitMember
from settings import PaletteConfig


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #

def _normalized(post_id: str) -> NormalizedContentItem:
    return NormalizedContentItem(
        source=ContentSource.INSTAGRAM,
        source_post_id=post_id,
        text_blob="",
        hashtags=[],
        image_urls=[],
        post_date=datetime(2026, 4, 24),
        engagement_raw=100,
    )


def _outfit(silhouette: Silhouette | None, area: float = 0.4) -> EthnicOutfit:
    return EthnicOutfit(
        person_bbox=(0.1, 0.1, 0.5, 0.7),
        person_bbox_area_ratio=area,
        upper_garment_type="kurta",
        lower_garment_type="palazzo",
        silhouette=silhouette,
        fabric="cotton",
        technique="plain",
        color_preset_picks_top3=[],
    )


def _canonical(
    index: int, silhouette: Silhouette | None, area: float = 0.4,
) -> CanonicalOutfit:
    rep = _outfit(silhouette, area=area)
    return CanonicalOutfit(
        canonical_index=index,
        representative=rep,
        members=[
            OutfitMember(
                image_id="img_0",
                outfit_index=0,
                person_bbox=(0.1, 0.1, 0.5, 0.7),
            )
        ],
    )


def _enriched(
    post_id: str,
    canonicals: list[CanonicalOutfit],
    *,
    cluster_key: str = "kurta_set/plain/cotton",
) -> EnrichedContentItem:
    return EnrichedContentItem(
        normalized=_normalized(post_id),
        canonicals=canonicals,
        trend_cluster_key=cluster_key,
    )


_PALETTE_CFG = PaletteConfig()


def _dist(items: list[EnrichedContentItem]) -> dict[str, float]:
    return make_drilldown(
        items=items,
        palette_cfg=_PALETTE_CFG,
        top_post_limit=3,
        top_video_ids=[],
    ).silhouette_distribution


# --------------------------------------------------------------------------- #
# canonical path — source of truth
# --------------------------------------------------------------------------- #

def test_single_post_single_canonical() -> None:
    # canonicals[0].rep.silhouette=STRAIGHT → {straight: 1.0}.
    item = _enriched("p1", [_canonical(0, Silhouette.STRAIGHT)])
    assert _dist([item]) == {"straight": 1.0}


def test_two_posts_different_silhouettes_split_half() -> None:
    a = _enriched("p1", [_canonical(0, Silhouette.STRAIGHT)])
    b = _enriched("p2", [_canonical(0, Silhouette.A_LINE)])
    assert _dist([a, b]) == {"straight": 0.5, "a_line": 0.5}


def test_post_with_multiple_canonicals_votes_only_for_canonical0() -> None:
    # canonicals[0] 은 area_ratio desc 0번 (대표). canonicals[1] 은 작은 outfit.
    # one-post-one-vote: canonicals[1].silhouette 는 분포에 기여하지 않는다.
    post = _enriched(
        "p1",
        [
            _canonical(0, Silhouette.A_LINE, area=0.45),
            _canonical(1, Silhouette.STRAIGHT, area=0.20),
        ],
    )
    assert _dist([post]) == {"a_line": 1.0}


def test_rep_silhouette_none_contributes_nothing() -> None:
    # canonicals 가 있어도 rep.silhouette=None 이면 그 post 는 분포에 기여하지 않는다.
    item = _enriched("p1", [_canonical(0, None)])
    assert _dist([item]) == {}


def test_no_canonicals_contributes_nothing() -> None:
    # vision 비활성 (fake extractor 경로) 또는 IG 아닌 경로 → canonicals=[].
    item = _enriched("p1", [])
    assert _dist([item]) == {}


def test_mixed_contributing_and_empty_posts() -> None:
    # 2 post 중 1 은 canonical 1개 (STRAIGHT), 다른 하나는 canonicals=[].
    # 기여 post 1 개만 카운트 → {straight: 1.0}.
    a = _enriched("p1", [_canonical(0, Silhouette.STRAIGHT)])
    b = _enriched("p2", [])
    assert _dist([a, b]) == {"straight": 1.0}


def test_three_posts_two_same_silhouette_one_other() -> None:
    # A_LINE 2 표 / STRAIGHT 1 표 → 2/3, 1/3.
    a = _enriched("p1", [_canonical(0, Silhouette.A_LINE)])
    b = _enriched("p2", [_canonical(0, Silhouette.A_LINE)])
    c = _enriched("p3", [_canonical(0, Silhouette.STRAIGHT)])
    result = _dist([a, b, c])
    assert result[Silhouette.A_LINE.value] == 2 / 3
    assert result[Silhouette.STRAIGHT.value] == 1 / 3
    assert sum(result.values()) == 1.0
