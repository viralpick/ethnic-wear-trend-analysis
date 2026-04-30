"""Phase β4 (2026-04-28) — make_drilldown share-weighted vote pinning.

cluster 안 (item, share) 페어가 distribution / top_posts / top_influencers / cluster_palette
모두에 share 가중으로 반영되는지 검증. multi-fan-out item 도 자기가 가장 많이 기여하는
cluster 에서 큰 vote, 적게 기여하는 cluster 에서 작은 vote 를 갖는다.

대상 함수:
- `silhouette_distribution`: canonicals[0].rep.silhouette × cluster 내 item share
- `occasion_distribution`: item.occasion.value × cluster 내 item share
- `styling_distribution`: item.styling_combo.value × cluster 내 item share
- `top_posts`: IG only, sort key = engagement_raw × share desc
- `top_influencers`: IG only, sort key = engagement_raw × share desc, account_handle dedup
- `color_palette`: cluster.share × item_share (per-post weight 직접 입력)
"""
from __future__ import annotations

from datetime import datetime

import pytest

pytest.importorskip("sklearn", reason="sklearn required (color_space deps for cluster_palette)")

from aggregation.build_cluster_summary import make_drilldown  # noqa: E402
from contracts.common import (  # noqa: E402
    ClassificationMethod,
    ColorFamily,
    ContentSource,
    Occasion,
    PaletteCluster,
    Silhouette,
    StylingCombo,
)
from contracts.enriched import EnrichedContentItem  # noqa: E402
from contracts.normalized import NormalizedContentItem  # noqa: E402
from contracts.vision import (  # noqa: E402
    CanonicalOutfit,
    EthnicOutfit,
    OutfitMember,
)
from settings import PaletteConfig  # noqa: E402

_PALETTE_CFG = PaletteConfig()


def _normalized(
    post_id: str,
    *,
    source: ContentSource = ContentSource.INSTAGRAM,
    engagement: int = 100,
    handle: str | None = None,
) -> NormalizedContentItem:
    return NormalizedContentItem(
        source=source,
        source_post_id=post_id,
        text_blob="",
        hashtags=[],
        image_urls=[],
        post_date=datetime(2026, 4, 28),
        engagement_raw_count=engagement,
        account_handle=handle,
        account_followers=1000,
    )


def _outfit(silhouette: Silhouette | None) -> EthnicOutfit:
    return EthnicOutfit(
        person_bbox=(0.1, 0.1, 0.5, 0.7),
        person_bbox_area_ratio=0.4,
        upper_garment_type="kurta",
        lower_garment_type="palazzo",
        silhouette=silhouette,
        fabric="cotton",
        technique="plain",
        color_preset_picks_top3=[],
    )


def _canonical(silhouette: Silhouette | None) -> CanonicalOutfit:
    return CanonicalOutfit(
        canonical_index=0,
        representative=_outfit(silhouette),
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
    *,
    silhouette: Silhouette | None = None,
    occasion: Occasion | None = None,
    styling: StylingCombo | None = None,
    engagement: int = 100,
    source: ContentSource = ContentSource.INSTAGRAM,
    handle: str | None = None,
    post_palette: list[PaletteCluster] | None = None,
    canonicals: list[CanonicalOutfit] | None = None,
) -> EnrichedContentItem:
    """canonicals override 를 받으면 그대로 사용 (vision-side styling_combo 테스트용).
    아니면 silhouette argument 로 single canonical 합성 — 기존 로직 유지.
    """
    if canonicals is None:
        canonicals = []
        if silhouette is not None:
            canonicals = [_canonical(silhouette)]
    method_map: dict[str, ClassificationMethod] = {}
    if styling is not None:
        # extract_text_attributes_llm 가 setting 하는 production state 반영.
        # method 없이 styling_combo enum 만 있는 enriched 는 비현실적 — text 가중치 0.
        method_map["styling_combo"] = ClassificationMethod.LLM
    return EnrichedContentItem(
        normalized=_normalized(post_id, source=source, engagement=engagement, handle=handle),
        canonicals=canonicals,
        occasion=occasion,
        styling_combo=styling,
        post_palette=post_palette or [],
        classification_method_per_attribute=method_map,
    )


# --------------------------------------------------------------------------- #
# distribution share-weighted vote
# --------------------------------------------------------------------------- #


def test_silhouette_distribution_share_weighted_two_items() -> None:
    """item A share=0.7 silhouette=A_LINE / item B share=0.3 silhouette=STRAIGHT.
    distribution = {A_LINE: 0.7, STRAIGHT: 0.3} (share-weighted, sum=1.0).
    """
    a = _enriched("p_a", silhouette=Silhouette.A_LINE)
    b = _enriched("p_b", silhouette=Silhouette.STRAIGHT)
    drill = make_drilldown(
        items_with_share=[(a, 0.7), (b, 0.3)],
        palette_cfg=_PALETTE_CFG,
        top_post_limit=3,
        top_video_ids=[],
    )
    assert drill.silhouette_distribution["a_line"] == pytest.approx(0.7)
    assert drill.silhouette_distribution["straight"] == pytest.approx(0.3)
    assert sum(drill.silhouette_distribution.values()) == pytest.approx(1.0)


def test_silhouette_distribution_share_weighted_same_silhouette_two_items() -> None:
    """같은 silhouette 두 item 의 share 가 합산.
    다른 silhouette 작은 share 도 정규화 후 비례 유지."""
    a = _enriched("p_a", silhouette=Silhouette.A_LINE)
    b = _enriched("p_b", silhouette=Silhouette.A_LINE)
    c = _enriched("p_c", silhouette=Silhouette.STRAIGHT)
    drill = make_drilldown(
        items_with_share=[(a, 0.5), (b, 0.5), (c, 1.0)],
        palette_cfg=_PALETTE_CFG,
        top_post_limit=3,
        top_video_ids=[],
    )
    # A_LINE = 0.5+0.5=1.0 / STRAIGHT = 1.0 / total = 2.0 → 0.5 / 0.5
    assert drill.silhouette_distribution["a_line"] == pytest.approx(0.5)
    assert drill.silhouette_distribution["straight"] == pytest.approx(0.5)


def test_occasion_distribution_share_weighted() -> None:
    a = _enriched("p_a", occasion=Occasion.OFFICE)
    b = _enriched("p_b", occasion=Occasion.OFFICE)
    c = _enriched("p_c", occasion=Occasion.FESTIVE_LITE)
    drill = make_drilldown(
        items_with_share=[(a, 0.6), (b, 0.6), (c, 0.4)],
        palette_cfg=_PALETTE_CFG,
        top_post_limit=3,
        top_video_ids=[],
    )
    # OFFICE = 1.2 / FESTIVE_LITE = 0.4 / total = 1.6 → 0.75 / 0.25
    assert drill.occasion_distribution[Occasion.OFFICE.value] == pytest.approx(0.75)
    assert drill.occasion_distribution[Occasion.FESTIVE_LITE.value] == pytest.approx(0.25)


def test_styling_distribution_share_weighted() -> None:
    a = _enriched("p_a", styling=StylingCombo.WITH_PALAZZO)
    b = _enriched("p_b", styling=StylingCombo.CO_ORD_SET)
    drill = make_drilldown(
        items_with_share=[(a, 0.8), (b, 0.2)],
        palette_cfg=_PALETTE_CFG,
        top_post_limit=3,
        top_video_ids=[],
    )
    assert drill.styling_distribution[StylingCombo.WITH_PALAZZO.value] == pytest.approx(0.8)
    assert drill.styling_distribution[StylingCombo.CO_ORD_SET.value] == pytest.approx(0.2)


def _vision_outfit(
    *,
    lower: str | None = None,
    is_co_ord_set: bool | None = None,
    dress_as_single: bool = False,
    outer_layer: str | None = None,
    upper: str = "kurta",
) -> EthnicOutfit:
    return EthnicOutfit(
        person_bbox=(0.1, 0.1, 0.5, 0.7),
        person_bbox_area_ratio=0.4,
        upper_garment_type=upper,
        lower_garment_type=lower,
        silhouette=None,
        fabric="cotton",
        technique="plain",
        is_co_ord_set=is_co_ord_set,
        dress_as_single=dress_as_single,
        outer_layer=outer_layer,
        color_preset_picks_top3=[],
    )


def _vision_canonical(index: int, outfit: EthnicOutfit) -> CanonicalOutfit:
    return CanonicalOutfit(
        canonical_index=index,
        representative=outfit,
        members=[
            OutfitMember(
                image_id=f"img_{index}",
                outfit_index=0,
                person_bbox=outfit.person_bbox,
            )
        ],
    )


def test_styling_distribution_vision_only_canonical() -> None:
    """canonicals 만으로 styling 파생 — text styling_combo 미설정 (로직 B 핵심).

    item A: lower=palazzo → WITH_PALAZZO, share=0.7
    item B: dress_as_single=True → STANDALONE, share=0.3
    """
    a = _enriched(
        "p_a",
        canonicals=[_vision_canonical(0, _vision_outfit(lower="palazzo"))],
    )
    b = _enriched(
        "p_b",
        canonicals=[_vision_canonical(0, _vision_outfit(dress_as_single=True))],
    )
    drill = make_drilldown(
        items_with_share=[(a, 0.7), (b, 0.3)],
        palette_cfg=_PALETTE_CFG,
        top_post_limit=3,
        top_video_ids=[],
    )
    assert drill.styling_distribution[StylingCombo.WITH_PALAZZO.value] == pytest.approx(0.7)
    assert drill.styling_distribution[StylingCombo.STANDALONE.value] == pytest.approx(0.3)


def test_styling_distribution_multi_canonical_per_post_split_within_item() -> None:
    """한 post 의 canonical 2개 → per-item distribution 안에서 둘이 분배 → cluster share
    가중. canonical 둘 다 같은 area, n_objects 같으면 정확히 50:50.

    item A (share=1.0): 2 canonicals, lower=palazzo / lower=churidar → 0.5/0.5
    """
    canonicals = [
        _vision_canonical(0, _vision_outfit(lower="palazzo")),
        _vision_canonical(1, _vision_outfit(lower="churidar")),
    ]
    a = _enriched("p_a", canonicals=canonicals)
    drill = make_drilldown(
        items_with_share=[(a, 1.0)],
        palette_cfg=_PALETTE_CFG,
        top_post_limit=3,
        top_video_ids=[],
    )
    assert drill.styling_distribution[StylingCombo.WITH_PALAZZO.value] == pytest.approx(0.5)
    assert drill.styling_distribution[StylingCombo.WITH_CHURIDAR.value] == pytest.approx(0.5)


def test_styling_distribution_text_plus_vision_blend() -> None:
    """text styling_combo (LLM=3.0) + vision-side derived styling (G=log2(2+1) ≈ 1.585)
    합산. text=WITH_PANTS, vision=WITH_PALAZZO 일 때 정규화 후 둘 다 양수 share.
    """
    canonicals = [_vision_canonical(0, _vision_outfit(lower="palazzo"))]
    a = _enriched(
        "p_a",
        styling=StylingCombo.WITH_PANTS,
        canonicals=canonicals,
    )
    drill = make_drilldown(
        items_with_share=[(a, 1.0)],
        palette_cfg=_PALETTE_CFG,
        top_post_limit=3,
        top_video_ids=[],
    )
    # text(LLM=3.0) + vision(log2(2+1) ≈ 1.585) → text 가 dominant 이지만 vision 도 살아남음
    assert drill.styling_distribution.get(StylingCombo.WITH_PANTS.value, 0) > 0.5
    assert drill.styling_distribution.get(StylingCombo.WITH_PALAZZO.value, 0) > 0.0
    assert sum(drill.styling_distribution.values()) == pytest.approx(1.0)


def test_distribution_zero_share_entry_excluded() -> None:
    """share<=0 인 entry 는 distribution 에 미기여 (β4 invariant)."""
    a = _enriched("p_a", silhouette=Silhouette.A_LINE)
    b = _enriched("p_b", silhouette=Silhouette.STRAIGHT)
    drill = make_drilldown(
        items_with_share=[(a, 1.0), (b, 0.0)],
        palette_cfg=_PALETTE_CFG,
        top_post_limit=3,
        top_video_ids=[],
    )
    assert drill.silhouette_distribution == {"a_line": 1.0}


def test_distribution_none_value_excluded() -> None:
    """value=None entry 는 분포에 미기여 (silhouette/occasion). styling 은 canonical 의
    lower_garment_type='palazzo' 에서 vision-side 로 자동 파생되므로 a 의 1표만 살아남음."""
    a = _enriched("p_a", silhouette=Silhouette.A_LINE)
    b = _enriched("p_b", silhouette=None, occasion=None, styling=None)
    drill = make_drilldown(
        items_with_share=[(a, 0.5), (b, 0.5)],
        palette_cfg=_PALETTE_CFG,
        top_post_limit=3,
        top_video_ids=[],
    )
    assert drill.silhouette_distribution == {"a_line": 1.0}
    assert drill.occasion_distribution == {}
    # 로직 B: a 의 canonical lower="palazzo" → WITH_PALAZZO; b 는 canonical 0개 → 미기여.
    assert drill.styling_distribution == {StylingCombo.WITH_PALAZZO.value: 1.0}


# --------------------------------------------------------------------------- #
# top_posts / top_influencers — engagement × share ranking
# --------------------------------------------------------------------------- #


def test_top_posts_engagement_times_share_ranking() -> None:
    """top_posts ranking key = engagement_raw × share. 작은 engagement × 큰 share 가 큰
    engagement × 작은 share 를 이길 수 있다 (β4 share-weighted invariant).
    """
    big = _enriched("p_big", engagement=200, handle="big_user")  # 200 × 0.3 = 60
    small = _enriched("p_small", engagement=80, handle="small_user")  # 80 × 1.0 = 80
    drill = make_drilldown(
        items_with_share=[(big, 0.3), (small, 1.0)],
        palette_cfg=_PALETTE_CFG,
        top_post_limit=3,
        top_video_ids=[],
    )
    # small (80) > big (60) — share 가중 ranking
    assert drill.top_posts == ["p_small", "p_big"]


def test_top_posts_excludes_youtube() -> None:
    ig = _enriched("p_ig", engagement=100, source=ContentSource.INSTAGRAM, handle="u")
    yt = _enriched("p_yt", engagement=10000, source=ContentSource.YOUTUBE)
    drill = make_drilldown(
        items_with_share=[(ig, 1.0), (yt, 1.0)],
        palette_cfg=_PALETTE_CFG,
        top_post_limit=3,
        top_video_ids=[],
    )
    assert drill.top_posts == ["p_ig"]


def test_top_influencers_dedup_handle_keeps_highest_score() -> None:
    """같은 account_handle 의 multi item — 첫 등장만 keep (engagement × share desc 정렬 후)."""
    h1 = _enriched("p1", engagement=100, handle="alpha")  # 100 × 1.0 = 100
    h1_low = _enriched("p1_low", engagement=200, handle="alpha")  # 200 × 0.1 = 20
    h2 = _enriched("p2", engagement=50, handle="beta")  # 50 × 1.0 = 50
    drill = make_drilldown(
        items_with_share=[(h1, 1.0), (h1_low, 0.1), (h2, 1.0)],
        palette_cfg=_PALETTE_CFG,
        top_post_limit=3,
        top_video_ids=[],
    )
    # alpha (100) > beta (50). dedup → [alpha, beta].
    assert drill.top_influencers == ["alpha", "beta"]


def test_top_influencers_skips_no_handle() -> None:
    """account_handle=None IG item 은 top_influencers 에 미기여."""
    no_handle = _enriched("p_anon", engagement=1000)  # handle=None
    with_handle = _enriched("p_named", engagement=10, handle="alpha")
    drill = make_drilldown(
        items_with_share=[(no_handle, 1.0), (with_handle, 1.0)],
        palette_cfg=_PALETTE_CFG,
        top_post_limit=3,
        top_video_ids=[],
    )
    assert drill.top_influencers == ["alpha"]


# --------------------------------------------------------------------------- #
# cluster_palette per-post weight
# --------------------------------------------------------------------------- #


def test_cluster_palette_share_weights_outweigh_post_internal_share() -> None:
    """post A 의 큰 cluster.share 도 item_share 작으면 post B 의 작은 cluster.share×큰 share
    에 밀린다.
    A: red 1.0, item_share=0.1 → weight=0.1
    B: blue 0.5, item_share=1.0 → weight=0.5
    C: blue 0.5, item_share=1.0 → weight=0.5
    → blue (1.0) > red (0.1).
    """
    red = PaletteCluster(hex="#CC0000", share=1.0, family=ColorFamily.JEWEL)
    blue1 = PaletteCluster(hex="#0000CC", share=0.5, family=ColorFamily.JEWEL)
    blue2 = PaletteCluster(hex="#0000CC", share=0.5, family=ColorFamily.JEWEL)
    a = _enriched("p_a", post_palette=[red])
    b = _enriched("p_b", post_palette=[blue1])
    c = _enriched("p_c", post_palette=[blue2])
    drill = make_drilldown(
        items_with_share=[(a, 0.1), (b, 1.0), (c, 1.0)],
        palette_cfg=_PALETTE_CFG,
        top_post_limit=3,
        top_video_ids=[],
    )
    # blue 가 더 큰 share 로 dominate
    palette = drill.color_palette
    assert len(palette) == 2  # red + blue (멀어서 merge X)
    top = max(palette, key=lambda c: c.share)
    assert top.family == ColorFamily.JEWEL
    # blue total weight = 0.5 + 0.5 = 1.0 / red weight = 0.1 → blue ≈ 0.909 / red ≈ 0.091
    assert top.share == pytest.approx(1.0 / 1.1, abs=1e-6)


def test_cluster_palette_zero_post_weight_skipped() -> None:
    """item_share=0 인 post 는 cluster_palette 에 미기여 (drop 되거나 빈 결과)."""
    red = PaletteCluster(hex="#CC0000", share=1.0, family=ColorFamily.JEWEL)
    a = _enriched("p_a", post_palette=[red])
    b = _enriched("p_b", post_palette=[red])  # 같은 hex
    drill = make_drilldown(
        items_with_share=[(a, 1.0), (b, 0.0)],
        palette_cfg=_PALETTE_CFG,
        top_post_limit=3,
        top_video_ids=[],
    )
    # b 가 미기여라 a 의 단일 색만 결과
    assert len(drill.color_palette) == 1
    assert drill.color_palette[0].share == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# multi-fan-out 의미 — cluster 별 다른 share 면 같은 item 도 다른 vote
# --------------------------------------------------------------------------- #


def test_same_item_different_clusters_have_different_share_weighted_votes() -> None:
    """같은 item 이 두 cluster 에 다른 share 로 등장하면, 각 cluster 의 distribution 에서
    동일 silhouette 에 다른 weight (정규화 후 단일 cluster 내에서는 100% 단독이지만,
    cluster 별 votes 분리 = score path 와 같은 의미).
    """
    a = _enriched("p_a", silhouette=Silhouette.A_LINE)
    drill_high = make_drilldown(
        items_with_share=[(a, 0.7)],
        palette_cfg=_PALETTE_CFG,
        top_post_limit=3,
        top_video_ids=[],
    )
    drill_low = make_drilldown(
        items_with_share=[(a, 0.3)],
        palette_cfg=_PALETTE_CFG,
        top_post_limit=3,
        top_video_ids=[],
    )
    # 둘 다 단일 item 단일 silhouette 라 normalize 후 100% 동일.
    assert drill_high.silhouette_distribution == {"a_line": 1.0}
    assert drill_low.silhouette_distribution == {"a_line": 1.0}
