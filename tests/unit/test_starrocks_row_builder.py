"""row_builder pinning — pipeline_spec_v1.0 §1 / §2 / §5 4-table 변환 invariant.

검증:
- representative_id = blake2b(key, 8) big-endian signed BIGINT — literal byte test.
- build_item_row: distribution dict pass-through, silhouette_dist 합성, NULL 정직 적재.
- build_group_rows: representative attribute, item_contribution_score = §2.7 공식.
- build_object_rows: per-member carry-over (Step 7.4a 결과 검증), object_to_group_contrib.
- build_representative_row: factor_contribution 키 ContentSource → str, 6 distribution
  passthrough, 빈 list/dict 도 NULL 화 (writer JSON null 처리).
- 정합성: object 행 수 = Σ canonical members.
"""
from __future__ import annotations

import math
from datetime import datetime

from aggregation.representative_builder import RepresentativeAggregate
from contracts.common import (
    ClassificationMethod,
    ColorFamily,
    ContentSource,
    Fabric,
    GarmentType,
    Occasion,
    PaletteCluster,
    Silhouette,
    Technique,
)
from contracts.enriched import EnrichedContentItem
from contracts.normalized import NormalizedContentItem
from contracts.vision import (
    CanonicalOutfit,
    EthnicOutfit,
    GarmentAnalysis,  # noqa: F401  (contract sanity import)
    OutfitMember,
)
from exporters.starrocks.row_builder import (
    SCHEMA_VERSION,
    build_group_rows,
    build_item_row,
    build_object_rows,
    build_representative_row,
    representative_id,
)


def _normalized(
    post_id: str = "p1",
    source: ContentSource = ContentSource.INSTAGRAM,
) -> NormalizedContentItem:
    return NormalizedContentItem(
        source=source,
        source_post_id=post_id,
        text_blob="cotton kurta in jaipur",
        hashtags=["#kurta"],
        image_urls=["https://example.com/img.jpg"],
        post_date=datetime(2026, 4, 27, 12, 0, 0),
        engagement_raw_count=4321,
        account_followers=1500,
        account_handle="brand_x",
    )


def _outfit(
    *,
    upper: str | None = "kurta",
    lower: str | None = "palazzo",
    fabric: str | None = "cotton",
    technique: str | None = "block_print",
    silhouette: Silhouette | None = Silhouette.A_LINE,
    bbox: tuple[float, float, float, float] = (0.1, 0.1, 0.5, 0.7),
    area_ratio: float = 0.35,
    upper_is_ethnic: bool | None = True,
    lower_is_ethnic: bool | None = True,
    dress_as_single: bool = False,
) -> EthnicOutfit:
    return EthnicOutfit(
        person_bbox=bbox,
        person_bbox_area_ratio=area_ratio,
        upper_garment_type=upper,
        upper_is_ethnic=upper_is_ethnic,
        lower_garment_type=lower,
        lower_is_ethnic=lower_is_ethnic,
        dress_as_single=dress_as_single,
        fabric=fabric,
        technique=technique,
        silhouette=silhouette,
        color_preset_picks_top3=[],
    )


def _canonical(
    index: int,
    *,
    outfit: EthnicOutfit,
    members: list[OutfitMember] | None = None,
    palette: list[PaletteCluster] | None = None,
) -> CanonicalOutfit:
    if members is None:
        members = [
            OutfitMember(
                image_id="img_0",
                outfit_index=0,
                person_bbox=outfit.person_bbox,
                garment_type=outfit.upper_garment_type,
                fabric=outfit.fabric,
                technique=outfit.technique,
                silhouette=outfit.silhouette,
            )
        ]
    return CanonicalOutfit(
        canonical_index=index,
        representative=outfit,
        members=members,
        palette=palette or [],
    )


def _enriched(canonicals: list[CanonicalOutfit]) -> EnrichedContentItem:
    return EnrichedContentItem(
        normalized=_normalized(),
        garment_type=GarmentType.KURTA_SET,
        fabric=Fabric.COTTON,
        technique=Technique.BLOCK_PRINT,
        occasion=Occasion.CASUAL,
        canonicals=canonicals,
        post_palette=[
            PaletteCluster(hex="#aabbcc", share=0.6, family=ColorFamily.NEUTRAL),
            PaletteCluster(hex="#112233", share=0.4, family=ColorFamily.EARTH),
        ],
        classification_method_per_attribute={
            "garment_type": ClassificationMethod.RULE,
            "fabric": ClassificationMethod.RULE,
            "technique": ClassificationMethod.LLM,
        },
    )


# --------------------------------------------------------------------------- #
# representative_id

def test_representative_id_is_signed_bigint_blake2b() -> None:
    # blake2b("kurta_set__block_print__cotton", 8) → big-endian → signed.
    rid = representative_id("kurta_set__block_print__cotton")
    # signed int64 range.
    assert -(2**63) <= rid <= 2**63 - 1
    # deterministic — 같은 key 는 항상 같은 id.
    assert rid == representative_id("kurta_set__block_print__cotton")
    # 다른 key 는 다른 id (collision 확률 무시).
    assert rid != representative_id("kurta_set__block_print__linen")


def test_representative_id_literal_byte_pinning() -> None:
    # blake2b 기본 fanout 등 파라미터 동결 — bytes 가 바뀌면 모든 적재 재계산.
    import hashlib

    digest = hashlib.blake2b(
        b"kurta_set__block_print__cotton", digest_size=8
    ).digest()
    expected = int.from_bytes(digest, "big", signed=True)
    assert representative_id("kurta_set__block_print__cotton") == expected


# --------------------------------------------------------------------------- #
# build_item_row

def test_build_item_row_full_distribution() -> None:
    canonical = _canonical(0, outfit=_outfit())
    enriched = _enriched([canonical])

    row = build_item_row(
        enriched,
        computed_at="2026-04-27 09:00:00",
        posted_at="2026-04-27 06:00:00",
    )

    assert row["source"] == "instagram"
    assert row["source_post_id"] == "p1"
    assert row["computed_at"] == "2026-04-27 09:00:00"
    assert row["posted_at"] == "2026-04-27 06:00:00"
    assert row["schema_version"] == SCHEMA_VERSION

    # text + vision distribution 채워짐. raw "kurta" → STRAIGHT_KURTA enum 매핑됨.
    assert row["garment_type_dist"] is not None
    assert "kurta_set" in row["garment_type_dist"]
    assert "straight_kurta" in row["garment_type_dist"]
    assert abs(sum(row["garment_type_dist"].values()) - 1.0) < 1e-9

    # silhouette vision-only.
    assert row["silhouette_dist"] == {"a_line": 1.0}

    # styling_combo: vision (lower="palazzo" → WITH_PALAZZO). text-LLM 측은 미설정.
    assert row["styling_combo_dist"] == {"with_palazzo": 1.0}

    # post_palette dump.
    assert row["color_palette"] == [
        {"hex": "#aabbcc", "share": 0.6, "family": "neutral"},
        {"hex": "#112233", "share": 0.4, "family": "earth"},
    ]

    # occasion enum unwrap.
    assert row["occasion"] == "casual"
    assert row["brands_mentioned"] is None

    # normalized passthrough.
    assert row["engagement_raw"] == 4321
    assert row["account_handle"] == "brand_x"
    assert row["account_follower_count"] == 1500


def test_build_item_row_empty_canonicals_silhouette_null() -> None:
    bare_normalized = NormalizedContentItem(
        source=ContentSource.YOUTUBE,
        source_post_id="yt_empty",
        text_blob="",
        hashtags=[],
        image_urls=[],
        post_date=datetime(2026, 4, 27, 0, 0, 0),
        engagement_raw_count=0,
    )
    enriched = EnrichedContentItem(normalized=bare_normalized, canonicals=[])
    row = build_item_row(
        enriched,
        computed_at="2026-04-27 09:00:00",
        posted_at=None,
    )
    # 모든 distribution 비어 있음 → NULL.
    assert row["garment_type_dist"] is None
    assert row["fabric_dist"] is None
    assert row["technique_dist"] is None
    assert row["silhouette_dist"] is None
    assert row["styling_combo_dist"] is None
    assert row["color_palette"] is None
    assert row["posted_at"] is None
    # account 없으면 follower_count 도 NULL (account_followers=0 default).
    assert row["account_follower_count"] is None
    assert row["account_handle"] is None
    assert row["source"] == "youtube"


# --------------------------------------------------------------------------- #
# build_group_rows

def test_build_group_rows_representative_attributes() -> None:
    c0 = _canonical(
        0,
        outfit=_outfit(
            upper="kurta",
            fabric="cotton",
            technique="block_print",
            silhouette=Silhouette.A_LINE,
        ),
        palette=[PaletteCluster(hex="#aabbcc", share=1.0)],
    )
    c1 = _canonical(
        1,
        outfit=_outfit(
            upper="saree",
            fabric="silk",
            technique="zardosi",
            silhouette=None,
        ),
    )
    enriched = _enriched([c0, c1])

    rows = build_group_rows(enriched, computed_at="2026-04-27 09:00:00")
    assert len(rows) == 2

    g0 = rows[0]
    assert g0["item_source"] == "instagram"
    assert g0["item_source_post_id"] == "p1"
    assert g0["canonical_index"] == 0
    assert g0["group_id"] == "instagram__p1__0"
    assert g0["garment_type"] == "kurta"
    assert g0["fabric"] == "cotton"
    assert g0["technique"] == "block_print"
    assert g0["silhouette"] == "a_line"
    # _outfit() default lower="palazzo" → WITH_PALAZZO 단일값.
    assert g0["styling_combo"] == "with_palazzo"
    assert g0["color_palette"] == [{"hex": "#aabbcc", "share": 1.0, "family": None}]
    assert g0["n_objects"] == 1
    # mean_area_ratio = 0.5 * 0.7 = 0.35 (단일 멤버 person_bbox area).
    assert abs(g0["mean_area_ratio"] - 0.35) < 1e-9
    # spec §2.7: log2(1+1) × log2(0.35×100+1) = 1.0 × log2(36).
    expected = math.log2(2) * math.log2(36.0)
    assert abs(g0["item_contribution_score"] - expected) < 1e-9

    g1 = rows[1]
    assert g1["silhouette"] is None
    assert g1["color_palette"] is None  # 빈 palette → NULL.


# --------------------------------------------------------------------------- #
# build_object_rows

def test_build_object_rows_carries_member_attributes() -> None:
    # 7.4a carry-over 검증: dress_as_single (silhouette=None) ↔ two_piece (A_LINE) 가
    # 같은 canonical 로 묶여도 멤버별 raw attr 그대로.
    members = [
        OutfitMember(
            image_id="img_0",
            outfit_index=0,
            person_bbox=(0.0, 0.0, 0.6, 0.8),
            garment_type="lehenga",
            fabric="silk",
            technique="block_print",
            silhouette=None,
        ),
        OutfitMember(
            image_id="img_1",
            outfit_index=0,
            person_bbox=(0.1, 0.1, 0.4, 0.6),
            garment_type="lehenga",
            fabric="cotton",
            technique="block_print",
            silhouette=Silhouette.A_LINE,
        ),
    ]
    canonical = _canonical(
        0,
        outfit=_outfit(upper="lehenga", silhouette=Silhouette.A_LINE),
        members=members,
    )
    enriched = _enriched([canonical])

    rows = build_object_rows(enriched, computed_at="2026-04-27 09:00:00")
    assert len(rows) == 2

    r0 = rows[0]
    assert r0["object_id"] == "instagram__p1__0__0"
    assert r0["group_id"] == "instagram__p1__0"
    assert r0["member_index"] == 0
    assert r0["garment_type"] == "lehenga"
    assert r0["fabric"] == "silk"
    assert r0["technique"] == "block_print"
    assert r0["silhouette"] is None
    assert r0["bbox"] == [0.0, 0.0, 0.6, 0.8]
    assert abs(r0["area_ratio"] - 0.48) < 1e-9
    # spec §2.7: log2(0.48×100+1) = log2(49).
    assert abs(r0["group_contribution_score"] - math.log2(49.0)) < 1e-9
    # media_ref / palette: 7.6 fix-up territory — 현재 NULL.
    assert r0["media_ref"] is None
    assert r0["color_palette"] is None

    r1 = rows[1]
    assert r1["fabric"] == "cotton"
    assert r1["silhouette"] == "a_line"


def test_build_object_rows_resolves_media_ref_by_basename() -> None:
    # image_id (path.name) 가 image_urls 중 하나의 basename 과 일치 → 그 URL 반환.
    members = [
        OutfitMember(
            image_id="01KP1234.jpg",
            outfit_index=0,
            person_bbox=(0.0, 0.0, 0.5, 0.5),
        ),
        OutfitMember(
            image_id="01KP5678.jpg",
            outfit_index=1,
            person_bbox=(0.0, 0.0, 0.4, 0.4),
        ),
    ]
    canonical = _canonical(0, outfit=_outfit(), members=members)
    enriched = EnrichedContentItem(
        normalized=NormalizedContentItem(
            source=ContentSource.INSTAGRAM,
            source_post_id="p1",
            text_blob="t",
            hashtags=[],
            image_urls=[
                "https://x.blob.core.windows.net/c/01KP1234.jpg?sv=2024&sig=AAA",
                "https://x.blob.core.windows.net/c/01KP5678.jpg",
            ],
            post_date=datetime(2026, 4, 27, 12, 0, 0),
            engagement_raw_count=0,
            account_followers=0,
        ),
        garment_type=GarmentType.KURTA_SET,
        fabric=Fabric.COTTON,
        technique=Technique.BLOCK_PRINT,
        occasion=Occasion.CASUAL,
        canonicals=[canonical],
        post_palette=[],
        classification_method_per_attribute={},
    )

    rows = build_object_rows(enriched, computed_at="2026-04-27 09:00:00")
    # SAS query string 보존된 원본 URL 그대로 반환.
    assert rows[0]["media_ref"] == (
        "https://x.blob.core.windows.net/c/01KP1234.jpg?sv=2024&sig=AAA"
    )
    assert rows[1]["media_ref"] == "https://x.blob.core.windows.net/c/01KP5678.jpg"


def test_build_object_rows_media_ref_none_when_no_match() -> None:
    members = [
        OutfitMember(
            image_id="missing.jpg",
            outfit_index=0,
            person_bbox=(0.0, 0.0, 0.5, 0.5),
        ),
    ]
    canonical = _canonical(0, outfit=_outfit(), members=members)
    enriched = EnrichedContentItem(
        normalized=NormalizedContentItem(
            source=ContentSource.INSTAGRAM,
            source_post_id="p1",
            text_blob="t",
            hashtags=[],
            image_urls=["https://x.blob.core.windows.net/c/01KP1234.jpg"],
            post_date=datetime(2026, 4, 27, 12, 0, 0),
            engagement_raw_count=0,
            account_followers=0,
        ),
        garment_type=GarmentType.KURTA_SET,
        fabric=Fabric.COTTON,
        technique=Technique.BLOCK_PRINT,
        occasion=Occasion.CASUAL,
        canonicals=[canonical],
        post_palette=[],
        classification_method_per_attribute={},
    )

    rows = build_object_rows(enriched, computed_at="2026-04-27 09:00:00")
    assert rows[0]["media_ref"] is None


def test_build_object_rows_resolves_media_ref_video_frame_match() -> None:
    """M3.G — image_id=`{stem}_f{idx}` (확장자 X) → video_urls 의 stem 매칭."""
    members = [
        OutfitMember(
            image_id="reel123_f30",
            outfit_index=0,
            person_bbox=(0.0, 0.0, 0.5, 0.5),
        ),
    ]
    canonical = _canonical(0, outfit=_outfit(), members=members)
    enriched = EnrichedContentItem(
        normalized=NormalizedContentItem(
            source=ContentSource.INSTAGRAM,
            source_post_id="p1",
            text_blob="t",
            hashtags=[],
            image_urls=[],
            video_urls=[
                "https://x.blob.core.windows.net/c/reel123.mp4?sv=2024&sig=AAA",
            ],
            post_date=datetime(2026, 4, 28, 12, 0, 0),
            engagement_raw_count=0,
            account_followers=0,
        ),
        garment_type=GarmentType.KURTA_SET,
        fabric=Fabric.COTTON,
        technique=Technique.BLOCK_PRINT,
        occasion=Occasion.CASUAL,
        canonicals=[canonical],
        post_palette=[],
        classification_method_per_attribute={},
    )

    rows = build_object_rows(enriched, computed_at="2026-04-28 09:00:00")
    # SAS query string 보존 (raw URL 그대로 적재).
    assert rows[0]["media_ref"] == (
        "https://x.blob.core.windows.net/c/reel123.mp4?sv=2024&sig=AAA"
    )


def test_build_object_rows_video_frame_id_no_false_positive_on_jpg() -> None:
    """확장자 포함 image basename 은 video frame regex 에 잡히지 않는다.

    `picture_f5.jpg` 는 image basename 패턴 — `^(.+)_f\\d+$` regex 에 매칭 안 됨
    (끝에 `.jpg` 있어서). image_urls 로만 fallback 되어, image_urls 비면 NULL.
    """
    members = [
        OutfitMember(
            image_id="picture_f5.jpg",
            outfit_index=0,
            person_bbox=(0.0, 0.0, 0.5, 0.5),
        ),
    ]
    canonical = _canonical(0, outfit=_outfit(), members=members)
    enriched = EnrichedContentItem(
        normalized=NormalizedContentItem(
            source=ContentSource.INSTAGRAM,
            source_post_id="p1",
            text_blob="t",
            hashtags=[],
            image_urls=[],
            # video_urls 가 stem 'picture' 인 영상이 있어도 image_id 는 .jpg 이라
            # video frame 패턴에 안 잡혀야 false-positive 차단.
            video_urls=["https://x.blob.core.windows.net/c/picture.mp4"],
            post_date=datetime(2026, 4, 28, 12, 0, 0),
            engagement_raw_count=0,
            account_followers=0,
        ),
        garment_type=GarmentType.KURTA_SET,
        fabric=Fabric.COTTON,
        technique=Technique.BLOCK_PRINT,
        occasion=Occasion.CASUAL,
        canonicals=[canonical],
        post_palette=[],
        classification_method_per_attribute={},
    )

    rows = build_object_rows(enriched, computed_at="2026-04-28 09:00:00")
    assert rows[0]["media_ref"] is None


def test_build_object_rows_video_frame_id_no_match_when_video_urls_empty() -> None:
    """video frame id 인데 video_urls 비면 NULL — IG 정적 포스트 케이스."""
    members = [
        OutfitMember(
            image_id="reel999_f12",
            outfit_index=0,
            person_bbox=(0.0, 0.0, 0.5, 0.5),
        ),
    ]
    canonical = _canonical(0, outfit=_outfit(), members=members)
    enriched = EnrichedContentItem(
        normalized=NormalizedContentItem(
            source=ContentSource.INSTAGRAM,
            source_post_id="p1",
            text_blob="t",
            hashtags=[],
            image_urls=["https://x.blob.core.windows.net/c/static.jpg"],
            video_urls=[],
            post_date=datetime(2026, 4, 28, 12, 0, 0),
            engagement_raw_count=0,
            account_followers=0,
        ),
        garment_type=GarmentType.KURTA_SET,
        fabric=Fabric.COTTON,
        technique=Technique.BLOCK_PRINT,
        occasion=Occasion.CASUAL,
        canonicals=[canonical],
        post_palette=[],
        classification_method_per_attribute={},
    )

    rows = build_object_rows(enriched, computed_at="2026-04-28 09:00:00")
    assert rows[0]["media_ref"] is None


def test_build_object_rows_media_ref_none_when_image_urls_empty() -> None:
    # YT 는 image_urls=[] (spec §7.2). YT canonical 멤버 → media_ref NULL.
    members = [
        OutfitMember(
            image_id="some.jpg",
            outfit_index=0,
            person_bbox=(0.0, 0.0, 0.5, 0.5),
        ),
    ]
    canonical = _canonical(0, outfit=_outfit(), members=members)
    enriched = EnrichedContentItem(
        normalized=NormalizedContentItem(
            source=ContentSource.YOUTUBE,
            source_post_id="v1",
            text_blob="t",
            hashtags=[],
            image_urls=[],
            post_date=datetime(2026, 4, 27, 12, 0, 0),
            engagement_raw_count=0,
            account_followers=0,
        ),
        garment_type=GarmentType.KURTA_SET,
        fabric=Fabric.COTTON,
        technique=Technique.BLOCK_PRINT,
        occasion=Occasion.CASUAL,
        canonicals=[canonical],
        post_palette=[],
        classification_method_per_attribute={},
    )

    rows = build_object_rows(enriched, computed_at="2026-04-27 09:00:00")
    assert rows[0]["media_ref"] is None


def test_build_group_rows_styling_combo_per_canonical() -> None:
    # canonical 별 styling_combo 단일값: dress_as_single → STANDALONE,
    # lower=jeans → WITH_JEANS, 매핑 표 외 (dupatta) → None.
    c_standalone = _canonical(
        0,
        outfit=EthnicOutfit(
            person_bbox=(0.1, 0.1, 0.5, 0.7),
            person_bbox_area_ratio=0.35,
            upper_garment_type="saree",
            lower_garment_type=None,
            dress_as_single=True,
            color_preset_picks_top3=[],
        ),
    )
    c_jeans = _canonical(
        1,
        outfit=_outfit(upper="kurta", lower="jeans"),
    )
    c_unknown = _canonical(
        2,
        outfit=_outfit(upper="kurta", lower="dupatta"),
    )
    enriched = _enriched([c_standalone, c_jeans, c_unknown])

    rows = build_group_rows(enriched, computed_at="2026-04-27 09:00:00")
    assert rows[0]["styling_combo"] == "standalone"
    assert rows[1]["styling_combo"] == "with_jeans"
    # 매핑 표 외 lower 는 None — fuzzy 추정 금지.
    assert rows[2]["styling_combo"] is None


def test_build_object_rows_styling_combo_distributed_to_members() -> None:
    # canonical 단일값 (representative 기반) 을 멤버 전체에 동일하게 배포.
    members = [
        OutfitMember(
            image_id="img_0",
            outfit_index=0,
            person_bbox=(0.0, 0.0, 0.5, 0.5),
        ),
        OutfitMember(
            image_id="img_1",
            outfit_index=1,
            person_bbox=(0.0, 0.0, 0.4, 0.4),
        ),
    ]
    canonical = _canonical(
        0,
        outfit=_outfit(upper="kurta", lower="churidar"),
        members=members,
    )
    enriched = _enriched([canonical])

    rows = build_object_rows(enriched, computed_at="2026-04-27 09:00:00")
    assert len(rows) == 2
    assert rows[0]["styling_combo"] == "with_churidar"
    assert rows[1]["styling_combo"] == "with_churidar"


def test_build_item_row_styling_combo_text_vision_combine() -> None:
    # text-LLM styling_combo (rule weight=6) + vision (with_palazzo) 합산.
    canonical = _canonical(0, outfit=_outfit(upper="kurta", lower="palazzo"))
    enriched = EnrichedContentItem(
        normalized=_normalized(),
        garment_type=GarmentType.KURTA_SET,
        styling_combo=__import__(
            "contracts.common", fromlist=["StylingCombo"]
        ).StylingCombo.WITH_DUPATTA,
        canonicals=[canonical],
        classification_method_per_attribute={
            "styling_combo": ClassificationMethod.RULE,
        },
    )

    row = build_item_row(
        enriched,
        computed_at="2026-04-27 09:00:00",
        posted_at=None,
    )
    dist = row["styling_combo_dist"]
    assert dist is not None
    # 두 키 모두 등장, 합 = 1.0, text(rule=6) > vision share.
    assert set(dist.keys()) == {"with_dupatta", "with_palazzo"}
    assert abs(sum(dist.values()) - 1.0) < 1e-9
    assert dist["with_dupatta"] > dist["with_palazzo"]


def test_build_item_row_styling_combo_text_only_when_no_canonicals() -> None:
    from contracts.common import StylingCombo as _SC

    bare_normalized = NormalizedContentItem(
        source=ContentSource.INSTAGRAM,
        source_post_id="p_text_only",
        text_blob="",
        hashtags=[],
        image_urls=[],
        post_date=datetime(2026, 4, 27, 0, 0, 0),
        engagement_raw_count=0,
    )
    enriched = EnrichedContentItem(
        normalized=bare_normalized,
        styling_combo=_SC.STANDALONE,
        canonicals=[],
        classification_method_per_attribute={
            "styling_combo": ClassificationMethod.LLM,
        },
    )
    row = build_item_row(
        enriched,
        computed_at="2026-04-27 09:00:00",
        posted_at=None,
    )
    assert row["styling_combo_dist"] == {"standalone": 1.0}


def test_build_object_rows_count_matches_total_members() -> None:
    c0 = _canonical(
        0,
        outfit=_outfit(),
        members=[
            OutfitMember(image_id=f"img_{i}", outfit_index=0, person_bbox=(0.0, 0.0, 0.5, 0.5))
            for i in range(3)
        ],
    )
    c1 = _canonical(1, outfit=_outfit())  # default 1 member.
    enriched = _enriched([c0, c1])

    rows = build_object_rows(enriched, computed_at="2026-04-27 09:00:00")
    assert len(rows) == 4


# --------------------------------------------------------------------------- #
# build_representative_row

def test_build_representative_row_full_payload() -> None:
    aggregate = RepresentativeAggregate(
        representative_key="kurta_set__block_print__cotton",
        total_item_contribution=12.5,
        factor_contribution={
            ContentSource.INSTAGRAM: 0.7,
            ContentSource.YOUTUBE: 0.3,
        },
        member_count=8,
    )
    palette = [
        PaletteCluster(hex="#aabbcc", share=0.5, family=ColorFamily.NEUTRAL),
    ]
    distributions = {
        "garment_type": {"kurta_set": 1.0},
        "fabric": {"cotton": 1.0},
        "technique": {"block_print": 1.0},
        "silhouette": {"a_line": 1.0},
        "occasion": {"casual": 1.0},
        # styling_combo 누락 → NULL.
    }

    row = build_representative_row(
        aggregate,
        week_start_date="2026-04-27",
        computed_at="2026-04-27 09:00:00",
        score_total=72.5,
        score_breakdown={"social": 30.0, "youtube": 20.0, "cultural": 12.5, "momentum": 10.0},
        lifecycle_stage="growth",
        weekly_change_pct=8.2,
        weekly_direction="up",
        color_palette=palette,
        distributions=distributions,
        evidence_ig_post_ids=["p1", "p2"],
        evidence_yt_video_ids=["yt1"],
        trajectory=[0.0] * 11 + [72.5],
        effective_item_count=42.5,
        display_name="Cotton Block Print Kurta Set",
    )

    # representative_id deterministic.
    assert row["representative_id"] == representative_id(
        "kurta_set__block_print__cotton"
    )
    assert row["week_start_date"] == "2026-04-27"
    assert row["granularity"] == "weekly"
    assert row["representative_key"] == "kurta_set__block_print__cotton"
    assert row["display_name"] == "Cotton Block Print Kurta Set"
    assert row["score_total"] == 72.5
    assert row["score_breakdown"] == {
        "social": 30.0,
        "youtube": 20.0,
        "cultural": 12.5,
        "momentum": 10.0,
    }
    assert row["lifecycle_stage"] == "growth"
    assert row["weekly_change_pct"] == 8.2
    assert row["weekly_direction"] == "up"

    # ContentSource enum → str 키.
    assert row["factor_contribution"] == {"instagram": 0.7, "youtube": 0.3}
    assert sum(row["factor_contribution"].values()) == 1.0

    assert row["evidence_ig_post_ids"] == ["p1", "p2"]
    assert row["evidence_yt_video_ids"] == ["yt1"]

    assert row["color_palette"] == [
        {"hex": "#aabbcc", "share": 0.5, "family": "neutral"},
    ]
    assert row["garment_type_distribution"] == {"kurta_set": 1.0}
    assert row["fabric_distribution"] == {"cotton": 1.0}
    assert row["technique_distribution"] == {"block_print": 1.0}
    assert row["silhouette_distribution"] == {"a_line": 1.0}
    assert row["occasion_distribution"] == {"casual": 1.0}
    # 누락된 키 → NULL.
    assert row["styling_combo_distribution"] is None

    assert row["trajectory"] == [0.0] * 11 + [72.5]
    assert len(row["trajectory"]) == 12
    assert row["total_item_contribution"] == 12.5
    assert row["effective_item_count"] == 42.5
    assert row["schema_version"] == SCHEMA_VERSION


def test_build_representative_row_empty_evidence_and_palette_nullable() -> None:
    aggregate = RepresentativeAggregate(
        representative_key="anarkali__chikankari__chanderi",
        total_item_contribution=1.0,
        factor_contribution={ContentSource.INSTAGRAM: 1.0, ContentSource.YOUTUBE: 0.0},
        member_count=1,
    )
    row = build_representative_row(
        aggregate,
        week_start_date="2026-04-27",
        computed_at="2026-04-27 09:00:00",
        score_total=None,
        score_breakdown=None,
        lifecycle_stage=None,
        weekly_change_pct=None,
        weekly_direction=None,
        color_palette=[],
        distributions={},
        evidence_ig_post_ids=[],
        evidence_yt_video_ids=[],
        trajectory=[0.0] * 12,
    )

    assert row["color_palette"] is None
    assert row["evidence_ig_post_ids"] == []
    assert row["evidence_yt_video_ids"] == []
    # 모든 distribution 누락 → 모두 NULL.
    for key in (
        "garment_type_distribution",
        "fabric_distribution",
        "technique_distribution",
        "silhouette_distribution",
        "occasion_distribution",
        "styling_combo_distribution",
    ):
        assert row[key] is None, key
    # display_name optional default.
    assert row["display_name"] is None
    # β1: effective_item_count default = None (호출자가 안 넘기면 NULL 적재).
    assert row["effective_item_count"] is None
