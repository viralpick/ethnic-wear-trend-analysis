"""Phase 4.5 — intra-post outfit dedup pinning tests.

focus:
- 같은 post 안 2 image 에 동일 의상 → canonical 1개 + members 2.
- 서로 다른 옷 (color_preset 겹침 <2 + garment_type 다름 + technique 다름) → canonical 2.
- 같은 image 내 2 outfit 은 signal 이 아무리 비슷해도 병합 금지 (transitive 포함).
- dress_as_single 브리징 (lehenga-as-single ↔ lehenga+choli).
- empty / non-ethnic-only post 는 빈 리스트.
"""
from __future__ import annotations

from contracts.common import ColorFamily, Silhouette
from contracts.vision import EthnicOutfit, GarmentAnalysis
from settings import OutfitDedupConfig
from vision.outfit_dedup import dedup_post


def _outfit(**overrides) -> EthnicOutfit:
    base = dict(
        person_bbox=(0.1, 0.1, 0.5, 0.7),
        person_bbox_area_ratio=0.35,
        upper_garment_type="kurta",
        lower_garment_type="palazzo",
        dress_as_single=False,
        silhouette=Silhouette.A_LINE,
        fabric="cotton",
        technique="chikankari",
        color_preset_picks_top3=["ivory", "saffron", "pool_05"],
    )
    base.update(overrides)
    return EthnicOutfit(**base)


def _wrap(outfits: list[EthnicOutfit]) -> GarmentAnalysis:
    return GarmentAnalysis(is_india_ethnic_wear=True, outfits=outfits)


def _cfg(**overrides) -> OutfitDedupConfig:
    return OutfitDedupConfig(**overrides)


# preset → family 매핑. dedup 테스트 편의상 hand-crafted (lab rule 분리).
_FAMILY_MAP: dict[str, ColorFamily] = {
    "ivory": ColorFamily.WHITE_ON_WHITE,
    "saffron": ColorFamily.BRIGHT,
    "pool_05": ColorFamily.NEUTRAL,
    "pool_12": ColorFamily.EARTH,
    "pool_20": ColorFamily.JEWEL,
    "rani_pink": ColorFamily.BRIGHT,
    "emerald": ColorFamily.JEWEL,
}


def test_same_outfit_across_two_images_merges() -> None:
    # carousel 의 2 image 모두에 동일 kurta+palazzo. signal 전부 매칭.
    o1 = _outfit(person_bbox_area_ratio=0.40)
    o2 = _outfit(person_bbox_area_ratio=0.30)
    result = dedup_post(
        [("img_0", _wrap([o1])), ("img_1", _wrap([o2]))],
        _cfg(),
        _FAMILY_MAP,
    )
    assert len(result) == 1
    canonical = result[0]
    assert canonical.canonical_index == 0
    # representative 는 더 큰 area (img_0)
    assert canonical.representative.person_bbox_area_ratio == 0.40
    assert len(canonical.members) == 2
    assert {m.image_id for m in canonical.members} == {"img_0", "img_1"}


def test_different_outfits_do_not_merge() -> None:
    # color_preset 겹침 0, garment_type 다름, technique 다름 → sim = 0 < threshold.
    o1 = _outfit(
        upper_garment_type="kurta",
        lower_garment_type="palazzo",
        technique="chikankari",
        color_preset_picks_top3=["ivory", "saffron", "pool_05"],
    )
    o2 = _outfit(
        upper_garment_type="saree",
        lower_garment_type=None,
        dress_as_single=True,
        technique="block_print",
        color_preset_picks_top3=["rani_pink", "emerald", "pool_20"],
    )
    result = dedup_post(
        [("img_0", _wrap([o1])), ("img_1", _wrap([o2]))],
        _cfg(),
        _FAMILY_MAP,
    )
    assert len(result) == 2
    # canonical_index 는 area_ratio desc — 둘 다 0.35 로 같으니 image_id asc
    assert result[0].members[0].image_id == "img_0"
    assert result[1].members[0].image_id == "img_1"


def test_same_image_two_outfits_never_merge_even_transitively() -> None:
    # A(img_0, idx=0) 와 C(img_0, idx=1) 은 signal 이 완전 동일해도 병합 금지.
    # B(img_1) 가 둘 다에 매칭돼도 transitive 차단 (component_images 교집합).
    a = _outfit(person_bbox_area_ratio=0.30)
    b = _outfit(person_bbox_area_ratio=0.40)
    c = _outfit(person_bbox_area_ratio=0.20)
    result = dedup_post(
        [("img_0", _wrap([a, c])), ("img_1", _wrap([b]))],
        _cfg(),
        _FAMILY_MAP,
    )
    # 결과: B ~ (A or C) 중 하나만 merged (transitive 차단), 나머지는 solo. 총 2 canonical.
    assert len(result) == 2
    # 두 canonical 의 member image_id 합치면 (img_0 A) + (img_1 B) + (img_0 C) 3 노드.
    all_members = [m for c in result for m in c.members]
    assert len(all_members) == 3
    # 같은 canonical 안에 img_0 두 outfit 이 공존하지 않음.
    for canonical in result:
        image_ids = [m.image_id for m in canonical.members]
        assert len(image_ids) == len(set(image_ids)), (
            f"canonical_index={canonical.canonical_index} 에 같은 image_id 중복: {image_ids}"
        )


def test_dress_as_single_bridging_matches_two_piece() -> None:
    # img_0: 단일 (lehenga-as-single) — upper="lehenga", dress_as_single=True
    # img_1: 2-piece (lehenga + choli) — upper="lehenga", lower="choli"
    # color/technique 매칭 시 garment_type 브리징 룰로 매치.
    single = _outfit(
        upper_garment_type="lehenga",
        lower_garment_type=None,
        dress_as_single=True,
        silhouette=None,
    )
    two_piece = _outfit(
        upper_garment_type="lehenga",
        lower_garment_type="choli",
        dress_as_single=False,
    )
    result = dedup_post(
        [("img_0", _wrap([single])), ("img_1", _wrap([two_piece]))],
        _cfg(),
        _FAMILY_MAP,
    )
    # color_preset 3/3, color_family 매칭, garment_type 브리징 매칭, technique 매칭 → sim=1.0
    assert len(result) == 1
    assert {m.image_id for m in result[0].members} == {"img_0", "img_1"}


def test_dominant_family_order_independent() -> None:
    # 같은 preset 집합이지만 순서가 다른 2 outfit. Counter.most_common 은 insertion-order
    # 의존 — ivory(WHITE_ON_WHITE) 1 + saffron(BRIGHT) 1 동률이면 먼저 들어온 쪽 승.
    # 명시적 total order (count desc, family value asc) 로 고친 후엔 같은 dominant family
    # → color_family 신호 매치 유지 → 병합 성공.
    o1 = _outfit(
        color_preset_picks_top3=["ivory", "saffron", "pool_05"],
        technique=None,  # technique=None 으로 만들어 sim 이 threshold 근처 확인
    )
    o2 = _outfit(
        color_preset_picks_top3=["saffron", "ivory", "pool_05"],
        technique=None,
    )
    # sim = color_preset(0.40, |{ivory,saffron,pool_05}|=3≥2) +
    #       color_family(0.25, dominant 같음) +
    #       garment_type(0.25, kurta+palazzo 동일) = 0.90 ≥ 0.60
    result = dedup_post(
        [("img_0", _wrap([o1])), ("img_1", _wrap([o2]))],
        _cfg(),
        _FAMILY_MAP,
    )
    assert len(result) == 1, (
        "preset 순서만 다른 동일 옷이 병합 안 되면 _dominant_family 가 순서 의존적"
    )
    assert {m.image_id for m in result[0].members} == {"img_0", "img_1"}


def test_member_carry_over_garment_fabric_technique_silhouette() -> None:
    # 7.4a: dedup 후에도 멤버별 raw attribute 보존 (canonical_object 행 검수용).
    # dress_as_single (silhouette=None, technique=block_print) ↔ two_piece (A_LINE, chikankari)
    # 가 같은 canonical 로 묶이지만 멤버별 attribute 는 그대로 원본 값.
    single = _outfit(
        upper_garment_type="lehenga",
        lower_garment_type=None,
        dress_as_single=True,
        silhouette=None,
        technique="block_print",
        fabric="silk",
    )
    two_piece = _outfit(
        upper_garment_type="lehenga",
        lower_garment_type="choli",
        dress_as_single=False,
        silhouette=Silhouette.A_LINE,
        technique="block_print",  # technique 매칭 유지 (브리징 신호)
        fabric="cotton",
    )
    result = dedup_post(
        [("img_0", _wrap([single])), ("img_1", _wrap([two_piece]))],
        _cfg(),
        _FAMILY_MAP,
    )
    assert len(result) == 1
    by_image = {m.image_id: m for m in result[0].members}

    assert by_image["img_0"].garment_type == "lehenga"
    assert by_image["img_0"].fabric == "silk"
    assert by_image["img_0"].technique == "block_print"
    assert by_image["img_0"].silhouette is None

    assert by_image["img_1"].garment_type == "lehenga"
    assert by_image["img_1"].fabric == "cotton"
    assert by_image["img_1"].technique == "block_print"
    assert by_image["img_1"].silhouette == Silhouette.A_LINE


def test_empty_and_non_ethnic_post_returns_empty() -> None:
    # non-ethnic analysis 는 skip. 전부 skip 이면 빈 리스트.
    empty_analysis = GarmentAnalysis(is_india_ethnic_wear=False, outfits=[])
    assert dedup_post([], _cfg(), _FAMILY_MAP) == []
    assert (
        dedup_post(
            [("img_0", empty_analysis), ("img_1", empty_analysis)],
            _cfg(),
            _FAMILY_MAP,
        )
        == []
    )
