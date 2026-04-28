"""EnrichedContentItem → ItemDistribution — pipeline_spec_v1.0 §2.1 + §2.4 사이 어댑터.

text 단일값 (post-level rule/LLM 추출) + vision multi-canonical 결과를 한 item 의 G/T/F
distribution 으로 합성. distribution_builder.build_distribution 을 G/T/F 각 attribute 마다
한 번씩 호출.

매핑 결정 (M3.I + spec §2.6, advisor 2026-04-27):
- garment_type group value = `canonical.representative.upper_garment_type`. dress_as_single
  여부와 무관 (dress 면 upper 슬롯이 곧 dress 자체).
- lower_garment_type 은 styling_combo 파생 입력 — distribution 에 미사용.
- fabric / technique group value = `canonical.representative.fabric` / `.technique`.
  upper/lower 는 spec §2.6 에서 단일값 가정 (canonical 단위).
- value=None canonical 은 자연 drop (build_distribution 내부에서 처리).
- text + vision 모두 빈 dict → ItemDistribution 의 해당 attribute = {} (빈 dict).
"""
from __future__ import annotations

from contracts.enriched import EnrichedContentItem
from contracts.vision import CanonicalOutfit, is_canonical_ethnic

from aggregation.distribution_builder import GroupSnapshot, build_distribution
from aggregation.representative_builder import ItemDistribution


def _canonical_area_ratio(canonical: CanonicalOutfit) -> float:
    """canonical 의 group_to_item_contrib 입력 — 멤버 person_bbox area 평균.

    OutfitMember.person_bbox = (x, y, w, h) normalized [0..1] → area = w * h.
    representative.person_bbox_area_ratio 와 분리 — 멤버 평균이 spec §2.7 의 area 항.
    """
    if not canonical.members:
        return 0.0
    total = 0.0
    for m in canonical.members:
        _, _, w, h = m.person_bbox
        total += w * h
    return total / len(canonical.members)


def _group_snapshot_for_garment(canonical: CanonicalOutfit) -> GroupSnapshot:
    return GroupSnapshot(
        value=canonical.representative.upper_garment_type,
        n_objects=len(canonical.members),
        mean_area_ratio=_canonical_area_ratio(canonical),
    )


def _group_snapshot_for_fabric(canonical: CanonicalOutfit) -> GroupSnapshot:
    return GroupSnapshot(
        value=canonical.representative.fabric,
        n_objects=len(canonical.members),
        mean_area_ratio=_canonical_area_ratio(canonical),
    )


def _group_snapshot_for_technique(canonical: CanonicalOutfit) -> GroupSnapshot:
    return GroupSnapshot(
        value=canonical.representative.technique,
        n_objects=len(canonical.members),
        mean_area_ratio=_canonical_area_ratio(canonical),
    )


def enriched_to_item_distribution(enriched: EnrichedContentItem) -> ItemDistribution:
    """spec §2.4 의 ItemDistribution 입력으로 변환.

    item_id = `{source}__{source_post_id}` (raw, hash 변환은 row_builder 단계에서).
    item_base_unit = 1.0 default — engagement 가중은 후속 작업.
    """
    normalized = enriched.normalized
    item_id = f"{normalized.source.value}__{normalized.source_post_id}"

    # 비-ethnic canonical 은 representative_weekly contribution 에서 차단 (2026-04-28).
    # canonical_extractor 의 라벨 보존 디자인 (pool=[] 인 채로 enriched 에 살아남음) 이
    # 이 단계에서 representative 매칭에 새면 비-ethnic 의 garment/fabric/technique 분포가
    # representative 점수에 섞여 운영 시그널 오염. group/object 적재 (검수 대시보드용) 는
    # build_group_rows / build_object_rows 가 그대로 처리.
    canonicals = [c for c in enriched.canonicals if is_canonical_ethnic(c)]
    garment_groups = [_group_snapshot_for_garment(c) for c in canonicals]
    fabric_groups = [_group_snapshot_for_fabric(c) for c in canonicals]
    technique_groups = [_group_snapshot_for_technique(c) for c in canonicals]

    method_map = enriched.classification_method_per_attribute

    garment_text = enriched.garment_type.value if enriched.garment_type is not None else None
    fabric_text = enriched.fabric.value if enriched.fabric is not None else None
    technique_text = enriched.technique.value if enriched.technique is not None else None

    return ItemDistribution(
        item_id=item_id,
        source=normalized.source,
        garment_type=build_distribution(
            text_value=garment_text,
            text_method=method_map.get("garment_type"),
            canonical_groups=garment_groups,
        ),
        fabric=build_distribution(
            text_value=fabric_text,
            text_method=method_map.get("fabric"),
            canonical_groups=fabric_groups,
        ),
        technique=build_distribution(
            text_value=technique_text,
            text_method=method_map.get("technique"),
            canonical_groups=technique_groups,
        ),
    )
