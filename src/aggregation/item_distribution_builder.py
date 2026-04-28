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

from aggregation.distribution_builder import GroupSnapshot, build_distribution
from aggregation.representative_builder import ItemDistribution
from attributes.derive_styling_from_vision import derive_styling_from_outfit
from contracts.common import Silhouette
from contracts.enriched import EnrichedContentItem
from contracts.vision import CanonicalOutfit, is_canonical_ethnic


def canonical_mean_area_ratio(canonical: CanonicalOutfit) -> float:
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


# 하위 호환 alias (외부 호출자 점진 이동용).
_canonical_area_ratio = canonical_mean_area_ratio


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


def _silhouette_groups(canonicals: list[CanonicalOutfit]) -> list[GroupSnapshot]:
    """vision-only silhouette distribution 입력. enum.value 는 str 로 unwrap.
    None 은 그대로 (build_distribution 이 자연 drop)."""
    out: list[GroupSnapshot] = []
    for c in canonicals:
        sil: Silhouette | None = c.representative.silhouette
        out.append(
            GroupSnapshot(
                value=sil.value if sil is not None else None,
                n_objects=len(c.members),
                mean_area_ratio=canonical_mean_area_ratio(c),
            )
        )
    return out


def _styling_combo_groups(canonicals: list[CanonicalOutfit]) -> list[GroupSnapshot]:
    """canonical 단위 styling_combo (derive_styling_from_outfit) distribution 입력.
    canonical 단일값을 GroupSnapshot.value 에 풀어둠. value=None canonical 은 자연 drop."""
    out: list[GroupSnapshot] = []
    for c in canonicals:
        combo = derive_styling_from_outfit(c.representative)
        out.append(
            GroupSnapshot(
                value=combo.value if combo is not None else None,
                n_objects=len(c.members),
                mean_area_ratio=canonical_mean_area_ratio(c),
            )
        )
    return out


def build_silhouette_distribution(
    enriched: EnrichedContentItem,
) -> dict[str, float]:
    """post-level silhouette distribution (vision_only).

    canonical.representative.silhouette 단일값을 멤버수×area 가중으로 분배. text 채널
    없음 (silhouette 은 vision-only 속성). 빈 dict = 미기여.
    """
    return build_distribution(
        text_value=None,
        text_method=None,
        canonical_groups=_silhouette_groups(list(enriched.canonicals)),
        vision_only=True,
    )


def build_styling_combo_distribution(
    enriched: EnrichedContentItem,
) -> dict[str, float]:
    """post-level styling_combo distribution (text + vision blend).

    text 채널: post-level `enriched.styling_combo` enum 을 rule(6.0)/LLM(3.0) 가중치로.
    vision 채널: 각 canonical 의 `derive_styling_from_outfit` 결과를 group_to_item_contrib
    로 분배. 합산 후 정규화. 둘 다 0 → 빈 dict.
    """
    method_map = enriched.classification_method_per_attribute
    text_value = (
        enriched.styling_combo.value if enriched.styling_combo is not None else None
    )
    return build_distribution(
        text_value=text_value,
        text_method=method_map.get("styling_combo"),
        canonical_groups=_styling_combo_groups(list(enriched.canonicals)),
    )
