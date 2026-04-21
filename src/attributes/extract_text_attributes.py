"""Stage 2a: 룰 기반 속성 추출 (spec §6.1 + §6.2).

AttributeExtractionState 는 stages 2a → 2b 간 공용 중간 상태다. Pydantic 도메인 모델은 frozen
이라 부분 채움에 부적절하므로 mutable dataclass 로 보관하다가, 마지막에 to_enriched() 로
EnrichedContentItem 으로 동결한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeVar

from attributes.mapping_tables import (
    FABRIC_KEYWORD_INDEX,
    FABRIC_TAG_INDEX,
    GARMENT_KEYWORD_INDEX,
    GARMENT_TAG_INDEX,
    OCCASION_KEYWORD_INDEX,
    OCCASION_TAG_INDEX,
    STYLING_KEYWORD_INDEX,
    STYLING_TAG_INDEX,
    TECHNIQUE_KEYWORD_INDEX,
    TECHNIQUE_TAG_INDEX,
)
from contracts.common import (
    ClassificationMethod,
    EmbellishmentIntensity,
    Fabric,
    GarmentType,
    Occasion,
    StylingCombo,
    Technique,
)
from contracts.enriched import BrandInfo, ColorInfo, EnrichedContentItem
from contracts.normalized import NormalizedContentItem

T = TypeVar("T")


@dataclass
class AttributeExtractionState:
    """stages 2a/2b 중간 상태. 마지막에 to_enriched() 로 frozen EnrichedContentItem 생성."""
    normalized: NormalizedContentItem
    garment_type: GarmentType | None = None
    fabric: Fabric | None = None
    technique: Technique | None = None
    embellishment_intensity: EmbellishmentIntensity | None = None
    color: ColorInfo | None = None
    silhouette: None = None  # VLM 영역 (spec §7). 이 스켈레톤에서 채우지 않는다.
    occasion: Occasion | None = None
    styling_combo: StylingCombo | None = None
    brand: BrandInfo | None = None
    method_per_attribute: dict[str, ClassificationMethod] = field(default_factory=dict)

    def to_enriched(self, cluster_key: str | None) -> EnrichedContentItem:
        return EnrichedContentItem(
            normalized=self.normalized,
            garment_type=self.garment_type,
            fabric=self.fabric,
            technique=self.technique,
            embellishment_intensity=self.embellishment_intensity,
            color=self.color,
            silhouette=None,
            occasion=self.occasion,
            styling_combo=self.styling_combo,
            brand=self.brand,
            trend_cluster_key=cluster_key,
            classification_method_per_attribute=dict(self.method_per_attribute),
        )


# --------------------------------------------------------------------------- #
# Match helpers (generic over enum type)
# --------------------------------------------------------------------------- #

def _normalize_tag(raw_tag: str) -> str:
    return raw_tag.lstrip("#").lower()


def _match_by_hashtag(tags: list[str], tag_index: dict[str, T]) -> T | None:
    for tag in tags:
        hit = tag_index.get(_normalize_tag(tag))
        if hit is not None:
            return hit
    return None


def _match_by_keyword(text_lower: str, keyword_index: dict[str, T]) -> T | None:
    for kw, value in keyword_index.items():
        if kw in text_lower:
            return value
    return None


def _match(
    tags: list[str],
    text_lower: str,
    tag_index: dict[str, T],
    keyword_index: dict[str, T],
) -> T | None:
    return _match_by_hashtag(tags, tag_index) or _match_by_keyword(text_lower, keyword_index)


# --------------------------------------------------------------------------- #
# Embellishment intensity derivation (spec §4.1 ③)
# --------------------------------------------------------------------------- #

# "heavy" 는 데모 스코프 밖 — rule 로 할당하지 않는다. LLM 이 명시적으로 분류하면 들어옴.
_EVERYDAY_TECHNIQUES: frozenset[Technique] = frozenset({
    Technique.SOLID,
    Technique.SELF_TEXTURE,
    Technique.FLORAL_PRINT,
    Technique.GEOMETRIC_PRINT,
    Technique.DIGITAL_PRINT,
    Technique.PINTUCK,
    Technique.CHIKANKARI,
    Technique.BLOCK_PRINT,
    Technique.LACE_CUTWORK,
})
_FESTIVE_LITE_TECHNIQUES: frozenset[Technique] = frozenset({
    Technique.MIRROR_WORK,
    Technique.THREAD_EMBROIDERY,
    Technique.GOTA_PATTI,
    Technique.ETHNIC_MOTIF,
})


def _derive_embellishment_intensity(technique: Technique) -> EmbellishmentIntensity | None:
    if technique in _EVERYDAY_TECHNIQUES:
        return EmbellishmentIntensity.EVERYDAY
    if technique in _FESTIVE_LITE_TECHNIQUES:
        return EmbellishmentIntensity.FESTIVE_LITE
    return None


# --------------------------------------------------------------------------- #
# Stage 2a entry point
# --------------------------------------------------------------------------- #

def extract_rule_based(item: NormalizedContentItem) -> AttributeExtractionState:
    """spec §6.2 매핑으로 속성 채우기. brand / color / silhouette 은 rule 에서 안 잡는다."""
    state = AttributeExtractionState(normalized=item)
    tags = item.hashtags
    text = item.text_blob.lower()

    state.garment_type = _match(tags, text, GARMENT_TAG_INDEX, GARMENT_KEYWORD_INDEX)
    state.fabric = _match(tags, text, FABRIC_TAG_INDEX, FABRIC_KEYWORD_INDEX)
    state.technique = _match(tags, text, TECHNIQUE_TAG_INDEX, TECHNIQUE_KEYWORD_INDEX)
    state.occasion = _match(tags, text, OCCASION_TAG_INDEX, OCCASION_KEYWORD_INDEX)
    state.styling_combo = _match(tags, text, STYLING_TAG_INDEX, STYLING_KEYWORD_INDEX)
    if state.technique is not None:
        state.embellishment_intensity = _derive_embellishment_intensity(state.technique)

    # TODO(§4.1 ⑧): brand 는 brand registry (외부 리스트) 도입 후 추출. 현재 rule 에서 skip.
    for key, value in (
        ("garment_type", state.garment_type),
        ("fabric", state.fabric),
        ("technique", state.technique),
        ("occasion", state.occasion),
        ("styling_combo", state.styling_combo),
        ("embellishment_intensity", state.embellishment_intensity),
    ):
        if value is not None:
            state.method_per_attribute[key] = ClassificationMethod.RULE

    return state
