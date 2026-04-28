"""분석 결과 → StarRocks 4 테이블 row dict 변환 — pipeline_spec_v1.0 §1 / §2 / §5.

DDL (`ddl/01_item.sql` ~ `04_representative_weekly.sql`) 의 컬럼 ↔ in-memory contract
매핑을 한 곳에서 다룬다. 실제 Stream Load 호출은 `StarRocksWriter` (Protocol) 측 책임.

설계 원칙:
- pure 함수 4개 — 입력 dataclass/contract → `dict[str, Any]`. 시간/시스템 의존 X
  (`computed_at` / `posted_at` / `week_start_date` 는 caller 가 주입).
- JSON 컬럼은 native python 컨테이너로 채움 — Stream Load writer 가 직렬화 담당
  (`json.dumps`).
- enum 은 `.value` 문자열로 unwrap (StrEnum 이라 그대로도 직렬화 되지만 dict 비교 시
  StrEnum vs str 동등 비교가 미묘하므로 명시적으로 풀어둔다).
- representative_id = `blake2b(representative_key, digest_size=8)` big-endian signed
  BIGINT — `xxhash` 의존성 없이 deterministic 64-bit 키. literal byte test 로 고정.

알려진 갭 (후속 fix-up):
- canonical_group 의 attr 단일값은 `canonical.representative` 사용 (largest-area
  member). spec §2.6 의 "다수결 + tie-break" 정식 구현은 별도 work item.

7.6 에서 해소된 갭:
- canonical_object.color_palette: B1 canonical_extractor + pipeline_b_adapter 가
  `finalize_object_palette` 로 멤버별 top_n cap palette + cut_off_share 채움 (spec §6.5).

7.7 후속 fix 에서 해소된 갭:
- canonical_object.media_ref: `OutfitMember.image_id = path.name` (basename) ↔
  `normalized.image_urls` 의 URL basename 매칭으로 채움. 매칭 없으면 NULL.
- styling_combo: `derive_styling_from_outfit` (P0 5 매핑) 로 canonical 단위 단일값
  파생 → group/object 행 적재 + item.styling_combo_dist 합성 (text + vision).
  P1 (co_ord_set / with_dupatta / with_jacket) 은 prompt slot 부재로 별도 work item.
"""
from __future__ import annotations

import hashlib
import math
import re
from pathlib import PurePosixPath
from typing import Any

from attributes.derive_styling_from_vision import derive_styling_from_outfit
from contracts.common import PaletteCluster, Silhouette
from contracts.enriched import BrandInfo, EnrichedContentItem
from contracts.vision import CanonicalOutfit, OutfitMember

from aggregation.distribution_builder import (
    GroupSnapshot,
    build_distribution,
    group_to_item_contrib,
)
from aggregation.item_distribution_builder import enriched_to_item_distribution
from aggregation.representative_builder import (
    ItemDistribution,
    RepresentativeAggregate,
)


SCHEMA_VERSION = "pipeline_v1.0"
GRANULARITY_WEEKLY = "weekly"

# `vision/frame_source.py::VideoFrameSource` 가 만든 Frame.id = "{video_stem}_f{global_idx}".
# `OutfitMember.image_id` 가 이 패턴이면 image_urls 가 아니라 video_urls 의 stem 으로 매칭.
_VIDEO_FRAME_ID_RE = re.compile(r"^(.+)_f\d+$")


def representative_id(representative_key: str) -> int:
    """`representative_key` → signed BIGINT.

    xxhash 의존성 부재로 `blake2b(digest_size=8)` 8-byte big-endian → signed int64.
    StarRocks BIGINT 도 signed 64-bit 라 fits. blake2b 는 collision 확률 충분히 낮음
    (key cardinality ~수만~수십만, 64-bit space 1.8e19).
    """
    digest = hashlib.blake2b(
        representative_key.encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, byteorder="big", signed=True)


def _palette_to_json(clusters: list[PaletteCluster]) -> list[dict[str, Any]]:
    """PaletteCluster list → JSON-friendly dict list.

    PaletteCluster 는 frozen pydantic — `model_dump()` 로 hex/share/family 추출.
    family 는 ColorFamily StrEnum / None. mode='json' 로 enum → str 풀기.
    """
    return [c.model_dump(mode="json") for c in clusters]


def _brands_to_json(brands: list[BrandInfo]) -> list[dict[str, Any]] | None:
    """multi-brand list → JSON-friendly. 빈 list → None (DB NULL)."""
    if not brands:
        return None
    return [b.model_dump(mode="json") for b in brands]


def _bbox_to_json(bbox: tuple[float, float, float, float]) -> list[float]:
    return [bbox[0], bbox[1], bbox[2], bbox[3]]


def _enum_value(value: Any) -> Any:
    """StrEnum → str, None → None, 그 외는 그대로."""
    if value is None:
        return None
    if hasattr(value, "value"):
        return value.value
    return value


def _silhouette_groups(canonicals: list[CanonicalOutfit]) -> list[GroupSnapshot]:
    """silhouette distribution (vision_only) 입력. enum 값을 str 로 풀어 build_distribution
    에 전달 — None 은 그대로 (자연 drop)."""
    out: list[GroupSnapshot] = []
    for c in canonicals:
        sil: Silhouette | None = c.representative.silhouette
        out.append(
            GroupSnapshot(
                value=sil.value if sil is not None else None,
                n_objects=len(c.members),
                mean_area_ratio=_canonical_mean_area_ratio(c),
            )
        )
    return out


def _styling_combo_for_canonical(canonical: CanonicalOutfit) -> str | None:
    """canonical.representative → StylingCombo 파생 → str 또는 None."""
    combo = derive_styling_from_outfit(canonical.representative)
    return combo.value if combo is not None else None


def _styling_combo_groups(canonicals: list[CanonicalOutfit]) -> list[GroupSnapshot]:
    """styling_combo distribution 입력. canonical 단일값을 GroupSnapshot.value 에 풀어둠."""
    out: list[GroupSnapshot] = []
    for c in canonicals:
        out.append(
            GroupSnapshot(
                value=_styling_combo_for_canonical(c),
                n_objects=len(c.members),
                mean_area_ratio=_canonical_mean_area_ratio(c),
            )
        )
    return out


def _canonical_mean_area_ratio(canonical: CanonicalOutfit) -> float:
    """멤버 person_bbox area 평균 — item_distribution_builder 와 동일 정의 (spec §2.7)."""
    if not canonical.members:
        return 0.0
    total = 0.0
    for m in canonical.members:
        _, _, w, h = m.person_bbox
        total += w * h
    return total / len(canonical.members)


def build_item_row(
    enriched: EnrichedContentItem,
    *,
    computed_at: str,
    posted_at: str | None,
    item_distribution: ItemDistribution | None = None,
) -> dict[str, Any]:
    """`item` 테이블 1 row.

    `item_distribution` 미주입 시 enriched 로부터 즉석 변환 (편의). silhouette/styling_combo
    distribution 은 ItemDistribution 에 없어서 enriched 에서 직접 합성.
    """
    if item_distribution is None:
        item_distribution = enriched_to_item_distribution(enriched)

    normalized = enriched.normalized

    silhouette_dist = build_distribution(
        text_value=None,
        text_method=None,
        canonical_groups=_silhouette_groups(enriched.canonicals),
        vision_only=True,
    )

    method_map = enriched.classification_method_per_attribute
    styling_combo_text = (
        enriched.styling_combo.value if enriched.styling_combo is not None else None
    )
    styling_combo_dist = build_distribution(
        text_value=styling_combo_text,
        text_method=method_map.get("styling_combo"),
        canonical_groups=_styling_combo_groups(enriched.canonicals),
    )

    return {
        "source": normalized.source.value,
        "source_post_id": normalized.source_post_id,
        "computed_at": computed_at,
        "posted_at": posted_at,
        "garment_type_dist": dict(item_distribution.garment_type) or None,
        "fabric_dist": dict(item_distribution.fabric) or None,
        "technique_dist": dict(item_distribution.technique) or None,
        "silhouette_dist": dict(silhouette_dist) or None,
        "styling_combo_dist": dict(styling_combo_dist) or None,
        "occasion": _enum_value(enriched.occasion),
        "brands_mentioned": _brands_to_json(enriched.brands),
        "color_palette": _palette_to_json(enriched.post_palette) or None,
        "engagement_raw": normalized.engagement_raw,
        "account_handle": normalized.account_handle,
        "account_follower_count": normalized.account_followers or None,
        "schema_version": SCHEMA_VERSION,
    }


def _group_id(source: str, source_post_id: str, canonical_index: int) -> str:
    return f"{source}__{source_post_id}__{canonical_index}"


def build_group_rows(
    enriched: EnrichedContentItem,
    *,
    computed_at: str,
) -> list[dict[str, Any]]:
    """`canonical_group` rows — enriched.canonicals 길이 = N rows.

    attr 단일값은 `canonical.representative` 에서 (largest-area member). spec §2.6 정식
    다수결 구현은 별도 work item — 현재 representative 로 stand-in.
    item_contribution_score = spec §2.7 group_to_item_contrib(n, mean_area).
    """
    normalized = enriched.normalized
    src = normalized.source.value
    pid = normalized.source_post_id

    rows: list[dict[str, Any]] = []
    for c in enriched.canonicals:
        rep = c.representative
        n = len(c.members)
        mean_area = _canonical_mean_area_ratio(c)
        rows.append({
            "item_source": src,
            "item_source_post_id": pid,
            "canonical_index": c.canonical_index,
            "computed_at": computed_at,
            "group_id": _group_id(src, pid, c.canonical_index),
            "garment_type": rep.upper_garment_type,
            "fabric": rep.fabric,
            "technique": rep.technique,
            "silhouette": _enum_value(rep.silhouette),
            "styling_combo": _styling_combo_for_canonical(c),
            "color_palette": _palette_to_json(c.palette) or None,
            "item_contribution_score": group_to_item_contrib(n, mean_area),
            "n_objects": n,
            "mean_area_ratio": mean_area,
            "schema_version": SCHEMA_VERSION,
        })
    return rows


def _object_to_group_contrib(area_ratio: float) -> float:
    """spec §2.7 — log2(area×100+1). 단축 (객체 등장 1회 고정)."""
    return math.log2(max(0.0, area_ratio) * 100 + 1)


def _member_area_ratio(member: OutfitMember) -> float:
    _, _, w, h = member.person_bbox
    return w * h


def _object_id(group_id_str: str, member_index: int) -> str:
    return f"{group_id_str}__{member_index}"


def _resolve_media_ref(
    image_id: str,
    image_urls: list[str],
    video_urls: list[str],
) -> str | None:
    """`OutfitMember.image_id` ↔ image/video URL basename 매칭 → raw URL.

    Image: `pipeline_b_adapter._load_images` 가 `image_id = path.name` (확장자 포함).
    Video frame: `vision/frame_source.py::VideoFrameSource` 가
    `image_id = "{video_stem}_f{global_idx}"` (확장자 X). 이 패턴이면 video_urls 의
    stem 과 매칭. Azure Blob URL 도 path 의 마지막 컴포넌트가 동일 파일명. SAS
    query string (`?sv=...&sig=...`) 은 split 으로 제거.

    매칭 없으면 NULL — caller 책임 X (Stream Load JSON null).
    """
    for url in image_urls:
        path_only = url.split("?", 1)[0]
        if PurePosixPath(path_only).name == image_id:
            return url
    match = _VIDEO_FRAME_ID_RE.match(image_id)
    if match:
        video_stem = match.group(1)
        for url in video_urls:
            path_only = url.split("?", 1)[0]
            if PurePosixPath(path_only).stem == video_stem:
                return url
    return None


def build_object_rows(
    enriched: EnrichedContentItem,
    *,
    computed_at: str,
) -> list[dict[str, Any]]:
    """`canonical_object` rows — Σ canonicals 멤버 수.

    멤버별 garment_type/fabric/technique/silhouette 는 OutfitMember 에 carry-over 된
    원본 (Step 7.4a 결과). canonical_object 가 검수 가능한 raw attr 행.

    media_ref: OutfitMember.image_id (path basename) ↔ normalized.image_urls 의 basename
    매칭으로 raw URL 채움. 매칭 없으면 NULL.
    """
    normalized = enriched.normalized
    src = normalized.source.value
    pid = normalized.source_post_id
    image_urls = list(normalized.image_urls)
    video_urls = list(normalized.video_urls)

    rows: list[dict[str, Any]] = []
    for c in enriched.canonicals:
        gid = _group_id(src, pid, c.canonical_index)
        # styling_combo 는 canonical 단위 단일값 (representative 기반) — OutfitMember 에
        # lower_garment_type slot 부재. 멤버 전체에 동일값 배포.
        canonical_styling = _styling_combo_for_canonical(c)
        for m in c.members:
            area = _member_area_ratio(m)
            rows.append({
                "item_source": src,
                "item_source_post_id": pid,
                "canonical_index": c.canonical_index,
                "member_index": m.outfit_index,
                "computed_at": computed_at,
                "object_id": _object_id(gid, m.outfit_index),
                "group_id": gid,
                "media_ref": _resolve_media_ref(m.image_id, image_urls, video_urls),
                "garment_type": m.garment_type,
                "fabric": m.fabric,
                "technique": m.technique,
                "silhouette": _enum_value(m.silhouette),
                "styling_combo": canonical_styling,
                "color_palette": _palette_to_json(m.palette) or None,
                "area_ratio": area,
                "group_contribution_score": _object_to_group_contrib(area),
                "bbox": _bbox_to_json(m.person_bbox),
                "schema_version": SCHEMA_VERSION,
            })
    return rows


def build_representative_row(
    aggregate: RepresentativeAggregate,
    *,
    week_start_date: str,
    computed_at: str,
    score_total: float | None,
    score_breakdown: dict[str, float] | None,
    lifecycle_stage: str | None,
    weekly_change_pct: float | None,
    weekly_direction: str | None,
    color_palette: list[PaletteCluster],
    distributions: dict[str, dict[str, float] | None],
    evidence_ig_post_ids: list[str],
    evidence_yt_video_ids: list[str],
    trajectory: list[float],
    display_name: str | None = None,
) -> dict[str, Any]:
    """`representative_weekly` 1 row.

    Args:
      distributions: {silhouette/occasion/styling_combo/garment_type/fabric/technique
        → {value:pct} or None}. caller 가 6개 키 모두 전달 — 키 누락 시 NULL.
      color_palette: representative 단위 palette (spec §2.3 — caller 가 cluster 합성).
      trajectory: 최근 12주 score (부족분 0). 길이 12 권장.
      score_breakdown: {social, youtube, cultural, momentum} 부분 점수.
    """
    factor_contribution = {
        source.value: pct for source, pct in aggregate.factor_contribution.items()
    }

    return {
        "representative_id": representative_id(aggregate.representative_key),
        "week_start_date": week_start_date,
        "computed_at": computed_at,
        "representative_key": aggregate.representative_key,
        "display_name": display_name,
        "granularity": GRANULARITY_WEEKLY,
        "score_total": score_total,
        "score_breakdown": score_breakdown,
        "lifecycle_stage": lifecycle_stage,
        "weekly_change_pct": weekly_change_pct,
        "weekly_direction": weekly_direction,
        "factor_contribution": factor_contribution,
        "evidence_ig_post_ids": list(evidence_ig_post_ids),
        "evidence_yt_video_ids": list(evidence_yt_video_ids),
        "color_palette": _palette_to_json(color_palette) or None,
        "silhouette_distribution": distributions.get("silhouette") or None,
        "occasion_distribution": distributions.get("occasion") or None,
        "styling_combo_distribution": distributions.get("styling_combo") or None,
        "garment_type_distribution": distributions.get("garment_type") or None,
        "fabric_distribution": distributions.get("fabric") or None,
        "technique_distribution": distributions.get("technique") or None,
        "trajectory": list(trajectory),
        "total_item_contribution": aggregate.total_item_contribution,
        "schema_version": SCHEMA_VERSION,
    }
