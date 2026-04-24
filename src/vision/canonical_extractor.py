"""Phase 3 canonical outfit pixel pool 추출기 — M3.A Step D Phase 3.

Pipeline B 재설계 (LLM-centric) 에서 BBOX 는 Phase 2 LLM (`VisionLLMClient.extract_garment`)
이 공급하고, 같은 post 내 동일 의상 병합은 Phase 4.5 (`outfit_dedup.dedup_post`) 가 담당한다.
이 모듈의 역할은 거기서 나온 CanonicalOutfit 의 members (image_id, outfit_index, person_bbox)
를 pixel 좌표로 변환해 crop → per-BBOX segformer → 2-layer skin drop → pixel union pool
을 만드는 것. KMeans (Phase 4) 는 이 pool 을 소비한다.

설계 원칙:
- pure function — rgb + analysis 가 입력, pixel ndarray 가 출력. Frame I/O / LLM 호출 /
  캐시는 호출부 (pipeline_b_adapter) 책임.
- contracts 에는 numpy 노출 X — `CanonicalOutfitPixels` 는 본 모듈 내부 dataclass.
- legacy `pipeline_b_extractor.extract_instances` 와 공존 — 양쪽 paths 동시에 살아있다가
  Phase 5 에서 adapter 가 canonical path 로 swap.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from contracts.common import ColorFamily
from contracts.vision import CanonicalOutfit, EthnicOutfit, GarmentAnalysis, OutfitMember
from settings import OutfitDedupConfig, VisionConfig
from vision.bbox_utils import normalized_xywh_to_pixel_xyxy
from vision.color_space import SkinDropConfig, drop_skin_2layer
from vision.outfit_dedup import dedup_post
from vision.pipeline_b_extractor import SegBundle, run_segformer
from vision.segformer_constants import SKIN_CLASS_IDS, WEAR_CLASS_IDS


@dataclass(frozen=True)
class CanonicalOutfitPixels:
    """canonical outfit 1개의 pool 된 pixel + 진단 정보.

    pooled_pixels 는 Phase 4 동적 k KMeans 입력. per_image_pixel_counts 는 smoke HTML 용
    (어느 image 가 몇 픽셀 기여했는지). skin_drop_* 은 2-layer drop 작동 검증용.
    """
    canonical_index: int
    representative: EthnicOutfit
    members_meta: list[OutfitMember]
    pooled_pixels: np.ndarray                  # shape (N, 3), dtype uint8
    per_image_pixel_counts: dict[str, int]
    skin_drop_primary_total: int
    skin_drop_secondary_total: int


def drop_small_outfits(
    analysis: GarmentAnalysis, min_area_ratio: float,
) -> GarmentAnalysis:
    """`person_bbox_area_ratio < min_area_ratio` outfit 제거.

    is_india_ethnic_wear 는 원본 유지 — outfits 가 전부 drop 돼도 True→False 로 전환하지
    않음. 검출 signal 자체는 있었다는 사실을 보존 (aggregation 쪽에서 outfit 0인 True post
    를 다룰 수 있게). 변경 없으면 동일 객체 반환.
    """
    kept = [
        o for o in analysis.outfits if o.person_bbox_area_ratio >= min_area_ratio
    ]
    if len(kept) == len(analysis.outfits):
        return analysis
    return GarmentAnalysis(
        is_india_ethnic_wear=analysis.is_india_ethnic_wear, outfits=kept,
    )


def _build_skin_drop_config(cfg: VisionConfig) -> SkinDropConfig:
    """VisionConfig → SkinDropConfig 조립. settings(core) ↔ color_space(vision) 분리 유지용."""
    return SkinDropConfig(
        lab_min=tuple(cfg.skin_lab_box.min),
        lab_max=tuple(cfg.skin_lab_box.max),
        secondary_drop_threshold_pct=cfg.skin_drop_threshold_pct,
        upper_ceiling_pct=cfg.skin_drop_upper_ceiling,
    )


def _extract_member_pixels(
    rgb: np.ndarray,
    person_bbox: tuple[float, float, float, float],
    bundle: SegBundle,
    skin_drop_cfg: SkinDropConfig,
) -> tuple[np.ndarray, int, int]:
    """member 1개 → cleaned pixel (N, 3) + (primary_drop, secondary_drop) count.

    person_bbox 가 너무 작아 crop 이 MIN_CROP_PX 미만이면 empty 반환 (skip signal).
    """
    h, w = rgb.shape[:2]
    pixel_box = normalized_xywh_to_pixel_xyxy(person_bbox, h, w)
    if pixel_box is None:
        return np.empty((0, 3), dtype=np.uint8), 0, 0
    x1, y1, x2, y2 = pixel_box
    crop_rgb = rgb[y1:y2, x1:x2]
    seg = run_segformer(bundle, crop_rgb)
    garment_mask = np.isin(seg, list(WEAR_CLASS_IDS))
    skin_mask = np.isin(seg, list(SKIN_CLASS_IDS))
    return drop_skin_2layer(crop_rgb, garment_mask, skin_mask, skin_drop_cfg)


def _pool_canonical(
    canonical: CanonicalOutfit,
    frame_rgb_map: dict[str, np.ndarray],
    bundle: SegBundle,
    skin_drop_cfg: SkinDropConfig,
) -> CanonicalOutfitPixels | None:
    """canonical.members 의 member 별 pixel 을 concat. empty pool → None."""
    pooled: list[np.ndarray] = []
    per_image: dict[str, int] = {}
    primary_total = 0
    secondary_total = 0
    for member in canonical.members:
        rgb = frame_rgb_map.get(member.image_id)
        if rgb is None:
            continue
        cleaned, primary, secondary = _extract_member_pixels(
            rgb, member.person_bbox, bundle, skin_drop_cfg,
        )
        primary_total += primary
        secondary_total += secondary
        if cleaned.shape[0] == 0:
            continue
        pooled.append(cleaned)
        per_image[member.image_id] = (
            per_image.get(member.image_id, 0) + cleaned.shape[0]
        )
    if not pooled:
        return None
    return CanonicalOutfitPixels(
        canonical_index=canonical.canonical_index,
        representative=canonical.representative,
        members_meta=list(canonical.members),
        pooled_pixels=np.concatenate(pooled, axis=0),
        per_image_pixel_counts=per_image,
        skin_drop_primary_total=primary_total,
        skin_drop_secondary_total=secondary_total,
    )


def extract_canonical_pixels(
    post_items: list[tuple[str, np.ndarray, GarmentAnalysis]],
    bundle: SegBundle,
    cfg: VisionConfig,
    dedup_cfg: OutfitDedupConfig,
    family_map: dict[str, ColorFamily],
) -> list[CanonicalOutfitPixels]:
    """post 1개 orchestrator — size drop → dedup → canonical 별 pixel pool.

    Args:
      post_items: [(image_id, rgb_hwc_uint8, GarmentAnalysis), ...]. 호출부가 Frame 로드
        + LLM 호출을 이미 수행해 넘긴다.
      bundle: SegBundle — YOLO 는 쓰지 않음, seg_processor/seg_model/device 만 사용.
      cfg: VisionConfig — skin_lab_box / skin_drop_* / min_person_bbox_area_ratio.
      dedup_cfg, family_map: Phase 4.5 dedup_post 에 전달.

    Returns:
      list[CanonicalOutfitPixels]. canonical_index asc 순. pooled_pixels 가 비는 canonical
      은 결과에서 제외 (Phase 4 KMeans 가 빈 입력을 받지 않게).
    """
    frame_rgb_map = {img_id: rgb for img_id, rgb, _ in post_items}
    filtered_items = [
        (img_id, drop_small_outfits(analysis, cfg.min_person_bbox_area_ratio))
        for img_id, _rgb, analysis in post_items
    ]
    canonicals = dedup_post(filtered_items, dedup_cfg, family_map)
    skin_drop_cfg = _build_skin_drop_config(cfg)
    out: list[CanonicalOutfitPixels] = []
    for canonical in canonicals:
        result = _pool_canonical(canonical, frame_rgb_map, bundle, skin_drop_cfg)
        if result is not None:
            out.append(result)
    return out
