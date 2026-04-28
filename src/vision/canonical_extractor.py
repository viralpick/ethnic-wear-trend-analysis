"""Phase 3 canonical outfit pixel pool 추출기 — β-hybrid per-object entry.

Pipeline B 재설계 (LLM-centric) 에서 BBOX 는 Phase 2 LLM (`VisionLLMClient.extract_garment`)
이 공급하고, 같은 post 내 동일 의상 병합은 Phase 4.5 (`outfit_dedup.dedup_post`) 가 담당한다.
이 모듈의 역할은 거기서 나온 CanonicalOutfit 의 members (image_id, outfit_index, person_bbox)
를 pixel 좌표로 변환해 crop → per-BBOX segformer → 2-layer skin drop → 멤버 별 ObjectPool
을 만드는 것. β-hybrid Phase 1 (`build_object_palette`) 가 이 pool 을 소비한다.

설계 원칙:
- pure function — rgb + analysis 가 입력, ObjectPool list 가 출력. Frame I/O / LLM 호출 /
  캐시는 호출부 (pipeline_b_adapter) 책임.
- contracts 에는 numpy 노출 X — `ObjectPool` 은 본 모듈 내부 dataclass.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

from contracts.common import ColorFamily
from contracts.vision import (
    CanonicalOutfit,
    EthnicOutfit,
    GarmentAnalysis,
    OutfitMember,
    is_ethnic_outfit,
)
from settings import OutfitDedupConfig, VisionConfig
from vision.bbox_utils import normalized_xywh_to_pixel_xyxy
from vision.color_space import SkinDropConfig, drop_skin_2layer
from vision.outfit_dedup import dedup_post
from vision.pipeline_b_extractor import SegBundle, run_segformer
from vision.segformer_constants import (
    DRESS_CLASS_IDS,
    LOWER_CLASS_IDS,
    SKIN_CLASS_IDS,
    UPPER_CLASS_IDS,
)


@dataclass(frozen=True)
class ObjectPool:
    """β-hybrid 재설계 (per-object) 의 오브젝트 1개 pool.

    오브젝트 = 1 OutfitMember (= 1 사진의 한 착장). BBOX 는 person 단위 1개 (Gemini
    스펙). picks 는 그 사진의 원본 EthnicOutfit `color_preset_picks_top3` — Phase 1
    β-hybrid 의 picks 입력. canonical 별 List[ObjectPool] 가 Phase 1/2 (per-object
    좌표 보존 머지) 의 단위.

    `frame_area` 는 멤버가 속한 이미지 frame 의 H×W (raw RGB ndarray 기준). β-hybrid
    Phase 1 의 weight 가 obj_pixel_count / frame_area × SCALE 로 frame normalize 되어
    같은 canonical 안에서도 obj 별 frame coverage 비율을 보존한다 (advisor A2 /
    2026-04-25 사용자 결정).
    """
    member: OutfitMember                       # 식별자 (image_id, outfit_index, bbox)
    rgb_pixels: np.ndarray                     # shape (N, 3), dtype uint8
    picks: list[str]                           # color_preset_picks_top3 from source outfit
    frame_area: int                            # 원본 frame 의 H × W
    skin_drop_primary: int
    skin_drop_secondary: int


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


def _select_wear_class_ids(rep: EthnicOutfit) -> frozenset[int]:
    """B1: Gemini ethnic 판정으로 segformer class pool 결정.

    규칙 (F-10, 2026-04-26 — segformer 의 의류 type 분리는 신뢰하지 않음):
      - dress_as_single=True + ethnic              → UPPER + LOWER + DRESS (전신 keep)
      - 2-piece + 양쪽 ethnic                       → UPPER + LOWER + DRESS (= WEAR_CLASS_IDS)
      - 2-piece + upper only ethnic                → UPPER + DRESS (segformer 가 saree top
        같은 ethnic 상의를 dress 로 분류해도 keep)
      - 2-piece + lower only ethnic                → LOWER + DRESS
      - 모두 False/None / dress_as_single non-eth  → frozenset() (pool 제외)

    핵심: segformer 의 dress vs upper-clothes 분리는 부정확 (Sridevi saree 의 maroon
    상의가 upper-clothes 로 분류돼 dress-only pool 에서 96% 손실). 액세서리 (hat/shoe/
    bag/belt/sunglasses) 는 `WEAR_KEEP` 에서 빠져 자동 drop, skin/hair 는 별도 drop —
    이 함수는 의류 클래스만 다룬다.

    None 은 보수적으로 False 취급 (`is_ethnic_outfit` 위임). WARNING 으로 가시화
    (CLAUDE.md #4 실패 숨김 금지).
    """
    if rep.dress_as_single and rep.upper_is_ethnic is None:
        log.warning(
            "canonical_ethnic_flag_missing dress_as_single=True upper_is_ethnic=None",
        )
    elif not rep.dress_as_single and (
        rep.upper_is_ethnic is None or rep.lower_is_ethnic is None
    ):
        log.warning(
            "canonical_ethnic_flag_missing upper=%s lower=%s — treated as False",
            rep.upper_is_ethnic, rep.lower_is_ethnic,
        )
    if not is_ethnic_outfit(rep):
        return frozenset()
    if rep.dress_as_single:
        return UPPER_CLASS_IDS | LOWER_CLASS_IDS | DRESS_CLASS_IDS
    upper = bool(rep.upper_is_ethnic)
    lower = bool(rep.lower_is_ethnic)
    if upper and lower:
        return UPPER_CLASS_IDS | LOWER_CLASS_IDS | DRESS_CLASS_IDS
    if upper:
        # 상의만 ethnic — 하의류 (pants/skirt) drop. dress 는 segformer 가 saree top
        # 같은 single-piece ethnic 의류를 dress 로 분류해도 빠지지 않게 keep.
        return UPPER_CLASS_IDS | DRESS_CLASS_IDS
    return LOWER_CLASS_IDS | DRESS_CLASS_IDS


def _build_skin_drop_config(cfg: VisionConfig) -> SkinDropConfig:
    """VisionConfig → SkinDropConfig 조립. settings(core) ↔ color_space(vision) 분리 유지용."""
    return SkinDropConfig(
        lab_min=tuple(cfg.skin_lab_box.min),
        lab_max=tuple(cfg.skin_lab_box.max),
        secondary_drop_threshold_pct=cfg.skin_drop_threshold_pct,
        upper_ceiling_pct=cfg.skin_drop_upper_ceiling,
        skin_dilate_iterations=cfg.skin_dilate_iterations,
    )


def _extract_member_pixels(
    rgb: np.ndarray,
    person_bbox: tuple[float, float, float, float],
    bundle: SegBundle,
    skin_drop_cfg: SkinDropConfig,
    wear_class_ids: frozenset[int],
) -> tuple[np.ndarray, int, int]:
    """member 1개 → cleaned pixel (N, 3) + (primary_drop, secondary_drop) count.

    person_bbox 가 너무 작아 crop 이 MIN_CROP_PX 미만이면 empty 반환 (skip signal).
    wear_class_ids 는 B1 ethnic-aware pool — `_select_wear_class_ids` 결과를 pool 당 1회
    계산해 member 호출마다 재전달.
    """
    h, w = rgb.shape[:2]
    pixel_box = normalized_xywh_to_pixel_xyxy(person_bbox, h, w)
    if pixel_box is None:
        return np.empty((0, 3), dtype=np.uint8), 0, 0
    x1, y1, x2, y2 = pixel_box
    crop_rgb = rgb[y1:y2, x1:x2]
    seg = run_segformer(bundle, crop_rgb)
    garment_mask = np.isin(seg, list(wear_class_ids))
    skin_mask = np.isin(seg, list(SKIN_CLASS_IDS))
    return drop_skin_2layer(crop_rgb, garment_mask, skin_mask, skin_drop_cfg)


def _build_picks_lookup(
    filtered_items: list[tuple[str, GarmentAnalysis]],
) -> dict[tuple[str, int], list[str]]:
    """filtered (image_id, GarmentAnalysis) → (image_id, outfit_index) → picks.

    `dedup_post` 가 박는 `OutfitMember.outfit_index` 는 **filtered** outfits 기준
    (drop_small_outfits 거친 후) 이므로 lookup 도 동일 기준으로 build. 빈 picks 도
    list 로 보존 (Gemini 가 0~3 가변 — `EthnicOutfit.color_preset_picks_top3` 기본
    `default_factory=list`).
    """
    lookup: dict[tuple[str, int], list[str]] = {}
    for image_id, analysis in filtered_items:
        for outfit_index, outfit in enumerate(analysis.outfits):
            lookup[(image_id, outfit_index)] = list(outfit.color_preset_picks_top3)
    return lookup


def _lookup_member_picks(
    picks_lookup: dict[tuple[str, int], list[str]], member: OutfitMember,
) -> list[str]:
    """OutfitMember → 원본 EthnicOutfit 의 picks. miss → raise.

    실패 숨김 금지 (CLAUDE.md #4) — dedup 정합성이 깨졌다는 신호이므로
    representative.picks 로 silent fallback 하지 않음.
    """
    key = (member.image_id, member.outfit_index)
    if key not in picks_lookup:
        raise KeyError(
            f"OutfitMember picks lookup miss: image_id={member.image_id!r} "
            f"outfit_index={member.outfit_index} — dedup ↔ filtered analysis 정합성 깨짐",
        )
    return picks_lookup[key]


def _pool_canonical_per_object(
    canonical: CanonicalOutfit,
    frame_rgb_map: dict[str, np.ndarray],
    picks_lookup: dict[tuple[str, int], list[str]],
    bundle: SegBundle,
    skin_drop_cfg: SkinDropConfig,
) -> list[ObjectPool]:
    """canonical.members 각각 → ObjectPool 1개. 빈 pool member 는 skip.

    wear_class_ids 는 representative 기준 1회 결정 — 같은 canonical 안에서 ethnic
    분기는 정의상 일치한다.
    """
    wear_class_ids = _select_wear_class_ids(canonical.representative)
    if not wear_class_ids:
        return []
    pools: list[ObjectPool] = []
    for member in canonical.members:
        rgb = frame_rgb_map.get(member.image_id)
        if rgb is None:
            continue
        cleaned, primary, secondary = _extract_member_pixels(
            rgb, member.person_bbox, bundle, skin_drop_cfg, wear_class_ids,
        )
        if cleaned.shape[0] == 0:
            continue
        h, w = rgb.shape[:2]
        pools.append(
            ObjectPool(
                member=member, rgb_pixels=cleaned,
                picks=_lookup_member_picks(picks_lookup, member),
                frame_area=int(h * w),
                skin_drop_primary=primary, skin_drop_secondary=secondary,
            ),
        )
    return pools


def extract_canonical_pixels_per_object(
    post_items: list[tuple[str, np.ndarray, GarmentAnalysis]],
    bundle: SegBundle,
    cfg: VisionConfig,
    dedup_cfg: OutfitDedupConfig,
    family_map: dict[str, ColorFamily],
) -> list[tuple[CanonicalOutfit, list[ObjectPool]]]:
    """β-hybrid (per-object) 진입점 — post 1개 orchestrator.

    Args:
      post_items: [(image_id, rgb_hwc_uint8, GarmentAnalysis), ...]. 호출부가 Frame 로드
        + LLM 호출을 이미 수행해 넘긴다.
      bundle: SegBundle — YOLO 는 쓰지 않음, seg_processor/seg_model/device 만 사용.
      cfg: VisionConfig — skin_lab_box / skin_drop_* / min_person_bbox_area_ratio.
      dedup_cfg, family_map: Phase 4.5 dedup_post 에 전달.

    Returns:
      list[(CanonicalOutfit, list[ObjectPool])]. `canonical_index` asc 순. pool list 가
      비면 (non-ethnic / 전 멤버 background-only) canonical 은 라벨 보존용으로 함께 반환.
      pool empty 는 palette 계산 skip signal (빈 입력을 KMeans 에 먹이지 않음).
    """
    frame_rgb_map = {img_id: rgb for img_id, rgb, _ in post_items}
    filtered_items = [
        (img_id, drop_small_outfits(analysis, cfg.min_person_bbox_area_ratio))
        for img_id, _rgb, analysis in post_items
    ]
    picks_lookup = _build_picks_lookup(filtered_items)
    canonicals = dedup_post(filtered_items, dedup_cfg, family_map)
    skin_drop_cfg = _build_skin_drop_config(cfg)
    out: list[tuple[CanonicalOutfit, list[ObjectPool]]] = []
    for canonical in canonicals:
        pools = _pool_canonical_per_object(
            canonical, frame_rgb_map, picks_lookup, bundle, skin_drop_cfg,
        )
        out.append((canonical, pools))
    return out
