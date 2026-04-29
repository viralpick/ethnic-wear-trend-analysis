"""Pipeline B — YOLOv8 person detect → segformer garment seg → LAB KMeans palette.

phase 3 (2026-04-22): instance 단위로 완전 재구성. (frame × person bbox × garment_class)
교집합을 GarmentInstance 1개로 보고 각자 독립 KMeans. post aggregate 는 instance 들을
duplicate 그룹으로 묶어 sub-linear weight 로 top-k 선별.

이 모듈은 top-level 로 torch / transformers / ultralytics 를 import. vision extras 미설치
시 ImportError — core 코드는 **절대 top-level import 금지** (ColorExtractor Protocol 뒤 DI).

spec §4.1 ④ / §7 대응. M3.G/H (2026-04-28) 이후 IG/YT 모두 frame_source 통해 동일
흐름으로 처리 — 이전 §7.2 의 "YT 는 thumbnail 만" 가정은 폐기됨.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from ultralytics import YOLO

from contracts.common import ColorPaletteItem
from settings import SceneFilterConfig, VisionConfig
from vision.color_family_preset import lab_to_family
from vision.color_space import (
    drop_skin_adaptive_spatial,
    extract_colors,
    hex_to_rgb,
)
from vision.frame_source import Frame, FrameSource
from vision.garment_instance import (
    GarmentInstance,
    aggregate_post_palette,
    classify_single_color,
)
from vision.scene_filter import FilterVerdict, NoopSceneFilter, PersonVerdict, SceneFilter

# ATR segformer class id / wear / skin 상수는 `vision.segformer_constants` single source.
# Phase 3 `canonical_extractor` 와 공유 — 이쪽에서 재정의하면 두 경로가 drift 할 위험.
from vision.segformer_constants import (  # noqa: E402
    ATR_LABELS,
    SKIN_CLASS_IDS,
    WEAR_CLASS_IDS,
    WEAR_KEEP,
)

MIN_CROP_PX = 32  # bbox 짧은 변이 이 값 미만이면 crop drop. smoke overlay 에서도 재사용.
_YOLO_CONF = 0.35
_YOLO_DEDUP_IOU = 0.45


@dataclass(frozen=True)
class SegBundle:
    """모델 번들. 한 번 load_models() 로 받아 여러 frame 에 재사용."""
    yolo: object
    seg_processor: object
    seg_model: object
    device: str
    scene_filter: SceneFilter  # disabled 면 NoopSceneFilter


def _pick_device(override: str | None) -> str:
    if override:
        return override
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_models(
    device: str | None = None,
    scene_filter_cfg: SceneFilterConfig | None = None,
) -> SegBundle:
    """YOLOv8n + segformer_b2_clothes + (optional) CLIPSceneFilter 로드.

    scene_filter_cfg 가 None 이거나 enabled=False 이면 NoopSceneFilter 부착. CLIPSceneFilter
    로드 경로는 lazy import (scene_filter_clip 모듈) — transformers 가 이미 vision extras
    에 있지만 CLIP 가중치 (~600MB) 다운로드는 enabled 때만.
    """
    dev = _pick_device(device)
    yolo = YOLO("yolov8n.pt")
    # Thread-safety pre-warm — ultralytics YOLO 의 첫 predict() 가 lazy 하게 setup_model
    # → fuse() (delattr 'bn') 호출. multi-thread 가 첫 predict 를 동시에 부르면 delattr
    # 가 두 번 일어나 'Conv' object has no attribute 'bn' AttributeError 발생
    # (vision-workers > 1 race). 32×32 dummy 으로 미리 1 회 predict 해서 fuse 끝낸 후
    # thread pool 진입.
    import numpy as np  # noqa: I001 — vision extras
    _warmup = np.zeros((32, 32, 3), dtype=np.uint8)
    yolo.predict(_warmup, classes=[0], conf=0.5, verbose=False)

    seg_processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
    seg_model = SegformerForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
    seg_model.eval()
    seg_model.to(dev)

    scene_filter: SceneFilter
    if scene_filter_cfg is not None and scene_filter_cfg.enabled:
        from vision.scene_filter_clip import load_clip_filter  # lazy
        scene_filter = load_clip_filter(scene_filter_cfg, dev)
    else:
        scene_filter = NoopSceneFilter()

    return SegBundle(
        yolo=yolo, seg_processor=seg_processor, seg_model=seg_model, device=dev,
        scene_filter=scene_filter,
    )


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def detect_people(
    yolo, rgb: np.ndarray,
) -> list[tuple[int, int, int, int]]:
    """YOLOv8 person detection (class 0) + IoU dedup → bbox 리스트."""
    result = yolo.predict(rgb, classes=[0], conf=_YOLO_CONF, verbose=False)[0]
    if result.boxes.conf is None:
        return []
    confs = result.boxes.conf.cpu().numpy()
    boxes = result.boxes.xyxy.cpu().numpy()
    scored = sorted(
        [
            ((int(b[0]), int(b[1]), int(b[2]), int(b[3])), float(c))
            for b, c in zip(boxes, confs)
        ],
        key=lambda x: -x[1],
    )
    kept: list[tuple[int, int, int, int]] = []
    for box, _ in scored:
        if all(_iou(box, k) < _YOLO_DEDUP_IOU for k in kept):
            kept.append(box)
    kept.sort(key=lambda b: b[0])
    return kept


def run_segformer(bundle: SegBundle, crop_rgb: np.ndarray) -> np.ndarray:
    """crop (H,W,3) uint8 → (H,W) int32 ATR label 배열."""
    from PIL import Image  # lazy

    crop_pil = Image.fromarray(crop_rgb)
    with torch.no_grad():
        inputs = bundle.seg_processor(images=crop_pil, return_tensors="pt").to(bundle.device)
        logits = bundle.seg_model(**inputs).logits
        upsampled = torch.nn.functional.interpolate(
            logits, size=crop_pil.size[::-1], mode="bilinear", align_corners=False,
        )
        return upsampled.argmax(dim=1)[0].cpu().numpy().astype(np.int32)


def _pixels_to_palette(
    combined: np.ndarray, cfg: VisionConfig,
) -> list[ColorPaletteItem]:
    """cleaned pixel concat → KMeans → ColorPaletteItem. pure helper.

    family 는 `color_family_preset.lab_to_family` rule 로 결정 (coarse 5-class).
    Phase 5 canonical path 와 legacy diagnostics 모두 같은 classifier 를 쓰게 해서
    HTML 비교 / Phase 4.5 dedup 과 drift 없도록 단일 source 유지.
    """
    colors = extract_colors(
        combined, k=cfg.extract_colors.k, min_pixels=cfg.extract_colors.min_pixels,
    )
    palette: list[ColorPaletteItem] = []
    for i, c in enumerate(colors):
        r, g, b = hex_to_rgb(c["hex"]).astype(int).tolist()
        L_lab, a_lab, b_lab = c["lab"]
        palette.append(
            ColorPaletteItem(
                r=int(r), g=int(g), b=int(b),
                hex_display=c["hex"],
                name=f"pipeline_b_{i}_{c['hex'].lstrip('#').lower()}",
                family=lab_to_family(float(L_lab), float(a_lab), float(b_lab)),
                pct=c["weight"],
            )
        )
    return palette


# --------------------------------------------------------------------------- #
# Phase 3: instance 추출
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class _ExtractCtx:
    """bbox 루프마다 재계산 피하려고 frame loop 진입 전 만드는 컨텍스트."""
    bundle: SegBundle
    cfg: VisionConfig
    lab_min: np.ndarray
    lab_max: np.ndarray


@dataclass(frozen=True)
class FrameDropRecord:
    """scene_filter Stage 1 에 의해 drop 된 frame 기록 — reason + prompt scores.

    HTML 에 bias 감사용으로 노출. frame_id 는 smoke 에서 원본 이미지 경로와 재매핑.
    """
    frame_id: str
    verdict: FilterVerdict


@dataclass(frozen=True)
class BBoxDropRecord:
    """scene_filter Stage 2 에 의해 drop 된 person BBOX 기록 — reason + prompt scores.

    stage=stage1_mix_needs_stage2 frame 에서만 발생. HTML bias 감사용.
    """
    frame_id: str
    bbox_idx: int
    verdict: PersonVerdict


@dataclass(frozen=True)
class ExtractStats:
    """extract_instances 부가 통계 — diagnostics 생성에 필요한 frame / bbox 단위 집계.

    filtered_out: Stage 1 frame drop. bbox_filtered_out: Stage 2 BBOX drop.
    """
    yolo_detected_persons: int
    fallback_triggered: bool
    filtered_out: list[FrameDropRecord]
    bbox_filtered_out: list[BBoxDropRecord]


def extract_instances(
    source: FrameSource, bundle: SegBundle, cfg: VisionConfig,
) -> tuple[list[GarmentInstance], ExtractStats]:
    """frame × person_bbox × garment_class 단위 GarmentInstance + scene filter drop 집계.

    Stage 1 (`scene_filter.accept`) 실패하면 frame skip (YOLO+segformer 미호출). pass 면
    stage 에 따라 BBOX 처리를 달리한다:
      - stage1_pass / disabled → 모든 bbox 진행 (기존 동작)
      - stage1_mix_needs_stage2 → YOLO detect 후 classify_persons 로 BBOX 재판정.
        passed BBOX 만 segformer 진행, 나머지는 BBoxDropRecord 로 기록
      - stage1_reject → frame drop (filtered_out 기록)

    Stage 2 토글은 SceneFilterConfig.stage2_enabled 가 아닌 pipeline_b_extractor 호출부에서
    강제할 수 있도록 cfg.scene_filter.stage2_enabled 체크. False 면 stage=mix 여도 bbox
    전체 진행 (Stage 2 옵트아웃 경로).

    단일 iter_frames 순회 — `ImageFrameSource.iter_frames()` 는 매 호출마다 PIL 로 이미지를
    다시 열어 disk 재로딩 + YOLO 이중 추론 위험이 있어 diagnostics 가 별도 순회하지 않음.
    """
    ctx = _ExtractCtx(
        bundle=bundle, cfg=cfg,
        lab_min=np.asarray(cfg.skin_lab_box.min, dtype=np.float32),
        lab_max=np.asarray(cfg.skin_lab_box.max, dtype=np.float32),
    )
    instances: list[GarmentInstance] = []
    yolo_count = 0
    fallback = False
    filtered: list[FrameDropRecord] = []
    bbox_filtered: list[BBoxDropRecord] = []
    stage2_enabled = cfg.scene_filter.stage2_enabled
    for frame in source.iter_frames():
        verdict = bundle.scene_filter.accept(frame.rgb, frame.id)
        if not verdict.passed:
            filtered.append(FrameDropRecord(frame_id=frame.id, verdict=verdict))
            continue
        boxes = detect_people(bundle.yolo, frame.rgb)
        yolo_count += len(boxes)
        if not boxes and cfg.fallback_full_image_on_no_person:
            fallback = True
            h, w = frame.rgb.shape[:2]
            boxes = [(0, 0, w, h)]
        # Stage 2 분기: mix signal + 토글 켜짐 + bbox 존재할 때만 per-bbox 판정.
        # Edge case: YOLO 미탐으로 fallback bbox (전체 이미지) 가 주입된 경우에도 분기 탐.
        # fallback bbox 는 per-person 의미가 퇴색되지만 stage2 의 stricter threshold 에서
        # drop 될 가능성이 높아 "애매한 frame 은 보수적 drop" 이라는 안전 동작.
        if verdict.stage == "stage1_mix_needs_stage2" and stage2_enabled and boxes:
            person_verdicts = bundle.scene_filter.classify_persons(frame.rgb, boxes)
            kept: list[tuple[int, tuple[int, int, int, int]]] = []
            for idx, pv in enumerate(person_verdicts):
                if pv.passed:
                    kept.append((idx, pv.bbox))
                else:
                    bbox_filtered.append(BBoxDropRecord(
                        frame_id=frame.id, bbox_idx=idx, verdict=pv,
                    ))
            for bbox_idx, bbox in kept:
                instances.extend(_instances_from_bbox(frame, bbox_idx, bbox, ctx))
        else:
            for bbox_idx, bbox in enumerate(boxes):
                instances.extend(_instances_from_bbox(frame, bbox_idx, bbox, ctx))
    stats = ExtractStats(
        yolo_detected_persons=yolo_count,
        fallback_triggered=fallback,
        filtered_out=filtered,
        bbox_filtered_out=bbox_filtered,
    )
    return instances, stats


def _is_person_bbox(seg: np.ndarray, cfg: VisionConfig) -> bool:
    """segformer 결과 기반 실 사람 bbox 판정. skin pixel 비율 낮거나 의류 비율 극단적이면 False.

    - skin 비율 < min_skin_ratio_for_person → 동상/마네킹 또는 사람 없음 (전체 이미지 fallback)
    - 의류 비율 > max_garment_ratio_for_person → 배경까지 의류 오분류 (제품샷)
    """
    area = int(seg.size)
    if area == 0:
        return False
    skin_count = int(np.isin(seg, tuple(SKIN_CLASS_IDS)).sum())
    garment_count = int(np.isin(seg, tuple(WEAR_CLASS_IDS)).sum())
    skin_ratio = skin_count / area
    garment_ratio = garment_count / area
    if skin_ratio < cfg.min_skin_ratio_for_person:
        return False
    if garment_ratio > cfg.max_garment_ratio_for_person:
        return False
    return True


def _instances_from_bbox(
    frame: Frame,
    bbox_idx: int,
    bbox: tuple[int, int, int, int],
    ctx: _ExtractCtx,
) -> list[GarmentInstance]:
    """하나의 bbox → (그 안의 garment_class 별) GarmentInstance 리스트."""
    cfg = ctx.cfg
    min_pixels = cfg.extract_colors.min_pixels
    x1, y1, x2, y2 = bbox
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(frame.rgb.shape[1], x2), min(frame.rgb.shape[0], y2)
    if x2c - x1c < MIN_CROP_PX or y2c - y1c < MIN_CROP_PX:
        return []
    crop_rgb = frame.rgb[y1c:y2c, x1c:x2c]
    seg = run_segformer(ctx.bundle, crop_rgb)
    # false-positive filter: 동상/마네킹/제품샷 방어. segformer 결과로 판정.
    if not _is_person_bbox(seg, cfg):
        return []
    out: list[GarmentInstance] = []
    skin_class_mask = np.isin(seg, list(SKIN_CLASS_IDS))
    for class_id, label in ATR_LABELS.items():
        if label not in WEAR_KEEP:
            continue
        garment_mask = seg == class_id
        if int(garment_mask.sum()) < min_pixels:
            continue
        cleaned, ratio, _kept = drop_skin_adaptive_spatial(
            crop_rgb, garment_mask, skin_class_mask,
            lab_min=ctx.lab_min, lab_max=ctx.lab_max,
            keep_threshold_pct=cfg.skin_drop_threshold_pct,
            upper_ceiling_pct=cfg.skin_drop_upper_ceiling,
            skin_dilate_iterations=cfg.skin_dilate_iterations,
        )
        if cleaned.shape[0] < min_pixels:
            continue
        palette = _pixels_to_palette(cleaned, cfg)
        if not palette:
            continue
        out.append(GarmentInstance(
            instance_id=f"{frame.id}:p{bbox_idx}:{label}",
            frame_id=frame.id,
            bbox=(x1c, y1c, x2c, y2c),
            garment_class=label,
            palette=palette,
            is_single_color=classify_single_color(
                palette, max_delta_e=cfg.instance.single_color_max_delta_e,
            ),
            pixel_count=int(cleaned.shape[0]),
            skin_drop_ratio=ratio,
        ))
    return out


# --------------------------------------------------------------------------- #
# Frame / post palette — instance 기반 재구성
# --------------------------------------------------------------------------- #

def _frame_palette_from_instances(
    instances: list[GarmentInstance], frame_id: str, top_k: int,
) -> "FramePalette":
    """frame 의 instance chip 들을 weight 없이 (pct × pixel_count) 로 top-k 선별."""
    frame_inst = [i for i in instances if i.frame_id == frame_id]
    if not frame_inst:
        return FramePalette(
            frame_id=frame_id, palette=[],
            garment_pixel_counts={}, skin_drop_ratios={},
        )
    scored: list[tuple[ColorPaletteItem, float]] = []
    for inst in frame_inst:
        for chip in inst.palette:
            scored.append((chip, chip.pct * inst.pixel_count))
    scored.sort(key=lambda x: -x[1])
    top = scored[: top_k]
    total = sum(s for _, s in top)
    palette = [
        ColorPaletteItem(
            r=c.r, g=c.g, b=c.b, hex_display=c.hex_display,
            name=c.name, family=c.family,
            pct=s / total if total > 0 else 0.0,
        )
        for c, s in top
    ]
    pixel_counts: dict[str, int] = {}
    drop_ratios: dict[str, float] = {}
    for inst in frame_inst:
        pixel_counts[inst.garment_class] = (
            pixel_counts.get(inst.garment_class, 0) + inst.pixel_count
        )
        drop_ratios[inst.garment_class] = inst.skin_drop_ratio
    return FramePalette(
        frame_id=frame_id, palette=palette,
        garment_pixel_counts=pixel_counts, skin_drop_ratios=drop_ratios,
    )


def extract_palette(
    source: FrameSource, bundle: SegBundle, cfg: VisionConfig,
) -> list[ColorPaletteItem]:
    """instance 기반 post-level aggregate palette (phase 3 backwards-compatible wrapper)."""
    instances, _stats = extract_instances(source, bundle, cfg)
    palette, _groups = aggregate_post_palette(
        instances,
        top_k=cfg.extract_colors.k,
        duplicate_max_delta_e=cfg.instance.duplicate_max_delta_e,
        weight_formula=cfg.instance.weight_formula,
    )
    return palette


@dataclass(frozen=True)
class FramePalette:
    """frame 1장의 aggregate palette — instance chip 들을 선별 (phase 3 재구성)."""
    frame_id: str
    palette: list[ColorPaletteItem]
    garment_pixel_counts: dict[str, int]
    skin_drop_ratios: dict[str, float]


@dataclass(frozen=True)
class ExtractionDiagnostics:
    """phase 3 — instance 중심 + post/frame aggregate 모두 노출."""
    palette: list[ColorPaletteItem]               # post-level aggregate (duplicate weighted)
    frame_palettes: list[FramePalette]            # frame 별 aggregate (no duplicate weight)
    instances: list[GarmentInstance]              # 원자 단위 instance
    duplicate_groups: list[list[GarmentInstance]] # 같은 옷 묶음
    yolo_detected_persons: int
    fallback_triggered: bool
    post_total_garment_pixels: int
    garment_class_counts: dict[str, int]
    # Scene filter Stage 1 drop (frame 단위) / Stage 2 drop (bbox 단위).
    # Noop 사용 시 둘 다 빈 list.
    filtered_out_frames: list[FrameDropRecord]
    filtered_out_bboxes: list[BBoxDropRecord]


def extract_palette_with_diagnostics(
    source: FrameSource, bundle: SegBundle, cfg: VisionConfig,
) -> ExtractionDiagnostics:
    """instance 단위 추출 → frame 별 palette + duplicate-weighted post palette."""
    instances, stats = extract_instances(source, bundle, cfg)

    # frame palette 를 instance 들로부터 재구성 (모델 호출 추가 없음).
    # 영향력 큰 frame (총 pixel 합 × instance 개수) 이 상단으로.
    frame_stats: dict[str, tuple[int, int]] = {}
    for inst in instances:
        px, cnt = frame_stats.get(inst.frame_id, (0, 0))
        frame_stats[inst.frame_id] = (px + inst.pixel_count, cnt + 1)
    frame_ids = sorted(frame_stats, key=lambda fid: (-frame_stats[fid][0], -frame_stats[fid][1]))
    frame_palettes = [
        _frame_palette_from_instances(instances, fid, cfg.extract_colors.k)
        for fid in frame_ids
    ]

    # post palette — duplicate 그룹 + sub-linear weight
    post_palette, dup_groups = aggregate_post_palette(
        instances,
        top_k=cfg.extract_colors.k,
        duplicate_max_delta_e=cfg.instance.duplicate_max_delta_e,
        weight_formula=cfg.instance.weight_formula,
    )

    class_counts: dict[str, int] = {}
    for inst in instances:
        class_counts[inst.garment_class] = (
            class_counts.get(inst.garment_class, 0) + inst.pixel_count
        )
    total = sum(inst.pixel_count for inst in instances)

    return ExtractionDiagnostics(
        palette=post_palette,
        frame_palettes=frame_palettes,
        instances=instances,
        duplicate_groups=dup_groups,
        yolo_detected_persons=stats.yolo_detected_persons,
        fallback_triggered=stats.fallback_triggered,
        post_total_garment_pixels=total,
        garment_class_counts=class_counts,
        filtered_out_frames=stats.filtered_out,
        filtered_out_bboxes=stats.bbox_filtered_out,
    )


def extract_garment_pixels(
    frame: Frame, bundle: SegBundle, cfg: VisionConfig,
) -> list[np.ndarray]:
    """frame 1개 → 의류 class 별 cleaned pixel list. legacy helper (phase 1/2 호환)."""
    lab_min = np.asarray(cfg.skin_lab_box.min, dtype=np.float32)
    lab_max = np.asarray(cfg.skin_lab_box.max, dtype=np.float32)
    boxes = detect_people(bundle.yolo, frame.rgb)
    if not boxes and cfg.fallback_full_image_on_no_person:
        h, w = frame.rgb.shape[:2]
        boxes = [(0, 0, w, h)]
    out: list[np.ndarray] = []
    for (x1, y1, x2, y2) in boxes:
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(frame.rgb.shape[1], x2), min(frame.rgb.shape[0], y2)
        if x2c - x1c < MIN_CROP_PX or y2c - y1c < MIN_CROP_PX:
            continue
        crop_rgb = frame.rgb[y1c:y2c, x1c:x2c]
        seg = run_segformer(bundle, crop_rgb)
        skin_class_mask = np.isin(seg, list(SKIN_CLASS_IDS))
        for class_id, label in ATR_LABELS.items():
            if label not in WEAR_KEEP:
                continue
            garment_mask = seg == class_id
            if int(garment_mask.sum()) < cfg.extract_colors.min_pixels:
                continue
            cleaned, _ratio, _kept = drop_skin_adaptive_spatial(
                crop_rgb, garment_mask, skin_class_mask,
                lab_min=lab_min, lab_max=lab_max,
                keep_threshold_pct=cfg.skin_drop_threshold_pct,
                upper_ceiling_pct=cfg.skin_drop_upper_ceiling,
                skin_dilate_iterations=cfg.skin_dilate_iterations,
            )
            if cleaned.shape[0] < cfg.extract_colors.min_pixels:
                continue
            out.append(cleaned)
    return out
