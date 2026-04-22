"""Pipeline B — YOLOv8 person detect → segformer garment seg → LAB KMeans palette.

phase 3 (2026-04-22): instance 단위로 완전 재구성. (frame × person bbox × garment_class)
교집합을 GarmentInstance 1개로 보고 각자 독립 KMeans. post aggregate 는 instance 들을
duplicate 그룹으로 묶어 sub-linear weight 로 top-k 선별.

이 모듈은 top-level 로 torch / transformers / ultralytics 를 import. vision extras 미설치
시 ImportError — core 코드는 **절대 top-level import 금지** (ColorExtractor Protocol 뒤 DI).

spec §4.1 ④ / §7 대응. spec §7.2 의 YT 경계는 frame_source 레이어에서 강제.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from ultralytics import YOLO

from contracts.common import ColorFamily, ColorPaletteItem
from settings import VisionConfig
from vision.color_space import (
    drop_skin_adaptive,
    extract_colors,
    hex_to_rgb,
)
from vision.frame_source import Frame, FrameSource
from vision.garment_instance import (
    GarmentInstance,
    aggregate_post_palette,
    classify_single_color,
)

# ATR 18-class segformer label 매핑 (동료 PoC 에서 인용).
# spec §4.1 ① GarmentType (kurta_set/anarkali 등) 과 직접 매칭 X — ATR 은 서양 복식.
# 이 모듈의 역할은 "의류 픽셀 vs 피부/배경 분리"만. garment_type 분류는 텍스트/LLM 담당.
ATR_LABELS: dict[int, str] = {
    0: "background",
    1: "hat",
    2: "hair",
    3: "sunglasses",
    4: "upper-clothes",
    5: "skirt",
    6: "pants",
    7: "dress",
    8: "belt",
    9: "left-shoe",
    10: "right-shoe",
    11: "bag",
    12: "skin-face",
    13: "skin-face",
    14: "skin-left-arm",
    15: "skin-right-arm",
    16: "skin-left-leg",
    17: "skin-right-leg",
}
WEAR_KEEP: frozenset[str] = frozenset({
    "upper-clothes", "pants", "skirt", "dress", "hat", "left-shoe", "right-shoe",
})
# ATR 의 피부 클래스 라벨 (bbox false positive 필터용). "진짜 사람" 검증에 사용 —
# 동상/마네킹/제품샷은 skin class pixel 이 거의 0.
SKIN_LABELS: frozenset[str] = frozenset({
    "skin-face", "skin-left-arm", "skin-right-arm", "skin-left-leg", "skin-right-leg",
})
_WEAR_CLASS_IDS: frozenset[int] = frozenset(
    {cid for cid, lbl in ATR_LABELS.items() if lbl in WEAR_KEEP}
)
_SKIN_CLASS_IDS: frozenset[int] = frozenset(
    {cid for cid, lbl in ATR_LABELS.items() if lbl in SKIN_LABELS}
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


def _pick_device(override: str | None) -> str:
    if override:
        return override
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_models(device: str | None = None) -> SegBundle:
    """YOLOv8n + segformer_b2_clothes 로드. 첫 호출 시 모델 파일 다운로드 (~250MB)."""
    dev = _pick_device(device)
    yolo = YOLO("yolov8n.pt")
    seg_processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
    seg_model = SegformerForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
    seg_model.eval()
    seg_model.to(dev)
    return SegBundle(yolo=yolo, seg_processor=seg_processor, seg_model=seg_model, device=dev)


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
    """cleaned pixel concat → KMeans → ColorPaletteItem. pure helper."""
    colors = extract_colors(
        combined, k=cfg.extract_colors.k, min_pixels=cfg.extract_colors.min_pixels,
    )
    palette: list[ColorPaletteItem] = []
    for i, c in enumerate(colors):
        r, g, b = hex_to_rgb(c["hex"]).astype(int).tolist()
        palette.append(
            ColorPaletteItem(
                r=int(r), g=int(g), b=int(b),
                hex_display=c["hex"],
                name=f"pipeline_b_{i}_{c['hex'].lstrip('#').lower()}",
                # TODO(§4.1 ④): LAB → ColorFamily 분류기 M4 에서 추가. 지금은 NEUTRAL 디폴트.
                family=ColorFamily.NEUTRAL,
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


def extract_instances(
    source: FrameSource, bundle: SegBundle, cfg: VisionConfig,
) -> tuple[list[GarmentInstance], int, bool]:
    """frame × person_bbox × garment_class 단위 GarmentInstance + YOLO 통계.

    반환: `(instances, yolo_detected_persons, fallback_triggered)`. diagnostics 별도 순회
    없이 단일 iter_frames 로 집계 — `ImageFrameSource.iter_frames()` 가 매 호출마다 PIL
    로 이미지를 다시 열기 때문에 FrameSource 를 두 번 도는 건 disk 재로딩 + YOLO 이중 추론.
    """
    ctx = _ExtractCtx(
        bundle=bundle, cfg=cfg,
        lab_min=np.asarray(cfg.skin_lab_box.min, dtype=np.float32),
        lab_max=np.asarray(cfg.skin_lab_box.max, dtype=np.float32),
    )
    instances: list[GarmentInstance] = []
    yolo_count = 0
    fallback = False
    for frame in source.iter_frames():
        boxes = detect_people(bundle.yolo, frame.rgb)
        yolo_count += len(boxes)
        if not boxes and cfg.fallback_full_image_on_no_person:
            fallback = True
            h, w = frame.rgb.shape[:2]
            boxes = [(0, 0, w, h)]
        for bbox_idx, bbox in enumerate(boxes):
            instances.extend(_instances_from_bbox(frame, bbox_idx, bbox, ctx))
    return instances, yolo_count, fallback


def _is_person_bbox(seg: np.ndarray, cfg: VisionConfig) -> bool:
    """segformer 결과 기반 실 사람 bbox 판정. skin pixel 비율 낮거나 의류 비율 극단적이면 False.

    - skin 비율 < min_skin_ratio_for_person → 동상/마네킹 또는 사람 없음 (전체 이미지 fallback)
    - 의류 비율 > max_garment_ratio_for_person → 배경까지 의류 오분류 (제품샷)
    """
    area = int(seg.size)
    if area == 0:
        return False
    skin_count = int(np.isin(seg, tuple(_SKIN_CLASS_IDS)).sum())
    garment_count = int(np.isin(seg, tuple(_WEAR_CLASS_IDS)).sum())
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
    for class_id, label in ATR_LABELS.items():
        if label not in WEAR_KEEP:
            continue
        pixels = crop_rgb[seg == class_id]
        if pixels.shape[0] < min_pixels:
            continue
        cleaned, ratio, _kept = drop_skin_adaptive(
            pixels, lab_min=ctx.lab_min, lab_max=ctx.lab_max,
            keep_threshold_pct=cfg.skin_drop_threshold_pct,
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
    instances, _yolo, _fallback = extract_instances(source, bundle, cfg)
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


def extract_palette_with_diagnostics(
    source: FrameSource, bundle: SegBundle, cfg: VisionConfig,
) -> ExtractionDiagnostics:
    """instance 단위 추출 → frame 별 palette + duplicate-weighted post palette."""
    instances, yolo_count, fallback = extract_instances(source, bundle, cfg)

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
        yolo_detected_persons=yolo_count,
        fallback_triggered=fallback,
        post_total_garment_pixels=total,
        garment_class_counts=class_counts,
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
        for class_id, label in ATR_LABELS.items():
            if label not in WEAR_KEEP:
                continue
            pixels = crop_rgb[seg == class_id]
            if pixels.shape[0] < cfg.extract_colors.min_pixels:
                continue
            cleaned, _ratio, _kept = drop_skin_adaptive(
                pixels, lab_min=lab_min, lab_max=lab_max,
                keep_threshold_pct=cfg.skin_drop_threshold_pct,
            )
            if cleaned.shape[0] < cfg.extract_colors.min_pixels:
                continue
            out.append(cleaned)
    return out
