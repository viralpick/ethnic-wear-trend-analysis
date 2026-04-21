"""Pipeline B — YOLOv8 person detect → segformer garment seg → LAB KMeans palette.

출처: ~/dev/clothing-color-extraction-poc/scripts/run_pipeline_b.py (2026-04-17 동료 PoC).
동료 PoC 는 영상 전용. 여기서는 FrameSource 추상화로 이미지/영상 공용화.

이 모듈은 top-level 로 torch / transformers / ultralytics 를 import. vision extras 미설치
시 ImportError — core 코드는 **절대 top-level import 금지** (Step D 에서 ColorExtractor
Protocol 뒤 DI 로 연결).

aggregation 전략: frame 별 garment 픽셀을 전부 cleaned → concat → 한 번의 KMeans.
계층적 (frame→cluster→reKMeans) 보다 단순 + 결정론.

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
    drop_skin,
    extract_colors,
    hex_to_rgb,
)
from vision.frame_source import Frame, FrameSource

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

_MIN_CROP_PX = 32
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


def extract_garment_pixels(
    frame: Frame, bundle: SegBundle, cfg: VisionConfig,
) -> list[np.ndarray]:
    """frame 1개 → 의류 class 별 skin-cleaned pixel (N,3) 리스트. 빈 리스트면 skip."""
    lab_min = np.asarray(cfg.skin_lab_box.min, dtype=np.float32)
    lab_max = np.asarray(cfg.skin_lab_box.max, dtype=np.float32)
    out: list[np.ndarray] = []
    for (x1, y1, x2, y2) in detect_people(bundle.yolo, frame.rgb):
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(frame.rgb.shape[1], x2), min(frame.rgb.shape[0], y2)
        if x2c - x1c < _MIN_CROP_PX or y2c - y1c < _MIN_CROP_PX:
            continue
        crop_rgb = frame.rgb[y1c:y2c, x1c:x2c]
        seg = run_segformer(bundle, crop_rgb)
        for class_id, label in ATR_LABELS.items():
            if label not in WEAR_KEEP:
                continue
            pixels = crop_rgb[seg == class_id]
            if pixels.shape[0] < cfg.extract_colors.min_pixels:
                continue
            cleaned = drop_skin(pixels, lab_min=lab_min, lab_max=lab_max)
            if cleaned.shape[0] < cfg.extract_colors.min_pixels:
                continue
            out.append(cleaned)
    return out


def extract_palette(
    source: FrameSource, bundle: SegBundle, cfg: VisionConfig,
) -> list[ColorPaletteItem]:
    """FrameSource → 모든 frame 의 garment cleaned pixel concat → KMeans → palette."""
    all_pixels: list[np.ndarray] = []
    for frame in source.iter_frames():
        all_pixels.extend(extract_garment_pixels(frame, bundle, cfg))
    if not all_pixels:
        return []
    combined = np.concatenate(all_pixels)
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
