"""Pipeline B 단계별 디버깅 — 특정 이미지/post 에서 어느 단계가 empty palette 의 원인인지 추적.

단계:
  1) 이미지 로드 + shape
  2) YOLO person detect (conf 조정 가능). 0 이면 더 낮은 conf 로 재시도
  3) 각 bbox 의 segformer class 분포 (ATR 18-class) — 어떤 class 가 주로 잡히는지
  4) WEAR_KEEP 의 각 garment class 에 대해: raw pixels → drop_skin → after. 어디서 떨어지는지
  5) 통과한 class 의 extract_colors top-3

전제: `uv sync --extra vision` 완료. 모델은 pipeline_b_smoke 돌리면서 이미 캐시됨.

실행:
  uv run python scripts/debug_pipeline_b.py --post-ulid 01KPNKR0JA5ZCS8FRW0RVCYHA1
  uv run python scripts/debug_pipeline_b.py --image sample_data/image/xxx.jpg --conf 0.20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from settings import load_settings  # noqa: E402
from vision.color_space import drop_skin, extract_colors  # noqa: E402
from vision.pipeline_b_extractor import (  # noqa: E402
    ATR_LABELS,
    WEAR_KEEP,
    SegBundle,
    load_models,
    run_segformer,
)


def _detect_with_conf(
    yolo, rgb: np.ndarray, conf: float
) -> list[tuple[tuple[int, int, int, int], float]]:
    """YOLO 호출 + (bbox, conf) pairs 반환 (desc)."""
    result = yolo.predict(rgb, classes=[0], conf=conf, verbose=False)[0]
    if result.boxes.conf is None:
        return []
    confs = result.boxes.conf.cpu().numpy().tolist()
    boxes = result.boxes.xyxy.cpu().numpy()
    pairs = [
        ((int(b[0]), int(b[1]), int(b[2]), int(b[3])), float(c))
        for b, c in zip(boxes, confs)
    ]
    pairs.sort(key=lambda x: -x[1])
    return pairs


def _debug_bbox(
    i: int, box: tuple[int, int, int, int], rgb: np.ndarray,
    bundle: SegBundle, cfg,
) -> None:
    x1, y1, x2, y2 = box
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(rgb.shape[1], x2), min(rgb.shape[0], y2)
    w, h = x2c - x1c, y2c - y1c
    if w < 32 or h < 32:
        print(f"    [{i}] SKIP — crop too small ({w}x{h})")
        return
    crop = rgb[y1c:y2c, x1c:x2c]
    print(f"    [{i}] crop: {w}x{h}")

    seg = run_segformer(bundle, crop)
    class_ids, counts = np.unique(seg, return_counts=True)
    print(f"    [{i}] segformer class 분포 (상위 8):")
    for cid, cnt in sorted(zip(class_ids, counts), key=lambda x: -x[1])[:8]:
        lbl = ATR_LABELS.get(int(cid), f"?{cid}")
        mark = "✓" if lbl in WEAR_KEEP else " "
        print(f"        {mark} cid={int(cid):2d} {lbl:18s} {int(cnt):6d}px")

    lab_min = np.asarray(cfg.skin_lab_box.min, dtype=np.float32)
    lab_max = np.asarray(cfg.skin_lab_box.max, dtype=np.float32)
    min_pixels = cfg.extract_colors.min_pixels
    print(f"    [{i}] garment drop_skin pipeline (min_pixels={min_pixels}):")
    any_pass = False
    for cid, cnt in zip(class_ids, counts):
        lbl = ATR_LABELS.get(int(cid))
        if lbl not in WEAR_KEEP:
            continue
        pixels = crop[seg == cid]
        before = pixels.shape[0]
        if before < min_pixels:
            print(f"        {lbl:18s}: {before:6d}px                    ❌ below-min before drop")
            continue
        cleaned = drop_skin(pixels, lab_min=lab_min, lab_max=lab_max)
        after = cleaned.shape[0]
        dropped = (1 - after / before) * 100 if before else 0
        ok = after >= min_pixels
        status = "✓ pass" if ok else "❌ below-min after drop"
        print(
            f"        {lbl:18s}: {before:6d}px → drop_skin "
            f"→ {after:6d}px ({dropped:5.1f}% dropped)  {status}"
        )
        if ok:
            any_pass = True
            colors = extract_colors(cleaned, k=cfg.extract_colors.k, min_pixels=min_pixels)
            for c in colors[:3]:
                print(f"            top → {c['hex']} weight={c['weight']:.2f}")
    if not any_pass:
        print(f"    [{i}] RESULT: empty palette — no garment class survived min_pixels + drop_skin")


def debug_image(image_path: Path, bundle: SegBundle, cfg, conf: float) -> None:
    img = Image.open(image_path).convert("RGB")
    rgb = np.array(img, dtype=np.uint8)
    print(f"\n=== {image_path.name} ===")
    print(f"  shape: {rgb.shape}")

    pairs = _detect_with_conf(bundle.yolo, rgb, conf)
    print(f"  YOLO conf≥{conf}: {len(pairs)} bbox")
    for i, (box, score) in enumerate(pairs):
        w, h = box[2] - box[0], box[3] - box[1]
        print(f"    [{i}] conf={score:.3f} bbox={box} size={w}x{h}")

    if not pairs:
        low = _detect_with_conf(bundle.yolo, rgb, 0.10)
        if low:
            top = low[0]
            print(
                f"  (retry conf≥0.10): {len(low)} bbox, top conf={top[1]:.3f} "
                f"bbox={top[0]}"
            )
            print("  → 결론: 기본 conf=0.35 가 너무 높음. 이 케이스에 0.20~0.30 권고.")
        else:
            print("  (retry conf≥0.10): 여전히 0 bbox — YOLO 자체가 사람 못 잡음")
        if cfg.fallback_full_image_on_no_person:
            print("  → fallback=True 이므로 전체 이미지를 bbox 로 간주하고 segformer 진행")
            h, w = rgb.shape[:2]
            _debug_bbox(0, (0, 0, w, h), rgb, bundle, cfg)
        return

    for i, (box, _) in enumerate(pairs):
        _debug_bbox(i, box, rgb, bundle, cfg)


def _resolve_paths(args) -> list[Path]:
    if args.image:
        return [args.image]
    if args.post_ulid:
        paths = sorted(args.image_root.glob(f"{args.post_ulid}*.jpg"))
        if not paths:
            print(f"No images for post_ulid={args.post_ulid} under {args.image_root}")
        return paths
    print("--image 또는 --post-ulid 필요")
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline B 단계별 디버깅")
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--post-ulid", type=str, default=None)
    parser.add_argument("--conf", type=float, default=0.35, help="YOLO conf threshold")
    parser.add_argument("--image-root", type=Path, default=_REPO / "sample_data" / "image")
    args = parser.parse_args()

    paths = _resolve_paths(args)
    if not paths:
        return

    settings = load_settings()
    print("Loading models...")
    bundle = load_models()
    print(f"Device: {bundle.device}")

    for p in paths:
        debug_image(p, bundle, settings.vision, args.conf)


if __name__ == "__main__":
    main()
