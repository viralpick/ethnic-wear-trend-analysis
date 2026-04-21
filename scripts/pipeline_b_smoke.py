"""Pipeline B end-to-end smoke — sample_data/image 20 JPG 처리 + comparison HTML (Step E).

흐름:
  image_root 에서 JPG 목록 로드 → filename prefix (post ULID) 로 그룹화 → 각 post 를
  ImageFrameSource 로 묶어 Pipeline B extract_palette 호출 → palette.json + mask overlay
  PNG + comparison.html 로 저장.

mask overlay: 각 이미지마다 segformer 의 WEAR_KEEP class 를 class 별 색으로 반투명 덧씌워
원본 옆에 표시 (분리 품질 눈으로 검증).

전제: `uv sync --extra vision` (torch + transformers + ultralytics) 완료.

실행:
  uv run python scripts/pipeline_b_smoke.py
  uv run python scripts/pipeline_b_smoke.py \
      --image-root sample_data/image \
      --output-dir outputs/pipeline_b_smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from settings import VisionConfig, load_settings  # noqa: E402
from vision.frame_source import ImageFrameSource  # noqa: E402
from vision.pipeline_b_extractor import (  # noqa: E402
    ATR_LABELS,
    WEAR_KEEP,
    SegBundle,
    detect_people,
    extract_palette,
    load_models,
    run_segformer,
)

# class 별 overlay 색상 (RGB). 시각 구분만 위한 것 (분리 품질 확인용).
_OVERLAY_COLORS: dict[str, tuple[int, int, int]] = {
    "upper-clothes": (255, 0, 0),       # red
    "pants": (0, 0, 255),               # blue
    "skirt": (0, 255, 255),             # cyan
    "dress": (255, 0, 255),             # magenta
    "hat": (255, 255, 0),               # yellow
    "left-shoe": (255, 128, 0),         # orange
    "right-shoe": (255, 128, 0),        # orange
}
_OVERLAY_ALPHA = 0.5


def group_by_post_ulid(image_dir: Path) -> dict[str, list[Path]]:
    """JPG 파일명 `{post_ULID}_{image_ULID}.jpg` → post_ULID 로 group."""
    grouped: dict[str, list[Path]] = defaultdict(list)
    for jpg in sorted(image_dir.glob("*.jpg")):
        parts = jpg.stem.split("_", 1)
        if len(parts) != 2:
            continue
        grouped[parts[0]].append(jpg)
    return dict(grouped)


def _relpath(target: Path, start: Path) -> str:
    try:
        return str(target.relative_to(start))
    except ValueError:
        import os
        return os.path.relpath(target, start)


def build_segformer_overlay(
    rgb: np.ndarray, bundle: SegBundle, cfg: VisionConfig,
) -> np.ndarray:
    """원본 rgb 위에 WEAR_KEEP class 를 class 별 색으로 반투명 overlay.

    YOLO 0 bbox + fallback 활성화 시 전체 이미지 crop 으로 동일 처리.
    여러 bbox 면 각 bbox 내 segformer 결과를 overlay 에 순차 write.
    """
    overlay = rgb.copy()
    boxes = detect_people(bundle.yolo, rgb)
    if not boxes and cfg.fallback_full_image_on_no_person:
        h, w = rgb.shape[:2]
        boxes = [(0, 0, w, h)]
    for (x1, y1, x2, y2) in boxes:
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(rgb.shape[1], x2), min(rgb.shape[0], y2)
        if x2c - x1c < 32 or y2c - y1c < 32:
            continue
        crop = rgb[y1c:y2c, x1c:x2c]
        seg = run_segformer(bundle, crop)
        patch = overlay[y1c:y2c, x1c:x2c].copy()
        for class_id, label in ATR_LABELS.items():
            if label not in WEAR_KEEP:
                continue
            mask = seg == class_id
            if not mask.any():
                continue
            color = np.array(_OVERLAY_COLORS.get(label, (0, 255, 0)), dtype=np.float32)
            blended = patch[mask].astype(np.float32) * (1 - _OVERLAY_ALPHA) + color * _OVERLAY_ALPHA
            patch[mask] = blended.astype(np.uint8)
        overlay[y1c:y2c, x1c:x2c] = patch
    return overlay


def save_overlay_png(
    image_path: Path, bundle: SegBundle, cfg: VisionConfig, output_path: Path,
) -> None:
    """이미지 1장 → segformer overlay PNG 저장."""
    img = Image.open(image_path).convert("RGB")
    rgb = np.asarray(img, dtype=np.uint8)
    overlay = build_segformer_overlay(rgb, bundle, cfg)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(output_path, format="PNG", optimize=True)


def _render_html(results: list[dict], output_path: Path) -> None:
    """post 당 썸네일 + overlay + palette chip 3열. 분리 품질 눈으로 검증."""
    rows: list[str] = []
    legend = " ".join(
        f'<span style="background:rgb{_OVERLAY_COLORS[lbl]};'
        f'color:white;padding:2px 6px;margin:2px">{lbl}</span>'
        for lbl in _OVERLAY_COLORS
    )
    for r in results:
        imgs_html = "".join(
            f'<img src="{_relpath(Path(p), output_path.parent)}" '
            f'style="width:120px;margin:2px;border:1px solid #ddd;">'
            for p in r["image_paths"]
        )
        overlay_html = "".join(
            f'<img src="{_relpath(Path(p), output_path.parent)}" '
            f'style="width:120px;margin:2px;border:1px solid #ddd;">'
            for p in r.get("overlay_paths", [])
        )
        chips_html = "".join(
            f'<div style="display:inline-block;margin:4px;text-align:center;font-size:11px">'
            f'<div style="background:{c["hex_display"]};width:48px;height:48px;'
            f'border:1px solid #888"></div>'
            f'<span>{c["hex_display"]}<br>{c["pct"]:.1%}</span></div>'
            for c in r["palette"]
        ) or '<span style="color:#888">(no palette)</span>'
        rows.append(
            f"<tr><td style='vertical-align:top;padding:8px'><code>{r['post_ulid']}</code>"
            f"<br><small>{len(r['image_paths'])} imgs</small></td>"
            f"<td style='padding:8px'>{imgs_html}</td>"
            f"<td style='padding:8px'>{overlay_html}</td>"
            f"<td style='padding:8px'>{chips_html}</td></tr>"
        )
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Pipeline B Smoke</title>
<style>body{{font-family:system-ui;margin:20px}}
table{{border-collapse:collapse}} td,th{{border:1px solid #ccc}}
.legend{{margin:12px 0;font-size:13px}}</style></head>
<body><h1>Pipeline B Smoke — sample_data/image</h1>
<p>Rows: {len(results)} posts. HEX = LAB KMeans centroid, pct = pixel weight.
overlay = segformer class mask 반투명 덧씌움 (분리 품질 검증).</p>
<div class="legend">Legend: {legend}</div>
<table><thead><tr><th>post ULID</th><th>carousel</th><th>segformer overlay</th>
<th>palette (top-k)</th></tr></thead><tbody>{''.join(rows)}</tbody></table></body></html>
"""
    output_path.write_text(html, encoding="utf-8")


def run_smoke(image_root: Path, output_dir: Path) -> dict:
    """실 진입점. palette.json + masks/*.png + comparison.html 쓰기."""
    settings = load_settings()
    output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    grouped = group_by_post_ulid(image_root)
    total_imgs = sum(len(v) for v in grouped.values())
    print(f"[smoke] image_root={image_root} posts={len(grouped)} images={total_imgs}")

    print("[smoke] loading YOLO + segformer (cached after first run)")
    bundle = load_models()
    print(f"[smoke] device={bundle.device}")

    results: list[dict] = []
    for post_ulid, paths in grouped.items():
        overlay_paths: list[Path] = []
        for p in paths:
            overlay_path = masks_dir / f"{p.stem}_overlay.png"
            save_overlay_png(p, bundle, settings.vision, overlay_path)
            overlay_paths.append(overlay_path)
        source = ImageFrameSource(paths)
        palette = extract_palette(source, bundle, settings.vision)
        results.append({
            "post_ulid": post_ulid,
            "image_paths": [str(p) for p in paths],
            "overlay_paths": [str(p) for p in overlay_paths],
            "palette": [item.model_dump(mode="json") for item in palette],
        })
        print(f"[smoke] {post_ulid}: {len(paths)} imgs → {len(palette)} chips")

    (output_dir / "palette.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )
    _render_html(results, output_dir / "comparison.html")
    return {
        "posts": len(results),
        "images": total_imgs,
        "output_dir": str(output_dir),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline B smoke on local JPG directory.")
    parser.add_argument(
        "--image-root", type=Path, default=_REPO_ROOT / "sample_data" / "image",
        help="JPG 디렉토리 (기본: sample_data/image)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=_REPO_ROOT / "outputs" / "pipeline_b_smoke",
        help="palette.json + masks/*.png + comparison.html 쓸 디렉토리",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run_smoke(args.image_root, args.output_dir)
    print(f"[smoke] done {summary}")


if __name__ == "__main__":
    main()
