"""Pipeline B end-to-end smoke — sample_data/image JPG 처리 + comparison HTML.

흐름:
  image_root 에서 JPG 목록 로드 → filename prefix (post ULID) 로 그룹화 → 각 post 를
  ImageFrameSource 로 묶어 extract_palette_with_diagnostics 호출 → palette.json +
  mask overlay PNG + comparison.html 로 저장.

phase 2 업데이트:
- overlay v2: drop_skin_adaptive kept/dropped 2색 표시 (흰색 = skin box 에 걸려 제거된 영역)
- comparison.html 의 palette column 에 frame-level palette 도 strip 으로 표시

전제: `uv sync --extra vision` (torch + transformers + ultralytics) 완료.

실행:
  uv run python scripts/pipeline_b_smoke.py
  uv run python scripts/pipeline_b_smoke.py \
      --image-root sample_data/image_cache \
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
from vision.color_space import rgb_to_lab  # noqa: E402
from vision.frame_source import ImageFrameSource  # noqa: E402
from vision.pipeline_b_extractor import (  # noqa: E402
    ATR_LABELS,
    WEAR_KEEP,
    SegBundle,
    detect_people,
    extract_palette_with_diagnostics,
    load_models,
    run_segformer,
)
from vision.quality import PostQuality, count_chip_similarity, count_skin_leaks  # noqa: E402

# class 별 overlay 색상 (RGB). 시각 구분만 위한 것.
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
# drop_skin 에 걸린 pixel 을 별도 색으로 — 흰색 강조 overlay 로 "의류로 잡혔지만 skin box
# 에 걸려 날아간 영역" 을 한눈에 보여준다. adaptive 가 "전체 유지" 결정한 class 는 이 영역 0.
_OVERLAY_DROPPED_COLOR: tuple[int, int, int] = (255, 255, 255)
_OVERLAY_DROPPED_ALPHA = 0.7


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


def _blend(patch: np.ndarray, mask: np.ndarray, color: np.ndarray, alpha: float) -> None:
    """in-place alpha blend. mask==True 위치만 color 와 합성."""
    if not mask.any():
        return
    patch[mask] = (
        patch[mask].astype(np.float32) * (1 - alpha) + color * alpha
    ).astype(np.uint8)


def _split_kept_dropped(
    crop: np.ndarray,
    class_mask: np.ndarray,
    lab_min: np.ndarray,
    lab_max: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """class_mask 를 kept/dropped 로 분할. adaptive 규칙 반영."""
    pixels = crop[class_mask]
    if pixels.shape[0] == 0:
        return class_mask, np.zeros_like(class_mask)
    lab = rgb_to_lab(pixels)
    inside = np.all((lab >= lab_min) & (lab <= lab_max), axis=-1)
    ratio = float(inside.sum()) / pixels.shape[0]
    if ratio > threshold:
        # adaptive: 전체 kept, dropped 없음
        return class_mask, np.zeros_like(class_mask)
    ys, xs = np.where(class_mask)
    kept_mask = np.zeros_like(class_mask)
    dropped_mask = np.zeros_like(class_mask)
    kept_mask[ys[~inside], xs[~inside]] = True
    dropped_mask[ys[inside], xs[inside]] = True
    return kept_mask, dropped_mask


def build_segformer_overlay(
    rgb: np.ndarray, bundle: SegBundle, cfg: VisionConfig,
) -> np.ndarray:
    """WEAR_KEEP class 를 kept / dropped 2색으로 반투명 overlay (phase 2)."""
    overlay = rgb.copy()
    boxes = detect_people(bundle.yolo, rgb)
    if not boxes and cfg.fallback_full_image_on_no_person:
        h, w = rgb.shape[:2]
        boxes = [(0, 0, w, h)]
    lab_min = np.asarray(cfg.skin_lab_box.min, dtype=np.float32)
    lab_max = np.asarray(cfg.skin_lab_box.max, dtype=np.float32)
    threshold = cfg.skin_drop_threshold_pct
    drop_color = np.array(_OVERLAY_DROPPED_COLOR, dtype=np.float32)

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
            class_mask = seg == class_id
            if not class_mask.any():
                continue
            class_color = np.array(_OVERLAY_COLORS.get(label, (0, 255, 0)), dtype=np.float32)
            kept_mask, dropped_mask = _split_kept_dropped(
                crop, class_mask, lab_min, lab_max, threshold,
            )
            _blend(patch, kept_mask, class_color, _OVERLAY_ALPHA)
            _blend(patch, dropped_mask, drop_color, _OVERLAY_DROPPED_ALPHA)
        overlay[y1c:y2c, x1c:x2c] = patch
    return overlay


def save_overlay_png(
    image_path: Path, bundle: SegBundle, cfg: VisionConfig, output_path: Path,
) -> None:
    """이미지 1장 → segformer overlay PNG 저장 (kept/dropped 2색)."""
    img = Image.open(image_path).convert("RGB")
    rgb = np.asarray(img, dtype=np.uint8)
    overlay = build_segformer_overlay(rgb, bundle, cfg)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(output_path, format="PNG", optimize=True)


def _quality_badge_html(q: dict) -> str:
    level = q.get("level", "ok")
    bg = {"ok": "#d4edda", "warning": "#fff3cd", "danger": "#f8d7da"}[level]
    icon = {"ok": "🟢", "warning": "🟡", "danger": "🔴"}[level]
    details = (
        f"skin_leak={q['skin_leak_count']} "
        f"sim_warn={q['chip_similarity_warning']} "
        f"px={q['post_total_garment_pixels']} "
        f"yolo={q['yolo_detected_persons']}"
        + (" fallback" if q.get("fallback_triggered") else "")
    )
    return (
        f'<div style="background:{bg};padding:4px 8px;border-radius:4px;'
        f'font-size:11px;margin-top:4px">{icon} <code>{details}</code></div>'
    )


def _chip_strip(palette_items: list[dict]) -> str:
    """palette → 색 칩 가로 strip. 작은 사이즈 (frame 단위용)."""
    if not palette_items:
        return '<span style="color:#aaa;font-size:10px">(empty)</span>'
    return "".join(
        f'<div title="{c["hex_display"]} {c["pct"]:.0%}" '
        f'style="display:inline-block;width:24px;height:24px;margin:1px;'
        f'background:{c["hex_display"]};border:1px solid #888"></div>'
        for c in palette_items
    )


def _frame_palettes_html(frame_palettes: list[dict]) -> str:
    """frame 별 palette strip — 캐러셀 frame 간 색 변화 시각 검증."""
    if not frame_palettes:
        return ""
    rows = []
    for fp in frame_palettes:
        pixel_summary = ", ".join(
            f"{lbl}={cnt}" for lbl, cnt in fp.get("garment_pixel_counts", {}).items()
        )
        rows.append(
            f'<div style="margin:2px 0;font-size:10px">'
            f'<code>{fp["frame_id"][:16]}…</code> {_chip_strip(fp["palette"])} '
            f'<span style="color:#888">{pixel_summary}</span></div>'
        )
    return "".join(rows)


def _render_html(results: list[dict], output_path: Path) -> None:
    """post 당: ULID+quality / carousel / overlay (kept+dropped) / post + frame palette."""
    rows: list[str] = []
    legend = " ".join(
        f'<span style="background:rgb{_OVERLAY_COLORS[lbl]};'
        f'color:white;padding:2px 6px;margin:2px">{lbl}</span>'
        for lbl in _OVERLAY_COLORS
    )
    dropped_legend = (
        f'<span style="background:rgb{_OVERLAY_DROPPED_COLOR};color:black;'
        'padding:2px 6px;margin:2px;border:1px solid #555">drop_skin removed</span>'
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
        badge_html = _quality_badge_html(r["quality"])
        frame_palettes_html = _frame_palettes_html(r.get("frame_palettes", []))
        palette_cell = (
            f'<div><strong style="font-size:11px">Post palette (aggregate):</strong>'
            f'<br>{chips_html}</div>'
            f'<div style="margin-top:8px"><strong style="font-size:11px">'
            f'Frame palettes (독립 KMeans):</strong>{frame_palettes_html}</div>'
        )
        rows.append(
            f"<tr><td style='vertical-align:top;padding:8px'><code>{r['post_ulid']}</code>"
            f"<br><small>{len(r['image_paths'])} imgs</small>{badge_html}</td>"
            f"<td style='padding:8px'>{imgs_html}</td>"
            f"<td style='padding:8px'>{overlay_html}</td>"
            f"<td style='padding:8px;vertical-align:top'>{palette_cell}</td></tr>"
        )
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Pipeline B Smoke</title>
<style>body{{font-family:system-ui;margin:20px}}
table{{border-collapse:collapse}} td,th{{border:1px solid #ccc}}
.legend{{margin:12px 0;font-size:13px}}</style></head>
<body><h1>Pipeline B Smoke — sample_data/image</h1>
<p>Rows: {len(results)} posts. HEX = LAB KMeans centroid, pct = pixel weight.
overlay = segformer class mask 반투명 + drop_skin 제거 영역 흰색. badge = post quality.</p>
<div class="legend">Class legend: {legend}<br>Dropped legend: {dropped_legend}</div>
<p style="font-size:12px">
Quality: 🟢 ok / 🟡 warning (skin leak or chip similarity or fallback) /
🔴 danger (total garment pixels &lt; 5000).<br>
Post palette = 전 frame 합쳐 한 번의 KMeans (cluster aggregate 용).<br>
Frame palettes = 각 frame 독립 KMeans (캐러셀 outfit 변화 검증용).
</p>
<table><thead><tr><th>post ULID + quality</th><th>carousel</th><th>overlay (kept + dropped)</th>
<th>palette</th></tr></thead><tbody>{''.join(rows)}</tbody></table></body></html>
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
        diag = extract_palette_with_diagnostics(source, bundle, settings.vision)
        quality = PostQuality(
            skin_leak_count=count_skin_leaks(diag.palette),
            chip_similarity_warning=count_chip_similarity(diag.palette),
            post_total_garment_pixels=diag.post_total_garment_pixels,
            yolo_detected_persons=diag.yolo_detected_persons,
            fallback_triggered=diag.fallback_triggered,
            garment_class_counts=dict(diag.garment_class_counts),
        )
        frame_palettes_json = [
            {
                "frame_id": fp.frame_id,
                "palette": [item.model_dump(mode="json") for item in fp.palette],
                "garment_pixel_counts": fp.garment_pixel_counts,
                "skin_drop_ratios": fp.skin_drop_ratios,
            }
            for fp in diag.frame_palettes
        ]
        results.append({
            "post_ulid": post_ulid,
            "image_paths": [str(p) for p in paths],
            "overlay_paths": [str(p) for p in overlay_paths],
            "palette": [item.model_dump(mode="json") for item in diag.palette],
            "frame_palettes": frame_palettes_json,
            "quality": {
                "level": quality.level,
                "skin_leak_count": quality.skin_leak_count,
                "chip_similarity_warning": quality.chip_similarity_warning,
                "post_total_garment_pixels": quality.post_total_garment_pixels,
                "yolo_detected_persons": quality.yolo_detected_persons,
                "fallback_triggered": quality.fallback_triggered,
                "garment_class_counts": quality.garment_class_counts,
            },
        })
        print(
            f"[smoke] {post_ulid}: {len(paths)} imgs → {len(diag.palette)} chips "
            f"(frame_palettes={len(diag.frame_palettes)}) {quality.badge} {quality.level}"
        )

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
