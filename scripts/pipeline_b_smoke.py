"""Pipeline B end-to-end smoke — sample_data/image JPG 처리 + comparison HTML.

흐름:
  image_root 에서 JPG 목록 로드 → filename prefix (post ULID) 로 그룹화 → 각 post 를
  ImageFrameSource 로 묶어 extract_palette_with_diagnostics 호출 → palette.json +
  mask overlay PNG + comparison.html 로 저장.

phase 2 업데이트:
- overlay v2: drop_skin_adaptive kept/dropped 2색 표시 (흰색 = skin box 에 걸려 제거된 영역)
- comparison.html 의 palette column 에 frame-level palette 도 strip 으로 표시

phase 3 업데이트:
- comparison.html 에 instance 단위 정보 추가 — (frame × person × class) 각각 독립 palette
- duplicate group 을 단일 블록으로 묶어 표시 (같은 옷 ΔE 기준 합쳐짐 + sub-linear weight)
- 각 instance 의 단색/다색 판정, pixel_count, skin_drop_ratio 노출

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
from vision.garment_instance import GarmentInstance, compute_group_weight  # noqa: E402
from vision.pipeline_b_extractor import (  # noqa: E402
    ATR_LABELS,
    MIN_CROP_PX,
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


def build_segformer_views(
    rgb: np.ndarray, bundle: SegBundle, cfg: VisionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """segformer 1회 호출로 overlay (RGB) + cloth-only (RGBA) 두 뷰 생성.

    - overlay: WEAR_KEEP class 반투명 (kept = class 색, dropped = 흰색) — phase 2 와 동일
    - cloth-only: WEAR_KEEP kept pixel 만 원래 색 + alpha=255, 나머지는 alpha=0 (완전 투명).
      HTML 은 체스판 CSS 배경 위에 이 PNG 를 올려 "투명 영역" 을 시각화.
    """
    overlay = rgb.copy()
    h, w = rgb.shape[:2]
    cloth_rgba = np.zeros((h, w, 4), dtype=np.uint8)  # 기본 완전 투명
    boxes = detect_people(bundle.yolo, rgb)
    if not boxes and cfg.fallback_full_image_on_no_person:
        boxes = [(0, 0, w, h)]
    lab_min = np.asarray(cfg.skin_lab_box.min, dtype=np.float32)
    lab_max = np.asarray(cfg.skin_lab_box.max, dtype=np.float32)
    threshold = cfg.skin_drop_threshold_pct
    drop_color = np.array(_OVERLAY_DROPPED_COLOR, dtype=np.float32)

    for (x1, y1, x2, y2) in boxes:
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(rgb.shape[1], x2), min(rgb.shape[0], y2)
        if x2c - x1c < MIN_CROP_PX or y2c - y1c < MIN_CROP_PX:
            continue
        crop = rgb[y1c:y2c, x1c:x2c]
        seg = run_segformer(bundle, crop)
        overlay_patch = overlay[y1c:y2c, x1c:x2c].copy()
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
            _blend(overlay_patch, kept_mask, class_color, _OVERLAY_ALPHA)
            _blend(overlay_patch, dropped_mask, drop_color, _OVERLAY_DROPPED_ALPHA)
            # cloth-only (RGBA): kept pixel 에 원 crop RGB + alpha=255
            ys, xs = np.where(kept_mask)
            cloth_rgba[y1c + ys, x1c + xs, :3] = crop[ys, xs]
            cloth_rgba[y1c + ys, x1c + xs, 3] = 255
        overlay[y1c:y2c, x1c:x2c] = overlay_patch
    return overlay, cloth_rgba


def save_views_png(
    image_path: Path, bundle: SegBundle, cfg: VisionConfig,
    overlay_path: Path, cloth_path: Path,
) -> None:
    """이미지 1장 → overlay (RGB) PNG + cloth-only (RGBA) PNG 저장 (segformer 1회)."""
    img = Image.open(image_path).convert("RGB")
    rgb = np.asarray(img, dtype=np.uint8)
    overlay, cloth_rgba = build_segformer_views(rgb, bundle, cfg)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(overlay_path, format="PNG", optimize=True)
    Image.fromarray(cloth_rgba, mode="RGBA").save(cloth_path, format="PNG", optimize=True)


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


def _chip_strip(palette_items: list[dict], width: int = 180, height: int = 16) -> str:
    """palette → pct 비례 width 의 가로 bar. frame/instance 단위 소형."""
    if not palette_items:
        return '<span style="color:#aaa;font-size:10px">(empty)</span>'
    segs = "".join(
        f'<div title="{c["hex_display"]} {c["pct"]:.0%}" '
        f'style="flex:{max(c["pct"], 0.001):.4f};height:{height}px;'
        f'background:{c["hex_display"]};"></div>'
        for c in palette_items
    )
    return (
        f'<div style="display:flex;width:{width}px;border:1px solid #888;'
        f'border-radius:2px;overflow:hidden">{segs}</div>'
    )


def _chip_bar(palette_items: list[dict]) -> str:
    """Post palette — 큰 pct-bar + hex + pct label. 시각적 주목."""
    if not palette_items:
        return '<span style="color:#888">(no palette)</span>'
    segs = "".join(
        f'<div title="{c["hex_display"]} {c["pct"]:.1%}" '
        f'style="flex:{max(c["pct"], 0.001):.4f};height:48px;'
        f'background:{c["hex_display"]};position:relative;'
        f'display:flex;flex-direction:column;justify-content:flex-end;'
        f'font-size:10px;color:#fff;text-shadow:0 0 2px #000;'
        f'padding:2px;text-align:center">'
        f'<span>{c["pct"]:.0%}</span></div>'
        for c in palette_items
    )
    labels = "".join(
        f'<div style="flex:{max(c["pct"], 0.001):.4f};font-size:10px;'
        f'color:#333;text-align:center;padding:2px 0">{c["hex_display"]}</div>'
        for c in palette_items
    )
    return (
        f'<div style="width:360px;border:1px solid #888;border-radius:3px;'
        f'overflow:hidden"><div style="display:flex">{segs}</div>'
        f'<div style="display:flex;background:#f4f4f4">{labels}</div></div>'
    )


def _views_cell_html(r: dict, html_dir: Path) -> str:
    """carousel / overlay / cloth-only 3 뷰 stack + post-row 단위 토글 버튼.

    클릭할 때마다 세 view 가 순환. post 안의 모든 이미지가 동시에 같은 view 로 전환.
    JS 는 페이지 공용 `cycleView` (_render_html 에서 <script> 한 번 삽입).
    """
    imgs = r["image_paths"]
    overlays = r.get("overlay_paths", [])
    cloths = r.get("cloth_paths", [])

    def _row(paths: list[str]) -> str:
        if not paths:
            return '<span style="color:#aaa;font-size:11px">(no data)</span>'
        return "".join(
            f'<img src="{_relpath(Path(p), html_dir)}" '
            f'style="width:120px;margin:2px;border:1px solid #ddd">'
            for p in paths
        )

    carousel = f'<div class="view view-active" data-view="carousel">{_row(imgs)}</div>'
    overlay = f'<div class="view" data-view="overlay">{_row(overlays)}</div>'
    cloth = f'<div class="view" data-view="cloth">{_row(cloths)}</div>'
    btn = (
        '<button class="view-toggle" onclick="cycleView(this)" '
        'style="margin-bottom:6px;padding:4px 10px;font-size:11px;cursor:pointer;'
        'border:1px solid #888;border-radius:3px;background:#f0f0f0">'
        '▶ carousel</button>'
    )
    return f'<div class="views-cell">{btn}{carousel}{overlay}{cloth}</div>'


def _frame_palettes_html(frame_palettes: list[dict]) -> str:
    """frame 별 palette bar — 영향력 desc 정렬 (extract_palette_with_diagnostics 순서 유지)."""
    if not frame_palettes:
        return ""
    rows = []
    for fp in frame_palettes:
        pc = fp.get("garment_pixel_counts", {})
        total_px = sum(pc.values())
        pixel_summary = ", ".join(f"{lbl}={cnt}" for lbl, cnt in pc.items())
        rows.append(
            f'<div style="margin:3px 0;font-size:10px">'
            f'<code>{fp["frame_id"][:16]}…</code> '
            f'<span style="color:#555">total_px={total_px}</span><br>'
            f'{_chip_strip(fp["palette"])} '
            f'<span style="color:#888">{pixel_summary}</span></div>'
        )
    return "".join(rows)


def _instance_to_json(inst: GarmentInstance) -> dict:
    """GarmentInstance → JSON-serializable dict (HTML 렌더 + palette.json 공용)."""
    return {
        "instance_id": inst.instance_id,
        "frame_id": inst.frame_id,
        "garment_class": inst.garment_class,
        "is_single_color": inst.is_single_color,
        "pixel_count": inst.pixel_count,
        "skin_drop_ratio": inst.skin_drop_ratio,
        "palette": [item.model_dump(mode="json") for item in inst.palette],
    }


def _instance_block_html(inst: dict) -> str:
    """instance 1 개 = class 1 개의 palette block. single/multi + drop ratio 표시."""
    single_badge = "단색" if inst["is_single_color"] else "다색"
    badge_bg = "#e7f3ff" if inst["is_single_color"] else "#fff0e5"
    short_id = inst["instance_id"].split(":", 1)[-1]
    return (
        f'<div style="display:inline-block;margin:2px;padding:4px;'
        f'border:1px solid #ddd;border-radius:4px;background:#fafafa;font-size:10px">'
        f'<div><strong>{inst["garment_class"]}</strong> '
        f'<span style="background:{badge_bg};padding:1px 4px;'
        f'border-radius:3px">{single_badge}</span></div>'
        f'<div style="color:#888">'
        f'<code>{short_id}</code> px={inst["pixel_count"]} '
        f'drop={inst["skin_drop_ratio"]:.2f}</div>'
        f'<div style="margin-top:2px">{_chip_strip(inst["palette"])}</div>'
        f'</div>'
    )


def _duplicate_groups_html(groups: list[dict]) -> str:
    """duplicate group 별 <details> 토글 — 같은 옷 묶음 + weight + chip 미리보기.

    기본 접힘 (closed). summary 에 class / count / weight / 대표 chip bar 를 노출해
    "열지 않고도" 어떤 옷인지 판별 가능하게.
    """
    if not groups:
        return '<span style="color:#aaa;font-size:10px">(no instances)</span>'
    blocks = []
    for g in groups:
        count = len(g["instances"])
        weight = g["weight"]
        insts = g["instances"]
        cls = insts[0]["garment_class"] if insts else "?"
        preview_palette = insts[0]["palette"] if insts else []
        weight_bg = "#ffeaa7" if count >= 2 else "#f0f0f0"
        inner = "".join(_instance_block_html(inst) for inst in insts)
        summary = (
            f'<summary style="cursor:pointer;font-size:11px;padding:3px 6px;'
            f'background:{weight_bg};border-radius:3px;display:flex;'
            f'align-items:center;gap:6px;list-style:revert">'
            f'<strong>{cls}</strong> × {count} '
            f'<span style="color:#555">w={weight:.2f}</span>'
            f'{_chip_strip(preview_palette, width=120, height=12)}'
            f'</summary>'
        )
        blocks.append(
            f'<details style="margin:3px 0;border-left:3px solid #888;'
            f'padding:2px 0 2px 4px">'
            f'{summary}<div style="margin-top:3px">{inner}</div></details>'
        )
    return "".join(blocks)


def _filtered_out_post_badge(filtered_out: list[dict], original_count: int) -> str:
    """post row 에 drop 된 frame 개수 + 사유 요약 badge. 없으면 빈 문자열."""
    if not filtered_out:
        return ""
    by_reason: dict[str, int] = {}
    for f in filtered_out:
        by_reason[f["reason"]] = by_reason.get(f["reason"], 0) + 1
    label = ", ".join(f"{r}×{n}" for r, n in by_reason.items())
    return (
        f'<div style="background:#fde0dc;padding:3px 6px;border-radius:3px;'
        f'font-size:10px;margin-top:3px" title="원본 {original_count} frame 중 filter drop">'
        f'🚫 {len(filtered_out)}/{original_count} dropped<br>'
        f'<span style="color:#555">{label}</span></div>'
    )


def _filtered_out_global_html(results: list[dict], html_dir: Path) -> str:
    """상단 <details> 접힘 섹션 — scene_filter drop 된 frame 전체 목록 (reason 별 groupby)."""
    by_reason: dict[str, list[dict]] = {}
    for r in results:
        for f in r.get("filtered_out", []):
            by_reason.setdefault(f["reason"], []).append({
                **f, "post_ulid": r["post_ulid"],
            })
    total = sum(len(v) for v in by_reason.values())
    if total == 0:
        return ""
    groups_html: list[str] = []
    for reason, items in sorted(by_reason.items(), key=lambda kv: -len(kv[1])):
        thumbs = "".join(
            f'<div style="display:inline-block;margin:2px;text-align:center;font-size:10px">'
            f'<img src="{_relpath(Path(f["image_path"]), html_dir)}" '
            f'style="width:90px;height:90px;object-fit:cover;border:1px solid #ddd"><br>'
            f'<code style="color:#888">{f["post_ulid"][:12]}…</code>'
            f'</div>'
            for f in items if f.get("image_path")
        )
        groups_html.append(
            f'<details style="margin:6px 0">'
            f'<summary style="cursor:pointer;font-weight:bold;padding:4px;'
            f'background:#fde0dc;border-radius:3px">'
            f'{reason} — {len(items)} frames</summary>'
            f'<div style="padding:6px">{thumbs}</div></details>'
        )
    return (
        f'<details style="margin:12px 0;border:1px solid #d44;border-radius:4px;padding:6px">'
        f'<summary style="cursor:pointer;font-weight:bold;color:#a00">'
        f'🚫 Filtered-out frames (scene_filter drop): {total} frames</summary>'
        f'<div style="padding:6px">{"".join(groups_html)}</div>'
        f'</details>'
    )


def _render_html(results: list[dict], output_path: Path) -> None:
    """post 당: ULID+quality / views / instances+groups / palette."""
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
        views_cell = _views_cell_html(r, output_path.parent)
        chips_html = _chip_bar(r["palette"])
        badge_html = _quality_badge_html(r["quality"])
        drop_badge = _filtered_out_post_badge(
            r.get("filtered_out", []),
            r.get("original_image_count", len(r["image_paths"])),
        )
        frame_palettes_html = _frame_palettes_html(r.get("frame_palettes", []))
        groups_html = _duplicate_groups_html(r.get("duplicate_groups", []))
        palette_cell = (
            f'<div><strong style="font-size:11px">Post palette (duplicate-weighted):</strong>'
            f'<br>{chips_html}</div>'
            f'<div style="margin-top:8px"><strong style="font-size:11px">'
            f'Frame palettes (독립 KMeans):</strong>{frame_palettes_html}</div>'
        )
        rows.append(
            f"<tr><td style='vertical-align:top;padding:8px'><code>{r['post_ulid']}</code>"
            f"<br><small>{len(r['image_paths'])} imgs</small>"
            f"{badge_html}{drop_badge}</td>"
            f"<td style='padding:8px;vertical-align:top'>{views_cell}</td>"
            f"<td style='padding:8px;vertical-align:top;max-width:360px'>{groups_html}</td>"
            f"<td style='padding:8px;vertical-align:top'>{palette_cell}</td></tr>"
        )
    global_drop_section = _filtered_out_global_html(results, output_path.parent)
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Pipeline B Smoke</title>
<style>
body{{font-family:system-ui;margin:20px}}
table{{border-collapse:collapse}} td,th{{border:1px solid #ccc}}
.legend{{margin:12px 0;font-size:13px}}
.views-cell .view{{display:none}}
.views-cell .view.view-active{{display:block}}
.views-cell .view[data-view="cloth"] img {{
  background-color:#eee;
  background-image:
    linear-gradient(45deg, #bbb 25%, transparent 25%),
    linear-gradient(-45deg, #bbb 25%, transparent 25%),
    linear-gradient(45deg, transparent 75%, #bbb 75%),
    linear-gradient(-45deg, transparent 75%, #bbb 75%);
  background-size: 12px 12px;
  background-position: 0 0, 0 6px, 6px -6px, -6px 0;
}}
</style></head>
<body><h1>Pipeline B Smoke</h1>
<p>Rows: {len(results)} posts. HEX = LAB KMeans centroid, pct = pixel weight.
view 버튼을 누르면 carousel → overlay → cloth-only 순환. badge = post quality.</p>
<div class="legend">Class legend: {legend}<br>Dropped legend: {dropped_legend}</div>
<p style="font-size:12px">
Quality: 🟢 ok / 🟡 warning (skin leak or chip similarity or fallback) /
🔴 danger (total garment pixels &lt; 5000).<br>
Post palette = instance duplicate-weighted aggregate. Frame palettes = 각 frame 독립 KMeans.
</p>
<p style="font-size:12px">
Instances = (frame × person × garment_class) 단위 독립 palette.
Groups = 같은 garment_class + top-1 chip ΔE &lt; threshold 인 instance 묶음.
weight = 1 + log(count) — sub-linear.
</p>
{global_drop_section}
<table><thead><tr><th>post ULID + quality</th>
<th>views (carousel / overlay / cloth)</th>
<th>instances + groups</th><th>palette</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<script>
const VIEW_ORDER = ['carousel', 'overlay', 'cloth'];
const VIEW_LABEL = {{ carousel: '▶ carousel', overlay: '▶ overlay', cloth: '▶ cloth' }};
function cycleView(btn) {{
  const cell = btn.parentElement;
  const current = cell.querySelector('.view.view-active');
  const currentKey = current ? current.dataset.view : 'carousel';
  const nextKey = VIEW_ORDER[(VIEW_ORDER.indexOf(currentKey) + 1) % VIEW_ORDER.length];
  cell.querySelectorAll('.view').forEach(v => {{
    v.classList.toggle('view-active', v.dataset.view === nextKey);
  }});
  btn.textContent = VIEW_LABEL[nextKey];
}}
</script>
</body></html>
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
    bundle = load_models(scene_filter_cfg=settings.vision.scene_filter)
    print(
        f"[smoke] device={bundle.device} "
        f"scene_filter={'CLIP' if settings.vision.scene_filter.enabled else 'Noop'}"
    )

    results: list[dict] = []
    for post_ulid, paths in grouped.items():
        # 먼저 diagnostics 를 돌려 scene_filter drop frame 집합 확보. overlay/cloth 는
        # pass 한 frame 만 생성 (drop frame 은 의미 없는 overlay 생성 비용 회피).
        source = ImageFrameSource(paths)
        diag = extract_palette_with_diagnostics(source, bundle, settings.vision)
        filtered_stems = {rec.frame_id for rec in diag.filtered_out_frames}
        kept_paths = [p for p in paths if p.stem not in filtered_stems]
        overlay_paths: list[Path] = []
        cloth_paths: list[Path] = []
        for p in kept_paths:
            overlay_path = masks_dir / f"{p.stem}_overlay.png"
            cloth_path = masks_dir / f"{p.stem}_cloth.png"
            save_views_png(
                p, bundle, settings.vision, overlay_path, cloth_path,
            )
            overlay_paths.append(overlay_path)
            cloth_paths.append(cloth_path)
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
        duplicate_groups_json = [
            {
                "instances": [_instance_to_json(inst) for inst in group],
                "weight": compute_group_weight(
                    len(group), settings.vision.instance.weight_formula,
                ),
            }
            for group in diag.duplicate_groups
        ]
        # scene_filter drop — frame_id 로 원본 image path 를 재매핑해 HTML thumbnail 에 사용.
        by_stem = {p.stem: p for p in paths}
        filtered_out_json = [
            {
                "frame_id": rec.frame_id,
                "reason": rec.verdict.reason,
                "scene_scores": rec.verdict.scene_scores,
                "gender_scores": rec.verdict.gender_scores,
                "age_scores": rec.verdict.age_scores,
                "image_path": str(by_stem[rec.frame_id])
                if rec.frame_id in by_stem else None,
            }
            for rec in diag.filtered_out_frames
        ]
        results.append({
            "post_ulid": post_ulid,
            "image_paths": [str(p) for p in kept_paths],
            "overlay_paths": [str(p) for p in overlay_paths],
            "cloth_paths": [str(p) for p in cloth_paths],
            "original_image_count": len(paths),
            "palette": [item.model_dump(mode="json") for item in diag.palette],
            "frame_palettes": frame_palettes_json,
            "duplicate_groups": duplicate_groups_json,
            "instances": [_instance_to_json(inst) for inst in diag.instances],
            "filtered_out": filtered_out_json,
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
        drop_n = len(filtered_out_json)
        drop_tag = f" ({drop_n} filtered)" if drop_n else ""
        print(
            f"[smoke] {post_ulid}: {len(paths)} imgs → {len(diag.palette)} chips "
            f"(frame_palettes={len(diag.frame_palettes)}) {quality.badge} {quality.level}"
            f"{drop_tag}"
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


def rerender_html_from_json(output_dir: Path) -> dict:
    """기존 palette.json 만 읽어서 comparison.html 재생성 (모델 호출 없음).

    HTML 렌더러 수정 후 full smoke 결과 재활용용. YOLO/segformer 안 돌림.
    """
    json_path = output_dir / "palette.json"
    if not json_path.exists():
        raise FileNotFoundError(f"palette.json 이 없다: {json_path}")
    results = json.loads(json_path.read_text())
    _render_html(results, output_dir / "comparison.html")
    return {"posts": len(results), "output_dir": str(output_dir), "mode": "html-only"}


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
    parser.add_argument(
        "--html-only", action="store_true",
        help="모델 호출 없이 기존 palette.json 으로 comparison.html 만 재생성",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.html_only:
        summary = rerender_html_from_json(args.output_dir)
    else:
        summary = run_smoke(args.image_root, args.output_dir)
    print(f"[smoke] done {summary}")


if __name__ == "__main__":
    main()
