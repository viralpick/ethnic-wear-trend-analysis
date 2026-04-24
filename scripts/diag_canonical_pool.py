"""Canonical pool 중간상태 진단 — Phase 5 Step C pool_02 수렴 디버깅 (2026-04-24).

단일 post (기본 Sridevi saree) 에 대해 canonical path 의 중간 단계 dump:
  crop → per-BBOX segformer → garment/skin mask → drop_skin_2layer 3단 분기 →
  canonical pool → KMeans(k=5) → merge → final palette.

목적: H1 (drop_skin 오버드롭) / H2 (seg mask 좁음) / H3 (KMeans 배경만 생존) 중 병목 특정.

실행:
  uv run python scripts/diag_canonical_pool.py
  uv run python scripts/diag_canonical_pool.py --post-id 01KPWFPXHWJVKPHKK0WKBZ1MGF  # Fabindia

출력: outputs/phase5_stepC_diag/{post_id}.json + {post_id}_visual.png + 콘솔 summary.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from io import BytesIO
from pathlib import Path

import numpy as np

from contracts.vision import GarmentAnalysis
from settings import load_settings
from vision.bbox_utils import normalized_xywh_to_pixel_xyxy
from vision.color_space import SkinDropConfig, drop_skin_2layer, rgb_to_lab
from vision.dynamic_palette import extract_dynamic_palette
from vision.llm_cache import compute_cache_key
from vision.pipeline_b_extractor import load_models, run_segformer
from vision.segformer_constants import ATR_LABELS, SKIN_CLASS_IDS, WEAR_CLASS_IDS

REPO = Path(__file__).resolve().parents[1]
DEFAULT_POST_ID = "01KPT74FM28H0GT6MQTNHFBY1Q"  # Sridevi saree
PROMPT_VERSION = "v0.3"
MODEL_ID = "gemini-2.5-flash"
OUT_DIR = REPO / "outputs" / "phase5_stepC_diag"


def _build_skin_drop_config(vcfg) -> SkinDropConfig:
    return SkinDropConfig(
        lab_min=tuple(vcfg.skin_lab_box.min),
        lab_max=tuple(vcfg.skin_lab_box.max),
        secondary_drop_threshold_pct=vcfg.skin_drop_threshold_pct,
        upper_ceiling_pct=vcfg.skin_drop_upper_ceiling,
    )


def _find_post(post_id: str, enriched_path: Path) -> dict:
    data = json.loads(enriched_path.read_text(encoding="utf-8"))
    for it in data:
        if it["normalized"].get("source_post_id") == post_id:
            return it
    raise SystemExit(f"post_id={post_id} 가 {enriched_path} 에 없음")


def _load_image(blob_cache: Path, blob_url: str) -> tuple[str, bytes, np.ndarray]:
    """image_urls 의 URL basename 으로 blob_cache 에서 이미지 로드."""
    from PIL import Image

    name = Path(blob_url).name
    path = blob_cache / name
    if not path.exists():
        raise SystemExit(f"image cache miss: {path}")
    data = path.read_bytes()
    rgb = np.array(Image.open(BytesIO(data)).convert("RGB"))
    return name, data, rgb


def _load_gemini(image_bytes: bytes, llm_cache: Path) -> GarmentAnalysis:
    key = compute_cache_key(
        image_bytes, prompt_version=PROMPT_VERSION, model_id=MODEL_ID,
    )
    p = llm_cache / MODEL_ID / f"{key}.json"
    if not p.exists():
        raise SystemExit(f"LLM cache miss: {p}")
    payload = json.loads(p.read_text(encoding="utf-8"))
    return GarmentAnalysis.model_validate(payload["garment_analysis"])


def _seg_class_hist(seg: np.ndarray, top_k: int = 10) -> list[tuple[str, int]]:
    """seg label → (label_name, count) 내림차순 top-k."""
    cnt = Counter(seg.flatten().tolist())
    return [
        (ATR_LABELS.get(int(label), f"cls{label}"), int(count))
        for label, count in cnt.most_common(top_k)
    ]


def _lab_inside_ratio(
    pixels: np.ndarray, cfg: SkinDropConfig
) -> float:
    """drop_skin_2layer 의 secondary 분기 판정 근거 재현 (진단용)."""
    if pixels.shape[0] == 0:
        return 0.0
    lab = rgb_to_lab(pixels)
    lo = np.asarray(cfg.lab_min, dtype=np.float32)
    hi = np.asarray(cfg.lab_max, dtype=np.float32)
    inside = np.all((lab >= lo) & (lab <= hi), axis=-1)
    return float(inside.sum()) / pixels.shape[0]


def _diag_outfit(
    rgb: np.ndarray, bbox: tuple[float, float, float, float],
    bundle, skin_cfg: SkinDropConfig, outfit_index: int,
) -> dict:
    """outfit (BBOX) 1개 진단. None 반환하지 않고 skip_reason 필드로 표시."""
    h, w = rgb.shape[:2]
    pixel_box = normalized_xywh_to_pixel_xyxy(bbox, h, w)
    if pixel_box is None:
        return {"outfit_index": outfit_index, "skip_reason": "bbox_too_small"}

    x1, y1, x2, y2 = pixel_box
    crop = rgb[y1:y2, x1:x2]
    seg = run_segformer(bundle, crop)
    garment_mask = np.isin(seg, list(WEAR_CLASS_IDS))
    skin_mask = np.isin(seg, list(SKIN_CLASS_IDS))
    effective = garment_mask & ~skin_mask
    effective_pixels = crop[effective] if effective.any() else np.empty((0, 3), np.uint8)
    drop_ratio = _lab_inside_ratio(effective_pixels, skin_cfg)
    cleaned, primary, secondary = drop_skin_2layer(crop, garment_mask, skin_mask, skin_cfg)
    return {
        "outfit_index": outfit_index,
        "pixel_box_xyxy": list(pixel_box),
        "crop_shape_hw": [crop.shape[0], crop.shape[1]],
        "crop_area": int(crop.shape[0] * crop.shape[1]),
        "seg_top_classes": _seg_class_hist(seg),
        "garment_mask_count": int(garment_mask.sum()),
        "skin_mask_count": int(skin_mask.sum()),
        "effective_garment_count": int(effective.sum()),
        "drop_ratio_lab_inside": drop_ratio,
        "skin_drop_branch": _skin_drop_branch(
            drop_ratio, skin_cfg, effective.sum(), cleaned.shape[0],
        ),
        "primary_drop_count": int(primary),
        "secondary_drop_count": int(secondary),
        "cleaned_count": int(cleaned.shape[0]),
        "cleaned_rgb_mean": _rgb_stat(cleaned, "mean"),
        "cleaned_rgb_std": _rgb_stat(cleaned, "std"),
        # cleaned 은 json 직렬화 제외 — 별도 저장. 시각화용으로만 반환.
        "_cleaned_pixels": cleaned,
        "_crop": crop,
        "_garment_mask": garment_mask,
        "_skin_mask": skin_mask,
        "_effective_mask": effective,
    }


def _skin_drop_branch(ratio: float, cfg: SkinDropConfig, effective: int, cleaned: int) -> str:
    if effective == 0:
        return "effective_empty"
    if ratio > cfg.upper_ceiling_pct:
        return "upper_ceiling_hit_drop_all"
    if ratio > cfg.secondary_drop_threshold_pct:
        return "kept_whole_skin_tone"
    return f"lab_dropped_{effective - cleaned}"


def _rgb_stat(pixels: np.ndarray, kind: str) -> list[float]:
    if pixels.shape[0] == 0:
        return [0.0, 0.0, 0.0]
    fn = np.mean if kind == "mean" else np.std
    return [round(float(fn(pixels[:, c])), 2) for c in range(3)]


def _kmeans_stages(pool: np.ndarray, dyn_cfg) -> dict:
    """KMeans(k=5) centers/counts 를 extract_dynamic_palette 와 별개로 추출해 비교."""
    if pool.shape[0] < dyn_cfg.min_pixels:
        return {"skip_reason": "pool_below_min_pixels", "pool_size": int(pool.shape[0])}
    from sklearn.cluster import KMeans

    lab = rgb_to_lab(pool)
    k_eff = min(dyn_cfg.initial_k, max(1, pool.shape[0] // 50))
    km = KMeans(n_clusters=k_eff, n_init=4, random_state=0).fit(lab)
    counts = np.bincount(km.labels_, minlength=k_eff).astype(int)
    centers_lab = km.cluster_centers_
    return {
        "pool_size": int(pool.shape[0]),
        "k_eff": int(k_eff),
        "centers_lab": [[round(float(v), 2) for v in c] for c in centers_lab],
        "counts": counts.tolist(),
        "shares": [round(float(c) / int(counts.sum()), 4) for c in counts],
    }


def _diag_palette(pool: np.ndarray, dyn_cfg) -> dict:
    """KMeans 진단 + extract_dynamic_palette 최종 결과."""
    out = {
        "pool_rgb_mean": _rgb_stat(pool, "mean"),
        "pool_rgb_std": _rgb_stat(pool, "std"),
        "kmeans_initial": _kmeans_stages(pool, dyn_cfg),
    }
    clusters = extract_dynamic_palette(pool, dyn_cfg)
    out["final_palette"] = [
        {
            "hex": c.hex, "rgb": list(c.rgb),
            "lab": [round(v, 2) for v in c.lab], "share": round(c.share, 4),
        }
        for c in clusters
    ]
    return out


def _hypotheses_summary(outfits_diag: list[dict], palette_diag: dict) -> str:
    """H1/H2/H3 의심도를 숫자 근거와 함께 텍스트로."""
    lines = ["=== Hypotheses ==="]
    # H1: upper_ceiling_hit or secondary_drop ratio high
    h1 = _judge_h1(outfits_diag)
    lines.extend(h1)
    # H2: garment_mask narrow or primary drop high
    h2 = _judge_h2(outfits_diag)
    lines.extend(h2)
    # H3: KMeans top cluster distribution
    h3 = _judge_h3(palette_diag)
    lines.extend(h3)
    return "\n".join(lines)


def _judge_h1(outfits: list[dict]) -> list[str]:
    lines = ["", "H1 drop_skin_2layer 오버드롭:"]
    for o in outfits:
        if "skip_reason" in o:
            lines.append(f"  outfit {o['outfit_index']}: skip ({o['skip_reason']})")
            continue
        branch = o["skin_drop_branch"]
        sec_ratio = (
            o["secondary_drop_count"] / o["effective_garment_count"]
            if o["effective_garment_count"] else 0.0
        )
        lines.append(
            f"  outfit {o['outfit_index']}: branch={branch} "
            f"secondary_drop_ratio={sec_ratio:.2f} "
            f"cleaned_mean_rgb={o['cleaned_rgb_mean']}"
        )
    return lines


def _judge_h2(outfits: list[dict]) -> list[str]:
    lines = ["", "H2 per-BBOX seg mask 너무 좁음:"]
    for o in outfits:
        if "skip_reason" in o:
            continue
        g_ratio = o["garment_mask_count"] / o["crop_area"]
        e_over_g = (
            o["effective_garment_count"] / o["garment_mask_count"]
            if o["garment_mask_count"] else 0.0
        )
        lines.append(
            f"  outfit {o['outfit_index']}: garment/crop={g_ratio:.2f} "
            f"effective/garment={e_over_g:.2f} "
            f"primary_drop={o['primary_drop_count']} skin_count={o['skin_mask_count']}"
        )
    return lines


def _judge_h3(palette: dict) -> list[str]:
    lines = ["", "H3 KMeans 배경 cluster 만 생존:"]
    km = palette.get("kmeans_initial", {})
    if "skip_reason" in km:
        lines.append(f"  skip ({km['skip_reason']})")
        return lines
    lines.append(f"  pool size={km['pool_size']} pool_mean_rgb={palette['pool_rgb_mean']}")
    for i, (c, s) in enumerate(zip(km["centers_lab"], km["shares"])):
        L, a, b = c
        is_warm = a > 8.0
        lines.append(
            f"  km[{i}] L={L:6.2f} a={a:6.2f} b={b:6.2f} share={s:.3f} "
            f"{'(warm/maroon?)' if is_warm else ''}"
        )
    lines.append(f"  final_palette n={len(palette['final_palette'])}")
    for c in palette["final_palette"]:
        lines.append(f"    {c['hex']} share={c['share']} lab={c['lab']}")
    return lines


def _write_visual(outfits: list[dict], palette: dict, out_png: Path) -> None:
    """outfit 별 [crop, garment, skin, effective, cleaned chip] + 최종 palette chip."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    drawable = [o for o in outfits if "_crop" in o]
    if not drawable:
        return
    n_rows = len(drawable) + 1
    fig, axes = plt.subplots(n_rows, 5, figsize=(18, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    for r, o in enumerate(drawable):
        _draw_outfit_row(axes[r], o)
    _draw_palette_row(axes[-1], palette)
    fig.tight_layout()
    fig.savefig(out_png, dpi=80)
    plt.close(fig)


def _draw_outfit_row(axes, o: dict) -> None:
    axes[0].imshow(o["_crop"]); axes[0].set_title(f"outfit {o['outfit_index']} crop")
    axes[1].imshow(o["_garment_mask"], cmap="gray")
    axes[1].set_title(f"garment {o['garment_mask_count']}")
    axes[2].imshow(o["_skin_mask"], cmap="gray")
    axes[2].set_title(f"skin {o['skin_mask_count']}")
    axes[3].imshow(o["_effective_mask"], cmap="gray")
    axes[3].set_title(f"effective {o['effective_garment_count']}")
    cleaned = o["_cleaned_pixels"]
    if cleaned.shape[0] > 0:
        side = max(1, int(np.sqrt(min(cleaned.shape[0], 4096))))
        chip = cleaned[: side * side].reshape(side, side, 3)
        axes[4].imshow(chip)
    axes[4].set_title(f"cleaned {cleaned.shape[0]}")
    for ax in axes:
        ax.axis("off")


def _draw_palette_row(axes, palette: dict) -> None:
    for ax in axes:
        ax.axis("off")
    final = palette.get("final_palette", [])
    if not final:
        axes[0].set_title("final palette empty")
        return
    chip = np.zeros((40, 40 * len(final), 3), dtype=np.uint8)
    for i, c in enumerate(final):
        chip[:, i * 40:(i + 1) * 40] = c["rgb"]
    axes[0].imshow(chip)
    axes[0].set_title("final palette " + " ".join(
        f"{c['hex']}({c['share']:.2f})" for c in final
    ))


def _serialize_dump(post_id: str, image_diag: list[dict], palette: dict) -> dict:
    """numpy 배열은 dump 제외. 기본 숫자/리스트만."""
    clean_images = []
    for img in image_diag:
        clean_outfits = []
        for o in img["outfits"]:
            clean_outfits.append({k: v for k, v in o.items() if not k.startswith("_")})
        clean_images.append({"image_id": img["image_id"], "outfits": clean_outfits})
    return {
        "post_id": post_id,
        "prompt_version": PROMPT_VERSION,
        "model_id": MODEL_ID,
        "images": clean_images,
        "palette": palette,
    }


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--post-id", default=DEFAULT_POST_ID)
    ap.add_argument("--enriched", default=str(REPO / "outputs/2026-04-24/enriched.json"))
    ap.add_argument("--llm-cache", default=str(REPO / "outputs/llm_cache"))
    ap.add_argument("--blob-cache", default=str(REPO / "sample_data/image_cache"))
    ap.add_argument("--out", default=str(OUT_DIR))
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    settings = load_settings()
    skin_cfg = _build_skin_drop_config(settings.vision)
    dyn_cfg = settings.vision.dynamic_palette

    post = _find_post(args.post_id, Path(args.enriched))
    blob_urls: list[str] = post["normalized"]["image_urls"]
    print(f"[diag] post_id={args.post_id} images={len(blob_urls)}")

    print(f"[diag] loading SegBundle (YOLO+segformer) ...")
    bundle = load_models()

    pool_parts: list[np.ndarray] = []
    image_diag: list[dict] = []
    for url in blob_urls:
        image_id, image_bytes, rgb = _load_image(Path(args.blob_cache), url)
        analysis = _load_gemini(image_bytes, Path(args.llm_cache))
        print(f"[diag] image={image_id} ethnic={analysis.is_india_ethnic_wear} "
              f"outfits={len(analysis.outfits)}")
        outfit_diag: list[dict] = []
        for i, outfit in enumerate(analysis.outfits):
            d = _diag_outfit(rgb, outfit.person_bbox, bundle, skin_cfg, i)
            outfit_diag.append(d)
            if "_cleaned_pixels" in d and d["_cleaned_pixels"].shape[0] > 0:
                pool_parts.append(d["_cleaned_pixels"])
        image_diag.append({"image_id": image_id, "outfits": outfit_diag})

    pool = (
        np.concatenate(pool_parts, axis=0)
        if pool_parts else np.empty((0, 3), np.uint8)
    )
    palette_diag = _diag_palette(pool, dyn_cfg)
    dump = _serialize_dump(args.post_id, image_diag, palette_diag)

    json_path = out_dir / f"{args.post_id}.json"
    json_path.write_text(json.dumps(dump, indent=2, ensure_ascii=False), encoding="utf-8")
    png_path = out_dir / f"{args.post_id}_visual.png"
    all_outfits = [o for img in image_diag for o in img["outfits"]]
    _write_visual(all_outfits, palette_diag, png_path)

    summary = _hypotheses_summary(all_outfits, palette_diag)
    print("\n" + summary)
    print(f"\n[diag] wrote {json_path}")
    print(f"[diag] wrote {png_path}")


if __name__ == "__main__":
    main()
