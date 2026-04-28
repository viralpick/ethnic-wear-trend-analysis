"""Canonical pool 중간상태 진단 — Phase 5 Step C pool_02 수렴 디버깅.

최초 2026-04-24: H1/H2/H3 가설 특정. 2026-04-25 업데이트: Color 3층 palette 재설계
(A1~C1) 결과 검증용으로 `post_palette_v2` (aggregated, max 3) dump 추가. 기존
combined-pool `final_palette` 는 legacy 비교용으로 유지. β-hybrid 재설계 (D-6,
2026-04-25) 에서 `canonical_palette_v2` sweep (build_canonical_palette 호출) 삭제.

E-1 (2026-04-25) 업데이트: β-hybrid 신규 path 검증용 `v3_hybrid` 섹션 추가 —
`extract_canonical_pixels_per_object` + `build_object_palette` (Phase 1+2)
+ `aggregate_canonical_palette` (Phase 3 통합 weighted KMeans) prod 함수를
그대로 호출해 per-object / per-canonical 덤프 생성. legacy combined-pool 진단은
보존 — 두 path 차이를 한 번에 보기 위함. `settings.vision.hybrid_palette` 는
명시 주입 (advisor #2 / feedback_load_models_scene_filter_cfg_leak 패턴).

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

from contracts.vision import CanonicalOutfit, EthnicOutfit, GarmentAnalysis
from settings import load_settings
from vision.bbox_utils import normalized_xywh_to_pixel_xyxy
from vision.canonical_extractor import (
    ObjectPool,
    extract_canonical_pixels_per_object,
)
from vision.canonical_palette_aggregator import aggregate_canonical_palette
from vision.color_family_preset import load_preset_views
from vision.color_space import SkinDropConfig, drop_skin_2layer, rgb_to_lab
from vision.dynamic_palette import extract_dynamic_palette
from vision.factory import build_vision_llm_client
from vision.hybrid_palette import (
    R3_DROP_DELTAE76,
    build_object_palette,
    filter_picks_by_pixel_evidence,
)
from vision.llm_client import VisionLLMClient
from vision.pipeline_b_extractor import load_models, run_segformer
from vision.prompts import PROMPT_VERSION
from vision.segformer_constants import ATR_LABELS, SKIN_CLASS_IDS, WEAR_CLASS_IDS

REPO = Path(__file__).resolve().parents[1]
DEFAULT_POST_ID = "01KPT74FM28H0GT6MQTNHFBY1Q"  # Sridevi saree
OUT_DIR = REPO / "outputs" / "phase5_stepC_diag"


def _build_skin_drop_config(vcfg) -> SkinDropConfig:
    return SkinDropConfig(
        lab_min=tuple(vcfg.skin_lab_box.min),
        lab_max=tuple(vcfg.skin_lab_box.max),
        secondary_drop_threshold_pct=vcfg.skin_drop_threshold_pct,
        upper_ceiling_pct=vcfg.skin_drop_upper_ceiling,
        skin_dilate_iterations=vcfg.skin_dilate_iterations,
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


def _load_gemini(
    image_bytes: bytes,
    client: VisionLLMClient,
    preset: list[dict[str, str]],
) -> GarmentAnalysis:
    """client.extract_garment 경유 — 내부에 LocalJSONCache cache-through 내장.

    cache hit → 즉시 반환. miss → live Gemini 호출 후 cache.put 자동 처리
    (`GeminiVisionLLMClient.extract_garment` 구현). VPN/API key 부재 시 이 호출에서
    예외 발생 — diag 의 중단은 사용자가 복구 후 재실행.
    """
    return client.extract_garment(image_bytes, preset=preset)


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


def _lab_a_gt20_stats(pixels: np.ndarray) -> dict:
    """maroon trace — lab_a>20 픽셀의 count + 평균 RGB/LAB."""
    n = int(pixels.shape[0])
    if n == 0:
        return {"total": 0, "lab_a_gt20": 0, "ratio": 0.0,
                "warm_rgb_mean": None, "warm_lab_mean": None}
    lab = rgb_to_lab(pixels.astype(np.float32))
    warm_mask = lab[..., 1] > 20.0
    warm_count = int(warm_mask.sum())
    out = {"total": n, "lab_a_gt20": warm_count,
           "ratio": round(warm_count / n, 4)}
    if warm_count:
        out["warm_rgb_mean"] = [
            round(float(pixels[warm_mask, c].mean()), 2) for c in range(3)
        ]
        out["warm_lab_mean"] = [
            round(float(lab[warm_mask, c].mean()), 2) for c in range(3)
        ]
    else:
        out["warm_rgb_mean"] = None
        out["warm_lab_mean"] = None
    return out


def _lab_a_gt20_per_seg_class(
    crop: np.ndarray, seg: np.ndarray, top_k: int = 8,
) -> list[dict]:
    """segformer class 별 lab_a>20 분포 — 진짜 maroon 이 어느 class 로 갔는지 추적."""
    cnt = Counter(seg.flatten().tolist())
    out = []
    for label_id, total in cnt.most_common(top_k):
        mask = seg == label_id
        pixels = crop[mask]
        stats = _lab_a_gt20_stats(pixels)
        out.append({
            "class_id": int(label_id),
            "class_name": ATR_LABELS.get(int(label_id), f"cls{label_id}"),
            "total": stats["total"],
            "lab_a_gt20": stats["lab_a_gt20"],
            "ratio": stats["ratio"],
            "warm_rgb_mean": stats["warm_rgb_mean"],
            "warm_lab_mean": stats["warm_lab_mean"],
        })
    return out


def _diag_outfit(
    rgb: np.ndarray, outfit: EthnicOutfit,
    bundle, skin_cfg: SkinDropConfig, outfit_index: int,
    dyn_cfg,
) -> dict:
    """outfit (BBOX) 1개 진단. None 반환하지 않고 skip_reason 필드로 표시."""
    h, w = rgb.shape[:2]
    pixel_box = normalized_xywh_to_pixel_xyxy(outfit.person_bbox, h, w)
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

    # F-10 maroon trace — stage 별 lab_a>20 픽셀 추적.
    # X₀ 는 _diag_outfit 밖 image-level 에서 별도 계산 (rgb 전체).
    crop_pixels = crop.reshape(-1, 3)
    garment_pixels = crop[garment_mask] if garment_mask.any() else np.empty((0, 3), np.uint8)
    skin_pixels = crop[skin_mask] if skin_mask.any() else np.empty((0, 3), np.uint8)
    maroon_trace = {
        "X1_bbox_crop": _lab_a_gt20_stats(crop_pixels),
        "X2_garment_mask": _lab_a_gt20_stats(garment_pixels),
        "X2b_skin_mask": _lab_a_gt20_stats(skin_pixels),
        "X3_effective_garment": _lab_a_gt20_stats(effective_pixels),
        "X4_cleaned": _lab_a_gt20_stats(cleaned),
        "per_seg_class": _lab_a_gt20_per_seg_class(crop, seg),
    }

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
        "maroon_trace": maroon_trace,
        "gemini_preset_picks_top3": list(outfit.color_preset_picks_top3),
        "person_bbox_area_ratio": float(outfit.person_bbox_area_ratio),
        "upper_is_ethnic": outfit.upper_is_ethnic,
        "lower_is_ethnic": outfit.lower_is_ethnic,
        # cleaned / mask 는 json 직렬화 제외 — 내부 aggregation 용.
        "_cleaned_pixels": cleaned,
        "_crop": crop,
        "_garment_mask": garment_mask,
        "_skin_mask": skin_mask,
        "_effective_mask": effective,
        "_outfit": outfit,
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


def _diag_v3_object(pool: ObjectPool, dyn_cfg, matcher_entries) -> dict:
    """β-hybrid 신규 path per-object 진단 — prod 함수 그대로 호출.

    advisor #1: 평행 구현 금지. `build_object_palette` 결과를 받아오고, R3 분기는
    `filter_picks_by_pixel_evidence` 공개 헬퍼로 재현. KMeans pool 자체는
    extract_dynamic_palette 로 한 번 더 확보 (R3 입력 가시화 용도).
    """
    rgb_pixels = pool.rgb_pixels
    obj_pixel_count = int(rgb_pixels.shape[0])
    frame_area = int(pool.frame_area)
    picks_input = list(pool.picks)

    # E-2b: a*>20 (red-leaning) histogram — Sridevi pool shrinkage file 가르기 (advisor)
    if obj_pixel_count:
        lab_all = rgb_to_lab(rgb_pixels.astype(np.float32))
        red_leaning_count = int(np.sum(lab_all[..., 1] > 20))
    else:
        red_leaning_count = 0

    pixel_clusters = extract_dynamic_palette(rgb_pixels, dyn_cfg) if obj_pixel_count else []
    picks_after_R3 = filter_picks_by_pixel_evidence(
        picks_input, pixel_clusters, matcher_entries, R3_DROP_DELTAE76,
    )
    weighted_clusters, etc_weight = build_object_palette(
        rgb_pixels, picks_input, dyn_cfg, matcher_entries, frame_area,
    )

    return {
        "image_id": pool.member.image_id,
        "outfit_index": pool.member.outfit_index,
        "obj_pixel_count": obj_pixel_count,
        "frame_area": frame_area,
        "frame_coverage_ratio": (
            round(obj_pixel_count / frame_area, 4) if frame_area else 0.0
        ),
        "skin_drop_primary": int(pool.skin_drop_primary),
        "skin_drop_secondary": int(pool.skin_drop_secondary),
        "lab_a_gt20_count": red_leaning_count,
        "picks_input": picks_input,
        "picks_after_R3": picks_after_R3,
        "kmeans_clusters_pre_hybrid": [
            {
                "hex": c.hex,
                "lab": [round(v, 2) for v in c.lab],
                "share": round(c.share, 4),
            }
            for c in pixel_clusters
        ],
        "weighted_clusters": [
            {
                "hex": wc.hex,
                "lab": [round(v, 2) for v in wc.lab],
                "weight": round(float(wc.weight), 6),
            }
            for wc in weighted_clusters
        ],
        "r2_etc_weight": round(float(etc_weight), 6),
    }


def _diag_v3_canonical(
    canonical: CanonicalOutfit,
    pools: list[ObjectPool],
    canonical_index: int,
    dyn_cfg,
    matcher_entries,
    hybrid_cfg,
) -> dict:
    """β-hybrid 신규 path canonical 단위 진단 — Phase 1+2+3 prod 함수 그대로.

    `aggregate_canonical_palette` 호출 시 `hybrid_cfg` 명시 주입
    (advisor #2 / feedback_load_models_scene_filter_cfg_leak).
    """
    rep = canonical.representative
    objects_diag: list[dict] = []
    per_object_results: list[tuple[list, float]] = []
    for pool in pools:
        obj_d = _diag_v3_object(pool, dyn_cfg, matcher_entries)
        objects_diag.append(obj_d)
        per_object_results.append(
            build_object_palette(
                pool.rgb_pixels, pool.picks, dyn_cfg, matcher_entries,
                int(pool.frame_area),
                chroma_vivid=hybrid_cfg.chroma_vivid,
                hue_near_deg=hybrid_cfg.hue_near_deg,
                r2_merge_deltae76=hybrid_cfg.r2_merge_deltae76,
            ),
        )

    aggregator_input_size = sum(len(c) for c, _ in per_object_results)
    palette, cut_off = aggregate_canonical_palette(
        per_object_results, matcher_entries, hybrid_cfg,
    )

    picks_union: list[str] = []
    seen: set[str] = set()
    for pool in pools:
        for p in pool.picks:
            if p not in seen:
                picks_union.append(p)
                seen.add(p)

    final_summary = " | ".join(
        f"{p.hex}({p.family.value if hasattr(p.family, 'value') else p.family},"
        f"{p.share:.2f})"
        for p in palette
    ) or "(empty)"

    return {
        "canonical_index": canonical_index,
        "representative_image_id": canonical.members[0].image_id if canonical.members else None,
        "representative_outfit_index": (
            canonical.members[0].outfit_index if canonical.members else None
        ),
        "members_count": len(canonical.members),
        "upper_is_ethnic": rep.upper_is_ethnic,
        "lower_is_ethnic": rep.lower_is_ethnic,
        "dress_as_single": rep.dress_as_single,
        "objects_count": len(pools),
        "objects": objects_diag,
        "aggregator_input_size": int(aggregator_input_size),
        "palette": [
            {
                "hex": p.hex,
                "share": round(float(p.share), 4),
                "family": p.family.value if hasattr(p.family, "value") else str(p.family),
            }
            for p in palette
        ],
        "cut_off_share": round(float(cut_off), 4),
        "side_by_side": {
            "gemini_picks_union": picks_union,
            "final_palette_summary": final_summary,
        },
    }


def _print_v3_summary(v3_canonicals: list[dict]) -> None:
    print("\n=== v3_hybrid (β-hybrid prod path) ===")
    if not v3_canonicals:
        print("[v3] (no canonicals)")
        return
    for c in v3_canonicals:
        print(
            f"[v3] canonical[{c['canonical_index']}] members={c['members_count']} "
            f"u_eth={c['upper_is_ethnic']} l_eth={c['lower_is_ethnic']} "
            f"objects={c['objects_count']} agg_input_clusters={c['aggregator_input_size']} "
            f"cut_off={c['cut_off_share']}"
        )
        for o in c["objects"]:
            picks_in = ",".join(o["picks_input"]) or "-"
            picks_r3 = ",".join(o["picks_after_R3"]) or "-"
            print(
                f"[v3]   obj img={o['image_id']} idx={o['outfit_index']} "
                f"pix={o['obj_pixel_count']} cov={o['frame_coverage_ratio']:.3f} "
                f"picks=[{picks_in}] R3=[{picks_r3}] wc={len(o['weighted_clusters'])}"
            )
        print(f"[v3]   palette: {c['side_by_side']['final_palette_summary']}")


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
    lines = [
        "",
        "H3 KMeans 배경 cluster 만 생존 (combined-pool legacy — D-6 이후 신규 path 진단은 "
        "후속 스크립트로 이관; 아래 final_palette 는 legacy 참고용):",
    ]
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


def _write_visual(
    outfits: list[dict], palette: dict, out_png: Path,
    v3_canonicals: list[dict] | None = None,
) -> None:
    """outfit 별 [crop, garment, skin, effective, cleaned chip] + legacy palette
    + v3 β-hybrid final palette (canonical 별, family 라벨 포함).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    drawable = [o for o in outfits if "_crop" in o]
    if not drawable:
        return
    v3_rows = len(v3_canonicals) if v3_canonicals else 0
    n_rows = len(drawable) + 1 + v3_rows
    fig, axes = plt.subplots(n_rows, 5, figsize=(18, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    for r, o in enumerate(drawable):
        _draw_outfit_row(axes[r], o)
    _draw_palette_row(axes[len(drawable)], palette)
    for r, canonical in enumerate(v3_canonicals or []):
        _draw_v3_palette_row(axes[len(drawable) + 1 + r], canonical, r)
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
        axes[0].set_title("legacy palette empty")
        return
    chip = np.zeros((40, 40 * len(final), 3), dtype=np.uint8)
    for i, c in enumerate(final):
        chip[:, i * 40:(i + 1) * 40] = c["rgb"]
    axes[0].imshow(chip)
    axes[0].set_title("legacy combined-pool: " + " ".join(
        f"{c['hex']}({c['share']:.2f})" for c in final
    ))


def _draw_v3_palette_row(axes, canonical: dict, idx: int) -> None:
    """β-hybrid (canonical 단위) final palette + family 라벨 시각화."""
    for ax in axes:
        ax.axis("off")
    palette = canonical.get("palette", [])
    cut_off = canonical.get("cut_off_share", 0.0)
    if not palette:
        axes[0].set_title(f"v3 canonical[{idx}] palette empty")
        return
    chip = np.zeros((40, 40 * len(palette), 3), dtype=np.uint8)
    for i, c in enumerate(palette):
        hx = c["hex"].lstrip("#")
        rgb = (int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16))
        chip[:, i * 40:(i + 1) * 40] = rgb
    axes[0].imshow(chip)
    axes[0].set_title(
        f"v3 β-hybrid canonical[{idx}] (cut_off={cut_off:.2f}): "
        + " ".join(
            f"{c['hex']}({c['family']},{c['share']:.2f})" for c in palette
        ),
        fontsize=9,
    )


def _serialize_dump(
    post_id: str,
    image_diag: list[dict],
    palette: dict,
    model_id: str,
    v3_canonicals: list[dict] | None = None,
) -> dict:
    """numpy 배열은 dump 제외. 기본 숫자/리스트만."""
    clean_images = []
    for img in image_diag:
        clean_outfits = []
        for o in img["outfits"]:
            clean_outfits.append({k: v for k, v in o.items() if not k.startswith("_")})
        clean_images.append({
            "image_id": img["image_id"],
            "outfits": clean_outfits,
            "X0_image_total": img.get("X0_image_total"),
        })
    return {
        "post_id": post_id,
        "prompt_version": PROMPT_VERSION,
        "model_id": model_id,
        "images": clean_images,
        "palette": palette,
        "v3_hybrid": {"canonicals": v3_canonicals or []},
    }


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--post-id", default=DEFAULT_POST_ID)
    ap.add_argument("--enriched", default=str(REPO / "outputs/2026-04-24/enriched.json"))
    ap.add_argument("--llm-cache", default=str(REPO / "outputs/llm_cache"))
    ap.add_argument("--blob-cache", default=str(REPO / "sample_data/image_cache"))
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--initial-k", type=int, default=None,
                    help="override DynamicPaletteConfig.initial_k for k sweep")
    ap.add_argument("--merge-threshold", type=float, default=None,
                    help="override DynamicPaletteConfig.merge_deltae76_threshold for threshold sweep")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    settings = load_settings()
    skin_cfg = _build_skin_drop_config(settings.vision)
    dyn_cfg = settings.vision.dynamic_palette
    dyn_overrides: dict = {}
    if args.initial_k is not None:
        dyn_overrides["initial_k"] = args.initial_k
    if args.merge_threshold is not None:
        dyn_overrides["merge_deltae76_threshold"] = args.merge_threshold
    if dyn_overrides:
        dyn_cfg = dyn_cfg.model_copy(update=dyn_overrides)
        print(f"[diag] dyn_cfg override: {dyn_overrides}")
    views = load_preset_views(settings.outfit_dedup.preset_path)

    # settings.vision_llm.cache_dir 이 args.llm_cache 와 다르면 CLI 값으로 덮어쓴다 —
    # diag 가 production 과 다른 cache dir 을 가리키던 기존 동작 보존.
    cli_cache = Path(args.llm_cache)
    vision_llm_cfg = settings.vision_llm
    if cli_cache != Path(vision_llm_cfg.cache_dir):
        vision_llm_cfg = vision_llm_cfg.model_copy(update={"cache_dir": cli_cache})
    vision_llm = build_vision_llm_client(vision_llm_cfg)

    post = _find_post(args.post_id, Path(args.enriched))
    blob_urls: list[str] = post["normalized"]["image_urls"]
    print(f"[diag] post_id={args.post_id} images={len(blob_urls)} "
          f"prompt_version={PROMPT_VERSION}")

    print(f"[diag] loading SegBundle (YOLO+segformer+SceneFilter) ...")
    # SceneFilterConfig 명시 주입 — yaml `enabled: true` 가 무시되던 leak 방어
    # (feedback_load_models_scene_filter_cfg_leak). prod (run_daily_pipeline.py:277)
    # 와 동일 path 보장: F smoke 결과가 prod 일반화에 사용 가능해야 함.
    bundle = load_models(scene_filter_cfg=settings.vision.scene_filter)

    pool_parts: list[np.ndarray] = []
    image_diag: list[dict] = []
    # IG reel post 의 image_urls 에는 .mp4 가 섞여 있어 PIL decode 불가. 이미지 확장자만 처리.
    image_urls = [
        u for u in blob_urls
        if Path(u).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".heic"}
    ]
    if not image_urls:
        raise SystemExit(f"post {args.post_id} has no image url (all video)")
    for url in image_urls:
        image_id, image_bytes, rgb = _load_image(Path(args.blob_cache), url)
        analysis = _load_gemini(image_bytes, vision_llm, views.llm_preset)
        print(f"[diag] image={image_id} ethnic={analysis.is_india_ethnic_wear} "
              f"outfits={len(analysis.outfits)}")
        outfit_diag: list[dict] = []
        for i, outfit in enumerate(analysis.outfits):
            d = _diag_outfit(
                rgb, outfit, bundle, skin_cfg, i, dyn_cfg,
            )
            picks = ", ".join(outfit.color_preset_picks_top3) or "(none)"
            print(
                f"[diag]   outfit[{i}] u_eth={outfit.upper_is_ethnic} "
                f"l_eth={outfit.lower_is_ethnic} "
                f"area={outfit.person_bbox_area_ratio:.3f} "
                f"preset=[{picks}] cleaned={d['cleaned_count']}"
            )
            outfit_diag.append(d)
            if "_cleaned_pixels" in d and d["_cleaned_pixels"].shape[0] > 0:
                pool_parts.append(d["_cleaned_pixels"])
        # F-10 X₀ image-level lab_a>20 (BBOX crop 전 단계).
        image_total_warm = _lab_a_gt20_stats(rgb.reshape(-1, 3))
        image_diag.append({
            "image_id": image_id,
            "outfits": outfit_diag,
            "X0_image_total": image_total_warm,
        })

    pool = (
        np.concatenate(pool_parts, axis=0)
        if pool_parts else np.empty((0, 3), np.uint8)
    )
    palette_diag = _diag_palette(pool, dyn_cfg)

    # v3_hybrid prod path: post_items 재구성 → prod 함수 호출.
    # _diag_outfit 가 이미 image 별 RGB / GarmentAnalysis 를 메모리에 들고 있지만,
    # advisor #1 (평행 구현 금지) 따라 prod entry 함수를 그대로 한 번 더 호출.
    post_items_v3: list[tuple[str, np.ndarray, GarmentAnalysis]] = []
    for url in image_urls:
        image_id, image_bytes, rgb = _load_image(Path(args.blob_cache), url)
        analysis = _load_gemini(image_bytes, vision_llm, views.llm_preset)
        post_items_v3.append((image_id, rgb, analysis))
    pairs = extract_canonical_pixels_per_object(
        post_items_v3, bundle, settings.vision,
        settings.outfit_dedup, views.family_map,
    )
    hybrid_cfg = settings.vision.hybrid_palette  # advisor #2: 명시 주입.
    v3_canonicals = [
        _diag_v3_canonical(
            canonical, pools, idx, dyn_cfg,
            views.matcher_entries, hybrid_cfg,
        )
        for idx, (canonical, pools) in enumerate(pairs)
    ]

    dump = _serialize_dump(
        args.post_id, image_diag, palette_diag,
        model_id=vision_llm_cfg.model_id,
        v3_canonicals=v3_canonicals,
    )

    json_path = out_dir / f"{args.post_id}.json"
    json_path.write_text(json.dumps(dump, indent=2, ensure_ascii=False), encoding="utf-8")
    png_path = out_dir / f"{args.post_id}_visual.png"
    all_outfits = [o for img in image_diag for o in img["outfits"]]
    _write_visual(all_outfits, palette_diag, png_path, v3_canonicals=v3_canonicals)

    summary = _hypotheses_summary(all_outfits, palette_diag)
    print("\n" + summary)
    _print_v3_summary(v3_canonicals)
    print(f"\n[diag] wrote {json_path}")
    print(f"[diag] wrote {png_path}")


if __name__ == "__main__":
    main()
