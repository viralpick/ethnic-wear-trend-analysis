"""50-color ethnic preset 생성 — M3.A Step D 선행 작업.

전략:
  1. StarRocks 에서 download_urls 있는 post 전량 pull (entry = hashtag OR profile)
  2. 각 post 의 첫 이미지에 YOLO + segformer → WEAR 클래스 pixel 만 수집
  3. per-instance KMeans(k=5) → instance 당 5 개 LAB centroid + pixel_count
  4. 모든 instance centroid 풀에서 LAB KMeans(k=35) — 우리 데이터가 실제로 담고 있는 색
  5. 15 색은 "있을 법한데 captured 못했을 수 있는" 전통 ethnic 색상을 수작업 보강
     (saffron, vermillion, henna green, indigo, peacock 등 — spec 참고 문헌)
  6. 출력: `configs/color_preset.yaml` 스니펫 + JSON 덤프 (분석용)

vision extras 와 blob extras 가 모두 필요. `uv sync --extra vision --extra blob`.

실행:
  uv run python scripts/build_color_preset.py
  uv run python scripts/build_color_preset.py --max-posts 500 --target-k 35
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

import numpy as np
import pymysql
from dotenv import load_dotenv
from PIL import Image
from sklearn.cluster import KMeans

from loaders.blob_downloader import BlobDownloader
from vision.color_space import (
    SKIN_LAB_MAX,
    SKIN_LAB_MIN,
    drop_skin_adaptive_spatial,
    hex_to_rgb,
    lab_to_rgb,
    rgb_to_hex,
    rgb_to_lab,
)
from vision.pipeline_b_extractor import (
    ATR_LABELS,
    MIN_CROP_PX,
    SKIN_CLASS_IDS,
    WEAR_KEEP,
    detect_people,
    load_models,
    run_segformer,
)

_CACHE_DIR = _REPO / "outputs" / "blob_cache_preset"
_OUT_DIR = _REPO / "outputs" / "color_preset"
_MAX_IMAGE_SIDE = 1280
_MIN_PIXELS_PER_INSTANCE = 600   # per-instance KMeans 최소 pixel (noise 제외)
_KMEANS_PER_INSTANCE = 5
_SKIN_KEEP_THRESHOLD = 0.5
_SKIN_UPPER_CEILING = 0.97       # drop_skin_adaptive 3단 방어 (configs/local.yaml 과 동기)
_SKIN_DILATE_ITERATIONS = 4      # spatial-aware drop margin (configs/local.yaml 과 동기)


# --------------------------------------------------------------------------- #
# 15 색 자체 보강 — 모수에서 captured 못했을 수 있는 전통 ethnic 색상.
# 근거: spec §4.1 ④ + 동료 PoC + Pantone textile (India Handloom) 참고.
# 이름은 나중에 LLM name 매핑에서 override 되므로 placeholder.
# --------------------------------------------------------------------------- #
_SELF_GENERATED_COLORS: list[dict[str, object]] = [
    {"name": "saffron",           "hex": "#FF9933"},  # 인도 국기 saffron
    {"name": "vermillion",        "hex": "#E34234"},  # sindoor 색
    {"name": "turmeric_yellow",   "hex": "#D9B500"},  # 강황
    {"name": "henna_green",       "hex": "#8B6E3F"},  # 헤나 건조 색
    {"name": "peacock_blue",      "hex": "#1F6D9E"},  # 공작새 청
    {"name": "rani_pink",         "hex": "#D04081"},  # Rajasthan rani
    {"name": "deep_indigo",       "hex": "#1C2958"},  # 전통 indigo dye
    {"name": "bottle_green",      "hex": "#0A5F38"},  # 전통 bottle green
    {"name": "maroon_red",        "hex": "#80030B"},  # bridal maroon
    {"name": "mustard_olive",     "hex": "#A27D28"},  # 머스타드/올리브 경계
    {"name": "blush_peach",       "hex": "#F2C9B4"},  # contemporary blush
    {"name": "lavender_mauve",    "hex": "#AE94C2"},  # contemporary lavender
    {"name": "mint_green",        "hex": "#98D4BB"},  # contemporary mint
    {"name": "charcoal_grey",     "hex": "#404347"},  # neutral charcoal
    {"name": "cream_ivory",       "hex": "#F3E5C3"},  # fabric 기본 ivory
]


def _connect() -> pymysql.Connection:
    """raw DB 연결 — `loaders.starrocks_connect.connect_raw` 위임 (drift 방지)."""
    from loaders.starrocks_connect import connect_raw
    return connect_raw()


def _load_all_urls(limit: int | None) -> list[str]:
    """download_urls 가 있는 모든 post 의 첫 URL. limit 이 있으면 random sample."""
    sql = """
        SELECT download_urls FROM india_ai_fashion_inatagram_posting
        WHERE download_urls IS NOT NULL AND download_urls != ''
    """
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
    urls: list[str] = []
    for row in rows:
        first = (row["download_urls"] or "").split(",")[0].strip()
        if first:
            urls.append(first)
    random.shuffle(urls)
    if limit is not None and limit < len(urls):
        urls = urls[:limit]
    return urls


def _load_rgb(path: Path) -> np.ndarray | None:
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            scale = min(1.0, _MAX_IMAGE_SIDE / max(w, h))
            if scale < 1.0:
                im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            return np.asarray(im, dtype=np.uint8)
    except Exception:  # noqa: BLE001
        return None


def _collect_instance_centroids(
    rgb: np.ndarray, bundle,
) -> list[tuple[np.ndarray, int]]:
    """이미지 한 장 → (LAB centroid, pixel_count) pair 리스트. 각 garment instance 당 최대 k 개."""
    lab_min = SKIN_LAB_MIN.astype(np.float32)
    lab_max = SKIN_LAB_MAX.astype(np.float32)
    boxes = detect_people(bundle.yolo, rgb)
    if not boxes:
        h, w = rgb.shape[:2]
        boxes = [(0, 0, w, h)]
    out: list[tuple[np.ndarray, int]] = []
    for x1, y1, x2, y2 in boxes:
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(rgb.shape[1], x2), min(rgb.shape[0], y2)
        if x2c - x1c < MIN_CROP_PX or y2c - y1c < MIN_CROP_PX:
            continue
        crop = rgb[y1c:y2c, x1c:x2c]
        seg = run_segformer(bundle, crop)
        skin_class_mask = np.isin(seg, list(SKIN_CLASS_IDS))
        for class_id, label in ATR_LABELS.items():
            if label not in WEAR_KEEP:
                continue
            garment_mask = seg == class_id
            if int(garment_mask.sum()) < _MIN_PIXELS_PER_INSTANCE:
                continue
            cleaned, _ratio, _kept = drop_skin_adaptive_spatial(
                crop, garment_mask, skin_class_mask,
                lab_min=lab_min, lab_max=lab_max,
                keep_threshold_pct=_SKIN_KEEP_THRESHOLD,
                upper_ceiling_pct=_SKIN_UPPER_CEILING,
                skin_dilate_iterations=_SKIN_DILATE_ITERATIONS,
            )
            if cleaned.shape[0] < _MIN_PIXELS_PER_INSTANCE:
                continue
            lab = rgb_to_lab(cleaned)
            k_eff = min(_KMEANS_PER_INSTANCE, max(1, cleaned.shape[0] // 120))
            km = KMeans(n_clusters=k_eff, n_init=4, random_state=0).fit(lab)
            counts = np.bincount(km.labels_, minlength=k_eff).astype(np.int64)
            for center, cnt in zip(km.cluster_centers_, counts):
                out.append((center.astype(np.float32), int(cnt)))
    return out


def _weighted_kmeans(
    centroids: np.ndarray, weights: np.ndarray, k: int, seed: int,
) -> np.ndarray:
    """sample_weight 지원 KMeans. centroids (N,3) + weights (N,) → (k,3) LAB."""
    km = KMeans(n_clusters=k, n_init=8, random_state=seed)
    km.fit(centroids, sample_weight=weights)
    return km.cluster_centers_


def _centroid_to_entry(lab: np.ndarray, idx: int, pool_size: int) -> dict[str, object]:
    rgb = lab_to_rgb(lab).astype(int)
    hex_code = rgb_to_hex(rgb)
    return {
        "name": f"pool_{idx:02d}",
        "hex": hex_code,
        "lab": [round(float(v), 2) for v in lab.tolist()],
        "origin": f"data_pool_n={pool_size}",
    }


def _dump_outputs(pool_entries: list[dict], all_entries: list[dict]) -> None:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    # JSON (분석용 — 50색 전량 + origin)
    json_path = _OUT_DIR / "color_preset.json"
    json_path.write_text(json.dumps(all_entries, indent=2, ensure_ascii=False), encoding="utf-8")
    # YAML snippet (configs/local.yaml 삽입용)
    yaml_path = _OUT_DIR / "color_preset.yaml"
    with yaml_path.open("w", encoding="utf-8") as fh:
        fh.write("# scripts/build_color_preset.py 산출물\n")
        fh.write("color_preset:\n")
        for entry in all_entries:
            fh.write(f"  - name: {entry['name']}\n")
            fh.write(f"    hex: '{entry['hex']}'\n")
            fh.write(f"    lab: {entry['lab']}\n")
            fh.write(f"    origin: {entry['origin']}\n")
    print(f"\n[preset] wrote {json_path} + {yaml_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-posts", type=int, default=500,
                        help="최대 post 샘플 수 (기본 500).")
    parser.add_argument("--target-k", type=int, default=35,
                        help="풀에서 뽑을 대표 색 수 (기본 35). 자체 15 합쳐 50.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"[preset] StarRocks URL pull (max_posts={args.max_posts})")
    urls = _load_all_urls(args.max_posts)
    print(f"[preset] URL pool size = {len(urls)}")
    if not urls:
        print("[preset] FAIL — URL 없음.")
        return 1

    print(f"[preset] 모델 로드 (device={args.device or 'auto'})")
    bundle = load_models(device=args.device, scene_filter_cfg=None)
    downloader = BlobDownloader.from_env()

    all_centroids: list[np.ndarray] = []
    all_weights: list[int] = []
    effective_posts = 0
    t_start = time.time()
    for idx, url in enumerate(urls, 1):
        dest = downloader.download(url, _CACHE_DIR)
        if dest is None:
            continue
        rgb = _load_rgb(dest)
        if rgb is None:
            continue
        per_img = _collect_instance_centroids(rgb, bundle)
        if not per_img:
            continue
        for center, cnt in per_img:
            all_centroids.append(center)
            all_weights.append(cnt)
        effective_posts += 1
        if idx % 20 == 0:
            elapsed = time.time() - t_start
            rate = idx / max(elapsed, 1e-6)
            eta = (len(urls) - idx) / max(rate, 1e-6)
            print(
                f"[preset] progress {idx}/{len(urls)} effective={effective_posts} "
                f"centroids={len(all_centroids)} elapsed={elapsed:.0f}s eta={eta:.0f}s"
            )

    if not all_centroids:
        print("[preset] FAIL — centroid 하나도 없음.")
        return 1

    centroids = np.stack(all_centroids, axis=0).astype(np.float32)
    weights = np.asarray(all_weights, dtype=np.float32)
    print(f"\n[preset] 총 centroid={centroids.shape[0]:,} 총 pixel={weights.sum():,.0f}")
    print(f"[preset] weighted KMeans(k={args.target_k}) on LAB …")
    pool_centers = _weighted_kmeans(centroids, weights, args.target_k, args.seed)

    pool_entries = [
        _centroid_to_entry(c, i, centroids.shape[0])
        for i, c in enumerate(pool_centers)
    ]
    supplemental = [
        {
            **entry,
            "lab": [
                round(float(v), 2)
                for v in rgb_to_lab(hex_to_rgb(entry["hex"])).tolist()
            ],
            "origin": "self_generated",
        }
        for entry in _SELF_GENERATED_COLORS
    ]
    all_entries = pool_entries + supplemental
    print(f"[preset] 풀에서 {len(pool_entries)} + 자체 {len(supplemental)} = {len(all_entries)}")

    print("\n[preset] 풀 유래 색 top-35")
    for entry in pool_entries:
        print(f"  {entry['name']:>10} {entry['hex']} LAB={entry['lab']}")
    print("\n[preset] 자체 보강 15")
    for entry in supplemental:
        print(f"  {entry['name']:>18} {entry['hex']}")

    _dump_outputs(pool_entries, all_entries)
    return 0


if __name__ == "__main__":
    sys.exit(main())
