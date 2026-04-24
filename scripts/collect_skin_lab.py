"""인도 skin tone LAB box 샘플 수집 — M3.A Step D 선행 작업.

전략: StarRocks 에서 hashtag entry 100 + profile entry 100 = 200 post 랜덤 추출,
각 포스트의 대표 이미지 한 장에 대해 YOLO+segformer 를 돌려 skin 클래스 (Face/Arm/Leg)
픽셀만 남기고 LAB 분포를 측정. L 은 느슨 (음영 포괄), a/b 는 타이트하게 잡는 것이 목표 —
L 이 과하게 좁으면 음영진 피부가 옷으로 새고, a/b 가 넓으면 옷 색이 drop 됨.

출력: L / a / b 축별 percentile + 추천 skin_lab_box YAML 스니펫.

실행:
  uv run python scripts/collect_skin_lab.py
  uv run python scripts/collect_skin_lab.py --max-posts 60 --device mps
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

import numpy as np
import pymysql
from dotenv import load_dotenv
from PIL import Image

from loaders.blob_downloader import BlobDownloader
from vision.color_space import SKIN_LAB_MAX, SKIN_LAB_MIN, rgb_to_lab
from vision.pipeline_b_extractor import (
    MIN_CROP_PX,
    SKIN_CLASS_IDS,
    detect_people,
    load_models,
    run_segformer,
)

_CACHE_DIR = _REPO / "outputs" / "blob_cache_skin_lab"
_MIN_SKIN_PX_PER_POST = 2000   # 이 미만이면 샘플 skip (bbox miss 또는 측면샷)
_MAX_IMAGE_SIDE = 1280          # segformer 입력 상한 (속도)
_HASHTAG_ENTRY = "hashtag"
_PROFILE_ENTRY = "profile"


def _connect() -> pymysql.Connection:
    load_dotenv()
    return pymysql.connect(
        host=os.environ["STARROCKS_HOST"],
        port=int(os.environ.get("STARROCKS_PORT", "9030")),
        user=os.environ["STARROCKS_USER"],
        password=os.environ["STARROCKS_PASSWORD"],
        database=os.environ.get("STARROCKS_DATABASE", "png"),
        connect_timeout=15,
        cursorclass=pymysql.cursors.DictCursor,
    )


def _load_post_urls(entry: str, limit: int) -> list[str]:
    """entry (hashtag|profile) 별로 download_urls 가 있는 post 의 첫 URL 을 반환."""
    sql = """
        SELECT download_urls FROM india_ai_fashion_inatagram_posting
        WHERE entry = %s AND download_urls IS NOT NULL AND download_urls != ''
    """
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(sql, (entry,))
        rows = cur.fetchall()
    urls: list[str] = []
    for row in rows:
        first = (row["download_urls"] or "").split(",")[0].strip()
        if first:
            urls.append(first)
    random.shuffle(urls)
    return urls[:limit]


def _load_rgb(path: Path) -> np.ndarray | None:
    """PIL 로 RGB 로드 + 최대 변 상한 리사이즈. 실패 시 None."""
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            scale = min(1.0, _MAX_IMAGE_SIDE / max(w, h))
            if scale < 1.0:
                im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            return np.asarray(im, dtype=np.uint8)
    except Exception:  # noqa: BLE001 — 손상된 이미지 skip
        return None


def _collect_skin_pixels(rgb: np.ndarray, bundle) -> np.ndarray:
    """한 이미지에서 skin class (12-17) 픽셀만 모아 (N,3) uint8 RGB 로 반환."""
    boxes = detect_people(bundle.yolo, rgb)
    if not boxes:
        h, w = rgb.shape[:2]
        boxes = [(0, 0, w, h)]
    skin_class_arr = np.asarray(tuple(SKIN_CLASS_IDS), dtype=np.int32)
    collected: list[np.ndarray] = []
    for x1, y1, x2, y2 in boxes:
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(rgb.shape[1], x2), min(rgb.shape[0], y2)
        if x2c - x1c < MIN_CROP_PX or y2c - y1c < MIN_CROP_PX:
            continue
        crop = rgb[y1c:y2c, x1c:x2c]
        seg = run_segformer(bundle, crop)
        skin_mask = np.isin(seg, skin_class_arr)
        if skin_mask.sum() > 0:
            collected.append(crop[skin_mask])
    if not collected:
        return np.zeros((0, 3), dtype=np.uint8)
    return np.concatenate(collected, axis=0)


def _report_distribution(lab: np.ndarray) -> dict[str, dict[str, float]]:
    """축별 percentile dict. L 은 loose (p1-p99), a/b 는 tight (p2.5-p97.5)."""
    axes = ("L", "a", "b")
    out: dict[str, dict[str, float]] = {}
    for idx, name in enumerate(axes):
        values = lab[:, idx]
        out[name] = {
            "p1": float(np.percentile(values, 1)),
            "p2_5": float(np.percentile(values, 2.5)),
            "p5": float(np.percentile(values, 5)),
            "median": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "p97_5": float(np.percentile(values, 97.5)),
            "p99": float(np.percentile(values, 99)),
            "mean": float(values.mean()),
            "std": float(values.std()),
        }
    return out


def _proposed_box(stats: dict[str, dict[str, float]]) -> tuple[list[float], list[float]]:
    """L 은 p1-p99 (loose), a/b 는 p2.5-p97.5 (tight). 설계 근거는 모듈 docstring."""
    min_box = [
        round(stats["L"]["p1"], 1),
        round(stats["a"]["p2_5"], 1),
        round(stats["b"]["p2_5"], 1),
    ]
    max_box = [
        round(stats["L"]["p99"], 1),
        round(stats["a"]["p97_5"], 1),
        round(stats["b"]["p97_5"], 1),
    ]
    return min_box, max_box


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-per-entry", type=int, default=100)
    parser.add_argument("--device", type=str, default=None,
                        help="mps/cuda/cpu (기본: 자동 감지)")
    parser.add_argument("--max-posts", type=int, default=200,
                        help="전체 상한 (샘플 풀 합계)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"[skin] StarRocks 에서 post URL 추출 (sample_per_entry={args.sample_per_entry})")
    hashtag_urls = _load_post_urls(_HASHTAG_ENTRY, args.sample_per_entry)
    profile_urls = _load_post_urls(_PROFILE_ENTRY, args.sample_per_entry)
    all_urls = (hashtag_urls + profile_urls)[: args.max_posts]
    print(f"[skin] hashtag={len(hashtag_urls)} profile={len(profile_urls)} total={len(all_urls)}")

    print(f"[skin] 모델 로드 (device={args.device or 'auto'}) — scene_filter 비활성")
    bundle = load_models(device=args.device, scene_filter_cfg=None)
    downloader = BlobDownloader.from_env()

    accumulated: list[np.ndarray] = []
    effective_posts = 0
    for idx, url in enumerate(all_urls, 1):
        dest = downloader.download(url, _CACHE_DIR)
        if dest is None:
            continue
        rgb = _load_rgb(dest)
        if rgb is None:
            continue
        skin = _collect_skin_pixels(rgb, bundle)
        if skin.shape[0] < _MIN_SKIN_PX_PER_POST:
            continue
        accumulated.append(skin)
        effective_posts += 1
        if idx % 10 == 0:
            print(f"[skin] progress {idx}/{len(all_urls)} effective={effective_posts}")

    if not accumulated:
        print("[skin] FAIL — skin pixel 하나도 못 모음. blob 접근 또는 segformer 문제 확인.")
        return 1

    pixels = np.concatenate(accumulated, axis=0)
    print(f"\n[skin] 유효 post={effective_posts}, 총 skin pixel={pixels.shape[0]:,}")
    lab = rgb_to_lab(pixels)
    stats = _report_distribution(lab)

    print("\n[skin] 축별 percentile")
    for axis, s in stats.items():
        print(
            f"  {axis}: p1={s['p1']:6.2f} p5={s['p5']:6.2f} med={s['median']:6.2f} "
            f"p95={s['p95']:6.2f} p99={s['p99']:6.2f} mean={s['mean']:6.2f} std={s['std']:5.2f}"
        )

    min_box, max_box = _proposed_box(stats)
    print("\n[skin] 기존 skin_lab_box (vision/color_space.py)")
    print(f"  min: {SKIN_LAB_MIN.tolist()}   max: {SKIN_LAB_MAX.tolist()}")
    print("[skin] 제안 skin_lab_box (L loose p1-p99, a/b tight p2.5-p97.5)")
    print(f"  min: {min_box}   max: {max_box}")
    print("\n[skin] configs/local.yaml YAML 스니펫:")
    print("  skin_lab_box:")
    print(f"    min: {min_box}")
    print(f"    max: {max_box}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
