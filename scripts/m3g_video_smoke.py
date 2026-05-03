"""M3.G — 1-post live smoke (IG video frame phase).

특정 IG post_id 를 StarRocks 에서 1건 pull 한 뒤
RawInstagramPost → normalize → PipelineBColorExtractor 로 흐름을 돌려
video frame 분석 (cv2 → JPEG 결정론 → Gemini live → canonical/post_palette) 을 검증.

목적:
1. cv2.VideoCapture 가 Azure Blob 에서 다운로드한 mp4 를 frame 으로 추출
2. _encode_jpeg_deterministic 이 결정론 bytes 생성 → VisionLLMClient cache 안정
3. _analyze_images 가 image+video frame 을 같은 흐름으로 받아 Gemini 호출
4. canonical/post_palette 결과 채워짐

비용: Gemini 2.5 Flash live 호출 N회 (image + N frame). 1-post 한정.

사용:
    uv run python scripts/m3g_video_smoke.py --post-id 01KQ6F1HM81FE1QGPPN8BN4FJC

전제조건:
    .env: STARROCKS_*, GEMINI_API_KEY, AZURE_STORAGE_CONNECTION_STRING, AZURE_STORAGE_CONTAINER
    Pritunl VPN ON (Azure Storage IP allow-list)
    uv sync --extra vision (cv2 / torch / transformers / ultralytics)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

load_dotenv()


def _fetch_one_post(post_id: str) -> dict:
    from loaders.starrocks_connect import connect_raw
    conn = connect_raw()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                p.id, p.user, p.url, p.posting_at, p.content,
                p.like_count, p.comment_count, p.entry,
                p.download_urls, p.created_at,
                COALESCE(prof.follower_count, 0) AS follower_count
            FROM india_ai_fashion_inatagram_posting p
            LEFT JOIN india_ai_fashion_inatagram_profile prof
                ON prof.user = p.user
            WHERE p.id = %s
            """,
            (post_id,),
        )
        row = cur.fetchone()
        if row is None:
            raise SystemExit(f"post_id={post_id} not found")
        return row
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="M3.G 1-post video smoke")
    parser.add_argument("--post-id", required=True, help="IG post_id (e.g. 01KQ6F1HM81FE1QGPPN8BN4FJC)")
    parser.add_argument("--blob-cache", default="sample_data/image_cache",
                        help="Azure Blob 다운로드 캐시 디렉토리")
    args = parser.parse_args()

    from loaders.starrocks_raw_loader import _build_ig_post  # noqa: E402
    from normalization.normalize_content import normalize_instagram_post  # noqa: E402
    from settings import load_settings  # noqa: E402
    from pipelines.run_daily_pipeline import _select_color_extractor  # noqa: E402

    print(f"== Fetch post_id={args.post_id} ==")
    row = _fetch_one_post(args.post_id)
    raw = _build_ig_post(row)
    if raw is None:
        raise SystemExit("raw post build failed")

    print(f"  account={raw.account_handle}  posted={raw.post_date}")
    print(f"  image_urls={len(raw.image_urls)}  video_urls={len(raw.video_urls)}")
    for vurl in raw.video_urls:
        print(f"    video: {vurl[:100]}")

    item = normalize_instagram_post(raw)
    print(f"== Normalized item: image={len(item.image_urls)} video={len(item.video_urls)} ==")

    settings = load_settings()
    cache_dir = Path(args.blob_cache)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("== Loading PipelineBColorExtractor (vision extras) ==")
    extractor = _select_color_extractor(
        choice="pipeline_b",
        settings=settings,
        image_root=None,
        blob_cache=cache_dir,
        vision_llm_choice="gemini",
    )

    print(f"== extract_visual([item]) — Gemini live, video_urls={len(item.video_urls)} ==")
    results = extractor.extract_visual([item])
    if not results:
        raise SystemExit("extract_visual returned empty")

    res = results[0]
    print(f"== Result ==")
    print(f"  canonicals: {len(res.canonicals)}")
    for i, can in enumerate(res.canonicals):
        rep = can.representative
        member_ids = [m.image_id for m in can.members]
        print(f"    [{i}] members={len(can.members)} image_ids={member_ids[:3]}{'…' if len(member_ids) > 3 else ''}")
        print(f"        upper={rep.upper_garment_type}/{rep.upper_is_ethnic} "
              f"lower={rep.lower_garment_type}/{rep.lower_is_ethnic} "
              f"picks={rep.color_preset_picks_top3}")
        for j, c in enumerate(can.palette[:3]):
            print(f"        cluster[{j}] hex={c.hex} family={c.family} share={c.share:.3f}")
    print(f"  post_palette ({len(res.post_palette)}):")
    for j, c in enumerate(res.post_palette):
        print(f"    [{j}] hex={c.hex} family={c.family} share={c.share:.3f}")


if __name__ == "__main__":
    main()
