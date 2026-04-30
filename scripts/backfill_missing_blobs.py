"""enriched JSON 의 image_urls / video_urls 중 blob 캐시에 없는 것만 다운로드.

`download_blobs.py` 는 posting.tsv 기반 (구 워크플로우). 이 스크립트는 enriched JSON 의
실제 사용 URL 을 모아 cache miss 만 골라 다운로드 — 검수 페이지 thumbnail 누락 해소용.

전제:
  - `uv sync --extra blob`
  - `.env` 에 AZURE_STORAGE_CONNECTION_STRING + AZURE_STORAGE_CONTAINER

실행:
  uv run python scripts/backfill_missing_blobs.py
  uv run python scripts/backfill_missing_blobs.py --dry-run

Phase 3 (2026-04-30): 12주 backfill 후 검수 페이지 thumbnail 진단:
  IG image cache miss 44 + IG Reel video miss 17 + YT video miss 408 = 469 blob.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import urlparse

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

_DEFAULT_GLOB = "outputs/backfill/page_*_enriched.json"
_DEFAULT_CACHE = _REPO / "sample_data" / "image_cache"


def _basename(url: str) -> str | None:
    if not url:
        return None
    path_only = url.split("?", 1)[0]
    bn = Path(urlparse(path_only).path).name
    return bn or None


def _collect_blob_paths(glob: str) -> tuple[list[str], list[str]]:
    """returns (image_blob_paths, video_blob_paths) — full blob path (container 상대).

    enriched 의 image_urls / video_urls 는 보통 'collectify/poc/...' 형태 (container
    상대). cache 의 file 은 basename 만 보존 (download_blobs 가 path tail 사용).
    """
    images: list[str] = []
    videos: list[str] = []
    seen_img: set[str] = set()
    seen_vid: set[str] = set()
    for p in sorted((_REPO).glob(glob)):
        try:
            items = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        for it in items:
            n = it.get("normalized") or {}
            for u in (n.get("image_urls") or []):
                bn = _basename(u)
                if bn and bn not in seen_img:
                    seen_img.add(bn)
                    images.append(u.split("?", 1)[0])
            for u in (n.get("video_urls") or []):
                bn = _basename(u)
                if bn and bn not in seen_vid:
                    seen_vid.add(bn)
                    videos.append(u.split("?", 1)[0])
    return images, videos


def _missing(blob_paths: list[str], cache_dir: Path) -> list[str]:
    cached = {p.name for p in cache_dir.iterdir() if not p.is_dir()}
    miss = []
    for bp in blob_paths:
        bn = Path(bp).name
        if not bn:
            continue
        if bn not in cached:
            miss.append(bp)
    return miss


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glob", default=_DEFAULT_GLOB)
    parser.add_argument("--dest", type=Path, default=_DEFAULT_CACHE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--include-yt", action="store_true",
        help="YT video 도 다운로드 (default: skip — YT mp4 는 보통 100MB+, 비싸다)",
    )
    args = parser.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)
    images, videos = _collect_blob_paths(args.glob)
    img_miss = _missing(images, args.dest)
    vid_miss = _missing(videos, args.dest)

    # YT/IG 분리 — YT 는 path 에 'youtube' 포함
    ig_video_miss = [v for v in vid_miss if "/youtube/" not in v]
    yt_video_miss = [v for v in vid_miss if "/youtube/" in v]

    print(f"[blobs] enriched files: {len(list(_REPO.glob(args.glob)))}")
    print(f"[blobs] distinct image basenames: {len(images)} miss: {len(img_miss)}")
    print(f"[blobs] distinct video basenames: {len(videos)} miss: {len(vid_miss)}")
    print(f"[blobs]   ↳ IG Reel video miss: {len(ig_video_miss)}")
    print(f"[blobs]   ↳ YT video miss: {len(yt_video_miss)} {'(skipped, use --include-yt)' if not args.include_yt else ''}")

    targets = img_miss + ig_video_miss
    if args.include_yt:
        targets += yt_video_miss
    print(f"[blobs] target downloads: {len(targets)}")

    if args.dry_run:
        for t in targets[:15]:
            print(f"  {t}")
        if len(targets) > 15:
            print(f"  ... +{len(targets) - 15} more")
        return 0

    if not targets:
        print("[blobs] nothing to download")
        return 0

    from loaders.blob_downloader import BlobDownloader
    downloader = BlobDownloader.from_env()
    print(f"[blobs] container={downloader.container} → dest={args.dest}")

    downloaded = 0
    failed = 0
    for i, p in enumerate(targets):
        if i % 50 == 0:
            print(f"  [{i}/{len(targets)}] {Path(p).name}")
        result = downloader.download(p, args.dest)
        if result is None:
            failed += 1
        else:
            downloaded += 1
    print(f"[blobs] done downloaded={downloaded} failed={failed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
