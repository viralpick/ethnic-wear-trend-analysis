"""posting.tsv 의 image_paths 를 Azure Blob 에서 로컬 캐시로 다운로드.

전제:
  - `uv sync --extra blob` (azure-storage-blob + python-dotenv)
  - `.env` 에 AZURE_STORAGE_CONNECTION_STRING 설정 (예시 .env.example)

실행:
  uv run python scripts/download_blobs.py
  uv run python scripts/download_blobs.py --dry-run
  uv run python scripts/download_blobs.py \
      --tsv-dir sample_data --dest sample_data/image_cache

이후 Pipeline B 를 전수 실행:
  uv run daily --source tsv --tsv-dir sample_data \
      --color-extractor pipeline_b --image-root sample_data/image_cache
  uv run python scripts/pipeline_b_smoke.py --image-root sample_data/image_cache
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

_POSTING_FILE = "png_india_ai_fashion_inatagram_posting.tsv"


def collect_blob_paths(tsv_path: Path) -> list[str]:
    """posting.tsv col [11] (image_paths, csv-joined) 를 펼쳐 blob path 리스트로."""
    if not tsv_path.exists():
        print(f"[blobs] TSV not found: {tsv_path}")
        return []
    paths: list[str] = []
    with tsv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            if len(row) < 11:
                continue
            for item in (row[10] or "").split(","):
                stripped = item.strip()
                if stripped:
                    paths.append(stripped)
    return paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download posting.tsv blobs to local cache.")
    parser.add_argument("--tsv-dir", type=Path, default=_REPO / "sample_data")
    parser.add_argument("--dest", type=Path, default=_REPO / "sample_data" / "image_cache")
    parser.add_argument("--dry-run", action="store_true", help="경로만 출력, 다운로드 안 함")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    tsv_path = args.tsv_dir / _POSTING_FILE
    paths = collect_blob_paths(tsv_path)
    print(f"[blobs] {len(paths)} blob paths from {tsv_path.name}")

    if args.dry_run:
        for p in paths[:10]:
            print(f"  {p}")
        if len(paths) > 10:
            print(f"  ... (+{len(paths) - 10} more)")
        return

    from loaders.blob_downloader import BlobDownloader
    downloader = BlobDownloader.from_env()
    print(f"[blobs] container={downloader.container} → dest={args.dest}")

    downloaded = 0
    cached_skip = 0
    failed = 0
    for p in paths:
        existed = (args.dest / Path(p).name).exists()
        result = downloader.download(p, args.dest)
        if result is None:
            failed += 1
        elif existed:
            cached_skip += 1
        else:
            downloaded += 1
    print(f"[blobs] done downloaded={downloaded} cached_skip={cached_skip} failed={failed}")


if __name__ == "__main__":
    main()
