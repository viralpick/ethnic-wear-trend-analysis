"""posting 이미지 blob 경로를 Azure Blob 에서 로컬 캐시로 다운로드.

소스 2가지:
  - tsv       — sample_data/png_india_ai_fashion_inatagram_posting.tsv 의 image_paths 컬럼
  - starrocks — png.india_ai_fashion_inatagram_posting 의 download_urls 컬럼 (실 DB)

전제:
  - `uv sync --extra blob` (+ starrocks 소스는 `--extra starrocks` 도)
  - `.env` 에 AZURE_STORAGE_CONNECTION_STRING 설정 (+ starrocks 는 STARROCKS_* 도)

실행:
  uv run python scripts/download_blobs.py                       # 기본 tsv 소스
  uv run python scripts/download_blobs.py --dry-run
  uv run python scripts/download_blobs.py --source starrocks    # 전체 DB 데이터
  uv run python scripts/download_blobs.py --source starrocks --date 2026-04-22
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import date, datetime, time, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

_POSTING_FILE = "png_india_ai_fashion_inatagram_posting.tsv"
_POSTING_TABLE = "india_ai_fashion_inatagram_posting"


def collect_from_tsv(tsv_path: Path) -> list[str]:
    """posting.tsv col [11] (image_paths, csv-joined) → blob path list."""
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


def collect_from_starrocks(target_date: date | None = None) -> list[str]:
    """StarRocks posting.download_urls → blob path list. target_date 면 당일 created_at 만."""
    from loaders.starrocks_reader import StarRocksReader

    reader = StarRocksReader.from_env()
    if target_date is not None:
        start = datetime.combine(target_date, time.min).replace(tzinfo=timezone.utc)
        end = datetime.combine(target_date, time.max).replace(tzinfo=timezone.utc)
        rows = reader.select(
            f"SELECT download_urls FROM {_POSTING_TABLE} "
            "WHERE created_at >= %s AND created_at <= %s",
            (start, end),
        )
    else:
        rows = reader.select(f"SELECT download_urls FROM {_POSTING_TABLE}")

    paths: list[str] = []
    for r in rows:
        for item in (r.get("download_urls") or "").split(","):
            stripped = item.strip()
            if stripped:
                paths.append(stripped)
    return paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download posting image blobs to local cache.")
    parser.add_argument(
        "--source", choices=["tsv", "starrocks"], default="tsv",
        help="blob 경로 소스. tsv=posting.tsv 파일, starrocks=DB posting.download_urls.",
    )
    parser.add_argument("--tsv-dir", type=Path, default=_REPO / "sample_data",
                        help="--source tsv 일 때 TSV 디렉토리.")
    parser.add_argument("--date", type=str, default=None,
                        help="--source starrocks 일 때 ISO 날짜 필터 (미지정=전체).")
    parser.add_argument("--dest", type=Path, default=_REPO / "sample_data" / "image_cache")
    parser.add_argument("--dry-run", action="store_true", help="경로만 출력, 다운로드 안 함")
    return parser.parse_args()


def _collect(args: argparse.Namespace) -> list[str]:
    if args.source == "starrocks":
        target = datetime.strptime(args.date, "%Y-%m-%d").date() if args.date else None
        print(f"[blobs] source=starrocks target_date={target}")
        return collect_from_starrocks(target)
    tsv_path = args.tsv_dir / _POSTING_FILE
    print(f"[blobs] source=tsv path={tsv_path}")
    return collect_from_tsv(tsv_path)


def main() -> None:
    args = _parse_args()
    paths = _collect(args)
    print(f"[blobs] {len(paths)} blob paths")

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
