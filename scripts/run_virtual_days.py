"""가상 7일치 배치 순차 실행 스크립트.

posting_at 기준 정렬된 전체 데이터를 200건씩 7개 가상 day로 나눠 파이프라인을 실행한다.
score_history.json이 누적되므로 실행 후 weekly direction과 momentum_post_growth가 실제 값을 가진다.

사용법:
  uv run python scripts/run_virtual_days.py              # 7일치 전부
  uv run python scripts/run_virtual_days.py --batches 3  # 처음 3일치만
  uv run python scripts/run_virtual_days.py --reset      # score_history 초기화 후 실행
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

_BASE_DATE = date(2026, 4, 21)
_DEFAULT_BATCHES = 7
_HISTORY_PATH = Path("outputs/score_history.json")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="가상 7일치 배치 파이프라인 순차 실행")
    parser.add_argument("--batches", type=int, default=_DEFAULT_BATCHES,
                        help=f"실행할 배치 수 (기본: {_DEFAULT_BATCHES})")
    parser.add_argument("--reset", action="store_true",
                        help="실행 전 score_history.json 초기화")
    parser.add_argument("--source", default="starrocks",
                        choices=["starrocks", "local", "tsv"],
                        help="raw source loader (기본: starrocks)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.reset and _HISTORY_PATH.exists():
        _HISTORY_PATH.unlink()
        print(f"[reset] {_HISTORY_PATH} 삭제 완료")

    for i in range(args.batches):
        d = _BASE_DATE + timedelta(days=i)
        print(f"\n{'='*50}")
        print(f"배치 {i}  날짜 {d}  ({i+1}/{args.batches})")
        print("=" * 50)
        result = subprocess.run(
            [sys.executable, "-m", "pipelines.run_daily_pipeline",
             "--source", args.source, "--date", d.isoformat()],
            check=False,
        )
        if result.returncode != 0:
            print(f"[ERROR] 배치 {i} 실패 (returncode={result.returncode}). 중단.")
            sys.exit(result.returncode)

    print(f"\n완료: {args.batches}개 배치 실행, score_history → {_HISTORY_PATH}")


if __name__ == "__main__":
    main()
