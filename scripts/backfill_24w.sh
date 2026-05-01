#!/usr/bin/env bash
# 24주 backfill orchestrator — backfill_12w.sh wrapper.
#
# 옛 12w manifest (outputs/backfill/) 는 그대로 보존하고 24w 는 별도 dir 사용.
# raw DB 의 newest first → oldest, page 80 부터 (24주 ≈ 7920 posts ≈ 80 pages).
# raw 데이터가 START_INDEX 보다 적으면 enriched empty 로 자연 종료.
#
# env override:
#   START_INDEX  default 80   (newest batch index)
#   END_INDEX    default 0    (oldest batch index)
#   PAGE_SIZE    default 100
#   DATE         default 2026-05-02
#   OUT_DIR      default outputs/backfill_24w
#
# 사용:
#   bash scripts/backfill_24w.sh                # 24주 full
#   START_INDEX=1 END_INDEX=0 bash scripts/backfill_24w.sh   # 1 page sanity
set -euo pipefail
cd "$(dirname "$0")/.."

export START_INDEX="${START_INDEX:-80}"
export END_INDEX="${END_INDEX:-0}"
export PAGE_SIZE="${PAGE_SIZE:-100}"
export DATE="${DATE:-2026-05-02}"
export OUT_DIR="${OUT_DIR:-outputs/backfill_24w}"

echo "[backfill_24w] START_INDEX=$START_INDEX END_INDEX=$END_INDEX OUT_DIR=$OUT_DIR DATE=$DATE"
exec bash scripts/backfill_12w.sh
