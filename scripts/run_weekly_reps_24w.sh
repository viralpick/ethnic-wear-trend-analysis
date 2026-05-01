#!/usr/bin/env bash
# 24주 weekly representative orchestrator — run_weekly_reps.sh wrapper.
#
# oldest → newest 순서로 24주 (2025-11-16 ~ 2026-04-26 Sunday). representative
# phase 가 emergence rule + hashtag_weekly + unknown_signal 동시 적재.
#
# env override:
#   N_WEEKS        default 24
#   LATEST_SUNDAY  default 2026-04-26  (가장 최신 anchor, oldest = LATEST - (N-1)*7)
#   GLOB           default outputs/backfill_24w/page_*_enriched.json
#   SINK           default starrocks
#
# 사용:
#   bash scripts/run_weekly_reps_24w.sh                       # 24w full
#   N_WEEKS=4 LATEST_SUNDAY=2026-04-26 bash scripts/run_weekly_reps_24w.sh   # 4w sanity
set -euo pipefail
cd "$(dirname "$0")/.."

export N_WEEKS="${N_WEEKS:-24}"
export LATEST_SUNDAY="${LATEST_SUNDAY:-2026-04-26}"
GLOB="${GLOB:-outputs/backfill_24w/page_*_enriched.json}"

echo "[run_weekly_reps_24w] N_WEEKS=$N_WEEKS LATEST_SUNDAY=$LATEST_SUNDAY GLOB=$GLOB"
exec bash scripts/run_weekly_reps.sh "$GLOB"
