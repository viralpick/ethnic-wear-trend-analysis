#!/usr/bin/env bash
# 16w 백필 종료 후 통합 파이프라인 — rep 산출 + review HTML + diff + quality 진단.
#
# 사용:
#   bash scripts/post_backfill_pipeline.sh
#
# 단계:
#   1. run_weekly_reps_24w.sh   — rep + emergence + LLM 분류
#   2. review HTML 빌드          — 16주 분 통합
#   3. post v2.3 snapshot        — 4월 monthly Top 재산출
#   4. pre/post diff             — Top1 동일성 / overlap / Δ score
#   5. v2.3 데이터 품질 진단     — 7 SQL queries
#
# 각 step 실패 시 즉시 exit (set -e). 결과 로그는 /tmp/post_backfill.log 에 누적.
set -euo pipefail
cd "$(dirname "$0")/.."

LOG="/tmp/post_backfill.log"
echo "=== post_backfill_pipeline start: $(date '+%Y-%m-%d %H:%M:%S %Z') ===" | tee -a "$LOG"

# ---- 1. weekly rep 산출 (rep + emergence + LLM 분류) ----
echo "[1/5] run_weekly_reps_24w.sh — rep + emergence + LLM 분류" | tee -a "$LOG"
N_WEEKS=16 LATEST_SUNDAY=2026-04-26 \
  GLOB="outputs/backfill_16w/page_*_enriched.json" \
  bash scripts/run_weekly_reps_24w.sh 2>&1 | tee -a "$LOG"

# ---- 2. review HTML 빌드 ----
echo "[2/5] review HTML build (16w)" | tee -a "$LOG"
uv run python scripts/build_review_html.py \
  --weeks 2026-04-26,2026-04-19,2026-04-12,2026-04-05,2026-03-29,2026-03-22,2026-03-15,2026-03-08,2026-03-01,2026-02-22,2026-02-15,2026-02-08,2026-02-01,2026-01-25,2026-01-18,2026-01-11 \
  --output outputs/weekly_review/review_16w.html 2>&1 | tee -a "$LOG"

# ---- 3. post v2.3 monthly Top snapshot ----
echo "[3/5] post v2.3 monthly Top snapshot" | tee -a "$LOG"
uv run python scripts/snapshot_april_monthly_top.py \
  --output outputs/april_monthly_top_post_v2.3.json \
  --note "post v2.3 rep run (16w backfill 완료)" 2>&1 | tee -a "$LOG"

# ---- 4. pre vs post diff ----
echo "[4/5] pre vs post v2.3 monthly Top diff" | tee -a "$LOG"
uv run python scripts/compare_april_monthly_top.py \
  --pre outputs/april_monthly_top_pre_v2.3.json \
  --post outputs/april_monthly_top_post_v2.3.json 2>&1 | tee -a "$LOG"

# ---- 5. v2.3 데이터 품질 진단 ----
echo "[5/5] v2.3 데이터 품질 자동 진단 (7 SQL queries)" | tee -a "$LOG"
uv run python scripts/diagnose_v2_3_quality.py \
  --output outputs/v2_3_quality_report.json 2>&1 | tee -a "$LOG"

echo "=== post_backfill_pipeline done: $(date '+%Y-%m-%d %H:%M:%S %Z') ===" | tee -a "$LOG"
echo "산출물:" | tee -a "$LOG"
echo "  - outputs/weekly_review/review_16w.html" | tee -a "$LOG"
echo "  - outputs/april_monthly_top_post_v2.3.json" | tee -a "$LOG"
echo "  - outputs/v2_3_quality_report.json" | tee -a "$LOG"
echo "  - 로그: $LOG" | tee -a "$LOG"
