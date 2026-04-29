#!/usr/bin/env bash
# 12 주 weekly representative orchestrator — oldest → newest 순서로 호출.
#
# score_history_weekly.json 이 매 호출마다 누적 write 되므로 순서가 중요:
# - oldest 부터 호출 → 마지막 (이번 주) rep 의 trajectory[12] 가 풀 채워짐
# - newest 부터 호출 → 마지막 rep 의 trajectory 는 자기 자신만 + zeros
#
# 각 주는 Mon~Sun IST. end_date = Sunday.
#
# 사용:
#   bash scripts/run_weekly_reps.sh
#   bash scripts/run_weekly_reps.sh outputs/backfill/page_*_enriched.json
set -euo pipefail
cd "$(dirname "$0")/.."

GLOB="${1:-outputs/backfill/page_*_enriched.json}"
SINK="${SINK:-starrocks}"

# 12 주차 — 각 주의 Sunday (end_date). oldest first.
WEEKS=(
  2026-02-15  # 1주차 (가장 과거)
  2026-02-22
  2026-03-01
  2026-03-08
  2026-03-15
  2026-03-22
  2026-03-29
  2026-04-05
  2026-04-12
  2026-04-19
  2026-04-26  # 12주차 (이번 주)
)

OUT_DIR="outputs/weekly_review"
mkdir -p "$OUT_DIR"

echo "============================================================"
echo "[weekly_reps] enriched_glob=$GLOB sink=$SINK"
echo "[weekly_reps] ${#WEEKS[@]} weeks (oldest→newest)"
echo "============================================================"

for end_date in "${WEEKS[@]}"; do
  # macOS date: -j (no time), -v-6d (-6 days), -f input format
  start_date=$(date -j -v-6d -f "%Y-%m-%d" "$end_date" "+%Y-%m-%d")
  echo ""
  echo "[weekly_reps] week: $start_date ~ $end_date"
  uv run python src/pipelines/run_daily_pipeline.py \
    --phase representative \
    --start-date "$start_date" \
    --end-date "$end_date" \
    --enriched-glob "$GLOB" \
    --sink "$SINK" 2>&1 | tail -5
done

echo ""
echo "============================================================"
echo "[weekly_reps] all 12 weeks done"
echo "============================================================"
echo ""
echo "이제 multi-week HTML 빌드:"
echo "  uv run python scripts/build_review_html.py \\"
echo "    --weeks $(IFS=,; echo "${WEEKS[*]}") \\"
echo "    --output $OUT_DIR/review_12w.html"
