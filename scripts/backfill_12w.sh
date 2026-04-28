#!/usr/bin/env bash
# 12주 backfill orchestrator — 100 post 단위로 page-index 36 → 0 까지 차례차례.
#
# 사용:
#   bash scripts/backfill_12w.sh
#
# 각 batch 후:
# - outputs/backfill/page_<idx>_smoke.log
# - outputs/backfill/page_<idx>_review.html (검수 HTML)
# - outputs/backfill/manifest.csv (시작/완료/exit code 누적 기록)
#
# Ctrl-C 또는 SIGTERM 으로 안전하게 중단 (현재 batch 끝까지 기다린 후 stop).
set -uo pipefail
cd "$(dirname "$0")/.."

# --------- config ---------
START_INDEX=36       # 가장 최신 batch (offset 3600~3661, 61 post)
END_INDEX=0          # 가장 오래된 batch
PAGE_SIZE=100
DATE="2026-04-29"
TEXT_WORKERS=4
VISION_WORKERS=8
OUT_DIR="outputs/backfill"
mkdir -p "$OUT_DIR"

MANIFEST="$OUT_DIR/manifest.csv"
if [[ ! -f "$MANIFEST" ]]; then
  echo "page_index,started_at,finished_at,exit_code,gemini_calls,enriched_items,clusters" > "$MANIFEST"
fi

# --------- signal handling ---------
STOP=0
trap 'echo "[backfill] stop requested — current batch 완료 후 종료"; STOP=1' INT TERM

# --------- resume: manifest 에 exit=0 으로 완료된 batch skip ---------
declare -A DONE
if [[ -f "$MANIFEST" ]]; then
  while IFS=, read -r p_idx _started _finished p_exit _gem _items _clu; do
    [[ "$p_idx" == "page_index" ]] && continue  # header
    if [[ "$p_exit" == "0" ]]; then
      DONE[$p_idx]=1
    fi
  done < "$MANIFEST"
  echo "[backfill] resume: ${#DONE[@]} batch already complete (exit=0), will skip"
fi

# --------- main loop ---------
for (( idx=START_INDEX; idx>=END_INDEX; idx-- )); do
  if [[ $STOP -eq 1 ]]; then
    echo "[backfill] stopped before page_index=$idx"
    break
  fi

  if [[ -n "${DONE[$idx]:-}" ]]; then
    echo "[backfill] page_index=$idx skip (이미 완료)"
    continue
  fi

  LOG="$OUT_DIR/page_${idx}_smoke.log"
  ENRICHED="$OUT_DIR/page_${idx}_enriched.json"
  SUMMARIES="$OUT_DIR/page_${idx}_summaries.json"
  REVIEW="$OUT_DIR/page_${idx}_review.html"

  STARTED=$(date -Iseconds)
  echo "============================================================"
  echo "[backfill] page_index=$idx page_size=$PAGE_SIZE date=$DATE"
  echo "[backfill] started=$STARTED"
  echo "============================================================"

  # 1. Pipeline run
  uv run python src/pipelines/run_daily_pipeline.py \
    --source starrocks \
    --color-extractor pipeline_b \
    --vision-llm gemini \
    --llm azure-openai \
    --sink starrocks \
    --window-mode count \
    --page-size "$PAGE_SIZE" \
    --page-index "$idx" \
    --blob-cache sample_data/image_cache \
    --text-workers "$TEXT_WORKERS" \
    --vision-workers "$VISION_WORKERS" \
    --date "$DATE" 2>&1 | tee "$LOG"

  EXIT_CODE=${PIPESTATUS[0]}
  FINISHED=$(date -Iseconds)

  # 2. enriched.json / summaries.json 보존 (default outputs/<DATE>/ 또는 outputs/)
  for cand in "outputs/$DATE/enriched.json" "outputs/enriched.json"; do
    if [[ -f "$cand" ]]; then
      cp "$cand" "$ENRICHED"
      break
    fi
  done
  for cand in "outputs/$DATE/summaries.json" "outputs/summaries.json"; do
    if [[ -f "$cand" ]]; then
      cp "$cand" "$SUMMARIES"
      break
    fi
  done

  # 3. 검수 HTML 빌드 (enriched/summaries 있을 때만)
  if [[ -f "$ENRICHED" && -f "$SUMMARIES" ]]; then
    uv run python scripts/build_review_html.py \
      --enriched "$ENRICHED" \
      --summaries "$SUMMARIES" \
      --output "$REVIEW" 2>&1 | tail -2
  else
    echo "[backfill] enriched/summaries 미생성 — review HTML skip"
  fi

  # 4. 메트릭 추출
  GEMINI=$(grep -c "generativelanguage" "$LOG" 2>/dev/null || echo 0)
  N_ITEMS=$(python3 -c "import json; print(len(json.load(open('$ENRICHED'))))" 2>/dev/null || echo 0)
  N_CLUSTERS=$(python3 -c "import json; print(len(json.load(open('$SUMMARIES'))))" 2>/dev/null || echo 0)

  # 5. manifest 기록
  echo "$idx,$STARTED,$FINISHED,$EXIT_CODE,$GEMINI,$N_ITEMS,$N_CLUSTERS" >> "$MANIFEST"
  echo "[backfill] page_index=$idx done — exit=$EXIT_CODE gemini=$GEMINI items=$N_ITEMS clusters=$N_CLUSTERS"
done

echo "============================================================"
echo "[backfill] all done. manifest: $MANIFEST"
echo "============================================================"
