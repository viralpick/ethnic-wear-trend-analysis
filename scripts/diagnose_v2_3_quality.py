"""Phase 2 v2.3 데이터 품질 자동 진단 — 7 SQL queries.

backfill + run_weekly_reps_24w.sh 종료 후 자동 실행. 결과 JSON 저장 + 콘솔 출력.

검수 포인트:
② Representative 결과 (drift / partial cluster 효과)
③ Vision 결과 (signal_type 분포, canonical 비율 proxy)
④ 데이터 분포 sanity (source 균형 IG/YT)
⑤ KPI 수치화 (LLM 분류율 / noise 감소율 / 시계열 패턴)
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone

import pymysql
from dotenv import load_dotenv

load_dotenv()


def _connect():
    return pymysql.connect(
        host=os.environ["STARROCKS_HOST"],
        port=int(os.environ["STARROCKS_PORT"]),
        user=os.environ["STARROCKS_USER"],
        password=os.environ["STARROCKS_PASSWORD"],
        database=os.environ.get("STARROCKS_RESULT_DATABASE", "ethnic_result"),
        autocommit=True,
    )


def _q(cur, sql: str, params: tuple = ()) -> list[dict]:
    cur.execute(sql, params)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _pct(n: int, total: int) -> str:
    if total == 0:
        return "—"
    return f"{n/total*100:.1f}%"


def main(out_path: str) -> None:
    conn = _connect()
    report: dict = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "queries": {},
    }
    try:
        with conn.cursor() as cur:
            # ================================================================
            # Q1. signal_type 분포 (Tier 4 효과 — vision 단어 inject 결과)
            # ================================================================
            print("=" * 70)
            print("Q1. signal_type 분포 (v2.3 row only)")
            print("=" * 70)
            rows = _q(cur, """
                SELECT COALESCE(signal_type, 'NULL_legacy') AS signal_type, COUNT(*) AS n
                FROM unknown_signal_latest
                WHERE schema_version = 'pipeline_v2.3'
                GROUP BY signal_type
                ORDER BY n DESC
            """)
            total = sum(r["n"] for r in rows) or 1
            for r in rows:
                print(f'  {r["signal_type"]:<25} {r["n"]:>5}  ({r["n"]/total*100:5.1f}%)')
            report["queries"]["q1_signal_type_distribution"] = rows

            # ================================================================
            # Q2. LLM 분류 비율 (Tier 3 효과)
            # ================================================================
            print()
            print("=" * 70)
            print("Q2. LLM 분류 적용 비율 (likely_category 채워진 row)")
            print("=" * 70)
            rows = _q(cur, """
                SELECT
                  COUNT(*) AS total_v23,
                  SUM(CASE WHEN likely_category IS NOT NULL THEN 1 ELSE 0 END) AS classified,
                  SUM(CASE WHEN likely_category IS NULL THEN 1 ELSE 0 END) AS unclassified
                FROM unknown_signal_latest
                WHERE schema_version = 'pipeline_v2.3'
            """)
            r = rows[0] if rows else {"total_v23": 0, "classified": 0, "unclassified": 0}
            t = r.get("total_v23") or 0
            c = r.get("classified") or 0
            u = r.get("unclassified") or 0
            print(f'  total v2.3:    {t}')
            print(f'  classified:    {c:>5} ({_pct(c, t)})')
            print(f'  unclassified:  {u:>5} ({_pct(u, t)})')
            report["queries"]["q2_llm_classification_rate"] = r

            # ================================================================
            # Q3. likely_category 카테고리 분포
            # ================================================================
            print()
            print("=" * 70)
            print("Q3. likely_category 분포 (LLM 분류 후 fashion 카테고리)")
            print("=" * 70)
            rows = _q(cur, """
                SELECT
                  CASE
                    WHEN likely_category LIKE '%%:%%'
                    THEN SUBSTRING_INDEX(likely_category, ':', 1)
                    ELSE likely_category
                  END AS category,
                  COUNT(*) AS n
                FROM unknown_signal_latest
                WHERE schema_version = 'pipeline_v2.3' AND likely_category IS NOT NULL
                GROUP BY category
                ORDER BY n DESC
            """)
            total = sum(r["n"] for r in rows) or 1
            for r in rows:
                print(f'  {r["category"] or "":<20} {r["n"]:>5}  ({r["n"]/total*100:5.1f}%)')
            report["queries"]["q3_category_distribution"] = rows

            # ================================================================
            # Q4. partial cluster 분포 (PR #63 옵션 C 롤백 효과)
            # ================================================================
            print()
            print("=" * 70)
            print("Q4. partial cluster 분포 (PR #63 옵션 C 롤백 검증)")
            print("=" * 70)
            rows = _q(cur, """
                SELECT
                  CASE
                    WHEN representative_key LIKE '%%__unknown%%'
                      OR representative_key LIKE 'unknown__%%'
                    THEN 'partial'
                    ELSE 'exact'
                  END AS cluster_type,
                  COUNT(*) AS n,
                  SUM(CASE WHEN score_total IS NULL THEN 1 ELSE 0 END) AS null_score,
                  ROUND(AVG(score_total), 2) AS avg_score
                FROM representative_weekly_latest
                WHERE week_start_date >= '2026-01-26'
                GROUP BY cluster_type
            """)
            for r in rows:
                drift = " ⚠️ score=NULL drift!" if (r["null_score"] or 0) > 0 else " ✅"
                print(f'  {r["cluster_type"]:<10} n={r["n"]:>5} null_score={r["null_score"]:>3} '
                      f'avg={r["avg_score"]}{drift}')
            report["queries"]["q4_partial_cluster_distribution"] = rows

            # ================================================================
            # Q5. source 균형 (IG vs YT)
            # ================================================================
            print()
            print("=" * 70)
            print("Q5. source 균형 — week 별 IG/YT factor_contribution 평균")
            print("=" * 70)
            rows = _q(cur, """
                SELECT week_start_date,
                  COUNT(*) AS n_clusters,
                  ROUND(AVG(get_json_double(CAST(factor_contribution AS VARCHAR), '$.instagram')), 3) AS avg_ig,
                  ROUND(AVG(get_json_double(CAST(factor_contribution AS VARCHAR), '$.youtube')), 3) AS avg_yt
                FROM representative_weekly_latest
                WHERE week_start_date >= '2026-01-26'
                GROUP BY week_start_date
                ORDER BY week_start_date DESC
                LIMIT 16
            """)
            for r in rows:
                print(f'  {r["week_start_date"]} clusters={r["n_clusters"]:>3} '
                      f'IG={r["avg_ig"]:.3f} YT={r["avg_yt"]:.3f}')
            report["queries"]["q5_source_balance"] = rows

            # ================================================================
            # Q6. 16w 시계열 surface 패턴 (균등 vs 폭발)
            # ================================================================
            print()
            print("=" * 70)
            print("Q6. 16w surface 시계열 (균등 분포 vs 폭발 주)")
            print("=" * 70)
            rows = _q(cur, """
                SELECT week_start_date, COUNT(*) AS n_surfaced
                FROM unknown_signal_latest
                WHERE schema_version = 'pipeline_v2.3'
                GROUP BY week_start_date
                ORDER BY week_start_date DESC
                LIMIT 16
            """)
            counts = [r["n_surfaced"] for r in rows]
            avg = sum(counts) / len(counts) if counts else 0
            for r in rows:
                bar = "█" * int(r["n_surfaced"])
                marker = " ⚠️ spike" if r["n_surfaced"] > avg * 2 else ""
                print(f'  {r["week_start_date"]} {r["n_surfaced"]:>3} {bar}{marker}')
            print(f'  평균: {avg:.1f}건/주, 총 {sum(counts)}건')
            report["queries"]["q6_weekly_surface_pattern"] = rows

            # ================================================================
            # Q7. hashtag_weekly schema_version 분포
            # ================================================================
            print()
            print("=" * 70)
            print("Q7. hashtag_weekly schema_version 분포 (v2.3 적재 검증)")
            print("=" * 70)
            rows = _q(cur, """
                SELECT schema_version, COUNT(*) AS n,
                  SUM(CASE WHEN is_known_mapping = 1 THEN 1 ELSE 0 END) AS known,
                  SUM(CASE WHEN is_known_mapping = 0 THEN 1 ELSE 0 END) AS unknown_mapping
                FROM hashtag_weekly_latest
                GROUP BY schema_version
                ORDER BY schema_version DESC
            """)
            for r in rows:
                total_r = (r["known"] or 0) + (r["unknown_mapping"] or 0)
                known_pct = _pct(r["known"] or 0, total_r) if total_r else "—"
                print(f'  {r["schema_version"]:<20} n={r["n"]:>5} '
                      f'known={r["known"]} ({known_pct}) unknown_mapping={r["unknown_mapping"]}')
            report["queries"]["q7_hashtag_weekly_schema_version"] = rows

            # ================================================================
            # 종합 KPI 요약
            # ================================================================
            print()
            print("=" * 70)
            print("KPI 요약 (PM/sales 자료 반영용)")
            print("=" * 70)
            q1 = report["queries"]["q1_signal_type_distribution"]
            q2 = report["queries"]["q2_llm_classification_rate"]
            total_v23 = sum(r["n"] for r in q1) or 0
            vision_n = sum(r["n"] for r in q1 if r["signal_type"].startswith("vision_"))
            hashtag_n = sum(r["n"] for r in q1 if r["signal_type"] == "hashtag")
            print(f'  16w v2.3 surface 총:        {total_v23}건')
            print(f'  hashtag source:             {hashtag_n}건  ({_pct(hashtag_n, total_v23)})')
            print(f'  vision_* source:            {vision_n}건  ({_pct(vision_n, total_v23)})')
            classified = (q2.get("classified") or 0) if isinstance(q2, dict) else 0
            print(f'  LLM 분류 적용:              {classified}건  ({_pct(classified, total_v23)})')
            partial = next(
                (r for r in report["queries"]["q4_partial_cluster_distribution"]
                 if r["cluster_type"] == "partial"),
                None,
            )
            if partial:
                exact = next(
                    (r for r in report["queries"]["q4_partial_cluster_distribution"]
                     if r["cluster_type"] == "exact"),
                    None,
                )
                exact_n = (exact["n"] if exact else 0) or 0
                partial_n = partial.get("n") or 0
                total_clusters = partial_n + exact_n
                print(f'  partial cluster 비율:       {_pct(partial_n, total_clusters)}')
                print(f'    └ score=NULL drift:       {partial.get("null_score") or 0} (0이면 PR #63 효과 OK)')

            report["summary"] = {
                "total_v23_signals": total_v23,
                "hashtag_source_n": hashtag_n,
                "vision_source_n": vision_n,
                "llm_classified_n": classified,
            }
    finally:
        conn.close()

    with open(out_path, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print()
    print(f'saved → {out_path}')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/v2_3_quality_report.json")
    args = parser.parse_args()
    sys.exit(main(args.output))
