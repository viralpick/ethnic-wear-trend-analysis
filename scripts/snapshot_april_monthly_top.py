"""4월 monthly Top 10 snapshot (4w aggregation, sparse 0-padding).

representative_weekly_latest 의 4월 4주 (IST Monday: 4/06, 4/13, 4/20, 4/27) 를
cluster_key 별 contribution-weighted 합산. avg_score = sum(score)/4 (sparse 0-pad).

용도: v2.3 rep 재산출 전후 비교. pre/post JSON 두 개 가지고 diff 측정.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime, timezone

import pymysql
from dotenv import load_dotenv

load_dotenv()


def parse_json(v):
    if v is None:
        return {}
    if isinstance(v, str):
        return json.loads(v) if v else {}
    return v


def weighted_dist(lst: list[dict], field: str) -> dict[str, float]:
    agg: dict[str, float] = defaultdict(float)
    for r in lst:
        d = parse_json(r.get(field))
        w = float(r.get("total_item_contribution") or 0)
        for k, v in d.items():
            agg[k] += float(v) * w
    total = sum(agg.values())
    return {k: v / total for k, v in agg.items()} if total > 0 else {}


def main(out_path: str, note: str) -> None:
    weeks = ["2026-04-06", "2026-04-13", "2026-04-20", "2026-04-27"]
    from loaders.starrocks_connect import connect_result
    conn = connect_result(autocommit=True, dict_cursor=False)
    rows: list[dict] = []
    try:
        with conn.cursor() as cur:
            for w in weeks:
                cur.execute(
                    """
                    SELECT representative_key, score_total, total_item_contribution,
                           effective_item_count, brand_distribution, technique_distribution,
                           weekly_direction, lifecycle_stage, schema_version
                    FROM representative_weekly_latest
                    WHERE week_start_date = %s
                    """,
                    (w,),
                )
                cols = [d[0] for d in cur.description]
                for r in cur.fetchall():
                    d = dict(zip(cols, r))
                    d["week_start_date"] = w
                    rows.append(d)
    finally:
        conn.close()

    by_key: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_key[r["representative_key"]].append(r)

    monthly: list[dict] = []
    for key, lst in by_key.items():
        avg_score = sum(float(r["score_total"] or 0) for r in lst) / 4.0
        sum_eic = sum(float(r["effective_item_count"] or 0) for r in lst)
        sum_contrib = sum(float(r["total_item_contribution"] or 0) for r in lst)
        brand_agg = weighted_dist(lst, "brand_distribution")
        tech_agg = weighted_dist(lst, "technique_distribution")
        monthly.append({
            "cluster_key": key,
            "avg_score": round(avg_score, 2),
            "n_weeks": len(lst),
            "sum_effective_item_count": round(sum_eic, 2),
            "sum_total_item_contribution": round(sum_contrib, 2),
            "top_brands": [(b, round(s, 4)) for b, s in
                           sorted(brand_agg.items(), key=lambda x: -x[1])[:5]],
            "top_techniques": [(t, round(s, 4)) for t, s in
                               sorted(tech_agg.items(), key=lambda x: -x[1])[:5]],
        })
    monthly.sort(key=lambda x: -x["avg_score"])

    print("=== April 2026 monthly Top 10 ===")
    print(f'{"#":<3} {"cluster_key":<40} {"avg":<7} {"weeks":<6} '
          f'{"Σ EIC":<8} {"top brand":<25} {"top tech":<40}')
    for i, m in enumerate(monthly[:10], 1):
        brands = ", ".join(b for b, _ in m["top_brands"][:3])[:24]
        techs = ", ".join(f"{t}({s:.2f})" for t, s in m["top_techniques"][:2])[:39]
        print(f'{i:<3} {m["cluster_key"]:<40} {m["avg_score"]:<7.2f} '
              f'{m["n_weeks"]}/4    {m["sum_effective_item_count"]:<8.2f} '
              f'{brands:<25} {techs}')

    out = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": note,
        "weeks_iso_monday": weeks,
        "monthly_top10": monthly[:10],
        "monthly_full": monthly,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print(f'\nsaved → {out_path} (rows={len(monthly)})')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/april_monthly_top_pre_v2.3.json")
    parser.add_argument("--note", default="pre v2.3 rep run")
    args = parser.parse_args()
    main(args.output, args.note)
