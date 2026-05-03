"""StarRocks 4 base + 4 _latest + 3 ethnic_latest view row count + sample 검증.

smoke 후 실 적재 결과 sanity check.

사용:
    uv run python scripts/verify_starrocks_views.py
    uv run python scripts/verify_starrocks_views.py --date 2026-04-29
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date as Date
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

import pymysql
from dotenv import load_dotenv

_VIEWS = [
    "item",
    "item_latest",
    "item_ethnic_latest",
    "canonical_group",
    "canonical_group_latest",
    "canonical_group_ethnic_latest",
    "canonical_object",
    "canonical_object_latest",
    "canonical_object_ethnic_latest",
    "representative_weekly",
    "representative_weekly_latest",
    "representative_weekly_ethnic_latest",
]


def _connect() -> pymysql.Connection:
    """result DB 연결 — `loaders.starrocks_connect.connect_result` 위임 (drift 방지)."""
    from loaders.starrocks_connect import connect_result
    return connect_result()


def _count(cur: pymysql.cursors.Cursor, table: str, where: str = "") -> int:
    sql = f"SELECT COUNT(*) AS c FROM {table}"
    if where:
        sql += f" WHERE {where}"
    cur.execute(sql)
    row = cur.fetchone()
    return int(row["c"]) if row else 0


def _sample_item(cur: pymysql.cursors.Cursor, target_date: Date | None) -> None:
    where = f"target_date = '{target_date}'" if target_date else ""
    sql = f"SELECT item_id, source, garment_type_dist, fabric_dist, brands_mentioned, computed_at FROM item_ethnic_latest"
    if where:
        sql += f" WHERE {where}"
    sql += " LIMIT 3"
    cur.execute(sql)
    print("  sample item rows:")
    for r in cur.fetchall():
        garment = json.loads(r["garment_type_dist"]) if r.get("garment_type_dist") else {}
        fabric = json.loads(r["fabric_dist"]) if r.get("fabric_dist") else {}
        brands = json.loads(r["brands_mentioned"]) if r.get("brands_mentioned") else []
        print(
            f"    {r['item_id']!r:50s} src={r['source']:9s} "
            f"garment={list(garment.keys())[:3]} fabric={list(fabric.keys())[:2]} "
            f"brands={[b.get('name') if isinstance(b, dict) else b for b in brands][:3]}"
        )


def _sample_representative(cur: pymysql.cursors.Cursor, target_date: Date | None) -> None:
    sql = "SELECT cluster_key, score, post_count_total, post_count_today, week_start_date FROM representative_weekly_ethnic_latest"
    sql += " ORDER BY score DESC LIMIT 5"
    cur.execute(sql)
    print("  top 5 representative rows (by score):")
    for r in cur.fetchall():
        print(
            f"    {r['cluster_key']:50s} score={r['score']:5.1f} "
            f"total={r['post_count_total']:6.1f} today={r['post_count_today']:6.1f} "
            f"week_start={r['week_start_date']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=Date.fromisoformat, default=None,
                        help="검증 대상 target_date (item/group/object 필터)")
    args = parser.parse_args()

    with _connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT DATABASE() AS db")
        db = cur.fetchone()["db"]
        print(f"\nDB: {db}\n")
        print(f"{'view':<45} {'rows':>10}")
        print("─" * 60)
        for v in _VIEWS:
            try:
                n = _count(cur, v)
                print(f"{v:<45} {n:>10,}")
            except Exception as e:
                print(f"{v:<45} ERROR: {e}")
        print("─" * 60)

        print("\n[sample] item_ethnic_latest")
        _sample_item(cur, args.date)
        print("\n[sample] representative_weekly_ethnic_latest")
        _sample_representative(cur, args.date)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
