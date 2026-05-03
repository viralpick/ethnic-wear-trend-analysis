"""StarRocks 적재 검증 1회용 — 9030 query 로 view 4개 행 수 + sample row 점검.

사용:
    uv run python scripts/verify_starrocks_emit.py
"""
from __future__ import annotations

import json
import os

import pymysql
from dotenv import load_dotenv

VIEWS = (
    "item_latest",
    "canonical_group_latest",
    "canonical_object_latest",
    "representative_weekly_latest",
)

ETHNIC_VIEWS = (
    ("item_latest", "item_ethnic_latest"),
    ("canonical_group_latest", "canonical_group_ethnic_latest"),
    ("canonical_object_latest", "canonical_object_ethnic_latest"),
)

JSON_SAMPLE_FIELDS = {
    "item_latest": (
        "source",
        "source_post_id",
        "garment_type_dist",
        "fabric_dist",
        "color_palette",
    ),
    "canonical_group_latest": (
        "group_id",
        "garment_type",
        "color_palette",
        "item_contribution_score",
    ),
    "canonical_object_latest": (
        "object_id",
        "media_ref",
        "color_palette",
    ),
    "representative_weekly_latest": (
        "representative_key",
        "score_total",
        "color_palette",
        "factor_contribution",
        "score_breakdown",
    ),
}


def _connect() -> pymysql.connections.Connection:
    """result DB 연결 — `loaders.starrocks_connect.connect_result` 위임 (drift 방지)."""
    from loaders.starrocks_connect import connect_result
    return connect_result(autocommit=True, dict_cursor=False)


def main() -> None:
    load_dotenv()
    db = os.environ["STARROCKS_RESULT_DATABASE"]
    print(f"# DB: {db}\n")
    with _connect() as conn:
        with conn.cursor() as cur:
            for view in VIEWS:
                cur.execute(f"SELECT COUNT(*) FROM `{db}`.`{view}`")
                (count,) = cur.fetchone()
                print(f"{view}: {count} rows")

            # ethnic view 카버리지 — base vs ethnic 격차 = 비-ethnic 라벨 보존 canonical 수.
            print()
            for base, ethnic in ETHNIC_VIEWS:
                cur.execute(f"SELECT COUNT(*) FROM `{db}`.`{base}`")
                (base_n,) = cur.fetchone()
                cur.execute(f"SELECT COUNT(*) FROM `{db}`.`{ethnic}`")
                (eth_n,) = cur.fetchone()
                gap = int(base_n) - int(eth_n)
                print(f"{ethnic}: {eth_n}/{base_n} (non-ethnic preserved = {gap})")

            # color_palette NOT NULL 비율 — pipeline_b vs fake 결과 분리 신호.
            print()
            for view in VIEWS:
                cur.execute(
                    f"SELECT COUNT(*), "
                    f"SUM(CASE WHEN color_palette IS NOT NULL THEN 1 ELSE 0 END) "
                    f"FROM `{db}`.`{view}`"
                )
                total, nn = cur.fetchone()
                nn = int(nn or 0)
                total = int(total)
                pct = (nn / total * 100) if total else 0.0
                print(f"{view}.color_palette NOT NULL: {nn}/{total} ({pct:.1f}%)")

            print()
            for view in VIEWS:
                fields = JSON_SAMPLE_FIELDS[view]
                col_clause = ", ".join(f"`{c}`" for c in fields)
                if view == "representative_weekly_latest":
                    cur.execute(
                        f"SELECT {col_clause} FROM `{db}`.`{view}` "
                        f"WHERE score_total IS NOT NULL "
                        f"ORDER BY score_total DESC LIMIT 1"
                    )
                elif view == "item_latest":
                    cur.execute(
                        f"SELECT {col_clause} FROM `{db}`.`{view}` "
                        f"WHERE color_palette IS NOT NULL LIMIT 1"
                    )
                else:
                    cur.execute(
                        f"SELECT {col_clause} FROM `{db}`.`{view}` LIMIT 1"
                    )
                row = cur.fetchone()
                if row is None:
                    print(f"## {view}: empty\n")
                    continue
                print(f"## {view} (first row):")
                for name, value in zip(fields, row):
                    if isinstance(value, str) and value.startswith(("{", "[")):
                        try:
                            parsed = json.loads(value)
                            value_repr = json.dumps(parsed, ensure_ascii=False)[:200]
                        except json.JSONDecodeError:
                            value_repr = f"!!! JSON parse FAIL: {value[:120]}"
                    else:
                        value_repr = repr(value)[:200]
                    print(f"  {name}: {value_repr}")
                print()


if __name__ == "__main__":
    main()
