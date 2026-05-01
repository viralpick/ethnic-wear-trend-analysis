"""score path ↔ item path cluster_key mismatch 진단.

12주 weekly run 의 enriched.json + summaries.json 을 읽고 양쪽 path 가 만드는
cluster_key 집합을 비교. agg_only (NULL row 후보) / sum_only (text fallback) 분리.

usage:
    uv run python scripts/diag_cluster_mismatch.py
    uv run python scripts/diag_cluster_mismatch.py outputs/2026-04-26
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from aggregation.item_distribution_builder import enriched_to_item_distribution
from aggregation.representative_builder import (
    aggregate_representatives,
    build_contributions,
)
from contracts.enriched import EnrichedContentItem


def diagnose(week_dir: Path) -> dict:
    enriched_raw = json.loads((week_dir / "enriched.json").read_text())
    summaries_raw = json.loads((week_dir / "summaries.json").read_text())
    enriched = [EnrichedContentItem.model_validate(item) for item in enriched_raw]
    distributions = [enriched_to_item_distribution(e) for e in enriched]
    contributions = build_contributions(distributions)
    aggregates = aggregate_representatives(contributions)
    agg_keys = {a.representative_key for a in aggregates}
    summary_keys = {s["cluster_key"] for s in summaries_raw}
    return {
        "week": week_dir.name,
        "n_enriched": len(enriched_raw),
        "agg": agg_keys,
        "summary": summary_keys,
        "matched": agg_keys & summary_keys,
        "agg_only": agg_keys - summary_keys,
        "sum_only": summary_keys - agg_keys,
    }


def main() -> int:
    if len(sys.argv) > 1:
        targets = [Path(sys.argv[1])]
    else:
        targets = sorted(
            p for p in (REPO / "outputs").glob("2026-*")
            if (p / "enriched.json").exists()
            and (p / "summaries.json").exists()
            and len(p.name) == 10
        )
    if not targets:
        print("no week directories found")
        return 1

    print(
        f"{'week':<12} {'enr':>5} {'agg':>5} {'sum':>5} "
        f"{'match':>6} {'agg_only':>9} {'sum_only':>9}"
    )
    print("-" * 68)

    total_agg_only = 0
    total_sum_only = 0
    total_matched = 0
    total_summaries = 0

    for week_dir in targets:
        r = diagnose(week_dir)
        print(
            f"{r['week']:<12} {r['n_enriched']:>5} {len(r['agg']):>5} "
            f"{len(r['summary']):>5} {len(r['matched']):>6} "
            f"{len(r['agg_only']):>9} {len(r['sum_only']):>9}"
        )
        total_agg_only += len(r["agg_only"])
        total_sum_only += len(r["sum_only"])
        total_matched += len(r["matched"])
        total_summaries += len(r["summary"])

        if len(targets) == 1:
            print()
            print(f"agg_only ({len(r['agg_only'])}): NULL row 발생 후보")
            for k in sorted(r["agg_only"]):
                print(f"  {k}")
            print()
            print(f"sum_only ({len(r['sum_only'])}): score path 만 산출")
            for k in sorted(r["sum_only"]):
                print(f"  {k}")

    if len(targets) > 1:
        print("-" * 68)
        print(
            f"{'TOTAL':<12} {'':>5} {'':>5} {total_summaries:>5} "
            f"{total_matched:>6} {total_agg_only:>9} {total_sum_only:>9}"
        )
        print()
        print(f"agg_only TOTAL = {total_agg_only}  (NULL 후보)")
        print(f"sum_only TOTAL = {total_sum_only}  (text fallback / score-only)")
        if total_summaries:
            rate = total_matched / (total_matched + total_agg_only) if (total_matched + total_agg_only) else 1.0
            print(f"rep_with_summary = {rate:.3f}  (1.000 = 모든 aggregate 가 summary 매칭)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
