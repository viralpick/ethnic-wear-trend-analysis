"""4주 weekly representative 를 G+F (technique 무시) 기준 rollup 후 monthly top N.

배경: 향후 cluster_key 가 G__T__F → G__F 로 변경 예정. 그 전에 4월 monthly top 1/2/3 을
미리 산출해야 함. 같은 (G, F) 인 cluster (T 만 다른) 들의 score 를 합산 → 4주 평균.

approximate match — minmax_same_run scope 와 N=2/N=3 multiplier 차이로 정확히 동일하지는
않지만 top ranking 은 거의 보존.

사용:
  uv run python scripts/monthly_gf_rollup.py
  uv run python scripts/monthly_gf_rollup.py --top 5
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]


def _parse_gf(cluster_key: str) -> tuple[str, str] | None:
    """`g__t__f` → (g, f). unknown 포함 또는 형식 안 맞으면 None."""
    parts = cluster_key.split("__")
    if len(parts) != 3:
        return None
    g, _t, f = parts
    if not g or not f or g == "unknown" or f == "unknown":
        return None  # partial cluster (T 만 unknown 인 N=2 도 일단 제외 — direct G__F 와 비교)
    return (g, f)


def _load_summaries(date_str: str) -> list[dict]:
    path = _REPO / "outputs" / date_str / "summaries.json"
    if not path.exists():
        raise SystemExit(f"summaries.json missing: {path}")
    with path.open() as f:
        return json.load(f)


def _rollup_week(summaries: list[dict]) -> dict[tuple[str, str], dict]:
    """주차 1개 — (G, F) 기준 cluster 합산. 합산값: score / post_count_today /
    contributor count (참고용).
    """
    out: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"score": 0.0, "post_count_today": 0.0, "n_clusters": 0}
    )
    for c in summaries:
        gf = _parse_gf(c["cluster_key"])
        if gf is None:
            continue
        out[gf]["score"] += c.get("score", 0.0)
        out[gf]["post_count_today"] += c.get("post_count_today", 0.0)
        out[gf]["n_clusters"] += 1
    return dict(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weeks", default="2026-04-05,2026-04-12,2026-04-19,2026-04-26",
        help="콤마 구분 end_date (Sunday). 4월 = 3/30~4/5 + 4/6~4/12 + 4/13~4/19 + 4/20~4/26",
    )
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    week_ends = [s.strip() for s in args.weeks.split(",") if s.strip()]
    print(f"\nMonthly rollup — {len(week_ends)} weeks: {week_ends}\n")

    # 주차별 (G, F) → score
    weekly_data: list[dict[tuple[str, str], dict]] = []
    for end in week_ends:
        summ = _load_summaries(end)
        rolled = _rollup_week(summ)
        weekly_data.append(rolled)
        n_full = sum(1 for c in summ if _parse_gf(c["cluster_key"]) is not None)
        print(f"  {end}: {len(summ)} G__T__F clusters → {len(rolled)} G__F groups "
              f"(N=3 cluster {n_full}/{len(summ)})")

    # union of all (G, F) keys
    all_gf = set()
    for w in weekly_data:
        all_gf.update(w.keys())

    # spec §3.3: monthly = 4-week rolling. sparse 0 padding (등장 안 한 주는 0 으로
    # 평균 — short-lived viral spike 가 monthly trend 로 부풀려지지 않게).
    monthly: dict[tuple[str, str], dict] = {}
    n_total_weeks = len(weekly_data)
    for gf in all_gf:
        weekly_scores = []
        post_counts = []
        n_clusters_acc = 0
        weeks_present = 0
        for w in weekly_data:
            if gf in w:
                weekly_scores.append(w[gf]["score"])
                post_counts.append(w[gf]["post_count_today"])
                n_clusters_acc += w[gf]["n_clusters"]
                weeks_present += 1
            else:
                weekly_scores.append(0.0)  # sparse 0 padding
        monthly[gf] = {
            "monthly_avg_score": sum(weekly_scores) / n_total_weeks,
            "weeks_present": weeks_present,
            "total_post_count": sum(post_counts),
            "n_clusters_total": n_clusters_acc,
        }

    # Top N (monthly_avg_score desc)
    sorted_gf = sorted(monthly.items(), key=lambda kv: -kv[1]["monthly_avg_score"])
    print(f"\nTop {args.top} monthly G__F (score desc):")
    print(f"{'rank':<5} {'G__F':<55} {'avg_score':>10} {'weeks':>6} {'posts':>8} {'rolled_clusters':>16}")
    print("─" * 110)
    for rank, (gf, m) in enumerate(sorted_gf[:args.top], 1):
        gf_str = f"{gf[0]}__{gf[1]}"
        print(
            f"{rank:<5} {gf_str:<55} {m['monthly_avg_score']:>10.2f} "
            f"{m['weeks_present']:>6} {m['total_post_count']:>8.1f} {m['n_clusters_total']:>16}"
        )
    print()

    # 추가 — 주차별 ranking 비교 (top 1/2/3 stability 점검)
    print("Weekly top 3 (each week's G__F rollup):")
    for end, w in zip(week_ends, weekly_data, strict=True):
        sorted_w = sorted(w.items(), key=lambda kv: -kv[1]["score"])[:3]
        print(f"  {end}:")
        for rank, (gf, d) in enumerate(sorted_w, 1):
            print(f"    {rank}. {gf[0]}__{gf[1]:<40} score={d['score']:.2f} "
                  f"(rolled from {d['n_clusters']} clusters)")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
