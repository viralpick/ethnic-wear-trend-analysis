"""4월 monthly Top 변동 비교 — pre v2.3 vs post v2.3.

snapshot_april_monthly_top.py 가 만든 두 JSON (pre/post) 을 받아 ranking diff,
score 변화, brand/technique 변동 출력.
"""
from __future__ import annotations

import argparse
import json


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def index_by_key(snapshot: dict) -> dict[str, dict]:
    return {m["cluster_key"]: m for m in snapshot["monthly_full"]}


def main(pre_path: str, post_path: str) -> None:
    pre = load(pre_path)
    post = load(post_path)
    pre_idx = index_by_key(pre)
    post_idx = index_by_key(post)
    pre_top10 = {m["cluster_key"]: i + 1 for i, m in enumerate(pre["monthly_top10"])}
    post_top10 = {m["cluster_key"]: i + 1 for i, m in enumerate(post["monthly_top10"])}

    print(f'pre  captured: {pre.get("captured_at_utc", "?")}')
    print(f'post captured: {post.get("captured_at_utc", "?")}')
    print()

    print("=== Top 10 ranking diff ===")
    print(f'{"#":<3} {"cluster":<40} {"pre rank":<9} {"post rank":<9} '
          f'{"pre avg":<8} {"post avg":<8} {"Δ avg":<7} {"note"}')
    all_top = sorted(set(pre_top10.keys()) | set(post_top10.keys()),
                     key=lambda k: post_top10.get(k, 99))
    for ck in all_top:
        pr = pre_top10.get(ck, "—")
        psr = post_top10.get(ck, "—")
        pre_m = pre_idx.get(ck)
        post_m = post_idx.get(ck)
        pre_avg = pre_m["avg_score"] if pre_m else 0.0
        post_avg = post_m["avg_score"] if post_m else 0.0
        delta = post_avg - pre_avg
        note = ""
        if pr == "—":
            note = "🆕 신규 진입"
        elif psr == "—":
            note = "❌ Top10 이탈"
        elif isinstance(pr, int) and isinstance(psr, int):
            if abs(pr - psr) >= 3:
                note = f"⚠️ {pr - psr:+d}위 변동"
        print(f'{psr if isinstance(psr, int) else "—":<3} {ck:<40} '
              f'{str(pr):<9} {str(psr):<9} '
              f'{pre_avg:<8.2f} {post_avg:<8.2f} {delta:+7.2f} {note}')

    print()
    print("=== Top 10 brand/technique 변동 (post 기준) ===")
    for i, m in enumerate(post["monthly_top10"], 1):
        ck = m["cluster_key"]
        pre_m = pre_idx.get(ck)
        post_brands = ", ".join(b for b, _ in m["top_brands"][:3])
        pre_brands = ", ".join(b for b, _ in pre_m["top_brands"][:3]) if pre_m else "—"
        post_techs = ", ".join(f"{t}({s:.2f})" for t, s in m["top_techniques"][:2])
        pre_techs = (", ".join(f"{t}({s:.2f})" for t, s in pre_m["top_techniques"][:2])
                     if pre_m else "—")
        diff_brand = "✅" if post_brands == pre_brands else "🔄"
        diff_tech = "✅" if post_techs == pre_techs else "🔄"
        print(f'#{i} {ck}')
        print(f'   {diff_brand} brand    pre={pre_brands}  →  post={post_brands}')
        print(f'   {diff_tech} technique pre={pre_techs}  →  post={post_techs}')

    # summary stats
    print()
    print("=== Summary ===")
    same_top1 = (
        pre["monthly_top10"][0]["cluster_key"] ==
        post["monthly_top10"][0]["cluster_key"]
    )
    pre_set = set(pre_top10.keys())
    post_set = set(post_top10.keys())
    overlap = len(pre_set & post_set)
    print(f'Top 1 동일?       {"✅ YES" if same_top1 else "❌ NO"}')
    print(f'Top 10 overlap:    {overlap}/10  (10이면 멤버 동일, 9 이하면 진입/이탈)')
    print(f'전체 cluster 수:   pre={len(pre_idx)}  post={len(post_idx)}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre", default="outputs/april_monthly_top_pre_v2.3.json")
    parser.add_argument("--post", default="outputs/april_monthly_top_post_v2.3.json")
    args = parser.parse_args()
    main(args.pre, args.post)
