"""백필 enriched 데이터 품질 breakdown — canonical 단위.

전체 post / canonical / ethnic canonical / garment enum / fabric enum / 둘다 enum
순으로 funnel 분석. cluster 기여 (exact) 비율 + partial 분포까지.
"""
from __future__ import annotations

import sys
from collections import Counter

from aggregation.vision_normalize import normalize_fabric, normalize_garment_for_cluster
from contracts.vision import is_canonical_ethnic
from pipelines.load_enriched import load_enriched_files


def main(glob_pattern: str) -> None:
    posts = load_enriched_files(glob_pattern)
    n_post = len(posts)

    total_canonical = 0
    ethnic_canonical = 0
    g_enum = 0
    f_enum = 0
    both_enum = 0
    g_only = 0
    f_only = 0
    neither = 0
    raw_g_words: Counter = Counter()
    raw_f_words: Counter = Counter()

    post_with_canonical = 0
    post_with_ethnic = 0
    post_with_exact = 0

    for post in posts:
        has_canonical = False
        has_ethnic = False
        has_exact = False
        for c in post.canonicals:
            total_canonical += 1
            has_canonical = True
            if not is_canonical_ethnic(c):
                continue
            ethnic_canonical += 1
            has_ethnic = True
            g_node = normalize_garment_for_cluster(c.representative)
            f_node = normalize_fabric(c.representative)
            g = g_node.value if g_node else None
            f = f_node.value if f_node else None
            if g is not None:
                g_enum += 1
            else:
                # raw word log
                rep = c.representative
                if rep.upper_garment_type:
                    raw_g_words[rep.upper_garment_type.lower()] += 1
                if rep.lower_garment_type:
                    raw_g_words[rep.lower_garment_type.lower()] += 1
            if f is not None:
                f_enum += 1
            else:
                rep = c.representative
                if rep.fabric:
                    raw_f_words[rep.fabric.lower()] += 1
            if g and f:
                both_enum += 1
                has_exact = True
            elif g and not f:
                g_only += 1
            elif f and not g:
                f_only += 1
            else:
                neither += 1
        if has_canonical:
            post_with_canonical += 1
        if has_ethnic:
            post_with_ethnic += 1
        if has_exact:
            post_with_exact += 1

    def pct(n: int, total: int) -> str:
        return f"{n/total*100:.1f}%" if total else "—"

    print("=" * 70)
    print("Funnel — Post 단위")
    print("=" * 70)
    print(f"  전체 post                              : {n_post:>6}")
    print(f"  └ canonical 1개 이상 (vision 분석 통과) : {post_with_canonical:>6}  ({pct(post_with_canonical, n_post)})")
    print(f"    └ ethnic canonical 1개 이상            : {post_with_ethnic:>6}  ({pct(post_with_ethnic, n_post)})")
    print(f"      └ exact cluster (G+F 둘다 enum) 1개 이상: {post_with_exact:>6}  ({pct(post_with_exact, n_post)})")

    print()
    print("=" * 70)
    print("Funnel — Canonical 단위")
    print("=" * 70)
    print(f"  전체 canonical (BBOX-detected outfit)  : {total_canonical:>6}")
    print(f"  └ ethnic canonical (여성+성인+에스닉)   : {ethnic_canonical:>6}  ({pct(ethnic_canonical, total_canonical)})")
    print(f"    │")
    print(f"    ├ garment enum 매칭                    : {g_enum:>6}  ({pct(g_enum, ethnic_canonical)})")
    print(f"    ├ fabric enum 매칭                     : {f_enum:>6}  ({pct(f_enum, ethnic_canonical)})")
    print(f"    │")
    print(f"    └ G+F 둘다 enum (exact cluster 기여)   : {both_enum:>6}  ({pct(both_enum, ethnic_canonical)})")

    print()
    print("=" * 70)
    print("Ethnic canonical 의 cluster 분류 분포")
    print("=" * 70)
    print(f"  G+F 둘다 enum  → exact cluster        : {both_enum:>6}  ({pct(both_enum, ethnic_canonical)})")
    print(f"  G만 enum (F=None) → kurta__unknown 류   : {g_only:>6}  ({pct(g_only, ethnic_canonical)})")
    print(f"  F만 enum (G=None) → unknown__cotton 류  : {f_only:>6}  ({pct(f_only, ethnic_canonical)})")
    print(f"  둘다 None → cluster 미참여               : {neither:>6}  ({pct(neither, ethnic_canonical)})")

    print()
    print("=" * 70)
    print("매핑 외 raw 단어 Top 10 (vision LLM raw 답)")
    print("=" * 70)
    print("  garment (enum 매핑 실패 시):")
    for w, n in raw_g_words.most_common(10):
        print(f"    {w:<35} {n:>4}")
    print()
    print("  fabric (enum 매핑 실패 시):")
    for w, n in raw_f_words.most_common(10):
        print(f"    {w:<35} {n:>4}")


if __name__ == "__main__":
    glob = sys.argv[1] if len(sys.argv) > 1 else "outputs/backfill_16w/page_*_enriched.json"
    main(glob)
