"""canonical=[] post 차단 사유 분해 — 어디서 62.8% 가 떨어지는가."""
from __future__ import annotations

import sys
from collections import Counter

from contracts.common import ContentSource
from pipelines.load_enriched import load_enriched_files


def main(glob_pattern: str) -> None:
    posts = load_enriched_files(glob_pattern)

    # source 별 split
    by_source: dict[str, dict[str, int]] = {
        "instagram": {"total": 0, "has_canonical": 0, "no_canonical": 0,
                      "no_image_url": 0, "video_only": 0, "image_post": 0},
        "youtube": {"total": 0, "has_canonical": 0, "no_canonical": 0,
                    "no_image_url": 0, "video_only": 0, "image_post": 0},
    }

    no_canonical_samples_ig: list = []
    no_canonical_samples_yt: list = []

    accounts_no_canonical: Counter = Counter()
    accounts_with_canonical: Counter = Counter()

    for post in posts:
        src = post.normalized.source.value
        if src not in by_source:
            continue
        s = by_source[src]
        s["total"] += 1
        has_can = bool(post.canonicals)
        if has_can:
            s["has_canonical"] += 1
            if post.normalized.account_handle:
                accounts_with_canonical[post.normalized.account_handle] += 1
        else:
            s["no_canonical"] += 1
            if post.normalized.account_handle:
                accounts_no_canonical[post.normalized.account_handle] += 1
            # image_urls 분석
            n_img = len(post.normalized.image_urls or [])
            if n_img == 0:
                s["no_image_url"] += 1
            else:
                # video frame 인지 image post 인지 — image_urls 의 url 확인
                # IG video frame 은 _f<num> 같은 패턴
                # 단순 추정: 어차피 enriched 단계 이후 post-vision 결과만 있어서 video/image 구분 어렵
                s["image_post"] += 1
            # sample
            if src == "instagram" and len(no_canonical_samples_ig) < 5:
                no_canonical_samples_ig.append({
                    "post_id": post.normalized.source_post_id,
                    "account": post.normalized.account_handle,
                    "n_img": n_img,
                    "hashtags": post.normalized.hashtags[:5],
                    "text_blob_head": (post.normalized.text_blob or "")[:80],
                })
            elif src == "youtube" and len(no_canonical_samples_yt) < 5:
                no_canonical_samples_yt.append({
                    "post_id": post.normalized.source_post_id,
                    "account": post.normalized.account_handle,
                    "n_img": n_img,
                    "title": (post.normalized.text_blob or "")[:80],
                })

    def pct(n: int, total: int) -> str:
        return f"{n/total*100:.1f}%" if total else "—"

    print("=" * 70)
    print("Source 별 canonical 보유율")
    print("=" * 70)
    grand_total = sum(s["total"] for s in by_source.values())
    grand_with = sum(s["has_canonical"] for s in by_source.values())
    print(f"  전체:        {grand_total:>5} post / canonical 보유 {grand_with} ({pct(grand_with, grand_total)})")
    for src, s in by_source.items():
        print(f"  {src:<12} {s['total']:>5} post / canonical 보유 {s['has_canonical']} ({pct(s['has_canonical'], s['total'])})")
        print(f"    └ canonical=[] : {s['no_canonical']:>4}")
        print(f"        ├ image_urls=0 (video / 텍스트 only)         : {s['no_image_url']}")
        print(f"        └ image post (vision 분석 통과 못함)         : {s['image_post']}")

    print()
    print("=" * 70)
    print("canonical=[] post 사례 — IG (top 5)")
    print("=" * 70)
    for s in no_canonical_samples_ig:
        print(f"  post={s['post_id'][:16]:<16} acc={s['account']:<25} n_img={s['n_img']}")
        print(f"    hashtags: {s['hashtags']}")
        print(f"    text:     {s['text_blob_head']}")

    print()
    print("=" * 70)
    print("canonical=[] post 사례 — YT (top 5)")
    print("=" * 70)
    for s in no_canonical_samples_yt:
        print(f"  post={s['post_id'][:16]:<16} acc={s['account']:<25} n_img={s['n_img']}")
        print(f"    title:    {s['title']}")

    print()
    print("=" * 70)
    print("canonical=[] post 가 많은 IG account top 10")
    print("=" * 70)
    print("  (비-ethnic / 비-fashion / 남성 콘텐츠 주력 계정 의심)")
    for acc, n in accounts_no_canonical.most_common(10):
        n_with = accounts_with_canonical.get(acc, 0)
        n_total = n + n_with
        ratio_drop = n / n_total if n_total else 0
        print(f"  {acc:<35} drop={n:>3}/{n_total} ({ratio_drop*100:.0f}%) keep={n_with}")


if __name__ == "__main__":
    glob = sys.argv[1] if len(sys.argv) > 1 else "outputs/backfill_16w/page_*_enriched.json"
    main(glob)
