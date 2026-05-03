"""M3.F — IG/YT mention 기반 ethnic wear 브랜드 seed list 추출.

전략:
1. IG content 에서 `@mentions` + `#hashtags` 정규식 추출
2. YT title+description+tags 에서 `@mentions` + `#hashtags` 추출
3. **Ethnic 필터** — 각 후보가 등장한 caption 모음에서 ethnic 키워드 동반 비율 (ethnic_share) 산출
4. CSV 출력 — 사용자 수동 검토용

기준 (튜닝 가능):
- min_mentions: 후보가 ≥ N개 caption 에 등장
- min_ethnic_share: ethnic 키워드 동반 비율 ≥ X
- 두 조건 모두 만족해야 keep

출력:
- outputs/brand_seed/brand_candidates.csv  (@mention 만 — 진짜 brand 후보)
- outputs/brand_seed/trend_hashtags.csv    (#hashtag 만 — trend keyword 후보)

사용:
    uv run python scripts/build_brand_seed_list.py
    uv run python scripts/build_brand_seed_list.py --min-mentions 3 --min-ethnic-share 0.5
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

load_dotenv()

_MENTION_RE = re.compile(r"@([A-Za-z0-9_.]{2,})")
_HASHTAG_RE = re.compile(r"#([A-Za-z0-9_]{2,})")

# Ethnic 키워드 — caption 안에 하나라도 있으면 ethnic context 로 카운트.
# garment_type / fabric / technique 어휘 (contracts/common.py) + 일반 desi 용어.
_ETHNIC_KEYWORDS = frozenset({
    # garment
    "saree", "sari", "lehenga", "kurta", "kurti", "anarkali", "salwar",
    "churidar", "palazzo", "sharara", "dupatta", "choli", "kameez",
    "sherwani", "blouse", "ghagra", "ghaghra",
    # generic
    "ethnic", "ethnicwear", "ethnicware", "indianwear", "indianware",
    "desi", "traditional", "festive", "wedding", "bridal", "haldi",
    "mehendi", "sangeet", "diwali", "rakhi", "navratri", "eid",
    "fusion",
    # fabric / technique 일부 (강한 ethnic signal)
    "chikankari", "bandhani", "banarasi", "kanjivaram", "ikat",
    "block_print", "blockprint", "zardosi", "gota", "mirror_work",
})

# Noise mentions (이메일 도메인 / 음악 / 비-ethnic 인물 등) — 명시 차단.
_BLOCKLIST = frozenset({
    "gmail.com", "outlook.com", "yahoo.com", "hotmail.com",
    "highlight", "mention", "all",
})


def _is_ethnic_context(text: str) -> bool:
    if not text:
        return False
    lower = text.lower()
    for kw in _ETHNIC_KEYWORDS:
        if kw in lower:
            return True
    return False


def _fetch_ig_contents() -> list[str]:
    from loaders.starrocks_connect import connect_raw
    conn = connect_raw()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT content FROM india_ai_fashion_inatagram_posting "
            "WHERE content IS NOT NULL AND content != ''"
        )
        return [r["content"] for r in cur.fetchall()]
    finally:
        conn.close()


def _fetch_yt_texts() -> list[str]:
    from loaders.starrocks_connect import connect_raw
    conn = connect_raw()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT title, description, tags FROM india_ai_fashion_youtube_posting "
            "WHERE title IS NOT NULL"
        )
        return [
            f"{r['title'] or ''} {r['description'] or ''} {r['tags'] or ''}"
            for r in cur.fetchall()
        ]
    finally:
        conn.close()


def _accumulate(
    texts: list[str],
    pattern: re.Pattern,
    candidate_to_contexts: dict[str, list[str]],
) -> None:
    """텍스트 리스트에서 pattern 으로 후보 추출 → 후보별 등장 caption 모음 누적."""
    for text in texts:
        for match in pattern.findall(text):
            handle = match.lower().rstrip(".")
            if handle in _BLOCKLIST or len(handle) < 2:
                continue
            candidate_to_contexts[handle].append(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="M3.F brand seed list")
    parser.add_argument("--min-mentions", type=int, default=3,
                        help="후보가 등장한 caption 최소 수 (default 3)")
    parser.add_argument("--min-ethnic-share", type=float, default=0.5,
                        help="ethnic 키워드 동반 비율 최소값 (default 0.5)")
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "outputs" / "brand_seed",
                        help="brand_candidates.csv / trend_hashtags.csv 출력 디렉터리")
    args = parser.parse_args()

    print("== Fetching IG contents ==")
    ig_contents = _fetch_ig_contents()
    print(f"  IG posts with content: {len(ig_contents)}")

    print("== Fetching YT title+description+tags ==")
    yt_texts = _fetch_yt_texts()
    print(f"  YT rows: {len(yt_texts)}")

    # source_key -> candidate -> [contexts]
    sources: dict[str, dict[str, list[str]]] = {
        "ig_mention": defaultdict(list),
        "ig_hashtag": defaultdict(list),
        "yt_mention": defaultdict(list),
        "yt_hashtag": defaultdict(list),
    }

    _accumulate(ig_contents, _MENTION_RE, sources["ig_mention"])
    _accumulate(ig_contents, _HASHTAG_RE, sources["ig_hashtag"])
    _accumulate(yt_texts, _MENTION_RE, sources["yt_mention"])
    _accumulate(yt_texts, _HASHTAG_RE, sources["yt_hashtag"])

    print()
    for src, cands in sources.items():
        print(f"  {src}: {sum(len(v) for v in cands.values())} mentions, "
              f"{len(cands)} unique")

    # 후보 (handle, source) 별 ethnic_share 계산
    rows: list[dict] = []
    for src, cand_map in sources.items():
        for handle, contexts in cand_map.items():
            n = len(contexts)
            if n < args.min_mentions:
                continue
            ethnic_n = sum(1 for c in contexts if _is_ethnic_context(c))
            share = ethnic_n / n
            if share < args.min_ethnic_share:
                continue
            example = next((c for c in contexts if _is_ethnic_context(c)), contexts[0])
            example_snippet = " ".join(example.split())[:240]
            rows.append({
                "handle": handle,
                "source": src,
                "mention_count": n,
                "ethnic_post_count": ethnic_n,
                "ethnic_share": round(share, 3),
                "example": example_snippet,
            })

    # mention vs hashtag 분리.
    brand_rows = [r for r in rows if r["source"].endswith("_mention")]
    hashtag_rows = [r for r in rows if r["source"].endswith("_hashtag")]

    # 각각 mention_count desc, ethnic_share desc 로 정렬 (brand 후보는 noise 가 많아 빈도 우선).
    brand_rows.sort(key=lambda r: (-r["mention_count"], -r["ethnic_share"]))
    hashtag_rows.sort(key=lambda r: (-r["mention_count"], -r["ethnic_share"]))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    brand_path = args.out_dir / "brand_candidates.csv"
    trend_path = args.out_dir / "trend_hashtags.csv"
    fieldnames = ["handle", "source", "mention_count",
                  "ethnic_post_count", "ethnic_share", "example"]
    for path, data in [(brand_path, brand_rows), (trend_path, hashtag_rows)]:
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

    print()
    print(f"  filter: mentions >= {args.min_mentions} AND ethnic_share >= {args.min_ethnic_share}")
    print()
    print(f"== Brand candidates: {brand_path} ({len(brand_rows)} rows) ==")
    print("  Top 30 (@mentions):")
    for r in brand_rows[:30]:
        print(f"    {r['handle']:35s} {r['source']:12s} "
              f"n={r['mention_count']:4d}  ethnic={r['ethnic_share']:.2f} "
              f"({r['ethnic_post_count']}/{r['mention_count']})")
    print()
    print(f"== Trend hashtags: {trend_path} ({len(hashtag_rows)} rows) ==")
    print("  Top 20 (#hashtags):")
    for r in hashtag_rows[:20]:
        print(f"    {r['handle']:35s} {r['source']:12s} "
              f"n={r['mention_count']:4d}  ethnic={r['ethnic_share']:.2f} "
              f"({r['ethnic_post_count']}/{r['mention_count']})")


if __name__ == "__main__":
    main()
