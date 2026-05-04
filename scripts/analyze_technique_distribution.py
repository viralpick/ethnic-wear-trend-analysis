"""technique_distribution 안 raw 단어 (enum 외) 표기 여부 검증."""
import json
from collections import Counter

from contracts.common import Technique
from loaders.starrocks_connect import connect_result


def main() -> None:
    enum_values = {t.value for t in Technique}
    print(f"Technique enum 값 ({len(enum_values)}개): {sorted(enum_values)}")
    print()

    conn = connect_result(dict_cursor=False)
    all_techniques: Counter = Counter()
    sample_with_raw: list[tuple] = []
    n_rows = 0
    n_with_dist = 0
    with conn.cursor() as cur:
        cur.execute("""
          SELECT representative_key, technique_distribution
          FROM representative_weekly_latest
          WHERE week_start_date >= '2026-01-26'
            AND technique_distribution IS NOT NULL
        """)
        for r in cur.fetchall():
            key, td = r
            n_rows += 1
            if not td:
                continue
            if isinstance(td, str):
                td = json.loads(td)
            if not td:
                continue
            n_with_dist += 1
            for t, share in td.items():
                all_techniques[t] += 1
                if t not in enum_values and len(sample_with_raw) < 10:
                    sample_with_raw.append((key, t, share, td))
    conn.close()

    enum_keys = {t for t in all_techniques if t in enum_values}
    raw_keys = {t for t in all_techniques if t not in enum_values}

    print(f"representative_weekly_latest 의 v2.3 row (week >= 2026-01-26): {n_rows}")
    print(f"  technique_distribution 채워진 row: {n_with_dist}")
    print()
    print(f"distribution 안 unique technique key 수: {len(all_techniques)}")
    print(f"  enum 매칭: {len(enum_keys)}")
    print(f"  enum 외 raw 단어: {len(raw_keys)}")
    print()
    print("Distribution 에 등장한 모든 key (등장 row 수 desc):")
    for t, n in all_techniques.most_common(30):
        marker = "✅ enum" if t in enum_values else "❌ enum 외"
        print(f"  {t:<25} {n:>4}  {marker}")

    if sample_with_raw:
        print()
        print("Raw 단어 등장 row 샘플:")
        for key, t, share, td in sample_with_raw[:5]:
            print(f"  cluster: {key}")
            print(f"    raw key '{t}' share={share:.3f}, full dist: {td}")


if __name__ == "__main__":
    main()
