"""run 별 분리 출력 — computed_at 기준."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from loaders.starrocks_connect import connect_result  # noqa: E402
conn = connect_result(autocommit=True, dict_cursor=False)
with conn.cursor() as cur:
    # raw row (latest view 아닌 base) 에서 v2.3 만, computed_at 분리
    cur.execute("""
      SELECT tag, week_start_date, count_recent_window, first_seen,
             likely_category, signal_type, computed_at
      FROM unknown_signal
      WHERE schema_version = 'pipeline_v2.3'
      ORDER BY computed_at DESC, week_start_date DESC, count_recent_window DESC
    """)
    cols = [d[0] for d in cur.description]
    print(f'{"computed_at (UTC)":<22} {"week":<12} {"tag":<22} {"count":<6} {"category":<22} {"signal_type":<18}')
    print('-' * 110)
    for r in cur.fetchall():
        d = dict(zip(cols, r))
        print(f'{str(d["computed_at"]):<22} {str(d["week_start_date"]):<12} {d["tag"]:<22} {d["count_recent_window"]:<6} '
              f'{str(d.get("likely_category") or "-"):<22} {d["signal_type"]:<18}')
conn.close()
