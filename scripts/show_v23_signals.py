import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from loaders.starrocks_connect import connect_result  # noqa: E402
conn = connect_result(autocommit=True, dict_cursor=False)
with conn.cursor() as cur:
    cur.execute("""
      SELECT tag, week_start_date, count_recent_window, first_seen,
             likely_category, signal_type, schema_version
      FROM unknown_signal_latest
      WHERE schema_version = 'pipeline_v2.3'
      ORDER BY week_start_date DESC, count_recent_window DESC
    """)
    cols = [d[0] for d in cur.description]
    print(f'{"week":<12} {"tag":<35} {"count":<6} {"first_seen":<12} {"category":<22} {"signal_type":<18}')
    print('-' * 110)
    for r in cur.fetchall():
        d = dict(zip(cols, r))
        print(f'{str(d["week_start_date"]):<12} {d["tag"]:<35} {d["count_recent_window"]:<6} '
              f'{str(d["first_seen"]):<12} {str(d.get("likely_category") or "-"):<22} {d["signal_type"]:<18}')
conn.close()
