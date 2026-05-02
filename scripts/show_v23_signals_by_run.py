"""run 별 분리 출력 — computed_at 기준."""
import os
from dotenv import load_dotenv
import pymysql
load_dotenv()
conn = pymysql.connect(host=os.environ['STARROCKS_HOST'], port=int(os.environ['STARROCKS_PORT']),
    user=os.environ['STARROCKS_USER'], password=os.environ['STARROCKS_PASSWORD'],
    database=os.environ.get('STARROCKS_RESULT_DATABASE','ethnic_result'), autocommit=True)
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
