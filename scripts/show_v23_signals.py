import os
from dotenv import load_dotenv
import pymysql
load_dotenv()
conn = pymysql.connect(host=os.environ['STARROCKS_HOST'], port=int(os.environ['STARROCKS_PORT']),
    user=os.environ['STARROCKS_USER'], password=os.environ['STARROCKS_PASSWORD'],
    database=os.environ.get('STARROCKS_RESULT_DATABASE','ethnic_result'), autocommit=True)
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
