-- unknown_signal_latest — (tag, week_start_date) 별 최신 1 row 노출.
-- v2.2: weekly cadence 라 같은 (tag, week_start_date) 가 재산출 시 computed_at 기준 dedup.
-- WHERE week_start_date IS NOT NULL 로 v1 row (NULL) 자동 제외.

CREATE OR REPLACE VIEW unknown_signal_latest AS
SELECT t.*
FROM unknown_signal AS t
INNER JOIN (
    SELECT tag, week_start_date, MAX(computed_at) AS max_at
    FROM unknown_signal
    WHERE week_start_date IS NOT NULL
    GROUP BY tag, week_start_date
) AS m
ON t.tag = m.tag
   AND t.week_start_date = m.week_start_date
   AND t.computed_at = m.max_at;
