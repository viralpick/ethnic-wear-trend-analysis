-- hashtag_weekly_latest — (tag, week_start_date) 별 최신 1 row 노출.
-- 같은 주 재산출 시 max(computed_at) row 만 보여 dedup.

CREATE OR REPLACE VIEW hashtag_weekly_latest AS
SELECT t.*
FROM hashtag_weekly AS t
INNER JOIN (
    SELECT tag, week_start_date, MAX(computed_at) AS max_at
    FROM hashtag_weekly
    GROUP BY tag, week_start_date
) AS m
ON t.tag = m.tag
   AND t.week_start_date = m.week_start_date
   AND t.computed_at = m.max_at;
