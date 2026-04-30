-- unknown_signal_latest — tag 별 최신 1 row 노출 (검수 대시보드용).

CREATE OR REPLACE VIEW unknown_signal_latest AS
SELECT t.*
FROM unknown_signal AS t
INNER JOIN (
    SELECT tag, MAX(computed_at) AS max_at
    FROM unknown_signal
    GROUP BY tag
) AS m
ON t.tag = m.tag
   AND t.computed_at = m.max_at;
