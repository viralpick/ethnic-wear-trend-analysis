-- Phase 4 (2026-05-01) — unknown_signal v2 → v3 (emergence rule + weekly cadence).
--
-- 변경:
--   1. column 추가: week_start_date DATE NULL — weekly anchor 의 주 시작일 (월요일 IST)
--   2. column 추가: count_recent_window INT NULL — spike window 안 등장 횟수 (옛 count_3day 의미 변경 대체)
--   3. _latest view 재정의 — (tag, week_start_date) 별 dedup
--
-- 옛 PK 유지 (tag, computed_at). PK 변경은 svc 계정 DROP 권한 부재로 불가 — 컬럼만 ADD.
-- 같은 (tag, week_start_date) 의 재산출은 computed_at 차이로 PK 충돌 회피, view dedup
-- 으로 최신 row 노출. 옛 count_3day 컬럼 (NOT NULL) 은 호환 유지 — sink 가 count_recent_window
-- 와 같은 값 dump (deprecated naming).
--
-- v2 row 가 0 건이라 옛 row NULL 처리 부담 없음. _latest INNER JOIN 이 NULL row 자동 제외.

ALTER TABLE unknown_signal
    ADD COLUMN week_start_date DATE NULL
    COMMENT 'weekly anchor 의 주 시작일 (월요일, IST). v2.2 신규.';

ALTER TABLE unknown_signal
    ADD COLUMN count_recent_window INT NULL
    COMMENT 'spike window (default 14일) 안 등장 횟수. v2.2 신규. 옛 count_3day 폐기.';

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
