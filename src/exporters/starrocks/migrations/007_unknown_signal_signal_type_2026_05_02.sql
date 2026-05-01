-- Phase 2 Tier 4 (2026-05-02) — unknown_signal 에 signal_type column 추가.
--
-- 변경:
--   1. column 추가: signal_type VARCHAR(32) NULL — source 분류
--      값: 'hashtag' / 'vision_garment' / 'vision_fabric' / 'vision_technique' / 'llm_classified'
--   2. 기존 row 는 NULL → 사용처 (BE) 가 default 'hashtag' 로 해석.
--      (StarRocks ALTER TABLE 가 NOT NULL DEFAULT 추가 권한을 svc 가 가지지 못해 NULL 허용)
--   3. _latest view 재정의 — column-list 갱신 (column 추가 시 view drift 방지).
--
-- 호환:
--   - sink 가 항상 signal_type 채워서 적재 (기본 'hashtag').
--   - 옛 row (v2.2) 는 NULL → BE 가 'hashtag' 로 read.
--   - DUPLICATE KEY (tag, computed_at) 변경 없음.

ALTER TABLE unknown_signal
    ADD COLUMN signal_type VARCHAR(32) NULL
    COMMENT 'v2.3 signal source — hashtag / vision_garment / vision_fabric / vision_technique / llm_classified. NULL = legacy v2.2 (=hashtag)';

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
