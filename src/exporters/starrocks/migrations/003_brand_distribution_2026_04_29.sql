-- 로직 C — representative_weekly.brand_distribution 컬럼 추가 (2026-04-29).
--
-- 의도: post 의 brands list (length N) 가 cluster C 로 share=s fan-out 될 때,
--   brand 별 raw 기여 = s × (1/log2(N+1)) × (1/N) 합산 → top 5 + share≥0.05 cut → 정규화.
--   대시보드 representative 화면의 brand 패널 데이터 source.
--
-- 적용 순서 (수동 실행):
--   1. ADD COLUMN (light schema change, 즉시)
--   2. SHOW ALTER TABLE COLUMN representative_weekly 으로 FINISHED 확인
--   3. representative_weekly_latest view CREATE OR REPLACE — view 는 column list frozen
--      (feedback_starrocks_view_column_frozen)
--
-- StarRocks 4.0.8 light schema change 지원 — 데이터 재작성 없음.

ALTER TABLE representative_weekly
    ADD COLUMN brand_distribution JSON
    COMMENT '로직 C — top 5 brand share, log-scale 균등 분배. {name: pct} 합=1.0 또는 NULL';

-- view column list refresh — 위 ALTER 가 SHOW ALTER 에서 FINISHED 확인 후 실행.
CREATE OR REPLACE VIEW representative_weekly_latest AS
SELECT t.*
FROM representative_weekly AS t
INNER JOIN (
    SELECT representative_id, week_start_date, MAX(computed_at) AS max_at
    FROM representative_weekly
    GROUP BY representative_id, week_start_date
) AS m
    ON t.representative_id = m.representative_id
   AND t.week_start_date = m.week_start_date
   AND t.computed_at = m.max_at;
