-- Phase β1 effective_item_count (2026-04-28).
-- representative_weekly 에 multiplier-scaled batch denominator 컬럼 추가.
-- 기존 row 는 NULL 로 남김 (재적재 시 새 값 채워짐).
--
-- 적용 순서 (수동 실행):
--   1. ADD COLUMN (light schema change, 즉시)
--   2. representative_weekly_latest view 는 SELECT t.* 라 자동 따라옴
--   3. representative_weekly_normalized view 신규 (05_views_latest.sql 재실행)
--
-- StarRocks 4.0.8 light schema change 지원 — 데이터 재작성 없음.

ALTER TABLE representative_weekly
    ADD COLUMN effective_item_count DOUBLE COMMENT 'batch 분모 (multiplier-scaled, view normalize 용)';
