-- M3.F brand 1:N 적용 (2026-04-28).
-- 기존 brand_mentioned VARCHAR(128) 컬럼을 multi-brand JSON 배열 컬럼으로 교체.
-- 데이터 보존: 기존 컬럼은 항상 NULL 적재 상태였으므로 데이터 손실 없음.
--
-- 적용 순서 (수동 실행):
--   1. ADD COLUMN (light schema change, 즉시)
--   2. DROP COLUMN (light schema change, 즉시)
--   3. item_latest view 는 SELECT t.* 라 자동 따라옴 (재생성 불필요)
--
-- StarRocks 4.0.8 light schema change 지원 — 데이터 재작성 없음.

ALTER TABLE item ADD COLUMN brands_mentioned JSON COMMENT 'M3.F multi-brand [{name, tier}, ...]';
ALTER TABLE item DROP COLUMN brand_mentioned;
