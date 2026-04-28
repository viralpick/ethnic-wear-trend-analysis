-- M3.F brand 1:N 적용 (2026-04-28).
-- 기존 brand_mentioned VARCHAR(128) 컬럼을 multi-brand JSON 배열 컬럼으로 교체.
-- 데이터 보존: 기존 컬럼은 항상 NULL 적재 상태였으므로 데이터 손실 없음.
--
-- 적용 순서 (수동 실행):
--   1. ADD COLUMN (light schema change, 즉시)
--   2. DROP COLUMN (light schema change. ADD job 종료 후 진행 — 같은 테이블에
--      schema change 동시 실행 불가, "schema change in progress" 1064 에러)
--   3. SHOW ALTER TABLE COLUMN ... 으로 두 잡 모두 FINISHED 확인
--   4. item_latest / item_ethnic_latest view 재생성 (StarRocks view 는 생성 시점
--      column list 가 frozen — `SELECT t.*` 라도 base 테이블 schema change 후
--      자동 반영 안 됨. CREATE OR REPLACE 로 갱신 필요)
--
-- StarRocks 4.0.8 light schema change 지원 — 데이터 재작성 없음.

ALTER TABLE item ADD COLUMN brands_mentioned JSON COMMENT 'M3.F multi-brand [{name, tier}, ...]';
ALTER TABLE item DROP COLUMN brand_mentioned;

-- view column list refresh — 위 ALTER 두 잡이 SHOW ALTER 에서 FINISHED 확인 후 실행.
CREATE OR REPLACE VIEW item_latest AS
SELECT t.*
FROM item AS t
INNER JOIN (
    SELECT source, source_post_id, MAX(computed_at) AS max_at
    FROM item
    GROUP BY source, source_post_id
) AS m
    ON t.source = m.source
   AND t.source_post_id = m.source_post_id
   AND t.computed_at = m.max_at;

CREATE OR REPLACE VIEW item_ethnic_latest AS
SELECT *
FROM item_latest
WHERE color_palette IS NOT NULL;
