-- 운영 메트릭 view — vision pipeline 통과 + ethnic canonical 1개 이상 적재된 post 만.
-- color_palette 는 `_apply_extraction_result` 에서 ethnic canonical pool 결과로만 채움.
-- 비-ethnic 라벨 보존 canonical 은 pools=[] → palette NULL → 자연 필터.
--
-- 정책 (2026-04-28):
--   - base `item_latest` 는 비-ethnic / YT text-only 추론 post 까지 모두 보존
--     (false negative 사후 분석 / 검수 대시보드용)
--   - 운영 query 는 `item_ethnic_latest` 만 사용 — color/silhouette dashboard 오염 방지
--
-- canonical_extractor 의 라벨 보존 디자인 (비-ethnic outfit 도 canonicals 에 살아남음) →
-- group/object 도 비-ethnic 적재 가능. ItemDistribution → representative_weekly 단계의
-- contribution 차단은 코드 가드 (`is_canonical_ethnic`) 에서 별도 처리.
-- representative_weekly 는 이미 ethnic-only 수렴이라 별도 view 불필요.

CREATE OR REPLACE VIEW item_ethnic_latest AS
SELECT *
FROM item_latest
WHERE color_palette IS NOT NULL;

CREATE OR REPLACE VIEW canonical_group_ethnic_latest AS
SELECT *
FROM canonical_group_latest
WHERE color_palette IS NOT NULL;

CREATE OR REPLACE VIEW canonical_object_ethnic_latest AS
SELECT *
FROM canonical_object_latest
WHERE color_palette IS NOT NULL;
