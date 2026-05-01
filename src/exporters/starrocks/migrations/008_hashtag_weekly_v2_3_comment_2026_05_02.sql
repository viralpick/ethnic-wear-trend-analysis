-- Phase 2 follow-up (2026-05-02) — hashtag_weekly silent drift 정정.
--
-- v2.3 으로 `n_posts_with_known_fashion` 컬럼 의미 변경됨 (옛 binary co-occurrence
-- → fashion_density >= 0.3 fashion-context post). DDL 컬럼명/타입 동일, 값 의미만
-- 변경. schema_version 으로 row 단위 구분.
--
-- 본 migration: 컬럼 comment 만 갱신 (prod SHOW CREATE TABLE 결과의 의미 정렬).
-- 데이터/스키마 영향 없음.

ALTER TABLE hashtag_weekly
    MODIFY COLUMN n_posts_with_known_fashion INT NOT NULL
    COMMENT 'v2.3: post 의 fashion_density >= 0.3 (fashion-context) 인 post 수. v2.2 row 는 옛 binary co-occurrence (known_fashion ≥1)';

ALTER TABLE hashtag_weekly
    MODIFY COLUMN schema_version VARCHAR(32) NOT NULL
    COMMENT 'pipeline_v2.3 (옛 v2.2 row 도 잔존 — semantic drift 구분 키)';
