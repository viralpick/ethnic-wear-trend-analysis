-- Phase 4 (2026-05-01) — hashtag_weekly 신규 테이블 + view.
--
-- 의도: hashtag 자체를 1급 entity 로 격상. weekly 단위 raw count + co-occurrence
-- 적재. emergence rule (unknown_signal) 평가 source 로 reuse, LLM 분류 도입 시
-- input cache 로 활용.
--
-- 적용:
--   1. CREATE TABLE hashtag_weekly
--   2. CREATE OR REPLACE VIEW hashtag_weekly_latest

CREATE TABLE IF NOT EXISTS hashtag_weekly (
    tag                          VARCHAR(128)  NOT NULL,
    week_start_date              DATE          NOT NULL,
    computed_at                  DATETIME      NOT NULL,
    n_posts                      INT           NOT NULL,
    n_instances                  INT           NOT NULL,
    n_posts_with_known_fashion   INT           NOT NULL,
    is_known_mapping             TINYINT       NOT NULL,
    schema_version               VARCHAR(32)   NOT NULL
) ENGINE = OLAP
DUPLICATE KEY (tag, week_start_date, computed_at)
DISTRIBUTED BY HASH (tag) BUCKETS 8
PROPERTIES (
    "replication_num" = "1"
);

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
