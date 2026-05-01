-- Hashtag weekly counters — spec §4.2/§8.3 v2.2 (2026-05-01).
-- 매핑 안/밖 모든 hashtag 의 주별 등장 카운트 + co-occurrence. emergence rule 평가의
-- pre-aggregated source. LLM 분류 도입 시 input 재사용.
--
-- 적재 cadence: weekly (representative phase 와 동일). DUPLICATE KEY append-only —
-- 같은 (tag, week_start_date) 의 재산출은 computed_at 차이로 누적, view 가 dedup.

CREATE TABLE IF NOT EXISTS hashtag_weekly (
    tag                          VARCHAR(128)  NOT NULL  COMMENT '해시태그 (# 미포함, lowercase)',
    week_start_date              DATE          NOT NULL  COMMENT 'IST 주 월요일 (anchor 의 주)',
    computed_at                  DATETIME      NOT NULL  COMMENT '적재 시각 (UTC)',
    n_posts                      INT           NOT NULL  COMMENT 'post-level dedup 카운트 (한 post 안 같은 tag 여러 번 = +1)',
    n_instances                  INT           NOT NULL  COMMENT 'raw instance 카운트 (post 안 중복 포함)',
    n_posts_with_known_fashion   INT           NOT NULL  COMMENT '같은 post 에 known fashion hashtag 도 있는 post 수 (co-occurrence)',
    is_known_mapping             TINYINT       NOT NULL  COMMENT '0=매핑 외, 1=매핑된 known hashtag (mapping_tables 에 있음)',
    schema_version               VARCHAR(32)   NOT NULL  COMMENT 'pipeline_v2.2'
) ENGINE = OLAP
DUPLICATE KEY (tag, week_start_date, computed_at)
DISTRIBUTED BY HASH (tag) BUCKETS 8
PROPERTIES (
    "replication_num" = "1"
);
