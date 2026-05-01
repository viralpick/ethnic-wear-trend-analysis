-- Hashtag weekly counters — spec §4.2/§8.3 v2.3 (2026-05-02).
-- 매핑 안/밖 모든 hashtag 의 주별 등장 카운트 + co-occurrence. emergence rule 평가의
-- pre-aggregated source. LLM 분류 도입 시 input 재사용.
--
-- 적재 cadence: weekly (representative phase 와 동일). DUPLICATE KEY append-only —
-- 같은 (tag, week_start_date) 의 재산출은 computed_at 차이로 누적, view 가 dedup.
--
-- v2.3 (2026-05-02) — `n_posts_with_known_fashion` 의미 변경:
-- 옛 v2.2: post 에 known_fashion tag 1개 이상이면 +1 (binary co-occurrence)
-- 새 v2.3: post 의 fashion_density >= 0.3 (fashion-context post) 일 때 +1
-- DDL 컬럼명/타입 동일, 값 의미만 변경. schema_version 으로 row 단위 구분 가능.

CREATE TABLE IF NOT EXISTS hashtag_weekly (
    tag                          VARCHAR(128)  NOT NULL  COMMENT '해시태그 (# 미포함, lowercase)',
    week_start_date              DATE          NOT NULL  COMMENT 'IST 주 월요일 (anchor 의 주)',
    computed_at                  DATETIME      NOT NULL  COMMENT '적재 시각 (UTC)',
    n_posts                      INT           NOT NULL  COMMENT 'post-level dedup 카운트 (한 post 안 같은 tag 여러 번 = +1)',
    n_instances                  INT           NOT NULL  COMMENT 'raw instance 카운트 (post 안 중복 포함)',
    n_posts_with_known_fashion   INT           NOT NULL  COMMENT 'v2.3: post 의 fashion_density >= 0.3 (fashion-context) 인 post 수. v2.2 row 는 옛 binary co-occurrence (known_fashion ≥1)',
    is_known_mapping             TINYINT       NOT NULL  COMMENT '0=매핑 외, 1=매핑된 known hashtag (mapping_tables 에 있음)',
    schema_version               VARCHAR(32)   NOT NULL  COMMENT 'pipeline_v2.3 (옛 v2.2 row 도 잔존 — semantic drift 구분 키)'
) ENGINE = OLAP
DUPLICATE KEY (tag, week_start_date, computed_at)
DISTRIBUTED BY HASH (tag) BUCKETS 8
PROPERTIES (
    "replication_num" = "1"
);
