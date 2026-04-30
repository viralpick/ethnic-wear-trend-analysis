-- Unknown attribute signal — spec §4.2 / §8.3.
-- 매핑 테이블에 없는 새 해시태그 (3일 ≥10건) 를 자동 surface. 사람이 매핑에 추가
-- 또는 noise 처리 결정.
--
-- DUPLICATE KEY append-only — 같은 tag 의 day-by-day 변화 추적용. _latest view 가
-- 최신 (computed_at MAX) 1 row 노출.

CREATE TABLE IF NOT EXISTS unknown_signal (
    tag                VARCHAR(128)  NOT NULL  COMMENT '해시태그 (# prefix 포함, lowercase)',
    computed_at        DATETIME      NOT NULL  COMMENT '적재 시각 (UTC), append-only sort key',
    count_3day         INT           NOT NULL  COMMENT '3일 윈도우 누적 빈도',
    first_seen         DATE          NOT NULL  COMMENT '최초 발견일 (IST)',
    likely_category    VARCHAR(64)   NULL      COMMENT 'technique? / fabric? 등 추정',
    reviewed           TINYINT       NOT NULL  COMMENT '0=pending, 1=reviewed',
    schema_version     VARCHAR(32)   NOT NULL  COMMENT 'pipeline_v1.0'
) ENGINE = OLAP
DUPLICATE KEY (tag, computed_at)
DISTRIBUTED BY HASH (tag) BUCKETS 4
PROPERTIES (
    "replication_num" = "1"
);
