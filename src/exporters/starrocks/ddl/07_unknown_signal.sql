-- Unknown attribute signal — spec §4.2 / §8.3 (v2.2, 2026-05-01).
-- emergence rule: baseline_window 부재 + spike_window ≥K + ethnic_co_share ≥R 통과한
-- 매핑 외 hashtag. weekly cadence 로 매 주 anchor 별 산출 (representative_weekly 와 정합).
--
-- DUPLICATE KEY append-only — 같은 tag day-by-day / week-by-week 변화 추적.
-- _latest view 가 (tag, week_start_date) 별 최신 (computed_at MAX) 1 row 노출.
--
-- count_3day 는 옛 v1 호환 컬럼 — sink 가 count_recent_window 와 같은 값 dump.
-- 신규 컬럼은 v2.2 에서 추가됨 (migration 005).

CREATE TABLE IF NOT EXISTS unknown_signal (
    tag                  VARCHAR(128)  NOT NULL  COMMENT '해시태그 (# prefix 포함, lowercase)',
    computed_at          DATETIME      NOT NULL  COMMENT '적재 시각 (UTC), append-only sort key',
    count_3day           INT           NOT NULL  COMMENT 'v1 호환 — count_recent_window 와 같은 값 dump',
    first_seen           DATE          NOT NULL  COMMENT '최초 발견일 (IST)',
    likely_category      VARCHAR(64)   NULL      COMMENT 'technique? / fabric? 등 추정',
    reviewed             TINYINT       NOT NULL  COMMENT '0=pending, 1=reviewed',
    schema_version       VARCHAR(32)   NOT NULL  COMMENT 'pipeline_v2.2',
    week_start_date      DATE          NULL      COMMENT 'weekly anchor 의 주 시작일 (월요일, IST). v2.2 신규',
    count_recent_window  INT           NULL      COMMENT 'spike window 안 등장 횟수. v2.2 신규'
) ENGINE = OLAP
DUPLICATE KEY (tag, computed_at)
DISTRIBUTED BY HASH (tag) BUCKETS 4
PROPERTIES (
    "replication_num" = "1"
);
