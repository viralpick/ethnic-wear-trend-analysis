-- pipeline_spec_v1.0 §1.1 / §5.1 — Representative weekly (G × T × F 조합).
-- representative_id = BIGINT(blake2b(representative_key, digest_size=8)) — deterministic.
-- xxhash 의존성 부재로 blake2b 채택 (row_builder.representative_id). 같은 logical row 는 항상
-- 같은 id 로 매주 적재 가능 (drilldown 안전). representative_key 가 사람 읽는 unique 식별.
-- DUPLICATE KEY = (representative_id, week_start_date, computed_at).
CREATE TABLE IF NOT EXISTS representative_weekly (
    representative_id          BIGINT        NOT NULL  COMMENT 'blake2b(representative_key, digest_size=8) signed',
    week_start_date            DATE          NOT NULL  COMMENT 'IST 월요일',
    computed_at                DATETIME      NOT NULL,
    representative_key         VARCHAR(255)  NOT NULL  COMMENT 'g__t__f',
    display_name               VARCHAR(255)  NULL,
    granularity                VARCHAR(16)   NOT NULL  COMMENT 'weekly 고정',
    score_total                DOUBLE        NULL      COMMENT '§9.2 0~100',
    score_breakdown            JSON          NULL      COMMENT '{social, youtube, cultural, momentum}',
    lifecycle_stage            VARCHAR(16)   NULL      COMMENT 'early|growth|maturity|decline',
    weekly_change_pct          DOUBLE        NULL      COMMENT '직전 주 대비 %',
    weekly_direction           VARCHAR(8)    NULL      COMMENT 'up|flat|down',
    factor_contribution        JSON          NULL      COMMENT '{instagram, youtube} 합=1.0',
    evidence_ig_post_ids       JSON          NULL      COMMENT 'top 4 contribution desc',
    evidence_yt_video_ids      JSON          NULL      COMMENT 'top 4 contribution desc',
    color_palette              JSON          NULL      COMMENT '§2.3 representative palette',
    silhouette_distribution    JSON          NULL,
    occasion_distribution      JSON          NULL,
    styling_combo_distribution JSON          NULL,
    garment_type_distribution  JSON          NULL,
    fabric_distribution        JSON          NULL,
    technique_distribution     JSON          NULL,
    brand_distribution         JSON          NULL      COMMENT '로직 C top 5 brand share, log-scale 균등 분배',
    trajectory                 JSON          NULL      COMMENT '최근 12주 score 시계열',
    total_item_contribution    DOUBLE        NULL      COMMENT 'sparse 적재 분모 (per-rep)',
    effective_item_count       DOUBLE        NULL      COMMENT 'batch 분모 (multiplier-scaled, view normalize 용)',
    schema_version             VARCHAR(32)   NOT NULL
)
ENGINE=OLAP
DUPLICATE KEY (representative_id, week_start_date, computed_at)
COMMENT 'pipeline_spec_v1.0 Representative weekly — append-only, read via representative_weekly_latest view'
DISTRIBUTED BY HASH(representative_id) BUCKETS 8
PROPERTIES (
    "replication_num" = "1"
);
