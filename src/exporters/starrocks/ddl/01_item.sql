-- pipeline_spec v2.2 §1.2 / §5.1 — Item (1 IG post 또는 1 YT video).
-- DUPLICATE KEY + computed_at append-only. read-side dedup 은 `item_latest` view.
-- raw png DB 의 (source, source_post_id) 와 1:1 join 가능.
-- v2 (migration 004): url_short_tag VARCHAR(64) — IG shortcode / YT video_id.
-- 같은 URL 의 multi-snapshot ULID 를 view 에서 1 row 로 dedup (COALESCE fallback).
CREATE TABLE IF NOT EXISTS item (
    source                  VARCHAR(16)    NOT NULL  COMMENT 'instagram|youtube',
    source_post_id          VARCHAR(64)    NOT NULL  COMMENT 'raw png DB ULID',
    computed_at             DATETIME       NOT NULL  COMMENT '적재 시각 (UTC), append-only sort key',
    posted_at               DATETIME       NULL      COMMENT 'IST',
    garment_type_dist       JSON           NULL      COMMENT '{value: pct} 로직 A',
    fabric_dist             JSON           NULL      COMMENT '{value: pct} 로직 A',
    technique_dist          JSON           NULL      COMMENT '{value: pct} 로직 A',
    silhouette_dist         JSON           NULL      COMMENT '{value: pct} 로직 B',
    styling_combo_dist      JSON           NULL      COMMENT '{value: pct} 로직 A (vision NULL)',
    occasion                VARCHAR(64)    NULL      COMMENT 'rule|gpt 단일값',
    brands_mentioned        JSON           NULL      COMMENT 'M3.F multi-brand [{name, tier}, ...] (§6.1)',
    color_palette           JSON           NULL      COMMENT 'post_palette top3 + etc',
    engagement_raw          BIGINT         NULL      COMMENT 'IG: like+comment / YT: views',
    account_handle          VARCHAR(255)   NULL,
    account_follower_count  BIGINT         NULL,
    schema_version          VARCHAR(32)    NOT NULL  COMMENT 'pipeline_v2.2',
    url_short_tag           VARCHAR(64)    NULL      COMMENT 'v2 (migration 004): IG shortcode / YT video_id, NULL=parse fail'
)
ENGINE=OLAP
DUPLICATE KEY (source, source_post_id, computed_at)
COMMENT 'pipeline_spec_v1.0 Item — append-only, read via item_latest view'
DISTRIBUTED BY HASH(source_post_id) BUCKETS 4
PROPERTIES (
    "replication_num" = "1",
    "enable_persistent_index" = "true"
);
