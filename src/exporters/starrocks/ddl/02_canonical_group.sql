-- pipeline_spec_v1.0 §1.3 / §5.1 — CanonicalGroup (한 item 내 동일 옷 군).
-- DUPLICATE KEY = (item key, canonical_index, computed_at). group_id 는 join 편의 redundant.
CREATE TABLE IF NOT EXISTS canonical_group (
    item_source             VARCHAR(16)    NOT NULL,
    item_source_post_id     VARCHAR(64)    NOT NULL,
    canonical_index         INT            NOT NULL  COMMENT 'item 내 0..n-1',
    computed_at             DATETIME       NOT NULL,
    group_id                VARCHAR(96)    NOT NULL  COMMENT '{source}__{post_id}__{idx}',
    garment_type            VARCHAR(64)    NULL      COMMENT '단일값 (다수결 §2.6)',
    fabric                  VARCHAR(64)    NULL,
    technique               VARCHAR(64)    NULL,
    silhouette              VARCHAR(64)    NULL,
    styling_combo           VARCHAR(64)    NULL      COMMENT 'vision NULL §6.1',
    color_palette           JSON           NULL      COMMENT 'canonical palette top3 + etc',
    item_contribution_score DOUBLE         NULL      COMMENT '§2.7 group → item',
    n_objects               INT            NULL,
    mean_area_ratio         DOUBLE         NULL,
    schema_version          VARCHAR(32)    NOT NULL
)
ENGINE=OLAP
DUPLICATE KEY (item_source, item_source_post_id, canonical_index, computed_at)
COMMENT 'pipeline_spec_v1.0 CanonicalGroup — append-only, read via canonical_group_latest view'
DISTRIBUTED BY HASH(item_source_post_id) BUCKETS 4
PROPERTIES (
    "replication_num" = "1"
);
