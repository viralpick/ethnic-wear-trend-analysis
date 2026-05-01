-- pipeline_spec_v1.0 §1.4 / §5.1 — CanonicalObject (group 의 1 멤버 = 1 BBOX).
-- DUPLICATE KEY = (item key, canonical_index, member_index, computed_at).
-- color_palette 는 §6.5 신규 — BBOX 별 segformer mask 픽셀 풀 KMeans top3.
CREATE TABLE IF NOT EXISTS canonical_object (
    item_source             VARCHAR(16)    NOT NULL,
    item_source_post_id     VARCHAR(64)    NOT NULL,
    canonical_index         INT            NOT NULL,
    member_index            INT            NOT NULL,
    computed_at             DATETIME       NOT NULL,
    object_id               VARCHAR(112)   NOT NULL  COMMENT '{group_id}__{member_idx}',
    group_id                VARCHAR(96)    NOT NULL,
    media_ref               VARCHAR(1024)  NULL      COMMENT 'IG image url / YT video_id (§6.2)',
    garment_type            VARCHAR(128)   NULL      COMMENT 'gemini 원시값',
    fabric                  VARCHAR(128)   NULL      COMMENT 'gemini 원시값',
    technique               VARCHAR(128)   NULL      COMMENT 'gemini 원시값',
    silhouette              VARCHAR(128)   NULL      COMMENT 'gemini 원시값 (Silhouette enum)',
    styling_combo           VARCHAR(128)   NULL      COMMENT 'vision NULL §6.1',
    color_palette           JSON           NULL      COMMENT 'object 단위 KMeans top3 + etc (§6.5)',
    area_ratio              DOUBLE         NULL,
    group_contribution_score DOUBLE        NULL      COMMENT '§2.7 object → group',
    bbox                    JSON           NULL      COMMENT '[x, y, w, h] normalized',
    schema_version          VARCHAR(32)    NOT NULL,
    url_short_tag           VARCHAR(64)    NULL      COMMENT 'v2 (migration 004): IG shortcode / YT video_id, NULL=parse fail'
)
ENGINE=OLAP
DUPLICATE KEY (item_source, item_source_post_id, canonical_index, member_index, computed_at)
COMMENT 'pipeline_spec_v1.0 CanonicalObject — append-only, read via canonical_object_latest view'
DISTRIBUTED BY HASH(item_source_post_id) BUCKETS 4
PROPERTIES (
    "replication_num" = "1"
);
