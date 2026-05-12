-- Phase 4 (2026-05-06) — canonical_object 에 frame_index + image_id 컬럼 추가.
--
-- 의도: BE / FE 가 cluster 매칭 BBOX 의 정확한 image / frame 트랙킹.
-- - IG: media_ref 가 이미 분석 image URL 직접 (frame_index NULL)
-- - YT: media_ref = video URL, frame_index = cv2 absolute frame number (정확 seek)
--
-- image_id (raw OutfitMember.image_id, e.g. "{video_stem}_f{N}") 도 디버그 / 검증용
-- 그대로 노출. row_builder.build_object_rows 에서 m.image_id 채움.
--
-- 적용 순서:
--   1. ALTER ADD COLUMN (async — SHOW ALTER 확인)
--   2. row_builder 가 image_id + frame_index 채움 (코드 patch 함께)
--   3. canonical_object_latest view 재정의 (SELECT t.* 라 자동 따라오지만 명시적 재생성)
--   4. 16w item-resync 로 신규 컬럼 백필

ALTER TABLE canonical_object
    ADD COLUMN image_id VARCHAR(128) NULL
    COMMENT 'OutfitMember.image_id raw — IG: filename, YT: {video_stem}_f{global_idx}';

ALTER TABLE canonical_object
    ADD COLUMN frame_index INT NULL
    COMMENT 'YT 비디오 cv2 absolute frame index (image_id 의 _f{N} suffix). IG 는 NULL.';

-- view 재정의 — base ALTER 후 column list frozen 우회 (feedback_starrocks_view_column_frozen).
CREATE OR REPLACE VIEW canonical_object_latest AS
SELECT t.*
FROM canonical_object AS t
INNER JOIN (
    SELECT item_source,
           COALESCE(url_short_tag, item_source_post_id) AS dedup_key,
           canonical_index, member_index,
           MAX(computed_at) AS max_at
    FROM canonical_object
    GROUP BY item_source, COALESCE(url_short_tag, item_source_post_id),
             canonical_index, member_index
) AS m
    ON t.item_source = m.item_source
   AND COALESCE(t.url_short_tag, t.item_source_post_id) = m.dedup_key
   AND t.canonical_index = m.canonical_index
   AND t.member_index = m.member_index
   AND t.computed_at = m.max_at;
