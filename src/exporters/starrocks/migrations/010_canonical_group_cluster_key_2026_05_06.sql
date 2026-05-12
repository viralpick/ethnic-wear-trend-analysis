-- Phase 4 (2026-05-06) — canonical_group 에 cluster_key 컬럼 추가.
--
-- 의도: BE 가 representative_weekly.representative_key 와 직접 JOIN 가능.
-- 기존 CONCAT(garment_type, '__', fabric) 매칭은 normalize 차이 (saree→casual_saree,
-- kurti→straight_kurta 등) 로 0 건 매칭. cluster_key 는 normalize 후 결과 직접 적재.
--
-- 적용 순서:
--   1. ALTER ADD COLUMN
--   2. row_builder.build_group_rows 가 normalize_garment_for_cluster + normalize_fabric
--      적용 후 cluster_key 채움 (코드 patch 함께)
--   3. canonical_group_latest view 재정의 (SELECT t.* 라 자동 포함)
--   4. 16w item-resync 로 신규 컬럼 백필

ALTER TABLE canonical_group
    ADD COLUMN cluster_key VARCHAR(128) NULL
    COMMENT 'normalize 후 G__F (representative_key 와 직접 JOIN). ethnic + 매핑 통과만 채움.';

-- view 재정의
CREATE OR REPLACE VIEW canonical_group_latest AS
SELECT t.*
FROM canonical_group AS t
INNER JOIN (
    SELECT item_source,
           COALESCE(url_short_tag, item_source_post_id) AS dedup_key,
           canonical_index, MAX(computed_at) AS max_at
    FROM canonical_group
    GROUP BY item_source, COALESCE(url_short_tag, item_source_post_id), canonical_index
) AS m
    ON t.item_source = m.item_source
   AND COALESCE(t.url_short_tag, t.item_source_post_id) = m.dedup_key
   AND t.canonical_index = m.canonical_index
   AND t.computed_at = m.max_at;
