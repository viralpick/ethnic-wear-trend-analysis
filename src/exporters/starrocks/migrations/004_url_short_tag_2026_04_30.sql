-- Phase 3 (2026-04-30) — item / canonical_group / canonical_object 에 url_short_tag
-- 컬럼 추가 + _latest view 의 dedup key 변경.
--
-- 의도: same URL 의 multi-snapshot ULID 가 view 에 multiple row 로 노출되던 문제 해소.
-- url_short_tag 가 진짜 unique key, source_post_id (ULID) 는 시계열 stamp.
--
-- 적용 순서:
--   1. ALTER ADD COLUMN (3 테이블, async — SHOW ALTER 확인 후 다음 단계)
--   2. row_builder 가 url_short_tag 채움 (이미 main 코드 반영)
--   3. _latest view 재정의 (column list frozen 우회)

ALTER TABLE item
    ADD COLUMN url_short_tag VARCHAR(64)
    COMMENT 'Phase 3: IG shortcode / YT video_id, NULL=parse fail';

ALTER TABLE canonical_group
    ADD COLUMN url_short_tag VARCHAR(64)
    COMMENT 'Phase 3: IG shortcode / YT video_id, NULL=parse fail';

ALTER TABLE canonical_object
    ADD COLUMN url_short_tag VARCHAR(64)
    COMMENT 'Phase 3: IG shortcode / YT video_id, NULL=parse fail';

-- view 재정의 — `COALESCE(url_short_tag, source_post_id)` fallback dedup.
-- 옛 row (url_short_tag NULL) 는 source_post_id 그대로 unique 식별, 신 row 는
-- url_short_tag 기준 dedup → same url 의 multi-ULID 자동 1개로 노출.

CREATE OR REPLACE VIEW item_latest AS
SELECT t.*
FROM item AS t
INNER JOIN (
    SELECT source,
           COALESCE(url_short_tag, source_post_id) AS dedup_key,
           MAX(computed_at) AS max_at
    FROM item
    GROUP BY source, COALESCE(url_short_tag, source_post_id)
) AS m
ON t.source = m.source
   AND COALESCE(t.url_short_tag, t.source_post_id) = m.dedup_key
   AND t.computed_at = m.max_at;

CREATE OR REPLACE VIEW canonical_group_latest AS
SELECT t.*
FROM canonical_group AS t
INNER JOIN (
    SELECT item_source,
           COALESCE(url_short_tag, item_source_post_id) AS dedup_key,
           canonical_index,
           MAX(computed_at) AS max_at
    FROM canonical_group
    GROUP BY item_source, COALESCE(url_short_tag, item_source_post_id), canonical_index
) AS m
ON t.item_source = m.item_source
   AND COALESCE(t.url_short_tag, t.item_source_post_id) = m.dedup_key
   AND t.canonical_index = m.canonical_index
   AND t.computed_at = m.max_at;

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
