-- pipeline_spec v2.2 §5.3 — read-side dedup (logical UPSERT).
-- 4 base 테이블이 DUPLICATE KEY append-only 라 같은 PK 의 history row 가 누적됨.
-- view 가 (PK) 기준 MAX(computed_at) row 만 노출. 모든 분석 쿼리는 *_latest 만 본다.
-- StarRocks logical view 라 base 테이블 변경 즉시 반영 (materialize 안함).
--
-- v2 갱신 (2026-04-30, migration 004):
--   - item / canonical_group / canonical_object: dedup key 가 source_post_id (raw ULID)
--     → COALESCE(url_short_tag, source_post_id). same URL multi-snapshot 의 multiple
--     ULID 를 1 row 로 dedup. 옛 row (url_short_tag NULL) 는 source_post_id 그대로.
--   - representative_weekly_latest: G__F 2축 filter (G__T__F 옛 cluster_key 데이터 leak 차단).
--     LENGTH(rk) - LENGTH(REPLACE(rk, '__', '')) = 2 → `__` 분리자 1 occurrence (g__f) 만 통과.

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

CREATE OR REPLACE VIEW representative_weekly_latest AS
SELECT t.*
FROM representative_weekly AS t
INNER JOIN (
    SELECT representative_id, week_start_date, MAX(computed_at) AS max_at
    FROM representative_weekly
    WHERE LENGTH(representative_key) - LENGTH(REPLACE(representative_key, '__', '')) = 2
    GROUP BY representative_id, week_start_date
) AS m
    ON t.representative_id = m.representative_id
   AND t.week_start_date = m.week_start_date
   AND t.computed_at = m.max_at;
