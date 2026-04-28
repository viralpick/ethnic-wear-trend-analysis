"""StarRocks exporter — pipeline_spec_v1.0 §5 4-table append-only adapter.

DDL: `ddl/01_item.sql` ~ `ddl/05_views_latest.sql` (sequential filenames = init 순서).
Schema model: DUPLICATE KEY + (PK..., computed_at). 같은 logical row 의 history 누적,
조회는 `*_latest` view 가 MAX(computed_at) 으로 logical UPSERT 흉내.
"""
