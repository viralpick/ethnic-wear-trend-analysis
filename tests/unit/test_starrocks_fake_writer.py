"""FakeStarRocksWriter pinning — Protocol 호환 + in-memory 적재 정합성."""
from __future__ import annotations

from exporters.starrocks.fake_writer import FakeStarRocksWriter
from exporters.starrocks.writer import StarRocksWriter


def test_protocol_compatibility() -> None:
    writer = FakeStarRocksWriter()
    assert isinstance(writer, StarRocksWriter)


def test_write_batch_returns_row_count() -> None:
    writer = FakeStarRocksWriter()
    n = writer.write_batch("item", [{"source": "instagram", "source_post_id": "x1"}])
    assert n == 1
    assert writer.call_count == 1


def test_write_batch_accumulates_across_calls() -> None:
    writer = FakeStarRocksWriter()
    writer.write_batch("item", [{"id": 1}, {"id": 2}])
    writer.write_batch("item", [{"id": 3}])
    assert writer.total_rows("item") == 3
    assert writer.call_count == 2
    # 적재 순서 보존.
    assert [r["id"] for r in writer.batches["item"]] == [1, 2, 3]


def test_total_rows_aggregates_all_tables() -> None:
    writer = FakeStarRocksWriter()
    writer.write_batch("item", [{"id": 1}])
    writer.write_batch("canonical_group", [{"id": 2}, {"id": 3}])
    assert writer.total_rows() == 3
    assert writer.total_rows("item") == 1
    assert writer.total_rows("canonical_group") == 2
    # 빈 table 은 0.
    assert writer.total_rows("representative_weekly") == 0


def test_reset_clears_state() -> None:
    writer = FakeStarRocksWriter()
    writer.write_batch("item", [{"id": 1}])
    writer.reset()
    assert writer.total_rows() == 0
    assert writer.call_count == 0
