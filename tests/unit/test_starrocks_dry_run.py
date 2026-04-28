"""dry_run sink 출력 헬퍼 pinning."""
from __future__ import annotations

import logging

from exporters.starrocks.dry_run import log_dry_run_summary
from exporters.starrocks.fake_writer import FakeStarRocksWriter


def test_log_dry_run_summary_emits_counts_and_order(caplog) -> None:
    writer = FakeStarRocksWriter()
    writer.write_batch("item", [{"id": 1}])
    writer.write_batch("canonical_group", [{"id": 2}, {"id": 3}])
    counts = {"item": 1, "canonical_group": 2}

    with caplog.at_level(logging.INFO):
        log_dry_run_summary(writer, counts)

    text = caplog.text
    assert "table=item rows=1" in text
    assert "table=canonical_group rows=2" in text
    # call_order 가 그대로 노출 (sequence 검증).
    assert "['item', 'canonical_group']" in text
    # 각 table sample 출력.
    assert "sample item[0]" in text
    assert "sample canonical_group[0]" in text


def test_log_dry_run_summary_truncates_long_collection_values(caplog) -> None:
    writer = FakeStarRocksWriter()
    long_list = [{"k": "v" * 50} for _ in range(20)]
    writer.write_batch("item", [{"id": 1, "huge": long_list}])

    with caplog.at_level(logging.INFO):
        log_dry_run_summary(writer, {"item": 1})

    assert "...(truncated)" in caplog.text


def test_log_dry_run_summary_handles_empty_table(caplog) -> None:
    writer = FakeStarRocksWriter()
    counts = {"item": 0}

    with caplog.at_level(logging.INFO):
        log_dry_run_summary(writer, counts)

    assert "table=item rows=0" in caplog.text
    assert "sample item: (empty)" in caplog.text
