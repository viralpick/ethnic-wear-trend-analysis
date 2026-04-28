"""run_daily_pipeline CLI 가드 pinning.

`--sink starrocks{,_insert}` + fake 조합 silent stale 방지 가드 (2026-04-27).
color_extractor / vision_llm 양쪽 모두 fake 면 빈 canonicals/palette 가 latest 로
적재돼 직전 정상 row 를 가리는 회귀 — 운영 sink 에 한해 reject.
"""
from __future__ import annotations

import pytest

from pipelines.run_daily_pipeline import _validate_sink_extractor


# --------------------------------------------------------------------------- #
# Reject: live sink + fake color_extractor (canonicals=[])

@pytest.mark.parametrize("sink", ["starrocks", "starrocks_insert"])
def test_live_sink_with_fake_color_extractor_raises(sink: str) -> None:
    with pytest.raises(SystemExit) as exc:
        _validate_sink_extractor(sink, "fake", "gemini")
    msg = str(exc.value)
    assert "--color-extractor pipeline_b" in msg
    assert "silent stale" in msg


# --------------------------------------------------------------------------- #
# Reject: live sink + fake vision_llm (palette skip via empty pools)

@pytest.mark.parametrize("sink", ["starrocks", "starrocks_insert"])
def test_live_sink_with_fake_vision_llm_raises(sink: str) -> None:
    with pytest.raises(SystemExit) as exc:
        _validate_sink_extractor(sink, "pipeline_b", "fake")
    msg = str(exc.value)
    assert "--vision-llm gemini" in msg
    assert "silent stale" in msg


# --------------------------------------------------------------------------- #
# Reject: live sink + 양쪽 fake (메시지에 둘 다 포함)

@pytest.mark.parametrize("sink", ["starrocks", "starrocks_insert"])
def test_live_sink_with_both_fake_lists_both(sink: str) -> None:
    with pytest.raises(SystemExit) as exc:
        _validate_sink_extractor(sink, "fake", "fake")
    msg = str(exc.value)
    assert "--color-extractor pipeline_b" in msg
    assert "--vision-llm gemini" in msg


# --------------------------------------------------------------------------- #
# Allow: live sink + pipeline_b + gemini (운영 정공법)

@pytest.mark.parametrize("sink", ["starrocks", "starrocks_insert"])
def test_live_sink_with_live_extractors_passes(sink: str) -> None:
    _validate_sink_extractor(sink, "pipeline_b", "gemini")


# --------------------------------------------------------------------------- #
# Allow: --sink none + 모든 조합 (snapshot/test path 보존)

@pytest.mark.parametrize(
    ("color_extractor", "vision_llm"),
    [
        ("fake", "fake"),
        ("fake", "gemini"),
        ("pipeline_b", "fake"),
        ("pipeline_b", "gemini"),
    ],
)
def test_sink_none_passes_any_combo(color_extractor: str, vision_llm: str) -> None:
    _validate_sink_extractor("none", color_extractor, vision_llm)
