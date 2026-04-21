"""Daily pipeline snapshot (spec §10.1 전체 흐름).

sample_data 에 대해 FakeLLM + FakeVLM 을 쓴 결정론적 출력을 golden_summaries.json 과 비교한다.
LLM/VLM 팩터를 갈아끼우지 않는 이상 출력이 바뀌면 바로 drift 로 잡는다.

갱신 절차: 코드 변경이 의도적 드리프트이면 env SNAPSHOT_UPDATE=1 로 실행해 golden 재생성.
"""
from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

import pytest

from attributes.extract_text_attributes_llm import DEFAULT_LLM_SEED, FakeLLMClient
from pipelines.run_daily_pipeline import run_pipeline
from settings import load_settings
from vision.extract_color_features import FakeVLMClient

_FIXED_DATE = date(2026, 4, 21)
_GOLDEN = Path(__file__).parent / "golden_summaries.json"


@pytest.fixture
def tmp_settings(tmp_path: Path):
    settings = load_settings()
    settings.paths.outputs = tmp_path / "outputs"
    settings.paths.outputs.mkdir(parents=True, exist_ok=True)
    return settings


def _run(settings) -> Path:
    llm = FakeLLMClient(seed=DEFAULT_LLM_SEED)
    vlm = FakeVLMClient(cfg=settings.vlm)
    run_pipeline(settings, _FIXED_DATE, llm, vlm)
    return settings.paths.outputs / _FIXED_DATE.isoformat() / "summaries.json"


def test_daily_pipeline_snapshot_matches_golden(tmp_settings) -> None:
    summaries_path = _run(tmp_settings)
    produced = json.loads(summaries_path.read_text())

    if os.environ.get("SNAPSHOT_UPDATE") == "1":
        _GOLDEN.write_text(json.dumps(produced, indent=2, ensure_ascii=False))
        pytest.skip("SNAPSHOT_UPDATE=1 — golden regenerated. Commit and re-run.")

    if not _GOLDEN.exists():
        pytest.fail(
            f"Golden missing at {_GOLDEN}. "
            "Run once with SNAPSHOT_UPDATE=1 then commit the generated file."
        )

    golden = json.loads(_GOLDEN.read_text())
    assert produced == golden, (
        "Daily pipeline output drifted from golden_summaries.json. "
        "If intentional, regenerate with SNAPSHOT_UPDATE=1 and commit."
    )


def test_daily_pipeline_writes_enriched(tmp_settings) -> None:
    _run(tmp_settings)
    enriched_path = (
        tmp_settings.paths.outputs / _FIXED_DATE.isoformat() / "enriched.json"
    )
    assert enriched_path.exists()
    data = json.loads(enriched_path.read_text())
    assert isinstance(data, list)
    assert len(data) > 0
