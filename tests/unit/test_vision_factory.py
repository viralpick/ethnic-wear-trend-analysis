"""vision.factory.build_vision_llm_client — provider 분기 + drift 경고.

vision extras 필요 — Gemini SDK 가 있어야 GeminiVisionLLMClient import 가능.
"""
from __future__ import annotations

import logging

import pytest

pytest.importorskip("google.genai", reason="vision extras required")
pytest.importorskip("json_repair", reason="vision extras required")

from pathlib import Path  # noqa: E402

from settings import VisionLLMConfig  # noqa: E402
from vision.factory import build_vision_llm_client  # noqa: E402
from vision.gemini_client import GeminiVisionLLMClient  # noqa: E402
from vision.llm_cache import LocalJSONCache  # noqa: E402


def test_build_gemini_returns_concrete_client(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    cfg = VisionLLMConfig(
        provider="gemini",
        model_id="gemini-2.5-flash",
        prompt_version="v0.3",
        cache_dir=tmp_path / "llm_cache",
    )
    client = build_vision_llm_client(cfg)
    assert isinstance(client, GeminiVisionLLMClient)
    assert client.model_id == "gemini-2.5-flash"
    assert client.prompt_version == "v0.3"


def test_build_gemini_default_creates_local_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    cfg = VisionLLMConfig(
        provider="gemini",
        model_id="gemini-2.5-flash",
        prompt_version="v0.3",
        cache_dir=tmp_path / "llm_cache",
    )
    client = build_vision_llm_client(cfg)
    # private 접근 — 구현 디테일이지만 factory 가 default cache 를 붙였는지 검증 필요
    assert isinstance(client._cache, LocalJSONCache)  # noqa: SLF001


def test_build_gemini_with_injected_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    cfg = VisionLLMConfig(
        provider="gemini",
        model_id="gemini-2.5-flash",
        prompt_version="v0.3",
        cache_dir=tmp_path / "unused",
    )
    injected = LocalJSONCache(
        base_dir=tmp_path / "custom",
        model_id=cfg.model_id,
        prompt_version=cfg.prompt_version,
    )
    client = build_vision_llm_client(cfg, cache=injected)
    assert client._cache is injected  # noqa: SLF001


def test_build_azure_openai_raises_not_implemented(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    cfg = VisionLLMConfig(
        provider="azure-openai",
        model_id="gpt-5-mini",
        prompt_version="v0.3",
        cache_dir=Path("/tmp/nope"),
    )
    with pytest.raises(NotImplementedError, match="azure-openai"):
        build_vision_llm_client(cfg)


def test_prompt_version_drift_emits_warning(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    cfg = VisionLLMConfig(
        provider="gemini",
        model_id="gemini-2.5-flash",
        prompt_version="v0.1",  # 코드의 PROMPT_VERSION="v0.6" 와 drift
        cache_dir=tmp_path / "llm_cache",
    )
    with caplog.at_level(logging.WARNING, logger="vision.factory"):
        build_vision_llm_client(cfg)
    assert any(
        "vision_llm_prompt_drift" in r.message for r in caplog.records
    )


def test_prompt_version_match_no_warning(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    cfg = VisionLLMConfig(
        provider="gemini",
        model_id="gemini-2.5-flash",
        prompt_version="v0.9",  # garment/fabric/technique 3-tier escape 적용 후 match 기준
        cache_dir=tmp_path / "llm_cache",
    )
    with caplog.at_level(logging.WARNING, logger="vision.factory"):
        build_vision_llm_client(cfg)
    assert not any(
        "vision_llm_prompt_drift" in r.message for r in caplog.records
    )
