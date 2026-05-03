"""Tier 3 — LLM signal classifier (Phase 2 v2.3, 2026-05-02).

FakeLLMSignalClassifier 는 substring 매칭 결정론. AzureOpenAI 구현은 이 테스트에서
제외 (extras + LLM 호출 필요).
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

from attributes.llm_signal_classifier import (
    CATEGORIES,
    FakeLLMSignalClassifier,
    LocalSignalCache,
    SignalClassification,
    annotate_signals,
)
from contracts.output import UnknownAttributeSignal


def test_fake_classifier_basic() -> None:
    clf = FakeLLMSignalClassifier()
    out = clf.classify(["chikankari", "love", "kurta", "randomword"])
    assert len(out) == 4
    by_word = {c.word: c for c in out}
    assert by_word["chikankari"].is_ethnic
    assert by_word["chikankari"].category == "technique"
    assert by_word["love"].category == "stoplist"
    assert not by_word["love"].is_ethnic
    assert by_word["kurta"].category == "garment"
    assert by_word["randomword"].category == "uncategorized"


def test_categories_are_known() -> None:
    """CATEGORIES list 는 frozen — DDL signal_type 와 정합."""
    expected = {
        "garment", "fabric", "technique", "styling_combo", "occasion",
        "brand", "stoplist", "uncategorized",
    }
    assert set(CATEGORIES) == expected


def test_local_cache_round_trip(tmp_path: Path) -> None:
    cache = LocalSignalCache(tmp_path, model_id="gpt-5-mini", prompt_version="v0.1")
    clf = SignalClassification(
        word="chikankari", is_ethnic=True, category="technique",
        variant_canonical="chikankari", confidence=0.95,
    )
    cache.put(clf)
    loaded = cache.get("chikankari")
    assert loaded == clf

    miss = cache.get("never_stored")
    assert miss is None


def test_local_cache_prompt_version_mismatch_invalidates(tmp_path: Path) -> None:
    cache = LocalSignalCache(tmp_path, model_id="gpt-5-mini", prompt_version="v0.1")
    clf = SignalClassification(
        word="aaa", is_ethnic=True, category="garment",
        variant_canonical=None, confidence=1.0,
    )
    cache.put(clf)
    # 다른 prompt_version 로 cache 열면 miss
    cache2 = LocalSignalCache(tmp_path, model_id="gpt-5-mini", prompt_version="v0.2")
    assert cache2.get("aaa") is None


def _signal(tag: str, signal_type: str = "hashtag") -> UnknownAttributeSignal:
    return UnknownAttributeSignal(
        tag=f"#{tag}", week_start_date=date(2026, 4, 20),
        count_recent_window=5, first_seen=date(2026, 4, 15),
        signal_type=signal_type,
    )


def test_annotate_signals_drops_non_ethnic() -> None:
    classifier = FakeLLMSignalClassifier()
    signals = [_signal("chikankari"), _signal("love")]
    annotated, dropped = annotate_signals(signals, classifier, drop_non_ethnic=True)
    assert len(annotated) == 1
    assert annotated[0].tag == "#chikankari"
    assert annotated[0].likely_category == "technique"
    assert dropped == ["love"]


def test_annotate_signals_keep_non_ethnic_when_disabled() -> None:
    classifier = FakeLLMSignalClassifier()
    signals = [_signal("chikankari"), _signal("love")]
    annotated, dropped = annotate_signals(signals, classifier, drop_non_ethnic=False)
    assert len(annotated) == 2
    assert dropped == []
    by_tag = {s.tag: s for s in annotated}
    assert by_tag["#love"].likely_category == "stoplist"


def test_annotate_signals_preserves_signal_type() -> None:
    classifier = FakeLLMSignalClassifier()
    signals = [_signal("chikankari", signal_type="vision_technique")]
    annotated, _ = annotate_signals(signals, classifier, drop_non_ethnic=True)
    assert len(annotated) == 1
    # signal_type 은 유지 (vision_technique)
    assert annotated[0].signal_type == "vision_technique"
    # likely_category 만 LLM 결과로 갱신
    assert annotated[0].likely_category == "technique"


# ---- run_representative_phase wire-up (Phase 2 v2.3 Tier 3) ----

def test_build_signal_classifier_env_disable(monkeypatch, tmp_path) -> None:
    """env UNKNOWN_SIGNAL_LLM_CLASSIFY=0 → None."""
    from pipelines.run_daily_pipeline import _build_signal_classifier
    from settings import load_settings
    monkeypatch.setenv("UNKNOWN_SIGNAL_LLM_CLASSIFY", "0")
    s = load_settings()
    assert _build_signal_classifier(s) is None


def test_build_signal_classifier_init_failure_graceful(monkeypatch) -> None:
    """ImportError / KeyError 만 graceful skip → None.

    그 외 (RuntimeError 등 실제 장애) 는 raise (룰 §4 silent drop 금지).
    """
    from pipelines.run_daily_pipeline import _build_signal_classifier
    from settings import load_settings
    import pytest
    monkeypatch.delenv("UNKNOWN_SIGNAL_LLM_CLASSIFY", raising=False)
    s = load_settings()
    import attributes.llm_signal_classifier as mod

    def _make_mock(exc_factory):
        class _Mock:
            MODEL_ID = "test-model"
            PROMPT_VERSION = "v0"
            def __init__(self, *a, **kw):
                raise exc_factory()
        return _Mock

    # KeyError (env 누락) → graceful None
    monkeypatch.setattr(
        mod, "AzureOpenAILLMSignalClassifier",
        _make_mock(lambda: KeyError("AZURE_OPENAI_API_KEY")),
    )
    assert _build_signal_classifier(s) is None

    # ImportError (extras 미설치) → graceful None
    monkeypatch.setattr(
        mod, "AzureOpenAILLMSignalClassifier",
        _make_mock(lambda: ImportError("openai extras missing")),
    )
    assert _build_signal_classifier(s) is None

    # 그 외 알려지지 않은 예외 → raise (silent drop 금지)
    monkeypatch.setattr(
        mod, "AzureOpenAILLMSignalClassifier",
        _make_mock(lambda: RuntimeError("unexpected — should propagate")),
    )
    with pytest.raises(RuntimeError, match="should propagate"):
        _build_signal_classifier(s)
