"""M3.F — brand registry loader + lookup.

목표:
- JSON 에서 brand registry 로드 → handle → BrandInfo lookup
- alias / case-insensitive 매핑
- text caption 의 `@mention` 에서 첫 brand match 추출
- 알 수 없는 handle → None
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from attributes.brand_registry import (
    BrandEntry,
    BrandRegistry,
    load_brand_registry,
)
from contracts.common import BrandTier
from contracts.enriched import BrandInfo


@pytest.fixture()
def sample_registry(tmp_path: Path) -> BrandRegistry:
    payload = {
        "_meta": {"source": "test"},
        "brands": [
            {
                "id": "myntra",
                "primary_handle": "myntra",
                "aliases": ["myntrafashion", "myntrasuit"],
                "display_name": "Myntra",
                "category": "marketplace",
                "tier": "mid",
                "country": "IN",
                "notes": "test",
            },
            {
                "id": "manish-malhotra",
                "primary_handle": "manishmalhotraworld",
                "aliases": [],
                "display_name": "Manish Malhotra",
                "category": "designer",
                "tier": "premium_everyday",
                "country": "IN",
                "notes": "",
            },
            {
                "id": "no-tier",
                "primary_handle": "notierbrand",
                "aliases": [],
                "display_name": "No Tier Brand",
                "category": "ethnic",
                "tier": None,
                "country": "IN",
                "notes": "",
            },
        ],
    }
    path = tmp_path / "brand_registry.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return load_brand_registry(path)


def test_load_brand_registry_parses_entries(sample_registry: BrandRegistry) -> None:
    assert len(sample_registry.entries) == 3
    entry = sample_registry.lookup_entry("myntra")
    assert isinstance(entry, BrandEntry)
    assert entry.display_name == "Myntra"
    assert entry.tier == BrandTier.MID


def test_lookup_primary_handle_returns_brand_info(sample_registry: BrandRegistry) -> None:
    info = sample_registry.lookup("manishmalhotraworld")
    assert info == BrandInfo(name="Manish Malhotra", tier=BrandTier.PREMIUM_EVERYDAY)


def test_lookup_alias_returns_primary_brand_info(sample_registry: BrandRegistry) -> None:
    info = sample_registry.lookup("myntrafashion")
    assert info == BrandInfo(name="Myntra", tier=BrandTier.MID)
    info2 = sample_registry.lookup("myntrasuit")
    assert info2 == BrandInfo(name="Myntra", tier=BrandTier.MID)


def test_lookup_case_insensitive(sample_registry: BrandRegistry) -> None:
    assert sample_registry.lookup("MYNTRA") is not None
    assert sample_registry.lookup("ManishMalhotraWorld") is not None


def test_lookup_strips_at_prefix(sample_registry: BrandRegistry) -> None:
    assert sample_registry.lookup("@myntra") is not None


def test_lookup_unknown_returns_none(sample_registry: BrandRegistry) -> None:
    assert sample_registry.lookup("rashmika_mandanna") is None


def test_lookup_brand_without_tier(sample_registry: BrandRegistry) -> None:
    info = sample_registry.lookup("notierbrand")
    assert info == BrandInfo(name="No Tier Brand", tier=None)


def test_extract_all_from_text_collects_brand_mentions(sample_registry: BrandRegistry) -> None:
    """caption 안의 모든 brand @mention 을 순서 보존, 비-brand mention 은 skip."""
    text = (
        "Festive look 💃 @rashmika_mandanna styled in @manishmalhotraworld "
        "x @myntrafashion for @vogueindia"
    )
    infos = sample_registry.extract_all_from_text(text)
    assert infos == [
        BrandInfo(name="Manish Malhotra", tier=BrandTier.PREMIUM_EVERYDAY),
        BrandInfo(name="Myntra", tier=BrandTier.MID),  # alias → primary 로 normalize
    ]


def test_extract_all_from_text_dedup_same_brand(sample_registry: BrandRegistry) -> None:
    """같은 brand (primary + alias) 가 여러 번 mention 돼도 1번만."""
    text = "@myntra collab @myntrafashion drop @myntrasuit"
    infos = sample_registry.extract_all_from_text(text)
    assert infos == [BrandInfo(name="Myntra", tier=BrandTier.MID)]


def test_extract_all_from_text_no_brand_mention_returns_empty(sample_registry: BrandRegistry) -> None:
    text = "Just a regular post @randominfluencer @anotherone"
    assert sample_registry.extract_all_from_text(text) == []


def test_extract_all_from_text_empty_returns_empty(sample_registry: BrandRegistry) -> None:
    assert sample_registry.extract_all_from_text("") == []
    assert sample_registry.extract_all_from_text(None) == []  # type: ignore[arg-type]


def test_alias_collision_detected(tmp_path: Path) -> None:
    """같은 handle 이 두 brand 의 primary/alias 양쪽에 있으면 load 단계에서 raise."""
    payload = {
        "brands": [
            {
                "id": "a",
                "primary_handle": "shared",
                "aliases": [],
                "display_name": "A",
                "category": "ethnic",
                "tier": "mid",
                "country": "IN",
                "notes": "",
            },
            {
                "id": "b",
                "primary_handle": "btwo",
                "aliases": ["shared"],
                "display_name": "B",
                "category": "ethnic",
                "tier": "mid",
                "country": "IN",
                "notes": "",
            },
        ],
    }
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate handle"):
        load_brand_registry(path)


def test_real_registry_loads(tmp_path: Path) -> None:
    """실 production registry 가 깨지지 않고 로드되는지 확인."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    real_path = repo_root / "data" / "brand_registry.json"
    if not real_path.exists():
        pytest.skip("data/brand_registry.json not present")
    registry = load_brand_registry(real_path)
    assert len(registry.entries) >= 30  # 60 brand 큐레이션
    # 대표 handle 들 lookup 확인
    assert registry.lookup("myntra") is not None
    assert registry.lookup("manishmalhotraworld") is not None
    assert registry.lookup("lashkaraa") is not None
