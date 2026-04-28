"""M3.F — brand registry: handle/alias 매핑 + caption text @mention 추출.

JSON schema (`data/brand_registry.json`):
- `brands`: [{id, primary_handle, aliases, display_name, category, tier, country, notes}]

사용:
    from attributes.brand_registry import load_brand_registry
    registry = load_brand_registry(Path("data/brand_registry.json"))
    info = registry.lookup("myntra")  # → BrandInfo(name="Myntra", tier=BrandTier.MID)
    info = registry.extract_from_text(caption)  # @mention 첫 brand match
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from contracts.common import BrandTier
from contracts.enriched import BrandInfo

_MENTION_RE = re.compile(r"@([A-Za-z0-9_.]{2,})")


@dataclass(frozen=True)
class BrandEntry:
    id: str
    primary_handle: str  # lowercase
    aliases: tuple[str, ...]  # lowercase
    display_name: str
    category: str
    tier: BrandTier | None
    country: str
    notes: str


@dataclass(frozen=True)
class BrandRegistry:
    """handle.lower() → BrandEntry 인덱스 + 추출 헬퍼."""
    entries: tuple[BrandEntry, ...]
    handle_index: dict[str, BrandEntry]  # primary + aliases 모두 포함

    @staticmethod
    def _normalize(handle: str | None) -> str:
        if not handle:
            return ""
        return handle.lstrip("@").strip().lower()

    def lookup_entry(self, handle: str | None) -> BrandEntry | None:
        return self.handle_index.get(self._normalize(handle))

    def lookup(self, handle: str | None) -> BrandInfo | None:
        entry = self.lookup_entry(handle)
        if entry is None:
            return None
        return BrandInfo(name=entry.display_name, tier=entry.tier)

    def extract_all_from_text(self, text: str | None) -> list[BrandInfo]:
        """caption 의 모든 `@mention` 중 brand registry 에 있는 것을 dedup 순서 보존 반환.

        같은 brand 가 여러 번 mention 돼도 1번만 (BrandEntry.id 기준 dedup).
        """
        if not text:
            return []
        seen_ids: set[str] = set()
        out: list[BrandInfo] = []
        for match in _MENTION_RE.findall(text):
            entry = self.lookup_entry(match)
            if entry is None or entry.id in seen_ids:
                continue
            seen_ids.add(entry.id)
            out.append(BrandInfo(name=entry.display_name, tier=entry.tier))
        return out


def load_brand_registry(path: Path) -> BrandRegistry:
    raw = json.loads(path.read_text(encoding="utf-8"))
    brands_raw = raw.get("brands", [])

    entries: list[BrandEntry] = []
    handle_index: dict[str, BrandEntry] = {}

    for b in brands_raw:
        tier_str = b.get("tier")
        tier = BrandTier(tier_str) if tier_str else None
        primary = b["primary_handle"].strip().lower()
        aliases = tuple(a.strip().lower() for a in b.get("aliases", []))
        entry = BrandEntry(
            id=b["id"],
            primary_handle=primary,
            aliases=aliases,
            display_name=b["display_name"],
            category=b["category"],
            tier=tier,
            country=b.get("country", ""),
            notes=b.get("notes", ""),
        )
        entries.append(entry)

        for handle in (primary, *aliases):
            if handle in handle_index:
                existing = handle_index[handle]
                raise ValueError(
                    f"duplicate handle {handle!r}: brand={entry.id} collides with {existing.id}"
                )
            handle_index[handle] = entry

    return BrandRegistry(entries=tuple(entries), handle_index=handle_index)
