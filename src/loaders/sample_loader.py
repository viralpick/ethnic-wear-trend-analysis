"""샘플 JSON 로더. 크롤러 레포의 실제 DB/API 로더로 교체될 인터페이스.

후속 단계 stub 자리:
    def load_crawled_instagram(since: datetime) -> list[RawInstagramPost]: ...
    def load_crawled_youtube(since: datetime) -> list[RawYouTubeVideo]: ...

각 함수는 TypeAdapter로 리스트 전체를 한 번에 검증한다 — 한 건이라도 스키마가 어긋나면
즉시 Pydantic ValidationError 로 크래시 (spec §project context: "크롤러 스키마 드리프트는
계약 shape이 요구 사항").
"""
from __future__ import annotations

from pathlib import Path

from pydantic import TypeAdapter

from contracts.raw import RawInstagramPost, RawYouTubeVideo

_instagram_adapter = TypeAdapter(list[RawInstagramPost])
_youtube_adapter = TypeAdapter(list[RawYouTubeVideo])


def load_instagram_samples(path: Path) -> list[RawInstagramPost]:
    return _instagram_adapter.validate_json(path.read_bytes())


def load_youtube_samples(path: Path) -> list[RawYouTubeVideo]:
    return _youtube_adapter.validate_json(path.read_bytes())
