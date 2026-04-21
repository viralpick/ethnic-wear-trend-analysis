"""Raw batch 로더. 크롤러 DB/API 로 교체될 인터페이스.

v1 실제 파이프라인에서는 이 Protocol 의 DB/API 구현을 붙인다. 현재 스켈레톤은 로컬
sample_data/ 디렉토리에서 읽는 구현만 제공 (크롤러 레포 준비 전).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Protocol, runtime_checkable

from contracts.raw import RawInstagramPost, RawYouTubeVideo
from loaders.sample_loader import load_instagram_samples, load_youtube_samples


@dataclass(frozen=True)
class RawDailyBatch:
    instagram: list[RawInstagramPost]
    youtube: list[RawYouTubeVideo]


@runtime_checkable
class RawLoader(Protocol):
    def load_batch(self, target_date: date) -> RawDailyBatch: ...


class LocalSampleLoader:
    """sample_data/ 의 고정 JSON 두 개를 읽어 RawDailyBatch 로 반환.

    TODO(§10.1): 실제 v1 파이프라인에서는 크롤러 DB 의 `collected_at BETWEEN target_date ...`
    쿼리로 대체. 지금은 target_date 를 무시하고 전체 샘플을 돌려준다.
    """

    def __init__(self, sample_dir: Path) -> None:
        self._sample_dir = sample_dir

    def load_batch(self, target_date: date) -> RawDailyBatch:  # noqa: ARG002 — stub 에서는 사용 X
        return RawDailyBatch(
            instagram=load_instagram_samples(
                self._sample_dir / "sample_instagram_posts.json",
            ),
            youtube=load_youtube_samples(
                self._sample_dir / "sample_youtube_videos.json",
            ),
        )
