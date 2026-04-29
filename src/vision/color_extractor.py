"""ColorExtractor Protocol + FakeColorExtractor — post 단위 color/canonical 추출 경계.

Step D (2026-04-21) 리네임: 이전 VLMClient / VLMVisualResult / FakeVLMClient.
배경: 동료 PoC verdict (2026-04-17) 로 pixel-level hex 는 VLM 아닌 Pipeline B 가 주 경로로
확정 → "VLM" 이름은 실 의미를 오도. Protocol 이름을 "ColorExtractor" 로 일반화 (Fake /
Pipeline B / 미래 VLM hybrid 모두 이 이름 뒤).

spec §7.2 / §7.3:
- M3.G/H (2026-04-28) 이후 IG/YT 둘 다 이 경로 진입. YT 는 video_urls 의 frame 을
  `VideoFrameSource` 가 sampling 해 Pipeline B 가 동일 흐름으로 처리.
- 실 구현체는 `vision.pipeline_b_adapter.PipelineBColorExtractor` (vision extras 필요).
- FakeColorExtractor 는 테스트 / color 담당자 아닌 환경의 기본값. source 무관 빈 결과.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from contracts.common import PaletteCluster
from contracts.normalized import NormalizedContentItem
from contracts.vision import CanonicalOutfit
from settings import VLMConfig


class ColorExtractionResult(BaseModel):
    """post 1건 vision 결과 — canonical 목록 + post-level palette.

    Color 파이프라인 3층 재설계 (2026-04-24, B3a/B3b/B3d):
      - 이전 단일 r/g/b/name/family/silhouette 는 post-level 단일값 금지 원칙 (feedback
        post_level_single_value) + LLM-centric canonical 경로로 전환하면서 제거.
      - canonicals: Phase 4.5 dedup 결과 CanonicalOutfit (palette + cut_off_share 포함)
        리스트. adapter 가 canonical_extractor → β-hybrid (per-object Phase 1+2 →
        Phase 3 통합 weighted KMeans, `aggregate_canonical_palette`) 로 채운다.
        Fake 구현은 빈 리스트.
      - post_palette: canonicals[*].palette 를 area_ratio × within-share 가중 ΔE76 merge
        해 만든 post 대표 3색.
      - silhouette 은 `canonicals[*].representative.silhouette` 로만 접근 (B3d 제거).
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_post_id: str
    canonicals: list[CanonicalOutfit] = Field(default_factory=list)
    post_palette: list[PaletteCluster] = Field(default_factory=list, max_length=3)


@runtime_checkable
class ColorExtractor(Protocol):
    """post 단위 색/실루엣 추출기. 구현체는 FakeColorExtractor 또는 PipelineBColorExtractor."""
    def extract_visual(
        self, items: list[NormalizedContentItem]
    ) -> list[ColorExtractionResult]: ...


class FakeColorExtractor:
    """테스트/기본용 결정론적 추출기. B3a 이후 빈 canonicals + 빈 post_palette 반환.

    용도: snapshot 테스트, color 담당 아닌 환경 (vision extras 미설치) 의 daily CLI 기본값.
    실 값이 필요한 검증은 Pipeline B (vision extras 필요) 또는 per-canonical fixture 로.
    YT video frame 처리는 PipelineBColorExtractor 만 담당 — Fake 는 source 무관 빈 결과.
    """

    def __init__(self, cfg: VLMConfig) -> None:
        self._cfg = cfg

    def extract_visual(
        self, items: list[NormalizedContentItem]
    ) -> list[ColorExtractionResult]:
        return [self._synthesize(item) for item in items]

    def _synthesize(self, item: NormalizedContentItem) -> ColorExtractionResult:
        # B3a: r/g/b/name/family 제거. Fake 는 빈 canonicals/post_palette 로 반환 — real
        # Pipeline B path (factory + loop wiring) 만 검증하는 smoke 는 이걸로 충분.
        return ColorExtractionResult(source_post_id=item.source_post_id)


def run_color_extraction(
    items: list[NormalizedContentItem],
    extractor: ColorExtractor,
    cap: int | None = None,
) -> list[ColorExtractionResult]:
    """Case 1 / Case 2 공용 entry. cap 이 있으면 앞에서 잘라 호출.

    M3.G/H (2026-04-28) 이후 IG/YT 둘 다 진입 가능 — source guard 없음. 호출자 측
    (`_case1_targets` / `_case2_targets`) 가 source 별 cap 적용 후 합친 batch 를 넘긴다.
    """
    bounded = items if cap is None else items[:cap]
    return extractor.extract_visual(bounded)
