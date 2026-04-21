"""ColorExtractor Protocol + FakeColorExtractor — post 단위 color/silhouette 추출 경계.

Step D (2026-04-21) 리네임: 이전 VLMClient / VLMVisualResult / FakeVLMClient.
배경: 동료 PoC verdict (2026-04-17) 로 pixel-level hex 는 VLM 아닌 Pipeline B 가 주 경로로
확정 → "VLM" 이름은 실 의미를 오도. Protocol 이름을 "ColorExtractor" 로 일반화 (Fake /
Pipeline B / 미래 VLM hybrid 모두 이 이름 뒤).

spec §7.2 / §7.3:
- YouTube 는 이 경로 호출 대상 아님 (type guard).
- 실 구현체는 `vision.pipeline_b_adapter.PipelineBColorExtractor` (vision extras 필요).
- FakeColorExtractor 는 테스트 / color 담당자 아닌 환경의 기본값.
"""
from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, ValidationError

from contracts.common import ColorFamily, ContentSource, Silhouette
from contracts.normalized import NormalizedContentItem
from settings import VLMConfig
from utils.logging import get_logger

logger = get_logger(__name__)

_RGB_MIN = 0
_RGB_MAX = 255


class ColorExtractionResult(BaseModel):
    """post 1건에서 뽑힌 대표 color + silhouette. 필드 모두 optional — enum-miss/실패 시 None."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_post_id: str
    r: int | None = None
    g: int | None = None
    b: int | None = None
    name: str | None = None
    family: ColorFamily | None = None
    silhouette: Silhouette | None = None


@runtime_checkable
class ColorExtractor(Protocol):
    """post 단위 색/실루엣 추출기. 구현체는 FakeColorExtractor 또는 PipelineBColorExtractor."""
    def extract_visual(
        self, items: list[NormalizedContentItem]
    ) -> list[ColorExtractionResult]: ...


class FakeColorExtractor:
    """테스트/기본용 결정론적 추출기. image_url (없으면 post_id) sha256 해시로 고정 RGB/silhouette.

    용도: snapshot 테스트, color 담당 아닌 환경 (vision extras 미설치) 의 daily CLI 기본값.
    """

    def __init__(self, cfg: VLMConfig) -> None:
        self._cfg = cfg

    def extract_visual(
        self, items: list[NormalizedContentItem]
    ) -> list[ColorExtractionResult]:
        _reject_youtube(items)
        return [self._synthesize(item) for item in items]

    def _synthesize(self, item: NormalizedContentItem) -> ColorExtractionResult:
        url_key = item.image_urls[0] if item.image_urls else item.source_post_id
        digest = hashlib.sha256(f"{self._cfg.seed}:{url_key}".encode()).digest()
        r, g, b = digest[0], digest[1], digest[2]
        family = _pick_enum(digest, 3, list(ColorFamily))
        silhouette = _pick_enum(digest, 4, list(Silhouette))
        try:
            return ColorExtractionResult(
                source_post_id=item.source_post_id,
                r=r, g=g, b=b,
                name=f"stub_{family.value}_{r:02x}{g:02x}{b:02x}",
                family=family,
                silhouette=silhouette,
            )
        except ValidationError as exc:
            logger.info("fake_drop post_id=%s reason=%s", item.source_post_id, exc)
            return ColorExtractionResult(source_post_id=item.source_post_id)


def _pick_enum(digest: bytes, offset: int, choices: list) -> object:
    return choices[digest[offset] % len(choices)]


def _reject_youtube(items: list[NormalizedContentItem]) -> None:
    """spec §7.2 type guard — YouTube 는 color 추출 대상이 아니다."""
    offenders = [i.source_post_id for i in items if i.source != ContentSource.INSTAGRAM]
    if offenders:
        raise ValueError(
            f"ColorExtractor invoked on non-Instagram items: {offenders}. spec §7.2 violation."
        )


def _validate_rgb(value: int | None) -> int | None:
    """0~255 밖이면 None. coerce 금지."""
    if value is None:
        return None
    if _RGB_MIN <= value <= _RGB_MAX:
        return value
    return None


def run_color_extraction(
    items: list[NormalizedContentItem],
    extractor: ColorExtractor,
    cap: int | None = None,
) -> list[ColorExtractionResult]:
    """Case 1 / Case 2 공용 entry. cap 이 있으면 앞에서 잘라 호출. RGB 범위 밖은 None 으로 drop."""
    _reject_youtube(items)
    bounded = items if cap is None else items[:cap]
    raw = extractor.extract_visual(bounded)
    return [
        result.model_copy(
            update={
                "r": _validate_rgb(result.r),
                "g": _validate_rgb(result.g),
                "b": _validate_rgb(result.b),
            }
        )
        for result in raw
    ]
