"""VLM 기반 color/silhouette 추출 (spec §7.2, §7.3).

이 모듈이 vision 패키지의 공용 Protocol + FakeVLMClient 를 소유한다.
extract_silhouette.py 는 여기서 재사용한다.

YouTube 가드: source != INSTAGRAM 아이템은 호출 전에 ValueError. spec §7.2 명시.
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


class VLMVisualResult(BaseModel):
    """spec §7.3 VLM 출력 JSON. 모든 필드 optional — 실패/enum-miss 시 None."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_post_id: str
    r: int | None = None
    g: int | None = None
    b: int | None = None
    name: str | None = None
    family: ColorFamily | None = None
    silhouette: Silhouette | None = None


@runtime_checkable
class VLMClient(Protocol):
    """실제 VLM HTTP 클라이언트의 얇은 interface. v1 에서는 FakeVLMClient 만 구현."""
    def extract_visual(
        self, items: list[NormalizedContentItem]
    ) -> list[VLMVisualResult]: ...


class FakeVLMClient:
    """테스트용 결정론적 VLM. image_url 해시로 RGB/silhouette 을 고정 생성 (spec §7 stub)."""

    def __init__(self, cfg: VLMConfig) -> None:
        self._cfg = cfg

    def extract_visual(
        self, items: list[NormalizedContentItem]
    ) -> list[VLMVisualResult]:
        _reject_youtube(items)
        return [self._synthesize(item) for item in items]

    def _synthesize(self, item: NormalizedContentItem) -> VLMVisualResult:
        url_key = item.image_urls[0] if item.image_urls else item.source_post_id
        digest = hashlib.sha256(
            f"{self._cfg.seed}:{url_key}".encode()
        ).digest()
        r, g, b = digest[0], digest[1], digest[2]
        family = _pick_enum(digest, 3, list(ColorFamily))
        silhouette = _pick_enum(digest, 4, list(Silhouette))
        try:
            return VLMVisualResult(
                source_post_id=item.source_post_id,
                r=r,
                g=g,
                b=b,
                name=f"stub_{family.value}_{r:02x}{g:02x}{b:02x}",
                family=family,
                silhouette=silhouette,
            )
        except ValidationError as exc:
            logger.info("vlm_drop post_id=%s reason=%s", item.source_post_id, exc)
            return VLMVisualResult(source_post_id=item.source_post_id)


def _pick_enum(digest: bytes, offset: int, choices: list) -> object:
    return choices[digest[offset] % len(choices)]


def _reject_youtube(items: list[NormalizedContentItem]) -> None:
    """spec §7.2 type guard — YouTube 는 VLM 컬러/실루엣 추출 대상이 아니다."""
    offenders = [i.source_post_id for i in items if i.source != ContentSource.INSTAGRAM]
    if offenders:
        raise ValueError(
            f"VLM invoked on non-Instagram items: {offenders}. spec §7.2 violation."
        )


def _validate_rgb(value: int | None) -> int | None:
    """0~255 밖이면 None. coerce 금지."""
    if value is None:
        return None
    if _RGB_MIN <= value <= _RGB_MAX:
        return value
    return None


def extract_color_batch(
    items: list[NormalizedContentItem],
    client: VLMClient,
    cap: int | None = None,
) -> list[VLMVisualResult]:
    """Case 2 / Case 1 공용 entry. cap 이 있으면 리스트를 앞에서 잘라 호출.

    TODO(§7.2): real HTTP client 구현 시 retry 금지, 타임아웃 가드 필요.
    """
    _reject_youtube(items)
    bounded = items if cap is None else items[:cap]
    raw = client.extract_visual(bounded)
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
