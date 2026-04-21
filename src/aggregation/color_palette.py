"""RGB → HEX 팔레트 (spec §4.1 ④, §5.4).

디자인 선택 (one-line docstring 수준으로 본 파일에 명시):
- v1 은 k-means 가 아니라 bucket_size 로 RGB 각 채널을 quantize 해서 bucket 빈도 상위 K 만 고른다.
- 이유: 결정론, 설명 가능, 의존성 zero. 팔레트 칩이 데모의 hero feature 라 stability 를 우선.
- 정확한 색상 분리는 4/24 이후 수요가 확정되면 교체 가능 (color science 가 아니라 explainable 칩).

이 모듈은 STUB 가 아니다 — 실제 파이프라인에서 사용되는 real 구현.
"""
from __future__ import annotations

from collections import Counter

from contracts.common import ColorFamily, ColorPaletteItem
from contracts.enriched import ColorInfo
from settings import PaletteConfig


def rgb_to_hex(r: int, g: int, b: int) -> str:
    """(184, 212, 195) → '#B8D4C3'. 입력은 0~255 가정 (검증은 호출자)."""
    return f"#{r:02X}{g:02X}{b:02X}"


def _bucket_channel(value: int, bucket_size: int) -> int:
    """채널 값을 bucket_size 로 quantize. 정수 나눗셈."""
    return (value // bucket_size) * bucket_size


def _bucket_midpoint(bucket_value: int, bucket_size: int) -> int:
    """버킷 시작값 → 중앙값 (0~255 clamp)."""
    mid = bucket_value + bucket_size // 2
    return min(max(mid, 0), 255)


def bucket_rgb(
    r: int, g: int, b: int, bucket_size: int
) -> tuple[int, int, int]:
    """RGB 를 quantize. 같은 버킷은 같은 튜플로 나온다."""
    return (
        _bucket_channel(r, bucket_size),
        _bucket_channel(g, bucket_size),
        _bucket_channel(b, bucket_size),
    )


def _color_family_of(rgbs: list[ColorInfo]) -> ColorFamily:
    """버킷의 대표 family — 포스트별 family 중 최빈값. 없으면 NEUTRAL."""
    families = [c.family for c in rgbs if c.family is not None]
    if not families:
        return ColorFamily.NEUTRAL
    counter: Counter[ColorFamily] = Counter(families)
    return counter.most_common(1)[0][0]


def build_palette(
    colors: list[ColorInfo], cfg: PaletteConfig
) -> list[ColorPaletteItem]:
    """color 목록 → top_k 팔레트. pct 는 해당 버킷 빈도 / 전체 색 수."""
    if not colors:
        return []

    bucket_to_posts: dict[tuple[int, int, int], list[ColorInfo]] = {}
    for color in colors:
        key = bucket_rgb(color.r, color.g, color.b, cfg.bucket_size)
        bucket_to_posts.setdefault(key, []).append(color)

    total = len(colors)
    sorted_buckets = sorted(
        bucket_to_posts.items(),
        key=lambda kv: (-len(kv[1]), kv[0]),
    )[: cfg.top_k]

    palette: list[ColorPaletteItem] = []
    for bucket, members in sorted_buckets:
        mid_r = _bucket_midpoint(bucket[0], cfg.bucket_size)
        mid_g = _bucket_midpoint(bucket[1], cfg.bucket_size)
        mid_b = _bucket_midpoint(bucket[2], cfg.bucket_size)
        family = _color_family_of(members)
        palette.append(
            ColorPaletteItem(
                r=mid_r,
                g=mid_g,
                b=mid_b,
                hex_display=rgb_to_hex(mid_r, mid_g, mid_b),
                name=f"bucket_{mid_r:02x}{mid_g:02x}{mid_b:02x}",
                family=family,
                pct=len(members) / total,
            )
        )
    return palette
