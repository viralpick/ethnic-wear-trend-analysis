"""Phase B3c + β4 cluster_palette pinning — share-weighted 병합.

post_palette 와 같은 ΔE76 greedy merge + small-share drop + top N cap 패턴을
cluster 레벨에서 재사용. 차이는 세 가지:
1. 입력이 `list[tuple[list[PaletteCluster], float]]` (post_palette + per-post weight).
2. weight = cluster.share × post_weight (β4 — post_weight 는 cluster 안 item share).
3. β4 이전 옵션 A 동작 (post 간 동등 = post_weight=1.0) 도 그대로 표현 가능.

post_palette 는 area_ratio × within-share 를 곱해 canonical 물리 질량을 반영했지만,
cluster 는 "얼마나 많은 post 에서 이 색이 나왔나" 를 share-weighted 다수결로 본다 —
스코어 도메인과 분리된 pixel 증거 기반 누적.
"""
from __future__ import annotations

import pytest

pytest.importorskip("sklearn", reason="sklearn required (color_space deps)")

from contracts.common import ColorFamily, PaletteCluster  # noqa: E402
from aggregation.cluster_palette import (  # noqa: E402
    CLUSTER_PALETTE_MERGE_DELTA_E,
    MAX_CLUSTER_PALETTE_CLUSTERS,
    MIN_CLUSTER_PALETTE_SHARE,
    build_cluster_palette,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _pc(hex_: str, share: float, family: ColorFamily | None = ColorFamily.JEWEL) -> PaletteCluster:
    return PaletteCluster(hex=hex_, share=share, family=family)


def _eq_weight(posts: list[list[PaletteCluster]]) -> list[tuple[list[PaletteCluster], float]]:
    """β4 이전 동작 (post 간 동등) — post_weight 1.0 으로 wrap."""
    return [(p, 1.0) for p in posts]


# --------------------------------------------------------------------------- #
# constants pinned (accidental threshold drift 차단)
# --------------------------------------------------------------------------- #

def test_constants_pinned() -> None:
    # cluster 도 post 와 같은 ΔE76=10.0 / min_share=0.05 를 씀.
    # max 는 5 (로드맵 확정, post 의 3 과 다름).
    assert CLUSTER_PALETTE_MERGE_DELTA_E == 10.0
    assert MIN_CLUSTER_PALETTE_SHARE == 0.05
    assert MAX_CLUSTER_PALETTE_CLUSTERS == 5


# --------------------------------------------------------------------------- #
# empty / trivial paths
# --------------------------------------------------------------------------- #

def test_empty_posts_returns_empty() -> None:
    assert build_cluster_palette([]) == []


def test_all_post_palettes_empty_returns_empty() -> None:
    # 여러 post 가 있어도 post_palette 가 전부 비면 flatten 결과 0.
    assert build_cluster_palette(_eq_weight([[], [], []])) == []


# --------------------------------------------------------------------------- #
# single post pass-through
# --------------------------------------------------------------------------- #

def test_single_post_pass_through() -> None:
    # post 1 개의 post_palette (sum=1.0) 이 그대로 cluster palette 로 나온다.
    # merge 없음, drop 없음 (모두 >=0.05), cap 없음 (<=5).
    posts = [[
        _pc("#CC0000", 0.6, ColorFamily.JEWEL),
        _pc("#0000CC", 0.3, ColorFamily.JEWEL),
        _pc("#00AA00", 0.1, ColorFamily.EARTH),
    ]]
    result = build_cluster_palette(_eq_weight(posts))
    assert len(result) == 3
    assert sum(c.share for c in result) == pytest.approx(1.0, abs=1e-6)
    # weight desc 정렬
    shares = [c.share for c in result]
    assert shares == sorted(shares, reverse=True)


# --------------------------------------------------------------------------- #
# merge — same hex across posts (옵션 A: one-post-one-vote)
# --------------------------------------------------------------------------- #

def test_two_posts_same_hex_merge_share_sum() -> None:
    # 두 post 가 각각 0.4 share 로 같은 hex 를 갖고 있음 → merge 후 단일 cluster.
    # 나머지 색은 distinct, small-share drop 안 걸리도록 크게.
    posts = [
        [_pc("#AABBCC", 0.4, ColorFamily.PASTEL), _pc("#CC0000", 0.6, ColorFamily.JEWEL)],
        [_pc("#AABBCC", 0.4, ColorFamily.PASTEL), _pc("#00AA00", 0.6, ColorFamily.EARTH)],
    ]
    result = build_cluster_palette(_eq_weight(posts))
    # 3 개의 distinct 색 (merged pastel + red + green). sum=1.0 재정규화.
    assert len(result) == 3
    assert sum(c.share for c in result) == pytest.approx(1.0, abs=1e-6)
    # merged pastel weight = 0.4 + 0.4 = 0.8 / total 2.0 = 0.4
    pastel = [c for c in result if c.family == ColorFamily.PASTEL]
    assert len(pastel) == 1
    assert pastel[0].share == pytest.approx(0.4, abs=1e-6)


# --------------------------------------------------------------------------- #
# distinct colors preserved
# --------------------------------------------------------------------------- #

def test_far_colors_do_not_merge() -> None:
    # red / blue / green — ΔE76 훨씬 > 10 → 3 clusters 유지.
    posts = [
        [_pc("#CC0000", 1.0, ColorFamily.JEWEL)],
        [_pc("#0000CC", 1.0, ColorFamily.JEWEL)],
        [_pc("#00AA00", 1.0, ColorFamily.EARTH)],
    ]
    result = build_cluster_palette(_eq_weight(posts))
    assert len(result) == 3
    assert sum(c.share for c in result) == pytest.approx(1.0, abs=1e-6)
    # 세 post 전부 share 1.0 (one-post-one-vote) → 균등 1/3
    for c in result:
        assert c.share == pytest.approx(1.0 / 3.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# cap at MAX_CLUSTER_PALETTE_CLUSTERS (5)
# --------------------------------------------------------------------------- #

def test_caps_at_max_clusters() -> None:
    # 6 개의 멀리 떨어진 색 → merge 없음 → share 전부 0.1666... → drop<0.05 없음 → top 5 cap.
    posts = [
        [_pc("#CC0000", 1.0)],
        [_pc("#0000CC", 1.0)],
        [_pc("#00AA00", 1.0)],
        [_pc("#E6E600", 1.0)],
        [_pc("#AA00AA", 1.0)],
        [_pc("#00AAAA", 1.0)],
    ]
    result = build_cluster_palette(_eq_weight(posts))
    assert len(result) == MAX_CLUSTER_PALETTE_CLUSTERS
    assert sum(c.share for c in result) == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# small-share drop
# --------------------------------------------------------------------------- #

def test_small_share_dropped_before_cap() -> None:
    # 세 개의 distinct 색, weight 합 = 1.0 + 1.0 + 0.02 = 2.02.
    # 0.02 / 2.02 ≈ 0.0099 < 0.05 → drop. 나머지 둘 재정규화.
    posts = [
        [_pc("#CC0000", 1.0)],
        [_pc("#0000CC", 1.0)],
        # 같은 post 안에 0.02 만 있는 건 post_palette 원칙 (sum=1) 위반이지만
        # 테스트는 순수하게 weight 만 본다. 실제 파이프라인에서는 post_palette 가
        # drop 을 이미 걸어서 여기 오는 건 >=0.05.
        [_pc("#00AA00", 0.02)],
    ]
    result = build_cluster_palette(_eq_weight(posts))
    assert len(result) == 2
    assert sum(c.share for c in result) == pytest.approx(1.0, abs=1e-6)
    # 1.0 / 2.0 = 0.5 각각
    for c in result:
        assert c.share == pytest.approx(0.5, abs=1e-6)


# --------------------------------------------------------------------------- #
# family tiebreak — equal weight, enum order decides
# --------------------------------------------------------------------------- #

def test_family_tiebreak_by_enum_order_on_equal_weight() -> None:
    # 같은 hex, 같은 share → weight 동점. ColorFamily 선언 순서: PASTEL 이 JEWEL 보다 앞.
    posts = [
        [_pc("#AABBCC", 0.5, ColorFamily.JEWEL), _pc("#CC0000", 0.5)],
        [_pc("#AABBCC", 0.5, ColorFamily.PASTEL), _pc("#0000CC", 0.5)],
    ]
    result = build_cluster_palette(_eq_weight(posts))
    # pastel hex merge → 1 cluster; red/blue 별개 → 총 3.
    assert len(result) == 3
    # merge 된 pastel cluster (weight 0.5+0.5=1.0, 나머지 0.5 씩) 가 최상위.
    # weight 동점이어야 tiebreak 이 PASTEL 로 확정된다.
    top = max(result, key=lambda c: c.share)
    assert top.family == ColorFamily.PASTEL


# --------------------------------------------------------------------------- #
# sum=1.0 invariant — 다양한 분기
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "posts",
    [
        # 1 post, 단일 색
        [[_pc("#CC0000", 1.0)]],
        # 2 post, 다 merge
        [[_pc("#AABBCC", 1.0)], [_pc("#AABBCC", 1.0)]],
        # 3 post, 전부 distinct + drop 발생
        [
            [_pc("#CC0000", 1.0)],
            [_pc("#0000CC", 1.0)],
            [_pc("#00AA00", 0.02)],
        ],
        # 6 post, top 5 cap
        [
            [_pc("#CC0000", 1.0)],
            [_pc("#0000CC", 1.0)],
            [_pc("#00AA00", 1.0)],
            [_pc("#E6E600", 1.0)],
            [_pc("#AA00AA", 1.0)],
            [_pc("#00AAAA", 1.0)],
        ],
    ],
    ids=["single", "all_merge", "drop", "cap"],
)
def test_sum_equals_one_invariant(posts: list[list[PaletteCluster]]) -> None:
    result = build_cluster_palette(_eq_weight(posts))
    if not result:
        return
    assert abs(sum(c.share for c in result) - 1.0) < 1e-6


# --------------------------------------------------------------------------- #
# determinism
# --------------------------------------------------------------------------- #

def test_is_deterministic() -> None:
    posts = [
        [_pc("#AA0000", 0.7, ColorFamily.JEWEL), _pc("#0000AA", 0.3, ColorFamily.JEWEL)],
        [_pc("#00AA00", 1.0, ColorFamily.EARTH)],
    ]
    r1 = build_cluster_palette(_eq_weight(posts))
    r2 = build_cluster_palette(_eq_weight(posts))
    assert [(c.hex, c.family) for c in r1] == [(c.hex, c.family) for c in r2]
    assert [c.share for c in r1] == pytest.approx([c.share for c in r2])
