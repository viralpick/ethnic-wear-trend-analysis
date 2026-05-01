"""Phase 3 canonical_palette_aggregator pinning — per-object β-hybrid 통합 weighted KMeans.

회귀 가드:
- 결정성 (sample_weight 동일 입력 2번 → byte-identical hex / share / family).
- N=0 / N=1 / 모든 obj k=1 / top_n ≥ surviving cluster 수 edge.
- cut_off_share = 1.0 − Σ(top_n shares before renormalize at canonical level).
- HybridPaletteConfig default ↔ vision/hybrid_palette 모듈 상수 일치 (drift 가드,
  격리 규칙으로 settings.py 가 vision import 못 하므로 hardcode + pinning 으로 보장).
"""
from __future__ import annotations

import numpy as np

from contracts.common import ColorFamily, PaletteCluster
from settings import HybridPaletteConfig
from vision.canonical_palette_aggregator import (
    aggregate_canonical_palette,
    finalize_object_palette,
)
from vision.color_family_preset import MatcherEntry
from vision.hybrid_palette import (
    R2_MIN_SHARE,
    R3_DROP_DELTAE76,
    WeightedCluster,
)


def _wc(
    lab: tuple[float, float, float], weight: float, hex_: str = "#808080",
    is_anchor: bool = False,
) -> WeightedCluster:
    """LAB + weight + is_anchor 빌더 — Phase 3 는 hex/rgb 무시 (centroid 재계산).

    weight 는 D-1 부터 float (frame_area normalized × WEIGHT_SCALE). 의미상 큰 weight =
    더 큰 frame coverage × share. 정수도 float 로 자연 승격되므로 호출부 부담 없음.
    `is_anchor` (F-13): True 면 centroid anchor flag propagation 가드.
    """
    return WeightedCluster(
        hex=hex_, rgb=(128, 128, 128), lab=lab, weight=float(weight),
        is_anchor=is_anchor,
    )


def _entry(
    name: str, lab: tuple[float, float, float], family: ColorFamily,
) -> MatcherEntry:
    return MatcherEntry(name=name, lab=lab, family=family)


def _cfg(top_n: int = 3) -> HybridPaletteConfig:
    return HybridPaletteConfig(
        pick_match_deltae76=R3_DROP_DELTAE76, r2_min_share=R2_MIN_SHARE, top_n=top_n,
    )


def _wrap(per_object_clusters: list[list[WeightedCluster]]) -> list[tuple[list[WeightedCluster], float]]:
    """기존 list[list[WC]] 테스트 fixture → list[(list[WC], etc=0.0)] tuple shape.

    R2 재설계 (2026-04-26) 이전 테스트 대부분은 etc=0 가정이라 그대로 흡수. etc 가 영향
    있는 테스트는 별도 케이스로 직접 tuple 빌드.
    """
    return [(clusters, 0.0) for clusters in per_object_clusters]


# --------------------------------------------------------------------------- #
# HybridPaletteConfig drift 가드 — settings.py hardcode ↔ hybrid_palette 모듈 상수
# --------------------------------------------------------------------------- #

def test_hybrid_config_default_matches_module_const() -> None:
    """settings 격리 규칙으로 settings.py 가 vision 을 import 못 함 → hardcode.
    drift 가 일어나면 pinning 으로 즉시 발견."""
    from vision.hybrid_palette import (
        CHROMA_RATIO_MIN, CHROMA_VIVID, HUE_NEAR_DEG,
        R2_MERGE_DELTAE76, R3_DROP_DELTAE76, R2_MIN_SHARE,
    )
    cfg = HybridPaletteConfig()
    assert cfg.pick_match_deltae76 == R3_DROP_DELTAE76 == 28.0
    assert cfg.r2_min_share == R2_MIN_SHARE == 0.10
    assert cfg.chroma_vivid == CHROMA_VIVID == 15.0
    assert cfg.hue_near_deg == HUE_NEAR_DEG == 30.0
    assert cfg.r2_merge_deltae76 == R2_MERGE_DELTAE76 == 40.0
    assert cfg.chroma_ratio_min == CHROMA_RATIO_MIN == 0.5
    assert cfg.top_n == 3


def test_post_palette_config_default_matches_module_const() -> None:
    """vision/post_palette.py 모듈 상수 ↔ PostPaletteConfig default drift 가드."""
    from settings import PostPaletteConfig
    from vision.post_palette import (
        MAX_POST_PALETTE_CLUSTERS,
        MIN_POST_PALETTE_SHARE,
        POST_PALETTE_MERGE_DELTA_E,
    )
    cfg = PostPaletteConfig()
    assert cfg.max_clusters == MAX_POST_PALETTE_CLUSTERS == 3
    assert cfg.merge_deltae76_threshold == POST_PALETTE_MERGE_DELTA_E == 10.0
    assert cfg.min_cluster_share == MIN_POST_PALETTE_SHARE == 0.05


# --------------------------------------------------------------------------- #
# Edge cases — N=0 / 모든 inner 빈 / top_n ≥ k
# --------------------------------------------------------------------------- #

def test_aggregate_empty_outer_returns_empty() -> None:
    palette, cut_off = aggregate_canonical_palette([], [], _cfg())
    assert palette == []
    assert cut_off == 0.0


def test_aggregate_all_inner_empty_returns_empty() -> None:
    """모든 오브젝트가 Phase 1 빈 출력 → ([], 0.0). max k=0 단락."""
    palette, cut_off = aggregate_canonical_palette(_wrap([[], [], []]), [], _cfg())
    assert palette == []
    assert cut_off == 0.0


def test_aggregate_single_object_single_cluster_share_one() -> None:
    """1 obj × 1 cluster → max k=1 → 단일 centroid → share=1.0 / cut_off=0.0."""
    entries = [_entry("maroon", (25.0, 32.0, 16.0), ColorFamily.JEWEL)]
    palette, cut_off = aggregate_canonical_palette(
        _wrap([[_wc((25.0, 32.0, 16.0), weight=100)]]), entries, _cfg(),
    )
    assert len(palette) == 1
    assert palette[0].share == 1.0
    assert cut_off == 0.0


def test_aggregate_top_n_ge_surviving_cut_off_zero() -> None:
    """top_n=5 인데 surviving cluster 가 2개면 cut_off=0.0 (float 오차 흡수)."""
    entries = [
        _entry("dark", (25.0, 0.0, 0.0), ColorFamily.NEUTRAL),
        _entry("light", (75.0, 0.0, 0.0), ColorFamily.NEUTRAL),
    ]
    per_object = [
        [_wc((25.0, 0.0, 0.0), weight=200), _wc((75.0, 0.0, 0.0), weight=100)],
    ]
    palette, cut_off = aggregate_canonical_palette(_wrap(per_object), entries, _cfg(top_n=5))
    assert len(palette) == 2
    assert cut_off == 0.0
    assert abs(sum(c.share for c in palette) - 1.0) < 1e-9


# --------------------------------------------------------------------------- #
# 결정성 — sample_weight 동일 입력 2번 byte-identical
# --------------------------------------------------------------------------- #

def test_aggregate_is_deterministic() -> None:
    """random_state=42 / n_init=10 고정 — 같은 입력 2번 → byte-identical hex+share+family."""
    entries = [
        _entry("maroon", (25.0, 32.0, 16.0), ColorFamily.JEWEL),
        _entry("cream", (88.0, 2.0, 12.0), ColorFamily.PASTEL),
        _entry("dark_brown", (15.0, 8.0, 12.0), ColorFamily.EARTH),
    ]
    per_object = [
        [_wc((25.0, 32.0, 16.0), weight=300), _wc((15.0, 8.0, 12.0), weight=120)],
        [_wc((88.0, 2.0, 12.0), weight=180), _wc((25.0, 32.0, 16.0), weight=200)],
    ]
    p1, c1 = aggregate_canonical_palette(_wrap(per_object), entries, _cfg())
    p2, c2 = aggregate_canonical_palette(_wrap(per_object), entries, _cfg())
    assert c1 == c2
    assert len(p1) == len(p2)
    for a, b in zip(p1, p2):
        assert a.hex == b.hex
        assert a.share == b.share
        assert a.family == b.family


# --------------------------------------------------------------------------- #
# 통합 KMeans — k=max(per-obj) / weight desc 정렬 / centroid 평균
# --------------------------------------------------------------------------- #

def test_aggregate_k_equals_max_per_object() -> None:
    """obj A 가 1 cluster, obj B 가 3 cluster → max k=3 → 통합 KMeans n_clusters=3."""
    entries = [
        _entry("dark", (20.0, 0.0, 0.0), ColorFamily.NEUTRAL),
        _entry("mid", (50.0, 0.0, 0.0), ColorFamily.NEUTRAL),
        _entry("light", (80.0, 0.0, 0.0), ColorFamily.NEUTRAL),
    ]
    per_object = [
        [_wc((50.0, 0.0, 0.0), weight=100)],
        [
            _wc((20.0, 0.0, 0.0), weight=80),
            _wc((50.0, 0.0, 0.0), weight=60),
            _wc((80.0, 0.0, 0.0), weight=40),
        ],
    ]
    palette, _cut_off = aggregate_canonical_palette(_wrap(per_object), entries, _cfg(top_n=5))
    assert len(palette) == 3


def test_aggregate_share_desc_order() -> None:
    """centroid 별 weight 합산 후 share desc 정렬 — top-1 이 가장 큰 weight 그룹."""
    entries = [
        _entry("dark", (20.0, 0.0, 0.0), ColorFamily.NEUTRAL),
        _entry("light", (80.0, 0.0, 0.0), ColorFamily.NEUTRAL),
    ]
    # dark cluster 들의 총 weight = 500, light = 100 → dark 가 share desc 1번
    per_object = [
        [_wc((20.0, 0.0, 0.0), weight=300), _wc((80.0, 0.0, 0.0), weight=50)],
        [_wc((20.0, 0.0, 0.0), weight=200), _wc((80.0, 0.0, 0.0), weight=50)],
    ]
    palette, _cut_off = aggregate_canonical_palette(_wrap(per_object), entries, _cfg())
    assert len(palette) == 2
    assert palette[0].share > palette[1].share


# --------------------------------------------------------------------------- #
# cut_off_share 정의 — 1.0 − Σ(top_n shares before renormalize at canonical level)
# --------------------------------------------------------------------------- #

def test_cut_off_share_top_n_truncation() -> None:
    """top_n=2 / surviving k=4 → cut_off = bottom 2 share 합. 재정규화 후 top-2 sum=1.0."""
    entries = [
        _entry("c1", (20.0, 0.0, 0.0), ColorFamily.NEUTRAL),
        _entry("c2", (40.0, 0.0, 0.0), ColorFamily.NEUTRAL),
        _entry("c3", (60.0, 0.0, 0.0), ColorFamily.NEUTRAL),
        _entry("c4", (80.0, 0.0, 0.0), ColorFamily.NEUTRAL),
    ]
    # 4 cluster — 400 / 300 / 200 / 100 (total 1000). top-2 = 700 / 1000 = 0.7
    # cut_off = 1.0 - 0.7 = 0.3
    per_object = [
        [
            _wc((20.0, 0.0, 0.0), weight=400),
            _wc((40.0, 0.0, 0.0), weight=300),
            _wc((60.0, 0.0, 0.0), weight=200),
            _wc((80.0, 0.0, 0.0), weight=100),
        ],
    ]
    palette, cut_off = aggregate_canonical_palette(_wrap(per_object), entries, _cfg(top_n=2))
    assert len(palette) == 2
    assert abs(cut_off - 0.3) < 1e-6
    # 재정규화 invariant
    assert abs(sum(c.share for c in palette) - 1.0) < 1e-9


# --------------------------------------------------------------------------- #
# Family resolve — preset ΔE76 ≤ 15 → MatcherEntry.family
# --------------------------------------------------------------------------- #

def test_aggregate_family_from_preset_match() -> None:
    """centroid LAB 이 preset entry 와 가까우면 그 family 채택.
    R1 anchor 임계 (25) 가 아니라 resolve_family 의 PRESET_MATCH_THRESHOLD (15) 사용."""
    entries = [_entry("maroon", (25.0, 32.0, 16.0), ColorFamily.JEWEL)]
    per_object = [[_wc((25.0, 32.0, 16.0), weight=500)]]
    palette, _ = aggregate_canonical_palette(_wrap(per_object), entries, _cfg())
    assert palette[0].family == ColorFamily.JEWEL


# --------------------------------------------------------------------------- #
# 픽셀 가중치 보존 — Phase 1 weight 가 centroid 평균에 영향
# --------------------------------------------------------------------------- #

def test_aggregate_normalize_by_frame_area() -> None:
    """advisor A3 (2026-04-25): per-obj weight 가 frame_area 로 normalize 되어야 한다.

    obj A: frame 의 1% 차지 (weight magnitude 작음). obj B: frame 100% 차지 (weight 큼).
    top_n=1 → 통합 centroid 가 큰 frame coverage 인 obj B LAB 쪽으로 끌려야 한다.

    aggregator 는 weight 절대치만 본다 (frame_area 인자 없음). build_object_palette 가
    frame_area 로 정규화한 결과를 그대로 받기 때문. 이 가드는 weight magnitude 차이가
    centroid 결과에 충실히 반영됨을 보장 — 통합 KMeans 의 sample_weight 가 작동한다는
    invariant.
    """
    entries = [_entry("placeholder", (50.0, 0.0, 0.0), ColorFamily.NEUTRAL)]
    # obj A coverage 1% → weight 5 ((10,0,0) cluster 단일)
    # obj B coverage 100% → weight 5000 ((90,0,0) cluster 단일)
    per_object = [
        [_wc((10.0, 0.0, 0.0), weight=5)],
        [_wc((90.0, 0.0, 0.0), weight=5000)],
    ]
    palette, _cut_off = aggregate_canonical_palette(_wrap(per_object), entries, _cfg(top_n=1))
    assert len(palette) == 1
    # 가중 평균 ≈ (10*5 + 90*5000) / 5005 ≈ 89.92 → obj B LAB 쪽
    # palette 는 LAB 직접 노출 안 함. hex 의 R 채널이 obj B (밝은 회색) 쪽에 가까운지로 검증.
    r = int(palette[0].hex[1:3], 16)
    # 90 LAB → ≈ 220 RGB (밝음). 50 (단순 평균) ≈ 119, 10 (작은 obj 우세) ≈ 0
    assert r > 200, f"frame normalize 실패: B 쪽으로 안 끌림, r={r}"


def test_aggregate_weight_pulls_centroid() -> None:
    """같은 LAB 영역에 weight 가 큰 cluster 가 있으면 centroid 가 그쪽으로 끌림.
    weighted KMeans 의 핵심 invariant — sample_weight 작동 확인."""
    entries = [_entry("placeholder", (50.0, 0.0, 0.0), ColorFamily.NEUTRAL)]
    # 단일 cluster max k=1 → 가중 평균. (10,0,0) weight=900, (90,0,0) weight=100
    # → 평균 ≈ (10*900 + 90*100) / 1000 = 18.0
    per_object = [
        [_wc((10.0, 0.0, 0.0), weight=900), _wc((90.0, 0.0, 0.0), weight=100)],
    ]
    palette, _cut_off = aggregate_canonical_palette(_wrap(per_object), entries, _cfg(top_n=1))
    assert len(palette) == 1
    # 가중 평균이 18 근처여야 함 — 단순 평균이면 50 이 되어 fail
    # palette 는 contracts type 이라 LAB 직접 노출 X. share 만 검증 가능 → 별도로 centroid
    # 영향은 hex 비교로 — 가중 평균 색은 어둡고 (저L), 단순 평균 (50,0,0) 은 회색.
    # 가중 결과 hex 가 회색 (#808080~#878787) 보다 어두워야 함.
    r, g, b = (
        int(palette[0].hex[1:3], 16),
        int(palette[0].hex[3:5], 16),
        int(palette[0].hex[5:7], 16),
    )
    avg_brightness = (r + g + b) / 3
    assert avg_brightness < 100  # 어두운 쪽 — weight 가 작용했음을 보장


# --------------------------------------------------------------------------- #
# etc bucket — R2 머지 못한 잔여가 cut_off_share 에 흡수된다 (2026-04-26)
# --------------------------------------------------------------------------- #

def test_aggregate_etc_weight_inflates_cut_off_share() -> None:
    """per-object etc weight 가 cut_off_share 에 합산된다. share 분모는 weights+etc."""
    entries = [_entry("maroon", (25.0, 32.0, 16.0), ColorFamily.JEWEL)]
    # 1 cluster weight 800 + etc weight 200 → top-1 share = 800/1000 = 0.8, cut_off = 0.2
    per_object_results = [(
        [_wc((25.0, 32.0, 16.0), weight=800)],
        200.0,
    )]
    palette, cut_off = aggregate_canonical_palette(per_object_results, entries, _cfg())
    assert len(palette) == 1
    assert palette[0].share == 1.0  # 재정규화 후 sum=1
    assert abs(cut_off - 0.2) < 1e-6


def test_aggregate_etc_only_returns_empty_palette_full_cut_off() -> None:
    """모든 obj clusters 빈데 etc>0 → ([], 1.0) — 전부 잔여."""
    per_object_results = [([], 100.0), ([], 50.0)]
    palette, cut_off = aggregate_canonical_palette(per_object_results, [], _cfg())
    assert palette == []
    assert cut_off == 1.0


def test_aggregate_etc_combines_with_top_n_truncation() -> None:
    """top_n cap 잘림 + etc 둘 다 cut_off 에 합산된다."""
    entries = [_entry("placeholder", (50.0, 0.0, 0.0), ColorFamily.NEUTRAL)]
    # 4 cluster (weight 400/300/200/100=1000) + etc 500 → grand_total 1500
    # top-2 = 700/1500 = 0.4666... cut_off = 1 - 0.4666 = 0.5333
    per_object_results = [(
        [
            _wc((20.0, 0.0, 0.0), weight=400),
            _wc((40.0, 0.0, 0.0), weight=300),
            _wc((60.0, 0.0, 0.0), weight=200),
            _wc((80.0, 0.0, 0.0), weight=100),
        ],
        500.0,
    )]
    palette, cut_off = aggregate_canonical_palette(
        per_object_results, entries, _cfg(top_n=2),
    )
    assert len(palette) == 2
    assert abs(cut_off - (1.0 - 700.0 / 1500.0)) < 1e-6
    assert abs(sum(c.share for c in palette) - 1.0) < 1e-9


# --------------------------------------------------------------------------- #
# F-13 anchor priority — anchor centroid 가 top_n 에 우선 보존된다
# --------------------------------------------------------------------------- #

def test_aggregate_anchor_priority_keeps_low_share_anchor_in_top_n() -> None:
    """anchor centroid 가 share 작아도 top_n 에 우선 보존.

    big non-anchor 2개 (LAB (50,0,0) / (60,0,0), weight 800/600) + 작은 anchor (LAB
    (25,32,16) maroon, weight 100). top_n=3 → 모두 들어감. top_n=2 → 큰 non-anchor 2개가
    들어가는 게 일반 share desc 룰. F-13 룰이면 anchor 가 우선이라 maroon 이 top_n=2 에
    들어가야 한다.
    """
    entries = [
        _entry("mid", (50.0, 0.0, 0.0), ColorFamily.NEUTRAL),
        _entry("light", (60.0, 0.0, 0.0), ColorFamily.NEUTRAL),
        _entry("maroon", (25.0, 32.0, 16.0), ColorFamily.JEWEL),
    ]
    per_object = [[
        _wc((50.0, 0.0, 0.0), weight=800),
        _wc((60.0, 0.0, 0.0), weight=600),
        _wc((25.0, 32.0, 16.0), weight=100, is_anchor=True),
    ]]
    palette, _cut = aggregate_canonical_palette(_wrap(per_object), entries, _cfg(top_n=2))
    assert len(palette) == 2
    families = [p.family for p in palette]
    assert ColorFamily.JEWEL in families  # anchor (maroon) 보존됨


def test_aggregate_anchor_priority_no_anchor_falls_back_to_share_desc() -> None:
    """anchor 없으면 일반 share desc top_n cap (기존 동작 유지)."""
    entries = [
        _entry("a", (20.0, 0.0, 0.0), ColorFamily.NEUTRAL),
        _entry("b", (50.0, 0.0, 0.0), ColorFamily.NEUTRAL),
        _entry("c", (80.0, 0.0, 0.0), ColorFamily.NEUTRAL),
    ]
    per_object = [[
        _wc((20.0, 0.0, 0.0), weight=400),
        _wc((50.0, 0.0, 0.0), weight=300),
        _wc((80.0, 0.0, 0.0), weight=100),
    ]]
    palette, _cut = aggregate_canonical_palette(_wrap(per_object), entries, _cfg(top_n=2))
    assert len(palette) == 2
    # share desc — 400 / 300 만 살고 100 은 잘림
    assert palette[0].share > palette[1].share


def test_aggregate_anchor_priority_too_many_anchors_truncates_by_share() -> None:
    """anchor 가 top_n 보다 많으면 share desc 로 top_n 만 보존."""
    entries = [_entry("placeholder", (50.0, 0.0, 0.0), ColorFamily.NEUTRAL)]
    per_object = [[
        _wc((20.0, 0.0, 0.0), weight=100, is_anchor=True),
        _wc((40.0, 0.0, 0.0), weight=200, is_anchor=True),
        _wc((60.0, 0.0, 0.0), weight=300, is_anchor=True),
        _wc((80.0, 0.0, 0.0), weight=400, is_anchor=True),
    ]]
    palette, _cut = aggregate_canonical_palette(_wrap(per_object), entries, _cfg(top_n=2))
    assert len(palette) == 2  # 4 anchor 중 share 큰 2개만


def test_aggregate_anchor_centroid_propagation_via_kmeans_label() -> None:
    """입력 WC 가 anchor 면 그 LAB 영역의 centroid 도 anchor 로 마킹된다.

    anchor cluster 와 non-anchor cluster 가 KMeans 로 같은 centroid 에 묶여도, 한 입력만
    anchor 면 centroid anchor (permissive). 결과: top_n cap 시 그 centroid 우선 보존.
    """
    entries = [_entry("placeholder", (50.0, 0.0, 0.0), ColorFamily.NEUTRAL)]
    # 같은 LAB (50,0,0) 영역에 anchor 와 non-anchor 가 같이 있음 → 하나의 centroid 로 묶임.
    # 별도 LAB (10,0,0) 영역에 큰 non-anchor 만 있음.
    per_object = [[
        _wc((10.0, 0.0, 0.0), weight=900, hex_="#100000"),       # 큰 non-anchor
        _wc((50.0, 0.0, 0.0), weight=50, is_anchor=True),         # 작은 anchor
        _wc((50.0, 0.0, 0.0), weight=50, hex_="#500000"),         # non-anchor 같이
    ]]
    # k = 3 인데 LAB 두 그룹 — anchor centroid (LAB 50 영역) + 큰 non-anchor (LAB 10 영역).
    # top_n=1 → anchor 가 share 작더라도 우선 보존 → 결과 hex 가 LAB 50 영역 (회색계)
    palette, _cut = aggregate_canonical_palette(_wrap(per_object), entries, _cfg(top_n=1))
    assert len(palette) == 1
    # LAB (50,0,0) 의 RGB 는 약 (119,119,119) — gray. LAB (10,0,0) 은 거의 (28,28,28) dark.
    r = int(palette[0].hex[1:3], 16)
    assert r > 80, f"anchor centroid 우선 보존 실패: r={r} (LAB 10 의 dark centroid 가 살아남음)"


# --------------------------------------------------------------------------- #
# finalize_object_palette — 멤버 단위 OutfitMember.palette 채움 (spec §6.5)
# --------------------------------------------------------------------------- #

def test_finalize_object_palette_empty_returns_zero_cut_off() -> None:
    """weighted=[], etc=0 → ([], 0.0). 멤버 pool 자체가 없는 케이스."""
    palette, cut_off = finalize_object_palette([], 0.0, [], _cfg())
    assert palette == []
    assert cut_off == 0.0


def test_finalize_object_palette_etc_only_returns_full_cut_off() -> None:
    """weighted=[], etc>0 → ([], 1.0). R2 머지 못한 잔여만 있는 케이스."""
    palette, cut_off = finalize_object_palette([], 100.0, [], _cfg())
    assert palette == []
    assert cut_off == 1.0


def test_finalize_object_palette_single_cluster_share_one() -> None:
    """1 cluster, etc=0 → top-1 share=1.0, cut_off=0.0."""
    entries = [_entry("maroon", (25.0, 32.0, 16.0), ColorFamily.JEWEL)]
    weighted = [_wc((25.0, 32.0, 16.0), weight=500, hex_="#5C201D")]
    palette, cut_off = finalize_object_palette(weighted, 0.0, entries, _cfg())
    assert len(palette) == 1
    assert palette[0].share == 1.0
    assert palette[0].hex == "#5C201D"
    assert palette[0].family == ColorFamily.JEWEL
    assert cut_off == 0.0


def test_finalize_object_palette_top_n_truncation_with_etc() -> None:
    """4 cluster (400/300/200/100=1000) + etc=500 → grand=1500. top_n=2 살림.
    cut_off = 1.0 − (400+300)/1500 = 0.5333..."""
    entries = [
        _entry("c1", (20.0, 0.0, 0.0), ColorFamily.NEUTRAL),
        _entry("c2", (40.0, 0.0, 0.0), ColorFamily.NEUTRAL),
        _entry("c3", (60.0, 0.0, 0.0), ColorFamily.NEUTRAL),
        _entry("c4", (80.0, 0.0, 0.0), ColorFamily.NEUTRAL),
    ]
    weighted = [
        _wc((20.0, 0.0, 0.0), weight=400, hex_="#202020"),
        _wc((40.0, 0.0, 0.0), weight=300, hex_="#404040"),
        _wc((60.0, 0.0, 0.0), weight=200, hex_="#606060"),
        _wc((80.0, 0.0, 0.0), weight=100, hex_="#808080"),
    ]
    palette, cut_off = finalize_object_palette(weighted, 500.0, entries, _cfg(top_n=2))
    assert len(palette) == 2
    assert abs(cut_off - (1.0 - 700.0 / 1500.0)) < 1e-6
    assert abs(sum(c.share for c in palette) - 1.0) < 1e-9
    # share desc — 가장 큰 weight 가 top-1
    assert palette[0].share > palette[1].share


def test_finalize_object_palette_anchor_priority() -> None:
    """anchor centroid 가 share 작아도 top_n 우선 보존 (canonical aggregator 와 동일 룰)."""
    entries = [
        _entry("mid", (50.0, 0.0, 0.0), ColorFamily.NEUTRAL),
        _entry("light", (60.0, 0.0, 0.0), ColorFamily.NEUTRAL),
        _entry("maroon", (25.0, 32.0, 16.0), ColorFamily.JEWEL),
    ]
    weighted = [
        _wc((50.0, 0.0, 0.0), weight=800, hex_="#808080"),
        _wc((60.0, 0.0, 0.0), weight=600, hex_="#9A9A9A"),
        _wc((25.0, 32.0, 16.0), weight=100, hex_="#5C201D", is_anchor=True),
    ]
    palette, _cut_off = finalize_object_palette(weighted, 0.0, entries, _cfg(top_n=2))
    assert len(palette) == 2
    families = {p.family for p in palette}
    assert ColorFamily.JEWEL in families  # anchor maroon 보존


def test_finalize_object_palette_share_renormalize_invariant() -> None:
    """capped share 합 = 1.0 (contracts.PaletteCluster invariant)."""
    entries = [_entry("placeholder", (50.0, 0.0, 0.0), ColorFamily.NEUTRAL)]
    weighted = [
        _wc((20.0, 0.0, 0.0), weight=300),
        _wc((50.0, 0.0, 0.0), weight=200),
        _wc((80.0, 0.0, 0.0), weight=100),
    ]
    palette, _cut = finalize_object_palette(weighted, 0.0, entries, _cfg(top_n=3))
    assert len(palette) == 3
    assert abs(sum(c.share for c in palette) - 1.0) < 1e-9


def test_finalize_object_palette_grand_total_zero_returns_empty() -> None:
    """weight 합 + etc 둘 다 0 → ([], 0.0). 방어."""
    weighted = [_wc((50.0, 0.0, 0.0), weight=0.0)]
    palette, cut_off = finalize_object_palette(weighted, 0.0, [], _cfg())
    assert palette == []
    assert cut_off == 0.0


def test_hybrid_config_new_thresholds_default_pinned() -> None:
    """HybridPaletteConfig 새 R2 임계값 default 확인 — settings/vision drift 가드."""
    from vision.hybrid_palette import (
        CHROMA_VIVID,
        HUE_NEAR_DEG,
        R2_MERGE_DELTAE76,
    )
    cfg = HybridPaletteConfig()
    assert cfg.chroma_vivid == CHROMA_VIVID == 15.0
    assert cfg.hue_near_deg == HUE_NEAR_DEG == 30.0
    assert cfg.r2_merge_deltae76 == R2_MERGE_DELTAE76 == 40.0
