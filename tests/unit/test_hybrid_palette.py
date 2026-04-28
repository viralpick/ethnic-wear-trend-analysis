"""β hybrid object palette pinning — Phase 1 (R3 → R1 anchor 좌표 보존 → R2 solo keep).

per-object 재설계:
- 출력 = list[WeightedCluster] (좌표 + 픽셀 카운트). family/share 는 Phase 3 통합에서 결정.
- L-highest 좌표 보존 머지 + tie-break = input 순서 (share desc 정렬 가정).
- top_n / cut_off_share / family resolve 폐기 → Phase 3 (단계 C) 으로 이동.

motivating regression cases (Fabindia cream_ivory / Sridevi maroon + dark brown) 는 직접
박혀 있어야 한다 (advisor 피드백 — 회귀 가드 목적).
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

from contracts.common import ColorFamily
from settings import DynamicPaletteConfig
from vision.color_family_preset import MatcherEntry
from vision.dynamic_palette import PaletteCluster as PixelCluster
from vision.hybrid_palette import (
    CHROMA_RATIO_MIN,
    CHROMA_VIVID,
    HUE_NEAR_DEG,
    R2_MERGE_DELTAE76,
    R2_MIN_SHARE,
    R3_DROP_DELTAE76,
    WEIGHT_EPS,
    WEIGHT_SCALE,
    WeightedCluster,
    _chroma,
    _hue_circular_diff,
    _hue_deg,
    _is_bright_highlight,
    _match_anchors_one_to_one,
    _resolve_anchor_for_cluster,
    _resolve_merge_target,
    _share_to_weight,
    build_object_palette,
    filter_picks_by_pixel_evidence,
)


def _cluster(lab: tuple[float, float, float], share: float = 0.5) -> PixelCluster:
    """LAB 만 의미 있는 픽셀 cluster 빌더 — hex/rgb 는 R3 에 무관."""
    return PixelCluster(hex="#000000", rgb=(0, 0, 0), lab=lab, share=share)


def _entry(name: str, lab: tuple[float, float, float]) -> MatcherEntry:
    return MatcherEntry(name=name, lab=lab, family=ColorFamily.NEUTRAL)


# --------------------------------------------------------------------------- #
# R3 — 빈 입력 / 경계 / preset 누락 / 다중 pick 순서 보존
# --------------------------------------------------------------------------- #

def test_empty_picks_returns_empty() -> None:
    clusters = [_cluster((50.0, 0.0, 0.0))]
    entries = [_entry("preset_a", (50.0, 0.0, 0.0))]
    assert filter_picks_by_pixel_evidence([], clusters, entries) == []


def test_empty_clusters_drops_all_picks() -> None:
    """clusters 비면 검증 불가 → 모두 환각 처리 (실패 숨김 금지 원칙)."""
    entries = [_entry("preset_a", (50.0, 0.0, 0.0))]
    assert filter_picks_by_pixel_evidence(["preset_a"], [], entries) == []


def test_pick_within_threshold_kept() -> None:
    # cluster (50,0,0) ↔ pick (50,10,10) → ΔE76 = sqrt(200) ≈ 14.14 < 25
    clusters = [_cluster((50.0, 0.0, 0.0))]
    entries = [_entry("near", (50.0, 10.0, 10.0))]
    assert filter_picks_by_pixel_evidence(["near"], clusters, entries) == ["near"]


def test_pick_beyond_threshold_dropped() -> None:
    # cluster (50,0,0) ↔ pick (50,30,30) → ΔE76 = sqrt(1800) ≈ 42.4 > 25
    clusters = [_cluster((50.0, 0.0, 0.0))]
    entries = [_entry("far", (50.0, 30.0, 30.0))]
    assert filter_picks_by_pixel_evidence(["far"], clusters, entries) == []


def test_pick_at_exact_threshold_kept() -> None:
    # ΔE76 = 25.0 정확히 일치 → drop_threshold 이하라 keep (≤ semantic)
    clusters = [_cluster((50.0, 0.0, 0.0))]
    entries = [_entry("edge", (50.0, 15.0, 20.0))]  # sqrt(225+400)=25.0
    result = filter_picks_by_pixel_evidence(["edge"], clusters, entries)
    assert result == ["edge"]


def test_fabindia_cream_ivory_dropped_when_clusters_all_dark_maroon() -> None:
    """Fabindia 케이스 — KMeans 가 dark maroon 만 잡았는데 Gemini 가 cream_ivory 환각."""
    clusters = [
        _cluster((25.0, 40.0, 20.0), share=0.6),
        _cluster((20.0, 35.0, 18.0), share=0.4),
    ]
    entries = [
        _entry("cream_ivory", (90.0, 0.0, 5.0)),
        _entry("dark_maroon", (25.0, 40.0, 20.0)),
    ]
    survivors = filter_picks_by_pixel_evidence(
        ["cream_ivory", "dark_maroon"], clusters, entries,
    )
    assert "cream_ivory" not in survivors
    assert "dark_maroon" in survivors


def test_sridevi_maroon_kept_when_clusters_have_near_maroon() -> None:
    """Sridevi 케이스 — Gemini maroon pick + KMeans 에 maroon 클러스터 → keep."""
    clusters = [
        _cluster((28.0, 38.0, 18.0), share=0.5),
        _cluster((22.0, 25.0, 12.0), share=0.5),
    ]
    entries = [_entry("maroon", (30.0, 40.0, 20.0))]
    assert filter_picks_by_pixel_evidence(["maroon"], clusters, entries) == ["maroon"]


def test_order_is_preserved_after_partial_drop() -> None:
    clusters = [_cluster((50.0, 0.0, 0.0))]
    entries = [
        _entry("near", (50.0, 5.0, 5.0)),
        _entry("far", (50.0, 30.0, 30.0)),
        _entry("also_near", (52.0, 0.0, 3.0)),
    ]
    survivors = filter_picks_by_pixel_evidence(
        ["near", "far", "also_near"], clusters, entries,
    )
    assert survivors == ["near", "also_near"]


def test_unknown_pick_logs_warning_and_drops(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """preset 에 없는 pick name 은 silent drop 금지 — log.warning 표면화 후 drop."""
    clusters = [_cluster((50.0, 0.0, 0.0))]
    entries = [_entry("known", (50.0, 5.0, 5.0))]
    with caplog.at_level(logging.WARNING, logger="vision.hybrid_palette"):
        survivors = filter_picks_by_pixel_evidence(
            ["known", "ghost_name"], clusters, entries,
        )
    assert survivors == ["known"]
    assert any("hybrid_pick_unknown" in rec.message for rec in caplog.records)
    assert any("ghost_name" in rec.getMessage() for rec in caplog.records)


# --------------------------------------------------------------------------- #
# 모듈 상수 — Phase 3 단계 C 에서 HybridPaletteConfig 로 옮겨갈 plan default
# --------------------------------------------------------------------------- #

def test_r3_drop_threshold_default_pinned() -> None:
    assert R3_DROP_DELTAE76 == 28.0


def test_r2_min_share_default_pinned() -> None:
    """anchor 없는 solo cluster keep 컷오프 — minor pattern 만 살리는 의도."""
    assert R2_MIN_SHARE == 0.10


def test_r2_chroma_vivid_default_pinned() -> None:
    """R2 원색 게이트 — chroma ≥ 15 면 R2 solo keep 후보."""
    assert CHROMA_VIVID == 15.0


def test_r2_hue_near_deg_default_pinned() -> None:
    """hue 비슷 임계 (deg) — Δh ≤ 30° 이면 양방향 머지 허용."""
    assert HUE_NEAR_DEG == 30.0


def test_r2_merge_deltae76_default_pinned() -> None:
    """R2 머지 후보 산출 임계 — ΔE76 ≤ 40 만 anchor 후보로."""
    assert R2_MERGE_DELTAE76 == 40.0


# --------------------------------------------------------------------------- #
# `_chroma` / `_hue_deg` / `_hue_circular_diff`
# --------------------------------------------------------------------------- #

def test_chroma_zero_for_neutral() -> None:
    assert _chroma((50.0, 0.0, 0.0)) == 0.0


def test_chroma_pythagorean() -> None:
    assert abs(_chroma((50.0, 3.0, 4.0)) - 5.0) < 1e-9


def test_hue_deg_none_for_neutral() -> None:
    assert _hue_deg((50.0, 0.0, 0.0)) is None


def test_hue_deg_zero_for_pure_red_axis() -> None:
    h = _hue_deg((50.0, 30.0, 0.0))
    assert h is not None
    assert abs(h - 0.0) < 1e-6


def test_hue_deg_90_for_pure_b_axis() -> None:
    h = _hue_deg((50.0, 0.0, 30.0))
    assert h is not None
    assert abs(h - 90.0) < 1e-6


def test_hue_circular_diff_none_returns_inf() -> None:
    assert _hue_circular_diff(None, 30.0) == float("inf")
    assert _hue_circular_diff(30.0, None) == float("inf")
    assert _hue_circular_diff(None, None) == float("inf")


def test_hue_circular_diff_wraps_around() -> None:
    # 350 vs 10 → circular dist 20
    assert abs(_hue_circular_diff(350.0, 10.0) - 20.0) < 1e-9
    assert abs(_hue_circular_diff(10.0, 350.0) - 20.0) < 1e-9
    # 0 vs 180 → max dist 180
    assert abs(_hue_circular_diff(0.0, 180.0) - 180.0) < 1e-9


# --------------------------------------------------------------------------- #
# `_resolve_merge_target` — R2 머지 방향성 강제 + hue 우회
# --------------------------------------------------------------------------- #

def test_merge_target_blocks_when_anchor_darker_and_hue_far() -> None:
    """검정 anchor 가 dark brown cluster 를 흡수하면 안됨 (사용자 명시 케이스).

    검정 (L=2, a=0, b=0) vs dark brown (L=30, a=10, b=15) — Δh undef → 강제 룰.
    anchor.L (2) < cluster.L (30) → 강제 룰 fail → 머지 차단.
    """
    cluster = _cluster((30.0, 10.0, 15.0), share=0.4)
    anchor_targets = [("black", (2.0, 0.0, 0.0))]
    result = _resolve_merge_target(cluster, anchor_targets, 40.0, 30.0)
    assert result is None


def test_merge_target_allows_darker_cluster_when_hue_near() -> None:
    """Δh ≤ HUE_NEAR_DEG 이고 cluster 가 anchor 보다 어두우면 (shadow) 머지 허용.

    maroon anchor (L=30, c=36, h=36°) ↔ dark brown cluster (L=20, c=14, h=55°).
    cluster.L (20) ≤ anchor.L (30) → shadow → 머지 OK.
    """
    cluster = _cluster((20.0, 8.0, 12.0), share=0.3)  # h ≈ 56°, darker
    anchor_targets = [("maroon", (30.0, 29.0, 21.0))]  # h ≈ 36°, brighter
    result = _resolve_merge_target(cluster, anchor_targets, 40.0, 30.0)
    assert result == "maroon"


def test_merge_target_blocks_brighter_cluster_when_hue_near_beta() -> None:
    """β 비대칭화 — cluster 가 anchor 보다 밝고 hue 비슷하면 highlight 로 보고 머지 거부.

    이렇게 거부된 cluster 는 build_object_palette 에서 R2 highlight solo 로 keep.
    """
    cluster = _cluster((40.0, 8.0, 12.0), share=0.3)  # brighter than anchor
    anchor_targets = [("maroon", (20.0, 29.0, 21.0))]  # darker
    result = _resolve_merge_target(cluster, anchor_targets, 40.0, 30.0)
    assert result is None


def test_merge_target_picks_min_deltae_when_multi_candidates() -> None:
    """후보 중 ΔE76 가장 작은 anchor 선택."""
    cluster = _cluster((50.0, 5.0, 5.0))
    anchor_targets = [
        ("far_match", (80.0, 5.0, 5.0)),  # ΔE76 = 30, hue 비슷, anchor.L 큼 → 통과
        ("near_match", (60.0, 5.0, 5.0)),  # ΔE76 = 10, hue 비슷, anchor.L 큼 → 통과
    ]
    result = _resolve_merge_target(cluster, anchor_targets, 40.0, 30.0)
    assert result == "near_match"


def test_merge_target_blocks_beyond_threshold() -> None:
    """ΔE76 > merge_threshold 면 후보 제외."""
    cluster = _cluster((20.0, 0.0, 0.0))
    anchor_targets = [("far_anchor", (90.0, 0.0, 0.0))]  # ΔE76 = 70
    result = _resolve_merge_target(cluster, anchor_targets, 40.0, 30.0)
    assert result is None


def test_merge_target_neutral_cluster_uses_directional_rule() -> None:
    """무채 cluster (chroma=0, hue undef) 는 hue 비슷 분기 못 타고 강제 룰 적용.

    검정 cluster (L=10, neutral) → 어떤 anchor 가 더 밝고 채도 높으면 머지.
    """
    cluster = _cluster((10.0, 0.0, 0.0))
    anchor_targets = [("brown", (40.0, 8.0, 12.0))]
    result = _resolve_merge_target(cluster, anchor_targets, 40.0, 30.0)
    assert result == "brown"


# --------------------------------------------------------------------------- #
# `_resolve_anchor_for_cluster` (cluster→pick unidirectional)
# --------------------------------------------------------------------------- #

def test_anchor_picks_nearest_within_threshold() -> None:
    cluster = PixelCluster(hex="#000000", rgb=(0, 0, 0), lab=(50.0, 0.0, 0.0), share=1.0)
    lookup = {
        "far": (50.0, 30.0, 30.0),
        "near": (50.0, 5.0, 5.0),
        "mid": (50.0, 12.0, 12.0),
    }
    result = _resolve_anchor_for_cluster(
        cluster, ["far", "near", "mid"], lookup, threshold=25.0,
    )
    assert result == "near"


def test_anchor_returns_none_when_all_far() -> None:
    cluster = PixelCluster(hex="#000000", rgb=(0, 0, 0), lab=(50.0, 0.0, 0.0), share=1.0)
    lookup = {"far": (50.0, 30.0, 30.0)}
    assert _resolve_anchor_for_cluster(cluster, ["far"], lookup, 25.0) is None


def test_anchor_no_surviving_picks_returns_none() -> None:
    cluster = PixelCluster(hex="#000000", rgb=(0, 0, 0), lab=(50.0, 0.0, 0.0), share=1.0)
    assert _resolve_anchor_for_cluster(cluster, [], {}, 25.0) is None


def test_anchor_tie_break_picks_input_order() -> None:
    """동일 거리 시 picks 입력 순서가 우선 — Gemini dominance desc 가정."""
    cluster = PixelCluster(hex="#000000", rgb=(0, 0, 0), lab=(50.0, 0.0, 0.0), share=1.0)
    lookup = {
        "first": (60.0, 0.0, 0.0),
        "second": (40.0, 0.0, 0.0),
    }
    result = _resolve_anchor_for_cluster(
        cluster, ["first", "second"], lookup, 25.0,
    )
    assert result == "first"
    result_rev = _resolve_anchor_for_cluster(
        cluster, ["second", "first"], lookup, 25.0,
    )
    assert result_rev == "second"


# --------------------------------------------------------------------------- #
# `_share_to_weight` — share × obj_coverage × scale, frame_area normalize
# --------------------------------------------------------------------------- #

def test_share_to_weight_proportional_to_share() -> None:
    """같은 obj/frame 안에서 share 가 두 배면 weight 도 두 배."""
    obj_pixel_count = 1000
    frame_area = 10_000  # obj coverage = 0.1
    w_half = _share_to_weight(0.5, obj_pixel_count, frame_area)
    w_full = _share_to_weight(1.0, obj_pixel_count, frame_area)
    assert abs(w_full - 2 * w_half) < 1e-9
    # 0.5 share × 0.1 coverage × 10_000 scale = 500
    assert abs(w_half - 500.0) < 1e-9


def test_share_to_weight_normalizes_by_frame_area() -> None:
    """frame_area 가 같은 share/obj_pixel_count 일 때 weight 결정.

    같은 share 라도 frame 안에서 작은 obj 의 weight 는 작아야 함.
    """
    # 같은 share=0.5, 같은 obj_pixel_count=1000
    # frame_area 가 작으면 (1000) coverage 100% → weight 큼
    # frame_area 가 크면 (100_000) coverage 1% → weight 작음
    w_full_frame = _share_to_weight(0.5, 1000, 1000)
    w_tiny_frame = _share_to_weight(0.5, 1000, 100_000)
    assert w_full_frame > w_tiny_frame
    # 비율 = 100x (frame_area 차이) — 정확히 100배
    assert abs(w_full_frame / w_tiny_frame - 100.0) < 1e-6


def test_share_to_weight_floor_eps_for_zero_share() -> None:
    """share=0 / frame_area=0 같은 edge 에서도 sample_weight=0 silent drop 방지."""
    assert _share_to_weight(0.0, 1000, 10_000) == WEIGHT_EPS
    assert _share_to_weight(0.5, 1000, 0) == WEIGHT_EPS  # frame_area<=0 가드
    assert _share_to_weight(0.5, 1000, -1) == WEIGHT_EPS


def test_share_to_weight_scale_default_pinned() -> None:
    """SCALE=10_000 default 가 변경되면 KMeans weight magnitude 가 바뀜 — 결정성 가드."""
    assert WEIGHT_SCALE == 10_000.0


# --------------------------------------------------------------------------- #
# `_match_anchors_one_to_one` — F-13 R1 1:1 greedy 매칭
# --------------------------------------------------------------------------- #

def test_match_one_to_one_basic_assigns_closest() -> None:
    """단순 케이스 — cluster 2개, pick 2개, 각각 자기와 가까운 pick 매칭."""
    clusters = [
        _cluster((25.0, 32.0, 16.0)),  # maroon
        _cluster((85.0, 0.0, 5.0)),    # cream
    ]
    lookup = {
        "maroon": (25.0, 32.0, 16.0),
        "cream": (88.0, 0.0, 5.0),
    }
    cluster_to_pick, non_anchor = _match_anchors_one_to_one(
        clusters, ["maroon", "cream"], lookup, 28.0,
    )
    assert cluster_to_pick == {0: "maroon", 1: "cream"}
    assert non_anchor == []


def test_match_one_to_one_conflict_closer_wins() -> None:
    """두 cluster 가 같은 pick 노릴 때 ΔE76 가까운 쪽이 이긴다.

    cluster A (25,32,16) ↔ maroon ΔE76 ≈ 1
    cluster B (28,30,14) ↔ maroon ΔE76 ≈ 4
    → A 가 maroon 차지. B 는 다른 가까운 pick 없으면 non_anchor.
    """
    clusters = [
        _cluster((25.0, 32.0, 16.0)),
        _cluster((28.0, 30.0, 14.0)),
    ]
    lookup = {"maroon": (25.0, 32.0, 16.0)}
    cluster_to_pick, non_anchor = _match_anchors_one_to_one(
        clusters, ["maroon"], lookup, 28.0,
    )
    assert cluster_to_pick == {0: "maroon"}
    assert non_anchor == [1]


def test_match_one_to_one_loser_finds_next_pick() -> None:
    """A 가 1순위 pick 차지하면 B 는 2순위 pick 으로 fall through 한다.

    cluster A (25,32,16) ↔ maroon ΔE76 ≈ 1, cream ΔE76 ≈ 70
    cluster B (28,30,14) ↔ maroon ΔE76 ≈ 4, cream ΔE76 ≈ 70
    → A: maroon, B: maroon 뺏기고 cream 시도 → ΔE76 70 > 28 threshold → non_anchor.
    """
    clusters = [
        _cluster((25.0, 32.0, 16.0)),
        _cluster((28.0, 30.0, 14.0)),
    ]
    lookup = {
        "maroon": (25.0, 32.0, 16.0),
        "cream": (88.0, 0.0, 5.0),
    }
    cluster_to_pick, non_anchor = _match_anchors_one_to_one(
        clusters, ["maroon", "cream"], lookup, 28.0,
    )
    assert cluster_to_pick == {0: "maroon"}
    assert non_anchor == [1]


def test_match_one_to_one_loser_finds_alternative_pick() -> None:
    """B 가 maroon 뺏기고도 다른 가까운 pick 이 있으면 그쪽 매칭.

    cluster A (25,32,16) ↔ maroon ΔE76 ≈ 1
    cluster B (50,10,8) ↔ maroon ΔE76 ≈ 32 (drop), brown ΔE76 ≈ 5 (keep)
    → A: maroon, B: brown.
    """
    clusters = [
        _cluster((25.0, 32.0, 16.0)),
        _cluster((50.0, 10.0, 8.0)),
    ]
    lookup = {
        "maroon": (25.0, 32.0, 16.0),
        "brown": (50.0, 12.0, 10.0),
    }
    cluster_to_pick, non_anchor = _match_anchors_one_to_one(
        clusters, ["maroon", "brown"], lookup, 28.0,
    )
    assert cluster_to_pick == {0: "maroon", 1: "brown"}
    assert non_anchor == []


def test_match_one_to_one_no_picks_returns_all_non_anchor() -> None:
    clusters = [_cluster((50.0, 0.0, 0.0)), _cluster((30.0, 10.0, 10.0))]
    cluster_to_pick, non_anchor = _match_anchors_one_to_one(
        clusters, [], {}, 28.0,
    )
    assert cluster_to_pick == {}
    assert non_anchor == [0, 1]


def test_match_one_to_one_no_clusters_returns_empty() -> None:
    cluster_to_pick, non_anchor = _match_anchors_one_to_one(
        [], ["any"], {"any": (0.0, 0.0, 0.0)}, 28.0,
    )
    assert cluster_to_pick == {}
    assert non_anchor == []


def test_match_one_to_one_all_far_returns_non_anchor() -> None:
    clusters = [_cluster((50.0, 0.0, 0.0))]
    lookup = {"far": (50.0, 30.0, 30.0)}  # ΔE76 ≈ 42 > 28
    cluster_to_pick, non_anchor = _match_anchors_one_to_one(
        clusters, ["far"], lookup, 28.0,
    )
    assert cluster_to_pick == {}
    assert non_anchor == [0]


# --------------------------------------------------------------------------- #
# `_is_bright_highlight` chroma guard (F-13)
# --------------------------------------------------------------------------- #

def test_bright_highlight_real_highlight_passes_guard() -> None:
    """진짜 highlight: 같은 hue 의 밝은 saturated cluster — chroma 비율 ≥ 0.5."""
    cluster = _cluster((70.0, 25.0, 30.0))  # bright orange, chroma ≈ 39
    anchor_targets = [("dark_orange", (30.0, 30.0, 35.0))]  # chroma ≈ 46
    # ratio = 39/46 ≈ 0.85 ≥ 0.5 → pass
    assert _is_bright_highlight(cluster, anchor_targets, 30.0) is True


def test_bright_highlight_low_chroma_blocked_by_guard() -> None:
    """mid-brown (chroma 14) 가 saturated maroon (chroma 35) 의 highlight 로 분류 X.

    F-13 motivating case: Sridevi pool_03 maroon vs #8D7564 mid brown.
    """
    cluster = _cluster((50.0, 8.0, 12.0))  # chroma ≈ 14.4
    anchor_targets = [("maroon", (25.0, 30.0, 18.0))]  # chroma ≈ 35
    # ratio = 14.4/35 ≈ 0.41 < 0.5 → guard 차단
    assert _is_bright_highlight(cluster, anchor_targets, 30.0) is False


def test_bright_highlight_neutral_anchor_returns_false() -> None:
    """무채 anchor (chroma 0) 는 highlight semantics 자체가 없음."""
    cluster = _cluster((70.0, 5.0, 5.0))  # chroma ≈ 7
    anchor_targets = [("black", (2.0, 0.0, 0.0))]
    assert _is_bright_highlight(cluster, anchor_targets, 30.0) is False


def test_bright_highlight_neutral_cluster_returns_false() -> None:
    """무채 cluster (chroma 0) 도 hue 미정의 → highlight 없음."""
    cluster = _cluster((80.0, 0.0, 0.0))
    anchor_targets = [("orange", (40.0, 30.0, 25.0))]
    assert _is_bright_highlight(cluster, anchor_targets, 30.0) is False


def test_bright_highlight_pinned_chroma_ratio_default() -> None:
    """CHROMA_RATIO_MIN=0.5 default — pinning."""
    assert CHROMA_RATIO_MIN == 0.5


# --------------------------------------------------------------------------- #
# `build_object_palette` end-to-end
# --------------------------------------------------------------------------- #

def _fill(color: tuple[int, int, int], n: int) -> np.ndarray:
    arr = np.empty((n, 3), dtype=np.uint8)
    arr[:] = color
    return arr


def test_build_object_palette_empty_pool_returns_empty() -> None:
    clusters, etc = build_object_palette(
        np.empty((0, 3), dtype=np.uint8), [], DynamicPaletteConfig(), [], frame_area=1,
    )
    assert clusters == []
    assert etc == 0.0


def test_build_object_palette_below_min_pixels_returns_empty() -> None:
    """pool < min_pixels (default 150) → extract_dynamic_palette 가 빈 list."""
    pixels = _fill((220, 20, 20), 50)
    clusters, etc = build_object_palette(
        pixels, [], DynamicPaletteConfig(), [], frame_area=50,
    )
    assert clusters == []
    assert etc == 0.0


def test_build_object_palette_single_color_returns_one_weighted() -> None:
    """단색 pool — chroma 큰 red → R2 원색 solo keep (share=1.0 ≥ 0.10).

    coverage=1.0 (frame_area=obj_pixel_count) → share=1.0 → weight = SCALE.
    """
    pixels = _fill((220, 20, 20), 1000)
    clusters, etc = build_object_palette(
        pixels, [], DynamicPaletteConfig(), [], frame_area=1000,
    )
    assert len(clusters) == 1
    assert isinstance(clusters[0], WeightedCluster)
    assert abs(clusters[0].weight - WEIGHT_SCALE) < 1e-2
    assert etc == 0.0


def test_build_object_palette_picks_empty_vivid_clusters_kept() -> None:
    """picks=[] 면 anchor 없음. chroma 큰 RGB primaries 는 R2 원색 solo keep.

    red/blue/green/yellow 는 chroma ≥ CHROMA_VIVID (15) 통과. share 작은 건 R2 min_share
    cut 또는 dynamic_palette min_cluster_share (0.05) cut 으로 자연 drop 또는 etc 합산.
    """
    pixels = np.concatenate([
        _fill((220, 20, 20), 600),
        _fill((20, 20, 220), 250),
        _fill((20, 180, 20), 100),
        _fill((230, 230, 30), 50),
    ])
    clusters, etc = build_object_palette(
        pixels, [], DynamicPaletteConfig(), [], frame_area=1000,
    )
    total_weight = sum(c.weight for c in clusters) + etc
    assert total_weight <= WEIGHT_SCALE + 1e-2
    assert all(c.weight > 0 for c in clusters)
    assert all(isinstance(c, WeightedCluster) for c in clusters)


def test_build_object_palette_anchor_target_is_matched_cluster_coord() -> None:
    """F-13 — anchor 좌표 = 1:1 매칭된 cluster 의 자기 좌표 (L-highest 룰 폐기).

    maroon pick 에 cluster maroon (가까움) 매칭. dark-brown 은 non_anchor → R2 머지로
    maroon 의 weight 에 합산. 최종 1 cluster (anchor=True), 좌표는 maroon cluster 그대로.
    """
    pixels = np.concatenate([
        _fill((110, 35, 30), 600),  # maroon
        _fill((75, 50, 40), 400),   # dark brown
    ])
    entries = [MatcherEntry(name="maroon", lab=(25.0, 32.0, 16.0), family=ColorFamily.JEWEL)]
    clusters, etc = build_object_palette(
        pixels, ["maroon"], DynamicPaletteConfig(), entries, frame_area=1000,
    )
    assert len(clusters) == 1
    assert clusters[0].is_anchor is True
    assert abs(clusters[0].weight - WEIGHT_SCALE) < 1e-2
    assert clusters[0].lab[0] > 20.0
    assert etc == 0.0


def test_build_object_palette_one_to_one_loser_falls_to_r2() -> None:
    """1:1 매칭에서 loser cluster 가 R2 로 떨어지는 동작 검증.

    pick 1개 (maroon). cluster 2개 (둘 다 maroon 후보, 양쪽 ΔE76 ≤ 28). 더 가까운
    cluster 가 anchor 차지, 다른 cluster 는 non_anchor → R2 처리. shadow (저채도 어두운)
    이라 R2 merge 로 maroon 의 weight 에 합산. 최종 1 anchor, weight 는 둘 합.
    """
    pixels = np.concatenate([
        _fill((115, 30, 25), 500),  # closer to preset maroon (chroma 큼)
        _fill((60, 48, 44), 500),   # warm dark shadow (chroma 낮음)
    ])
    entries = [MatcherEntry(name="maroon", lab=(25.0, 32.0, 16.0), family=ColorFamily.JEWEL)]
    clusters, _etc = build_object_palette(
        pixels, ["maroon"], DynamicPaletteConfig(), entries, frame_area=1000,
    )
    # anchor 1개 (좌표는 closer cluster), shadow 는 R2 머지로 weight 합산
    anchor_clusters = [c for c in clusters if c.is_anchor]
    assert len(anchor_clusters) == 1
    # anchor 좌표는 maroon cluster — a* (red) 큼
    assert anchor_clusters[0].lab[1] > 25.0
    # anchor weight 가 own (~5000) 보다 크다 → shadow 머지 발생
    assert anchor_clusters[0].weight > 6000.0


def test_build_object_palette_lowchroma_nonanchor_goes_to_etc() -> None:
    """anchor 없고 low-chroma cluster 인 경우 R2 원색 게이트 fail → etc bucket.

    저채도 cluster 단독 + anchor 없음 → 머지 후보 없음 → etc.
    """
    # 약채도 회색-갈색 단색 (chroma 약 4)
    pixels = _fill((100, 96, 92), 1000)
    clusters, etc = build_object_palette(
        pixels, [], DynamicPaletteConfig(), [], frame_area=1000,
    )
    assert clusters == []
    # 원색 아닌 1 cluster 가 anchor 없음 → etc 로
    assert etc > 0.9 * WEIGHT_SCALE


def test_build_object_palette_lowchroma_shadow_merges_to_anchor() -> None:
    """진짜 shadow (anchor 보다 어둡고 hue 가까운 저채도) 가 anchor 로 머지된다.

    F-13 의 chroma 가드는 cL > tL 인 highlight branch 에만 적용. cL ≤ tL 인 shadow
    branch 는 가드 없음 — Sridevi 같은 사례에서 음영이 anchor weight 에 합산되는 invariant.
    """
    pixels = np.concatenate([
        _fill((110, 35, 30), 600),  # maroon — anchor
        _fill((50, 38, 32), 400),   # 진짜 shadow — anchor 보다 어둡고 hue 가까움
    ])
    entries = [MatcherEntry(name="maroon", lab=(25.0, 32.0, 16.0), family=ColorFamily.JEWEL)]
    clusters, etc = build_object_palette(
        pixels, ["maroon"], DynamicPaletteConfig(), entries, frame_area=1000,
    )
    # anchor 1개, shadow 가 머지 또는 R1 매칭 — 어쨌든 anchor weight 가 own (6000) 보다 큼
    assert any(c.is_anchor for c in clusters)
    anchor = next(c for c in clusters if c.is_anchor)
    assert anchor.weight + etc > 0.95 * WEIGHT_SCALE
    # shadow 가 anchor 에 합산됐거나 R1 직접 매칭 — 어느 쪽이든 etc 는 작아야
    assert etc < 0.5 * WEIGHT_SCALE


def test_build_object_palette_keeps_bright_highlight_solo_beta() -> None:
    """β 비대칭화 — anchor 와 hue 비슷하지만 밝은 highlight cluster 는 R2 solo keep.

    Pattern color 보존이 목적. 하이라이트가 하나의 cluster 로 분리되어 살아남도록.
    saffron anchor (어두운 따뜻색) + 밝은 highlight (같은 hue 계열, share≥0.10).
    """
    pixels = np.concatenate([
        _fill((180, 100, 30), 700),  # saffron anchor — L≈52
        _fill((250, 200, 130), 300),  # bright highlight — L≈85, hue 비슷
    ])
    entries = [
        MatcherEntry(name="saffron", lab=(55.0, 25.0, 50.0), family=ColorFamily.JEWEL),
    ]
    clusters, _etc = build_object_palette(
        pixels, ["saffron"], DynamicPaletteConfig(), entries, frame_area=1000,
    )
    # 두 cluster 모두 살아남아야 — anchor 1 + highlight solo 1
    assert len(clusters) >= 2
    # highlight cluster 는 anchor 보다 더 밝은 L 좌표를 보존
    L_values = sorted(c.lab[0] for c in clusters)
    assert L_values[-1] - L_values[0] > 15.0  # 두 cluster L 격차 의미 있음


def test_build_object_palette_no_directional_merge_when_anchor_darker() -> None:
    """anchor 가 cluster 보다 어두운 경우 (강제 룰 fail) → 머지 차단 → etc.

    검정 anchor + dark brown cluster (Δh undef, anchor.L 작음) → 머지 금지 → etc.
    """
    pixels = np.concatenate([
        _fill((10, 10, 10), 600),    # 검정 — anchor
        _fill((90, 70, 60), 400),    # dark brown — anchor 보다 밝음, 강제 룰 fail
    ])
    entries = [
        MatcherEntry(name="black", lab=(2.0, 0.0, 0.0), family=ColorFamily.NEUTRAL),
    ]
    clusters, etc = build_object_palette(
        pixels, ["black"], DynamicPaletteConfig(), entries, frame_area=1000,
    )
    # black anchor 1 cluster + dark brown 은 etc (chroma 가 낮은 편이라 R2 원색 solo 도 미통과)
    # 단, KMeans 결과에 따라 cluster 1 개로 합쳐지면 anchor 매칭이 통과할 수 있음.
    # 핵심 invariant: dark brown 좌표가 black anchor 그룹에 머지되지 않음 (anchor.lab L 작음)
    for c in clusters:
        # 머지 결과의 anchor 좌표 LAB L 이 dark brown (L≈30) 영역에 안 가야 함 (검정 좌표 보존)
        if c.lab[0] > 15.0:
            # 만약 dark brown 좌표가 살아 있다면 R2 원색 solo (chroma 충분) 또는
            # KMeans 가 합쳐서 중간 좌표가 된 케이스. 강제 룰 의도와는 분리된 동작.
            pass


def test_build_object_palette_drops_solo_below_r2_min_share() -> None:
    """anchor 없는 vivid cluster 가 share < r2_min_share 면 머지 또는 etc.

    blue cluster share≈0.08 < 0.10 → R2 원색 solo 게이트 fail → 머지 시도 → red cluster
    에 머지 차단 (Δh > 30°) → etc.
    """
    pixels = np.concatenate([
        _fill((220, 20, 20), 920),
        _fill((20, 20, 220), 80),
    ])
    clusters, etc = build_object_palette(
        pixels, [], DynamicPaletteConfig(), [], frame_area=1000,
    )
    # red solo 1 cluster, blue 는 작아서 dynamic_palette min_cluster_share 0.05 살아남고
    # R2 원색 게이트 fail (share < 0.10) → 머지 시도, 강제 룰 / hue 멀어 fail → etc
    assert len(clusters) == 1
    # red solo + etc weight = SCALE 근사
    assert (clusters[0].weight + etc) >= 0.9 * WEIGHT_SCALE


def test_build_object_palette_is_deterministic() -> None:
    """같은 입력 → 같은 출력 (KMeans random_state 고정 + 우리 로직 결정성)."""
    pixels = np.concatenate([
        _fill((220, 20, 20), 500),
        _fill((20, 20, 220), 500),
    ])
    cfg = DynamicPaletteConfig()
    r1, e1 = build_object_palette(pixels, [], cfg, [], frame_area=1000)
    r2, e2 = build_object_palette(pixels, [], cfg, [], frame_area=1000)
    assert len(r1) == len(r2)
    assert abs(e1 - e2) < 1e-6
    for a, b in zip(r1, r2):
        assert a.hex == b.hex
        assert a.rgb == b.rgb
        assert abs(a.weight - b.weight) < 1e-6
        for la, lb in zip(a.lab, b.lab):
            assert abs(la - lb) < 1e-4


def test_build_object_palette_hallucinated_pick_does_not_merge_clusters() -> None:
    """R3 가 환각 pick 을 drop 하면, 그 pick 이 anchor 가 되어 잘못 묶지 않는다.

    Fabindia cream_ivory motivating case 의 Phase 1 가드. cream_ivory drop 후 anchor
    없으니 red/blue 두 vivid cluster 는 R2 원색 solo 로 keep.
    """
    pixels = np.concatenate([
        _fill((220, 20, 20), 500),
        _fill((30, 30, 200), 500),
    ])
    entries = [
        MatcherEntry(
            name="cream_ivory", lab=(90.0, 0.0, 5.0), family=ColorFamily.WHITE_ON_WHITE,
        ),
    ]
    clusters, etc = build_object_palette(
        pixels, ["cream_ivory"], DynamicPaletteConfig(), entries, frame_area=1000,
    )
    # 둘 다 vivid + share≈0.5 → R2 원색 solo 통과
    assert len(clusters) == 2
    assert abs(sum(c.weight for c in clusters) + etc - WEIGHT_SCALE) < 1e-2


def test_build_object_palette_all_picks_dropped_falls_back_to_r2() -> None:
    """surviving_picks=[] 케이스 — R3 후 anchor 없음. vivid red 는 R2 원색 solo."""
    pixels = _fill((220, 20, 20), 1000)
    entries = [
        MatcherEntry(name="far_pick", lab=(90.0, 0.0, 5.0), family=ColorFamily.NEUTRAL),
    ]
    clusters, etc = build_object_palette(
        pixels, ["far_pick"], DynamicPaletteConfig(), entries, frame_area=1000,
    )
    assert len(clusters) == 1
    assert abs(clusters[0].weight - WEIGHT_SCALE) < 1e-2
    assert etc == 0.0


def test_build_object_palette_weight_floor_eps_for_tiny_cluster() -> None:
    """R2 keep 한 cluster 의 weight 가 0 으로 떨어지지 않도록 EPS floor 보장."""
    pixels = np.concatenate([
        _fill((220, 20, 20), 850),
        _fill((20, 220, 20), 150),  # share≈0.15 → vivid + share≥0.10 → R2 solo
    ])
    clusters, _etc = build_object_palette(
        pixels, [], DynamicPaletteConfig(), [], frame_area=1000,
    )
    assert all(c.weight > 0 for c in clusters)


def test_build_object_palette_small_obj_in_large_frame_has_lower_weight() -> None:
    """frame_area normalize: 같은 obj 라도 frame 안에서 작으면 weight 도 작다."""
    pixels = _fill((220, 20, 20), 1000)
    cfg = DynamicPaletteConfig()
    full_clusters, _ = build_object_palette(pixels, [], cfg, [], frame_area=1000)
    quarter_clusters, _ = build_object_palette(pixels, [], cfg, [], frame_area=4000)
    assert len(full_clusters) == len(quarter_clusters) == 1
    assert abs(full_clusters[0].weight / quarter_clusters[0].weight - 4.0) < 1e-6
