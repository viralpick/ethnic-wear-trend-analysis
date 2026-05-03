"""β hybrid object palette — Phase 1 of per-object β-hybrid → weighted KMeans.

오브젝트 = 1 OutfitMember (1 사진의 1 person BBOX 안 ethnic wear set).

Phase 1 (이 모듈):
  R3: KMeans cluster pool 과 ΔE76 > threshold 인 Gemini pick 을 환각으로 판단해 drop.
  R1 (2026-04-27 F-13 재설계 — 1:1 매칭): pick 1개 = cluster 1개 greedy 매칭.
       (i) 모든 (cluster, surviving_pick) 쌍 중 ΔE76 ≤ drop_threshold 인 것을 후보로.
       (ii) ΔE76 가장 작은 후보부터 양쪽 미할당이면 (cluster→pick) 확정.
       (iii) 같은 pick 을 노린 다른 cluster 들은 자동 후순위 → 다음 후보 처리.
            cluster 입장에선 자기 1순위 pick 이 다른 cluster 한테 뺏기면 다음 가까운
            미점유 pick 시도 (greedy 자연스럽게 처리).
       (iv) 매칭 못한 cluster 는 R2 진입.
       L-highest 룰 폐기 — 그룹 크기 = 1 이라 자체 좌표 = anchor 좌표.
  R2 (2026-04-26 재설계 + 2026-04-27 β 비대칭화 + F-13 chroma 가드): anchor 못 잡은 cluster.
       (a) 원색 (chroma ≥ CHROMA_VIVID) 이고 share ≥ R2_MIN_SHARE 면 독립 보존.
       (a-β) 같은-hue anchor 보다 밝은 cluster (Δh ≤ HUE_NEAR_DEG AND cluster.L >
           anchor.L) 가 share ≥ R2_MIN_SHARE AND chroma 비율 ≥ CHROMA_RATIO_MIN 면
           chroma 임계 무시 독립 보존 (F-13: chroma 가드 추가 — mid brown 이 saturated
           maroon 의 highlight 로 오인되는 것을 차단).
       (b) 그 외는 anchor 로 머지 — Δh ≤ HUE_NEAR_DEG AND cluster.L ≤ anchor.L 면 음영,
           Δh > HUE_NEAR_DEG 면 "음영 = base 보다 어둡고 채도 낮다" 강제 (anchor.L >
           cluster.L AND anchor.chroma > cluster.chroma).
       (c) 머지 후보 중 ΔE76 ≤ R2_MERGE_DELTAE76 만 통과. 후보 중 ΔE76 가장 작은
           anchor 로 합산. anchor weight 에는 머지된 cluster 의 weight 가 합산되므로
           Phase 3 의 share 는 "흡수 후" 기준 (사용자 결정 2026-04-27 F-13).
       (d) 머지 못한 잔여는 etc bucket weight 에 합산 (좌표 버림).

Phase 1 출력 = `(list[WeightedCluster], etc_weight)`. anchor 매칭 cluster 는
`is_anchor=True`. Phase 3 가 centroid 단위로 anchor 여부 propagate 해 top_n cap 에서
anchor centroid 우선 보존 (사용자 결정 2026-04-27 F-13).

설계 원칙:
- pure function — pixels (RGB N,3) + Gemini picks → (list[WeightedCluster], etc_weight).
- Gemini pick 은 hex/share 주입 X. anchor 판단에만 활용.
- preset name → LAB lookup 은 `MatcherEntry` 그대로 재사용.
- preset 에서 누락된 pick name 은 silent drop 금지 — `log.warning` 후 drop.
- 1:1 매칭 후보 정렬 tie-break: ΔE76 동률이면 picks 입력 순서 (Gemini dominance) 우선.
- R2 머지 규칙은 nearest anchor 가 아니라 "음영의 정의" 강제 — 검정 anchor 가 dark
  brown cluster 를 흡수하는 잘못된 머지 차단 (2026-04-26).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from settings import DynamicPaletteConfig
from vision.color_family_preset import MatcherEntry
from vision.color_space import delta_e76_tuple
from vision.dynamic_palette import PaletteCluster as PixelCluster
from vision.dynamic_palette import extract_dynamic_palette

log = logging.getLogger(__name__)


R3_DROP_DELTAE76: float = 28.0
R2_MIN_SHARE: float = 0.10
CHROMA_VIVID: float = 15.0
HUE_NEAR_DEG: float = 30.0
R2_MERGE_DELTAE76: float = 40.0
CHROMA_RATIO_MIN: float = 0.5
WEIGHT_SCALE: float = 10_000.0
WEIGHT_EPS: float = 1e-6


@dataclass(frozen=True)
class WeightedCluster:
    """Phase 1 출력. 좌표 (hex/rgb/lab) + frame-area normalized weight.

    family / share 는 의도적으로 없음 — Phase 3 통합 KMeans 가 결정한다.
    weight 는 frame_area 로 normalize 된 float — `share * obj_pixel_count / frame_area`
    × `WEIGHT_SCALE` 로 계산해 sklearn KMeans `sample_weight` 가 정밀도를 잃지 않도록 한다.
    이미지 frame 크기에 비례해 영향력을 부여하므로 작은 obj 가 큰 obj 를 압도하지 않는다
    (advisor A2 / 사용자 결정 2026-04-25).
    weight > 0 이 invariant — `_share_to_weight` 가 `WEIGHT_EPS` floor 보장.

    `is_anchor` (F-13): R1 1:1 매칭으로 Gemini pick 과 짝지어진 cluster 인지. Phase 3
    aggregator 가 centroid 단위로 propagate 해 top_n cap 시 anchor 우선 보존
    (사용자 결정 2026-04-27 — Gemini ∩ KMeans 신호는 환각 가능성이 낮으므로 우선).
    """

    hex: str
    rgb: tuple[int, int, int]
    lab: tuple[float, float, float]
    weight: float
    is_anchor: bool = False


def _build_lab_lookup(
    matcher_entries: list[MatcherEntry],
) -> dict[str, tuple[float, float, float]]:
    """preset name → LAB. 같은 name 중복 시 첫 entry 우선 (preset 자체에 unique 가정)."""
    lookup: dict[str, tuple[float, float, float]] = {}
    for entry in matcher_entries:
        if entry.name not in lookup:
            lookup[entry.name] = entry.lab
    return lookup


def _min_deltae76_to_clusters(
    pick_lab: tuple[float, float, float], clusters: list[PixelCluster],
) -> float:
    """pick LAB 과 KMeans clusters 중 가장 가까운 ΔE76. clusters 비면 inf."""
    if not clusters:
        return float("inf")
    best = float("inf")
    for c in clusters:
        d = delta_e76_tuple(pick_lab, c.lab)
        if d < best:
            best = d
    return best


def filter_picks_by_pixel_evidence(
    picks: list[str],
    clusters: list[PixelCluster],
    matcher_entries: list[MatcherEntry],
    drop_threshold: float = R3_DROP_DELTAE76,
) -> list[str]:
    """R3 — pick 별 KMeans pool 과 ΔE76 비교, 가장 가까운 cluster 와도 멀면 drop.

    Args:
      picks: Gemini `color_preset_picks_top3` (preset name list, max 3).
      clusters: 동적 k KMeans 결과 (post merge/drop). 비면 모든 pick drop.
      matcher_entries: preset name → LAB lookup 을 위한 MatcherEntry list.
      drop_threshold: ΔE76 cutoff. plan default 25.0.

    Returns:
      입력 순서를 유지한 생존 pick list. preset 누락 / clusters 비면 빈 list 가능.

    실패 모드:
      - pick name 이 matcher_entries 에 없으면 `log.warning` 후 drop. preset stale 또는
        whitelist 우회 신호이므로 silent drop 금지.
      - clusters 비면 어떤 pick 도 검증 불가 → 모두 drop (전부 환각으로 간주).
    """
    if not picks or not clusters:
        return []
    lookup = _build_lab_lookup(matcher_entries)
    survivors: list[str] = []
    for pick in picks:
        pick_lab = lookup.get(pick)
        if pick_lab is None:
            log.warning(
                "hybrid_pick_unknown name=%s — preset stale 또는 whitelist 우회 가능", pick,
            )
            continue
        if _min_deltae76_to_clusters(pick_lab, clusters) <= drop_threshold:
            survivors.append(pick)
    return survivors


def _resolve_anchor_for_cluster(
    cluster: PixelCluster,
    surviving_picks: list[str],
    pick_lab_lookup: dict[str, tuple[float, float, float]],
    threshold: float,
) -> str | None:
    """cluster.lab 에 가장 가까운 surviving pick 을 반환. 모두 threshold 초과면 None.

    F-13 1:1 매칭 도입 후에도 후방 호환을 위해 유지 (단위 테스트 다수 의존). 실제
    `build_object_palette` 의 R1 매핑은 `_match_anchors_one_to_one` 으로 대체됨.
    """
    if not surviving_picks:
        return None
    best: tuple[str, float] | None = None
    for pick in surviving_picks:
        d = delta_e76_tuple(cluster.lab, pick_lab_lookup[pick])
        if d > threshold:
            continue
        if best is None or d < best[1]:
            best = (pick, d)
    return best[0] if best else None


def _match_anchors_one_to_one(
    pixel_clusters: list[PixelCluster],
    surviving_picks: list[str],
    pick_lab_lookup: dict[str, tuple[float, float, float]],
    threshold: float,
) -> tuple[dict[int, str], list[int]]:
    """F-13: 1:1 greedy ΔE76 매칭. pick 1개 = cluster 1개 (다대다 X).

    알고리즘:
      1. 모든 (cluster_idx, pick) 쌍 중 ΔE76 ≤ threshold 인 것을 후보로 수집.
      2. 후보를 (ΔE76 asc, pick_priority asc, cluster_idx asc) 로 정렬.
         pick_priority = surviving_picks 입력 순서 (Gemini dominance desc) — tie-break 안정.
      3. 순서대로 처리하면서 양쪽 미할당이면 (cluster→pick) 확정. 이미 점유된 쪽은 skip.
         자연 결과: 각 pick 은 자기와 ΔE76 가장 가까운 cluster 를 갖고, 충돌 시 거리
         더 작은 (cluster, pick) 쌍이 우선 — 거리 큰 쌍은 "loser cluster" 가 다음 가까운
         미점유 pick 시도 (greedy 가 자연스럽게 처리).

    Args:
      pixel_clusters: KMeans cluster pool (R3 후 surviving 가정).
      surviving_picks: R3 가 keep 한 pick name list.
      pick_lab_lookup: preset name → LAB.
      threshold: R3_DROP_DELTAE76 (anchor 매칭 cutoff).

    Returns:
      (cluster_to_pick dict, non_anchor_idx list).
      cluster_to_pick: cluster_idx → pick_name 매칭. 1:1 보장.
      non_anchor_idx: 매칭 못한 cluster_idx (pixel_clusters 입력 순서 보존).
    """
    if not surviving_picks or not pixel_clusters:
        return {}, list(range(len(pixel_clusters)))
    pick_priority = {pick: i for i, pick in enumerate(surviving_picks)}
    candidates: list[tuple[float, int, int, str]] = []
    for c_idx, cluster in enumerate(pixel_clusters):
        for pick in surviving_picks:
            d = delta_e76_tuple(cluster.lab, pick_lab_lookup[pick])
            if d > threshold:
                continue
            candidates.append((d, pick_priority[pick], c_idx, pick))
    candidates.sort()
    cluster_to_pick: dict[int, str] = {}
    pick_used: set[str] = set()
    for _d, _p, c_idx, pick in candidates:
        if c_idx in cluster_to_pick or pick in pick_used:
            continue
        cluster_to_pick[c_idx] = pick
        pick_used.add(pick)
    non_anchor_idx = [
        i for i in range(len(pixel_clusters)) if i not in cluster_to_pick
    ]
    return cluster_to_pick, non_anchor_idx


def _share_to_weight(
    share: float,
    obj_pixel_count: int,
    frame_area: int,
    scale: float = WEIGHT_SCALE,
) -> float:
    """share (0..1) × obj_pixel_count / frame_area × scale → float weight.

    frame_area normalize: 이미지 frame 안에서 obj 가 차지하는 면적 비율을 그대로
    weight 의 magnitude 로 부여한다. 작은 obj (frame 의 1%) 의 share=1.0 cluster 가
    큰 obj (frame 100%) 의 share=0.05 cluster 에 압도되지 않도록 obj 자신의
    frame coverage 를 곱한다 (advisor A2 / 사용자 결정 2026-04-25).

    `scale` (default 10_000) 은 sklearn KMeans 의 `sample_weight` 에서 0.0001 같은
    작은 값이 정밀도를 잃지 않도록 곱하는 magnitude. 같은 frame_area 안에서는 의미
    동일.

    `WEIGHT_EPS` floor: sklearn `sample_weight=0` silent drop 회피. frame_area 가
    비정상적으로 작거나 share 가 0 일 때도 cluster 가 살아있게 보장.
    """
    if frame_area <= 0:
        return WEIGHT_EPS
    raw = share * obj_pixel_count / frame_area * scale
    return max(WEIGHT_EPS, raw)


def _chroma(lab: tuple[float, float, float]) -> float:
    """LAB chroma = sqrt(a^2 + b^2). 무채 (a=b=0) 는 0."""
    _, a, b = lab
    return (a * a + b * b) ** 0.5


def _hue_deg(lab: tuple[float, float, float]) -> float | None:
    """LAB → hue (degrees, 0~360). chroma=0 (무채) 면 None — hue 정의 안됨."""
    _, a, b = lab
    if (a * a + b * b) ** 0.5 < 1e-6:
        return None
    h = math.degrees(math.atan2(b, a))
    if h < 0:
        h += 360.0
    return h


def _hue_circular_diff(h1: float | None, h2: float | None) -> float:
    """두 hue (deg) 의 circular 거리 0~180. None 이 하나라도 있으면 inf 반환.

    R2 머지에서 무채 cluster (hue=None) 는 hue 비슷 분기를 못 타게 해서 강제 룰
    적용 — anchor.L > cluster.L AND anchor.chroma > cluster.chroma 만 머지 허용.
    """
    if h1 is None or h2 is None:
        return float("inf")
    d = abs(h1 - h2) % 360.0
    return min(d, 360.0 - d)


def _resolve_merge_target(
    cluster: PixelCluster,
    anchor_targets: list[tuple[str, tuple[float, float, float]]],
    merge_threshold: float,
    hue_near_deg: float,
    chroma_ratio_min: float = CHROMA_RATIO_MIN,
) -> str | None:
    """non-anchor cluster 를 어떤 anchor 로 머지할지 결정.

    각 anchor 후보에 대해:
      - ΔE76 > merge_threshold → 제외.
      - Δh ≤ hue_near_deg AND cluster.L ≤ anchor.L → hue 비슷한 음영 → 머지 허용.
      - Δh ≤ hue_near_deg AND cluster.L > anchor.L:
          (β) chroma 비교 (F-13 가드) — min(c, a) / max(c, a) ≥ chroma_ratio_min 이면
              "진짜 highlight" → 머지 거부 (호출자 R2 (a-β) highlight solo 분기).
              그 미만이면 hue 만 비슷하지 saturation 이 다른 색 → highlight 분기에서 빠짐.
              이 경우는 "음영 정의" 분기로 떨어뜨려 anchor 보다 어둡고 채도 높을 때만 머지.
              실제로 cL > tL 이라 어두운 조건 fail → 머지 거부 (etc 로).
      - Δh > hue_near_deg → "음영의 정의" 강제: anchor.L > cluster.L AND
          anchor.chroma > cluster.chroma 일 때만 허용.

    통과 후보 중 ΔE76 가장 작은 anchor name 반환. 후보 0 이면 None.

    F-13: chroma_ratio_min 가드는 mid-brown (chroma 14) 같은 cluster 가 saturated maroon
    (chroma 30) 의 highlight 로 잘못 분류돼 의미 없는 lone bright cluster 가 palette 에
    들어가는 것을 차단.

    Args:
      cluster: non-anchor PixelCluster.
      anchor_targets: list of (anchor_name, target_lab).
      merge_threshold: R2_MERGE_DELTAE76. 후보 산출 cutoff.
      hue_near_deg: HUE_NEAR_DEG. hue 비슷 임계 (deg).
      chroma_ratio_min: same-hue brighter 분기에서 highlight 로 인정할 chroma 비율 하한.

    Returns:
      매칭된 anchor name, 또는 None.
    """
    if not anchor_targets:
        return None
    cL, _ca, _cb = cluster.lab
    c_chroma = _chroma(cluster.lab)
    c_hue = _hue_deg(cluster.lab)
    best: tuple[str, float] | None = None
    for anchor_name, target_lab in anchor_targets:
        tL, _ta, _tb = target_lab
        d = delta_e76_tuple(cluster.lab, target_lab)
        if d > merge_threshold:
            continue
        t_hue = _hue_deg(target_lab)
        t_chroma = _chroma(target_lab)
        hue_diff = _hue_circular_diff(c_hue, t_hue)
        if hue_diff <= hue_near_deg:
            if cL > tL:
                # F-13: chroma 비교가 충분히 가까울 때만 "진짜 highlight" 로 보고 머지 거부.
                # 멀면 highlight 가 아니라서 다음 검증 (어두운 음영) 으로 보내는데,
                # 이미 cL > tL 라 어두운 조건도 fail → 결과적으로 anchor 후보에서 제외.
                if (
                    t_chroma > 1e-6
                    and c_chroma > 1e-6
                    and min(c_chroma, t_chroma) / max(c_chroma, t_chroma)
                    >= chroma_ratio_min
                ):
                    continue
                continue
        else:
            if not (tL > cL and t_chroma > c_chroma):
                continue
        if best is None or d < best[1]:
            best = (anchor_name, d)
    return best[0] if best else None


def _is_bright_highlight(
    cluster: PixelCluster,
    anchor_targets: list[tuple[str, tuple[float, float, float]]],
    hue_near_deg: float,
    chroma_ratio_min: float = CHROMA_RATIO_MIN,
) -> bool:
    """β: cluster 가 같은-hue anchor 보다 밝은 highlight 영역인지.

    R2 (a-β) 의 highlight 보존 분기. 같은 hue (Δh ≤ hue_near_deg) 인 anchor 중 cluster
    보다 어두운 게 있고 chroma 비율이 chroma_ratio_min 이상이면 True.

    F-13 chroma 가드: chroma 비율 (min/max) ≥ chroma_ratio_min 미충족이면 False —
    saturated anchor 와 grey-ish cluster 의 잘못된 highlight 매칭 차단.
    어느 한쪽 chroma 가 0 (무채) 이어도 highlight semantics 가 의미 없으니 False.

    무채 cluster (hue=None) 는 항상 False — hue near 분기 자체가 동작 안 함.
    """
    if not anchor_targets:
        return False
    cL = cluster.lab[0]
    c_chroma = _chroma(cluster.lab)
    c_hue = _hue_deg(cluster.lab)
    if c_chroma < 1e-6:
        return False
    for _name, target_lab in anchor_targets:
        if cL <= target_lab[0]:
            continue
        t_chroma = _chroma(target_lab)
        if t_chroma < 1e-6:
            continue
        t_hue = _hue_deg(target_lab)
        if _hue_circular_diff(c_hue, t_hue) > hue_near_deg:
            continue
        if min(c_chroma, t_chroma) / max(c_chroma, t_chroma) < chroma_ratio_min:
            continue
        return True
    return False


def build_object_palette(
    rgb_pixels: np.ndarray,
    picks: list[str],
    dyn_cfg: DynamicPaletteConfig,
    matcher_entries: list[MatcherEntry],
    frame_area: int,
    drop_threshold: float = R3_DROP_DELTAE76,
    r2_min_share: float = R2_MIN_SHARE,
    chroma_vivid: float = CHROMA_VIVID,
    hue_near_deg: float = HUE_NEAR_DEG,
    r2_merge_deltae76: float = R2_MERGE_DELTAE76,
    chroma_ratio_min: float = CHROMA_RATIO_MIN,
) -> tuple[list[WeightedCluster], float]:
    """object pool (RGB N,3) + Gemini picks + frame_area → (list[WeightedCluster], etc_weight).

    pipeline (F-13 재설계):
      1. extract_dynamic_palette — pixel cluster pool (share desc 정렬).
      2. R3 — `filter_picks_by_pixel_evidence` (mismatched Gemini pick drop).
      3. R1 1:1 매칭 — `_match_anchors_one_to_one`. 각 pick 은 자기와 ΔE76 가장 가까운
         cluster 1개와만 매칭 (충돌 시 거리 더 작은 쪽 우선). loser cluster 는 R2 진입.
      4. anchor cluster 좌표 = 자기 cluster 좌표 (그룹 크기 1, L-highest 룰 폐기).
      5. R2 (2026-04-26 재설계 + 2026-04-27 β + F-13 chroma 가드) — non-anchor 분류:
         (a) chroma ≥ chroma_vivid 이고 share ≥ r2_min_share 면 R2 vivid solo.
         (a-β) `_is_bright_highlight` (chroma ratio guard) AND share ≥ r2_min_share 면
              highlight solo.
         (b) 그 외는 `_resolve_merge_target` 로 머지 — 통과 anchor 가 있으면 weight 합산.
             없으면 etc weight 에 합산 (좌표 버림).

    `frame_area` 는 멤버가 속한 이미지 frame 의 H×W (advisor A2 / 2026-04-25).

    출력 순서:
      - anchor merged: surviving_picks 순서 (Gemini pick 순서 보존, 결정성).
      - R2 solo 부분: pixel_clusters 순서 (share desc).
    anchor cluster 의 `is_anchor=True`. solo 와 etc 는 False.

    Returns:
      (clusters, etc_weight) — etc_weight 는 머지 못한 잔여 weight 합.
      Phase 3 aggregator 가 cut_off_share 에 흡수.

    빈 pool / 미만 pixel / R3 후 picks 전부 환각 + R2 후 전부 drop 인 경우 안전하게
    ([], 0.0) 반환.
    """
    if rgb_pixels.size == 0:
        return [], 0.0
    pixel_clusters = extract_dynamic_palette(rgb_pixels, dyn_cfg)
    if not pixel_clusters:
        return [], 0.0

    obj_pixel_count = int(rgb_pixels.shape[0])

    surviving_picks = filter_picks_by_pixel_evidence(
        picks, pixel_clusters, matcher_entries, drop_threshold,
    )
    pick_lab_lookup = _build_lab_lookup(matcher_entries)

    # R1 1:1 greedy 매칭
    cluster_to_pick, non_anchor_idx = _match_anchors_one_to_one(
        pixel_clusters, surviving_picks, pick_lab_lookup, drop_threshold,
    )
    pick_to_cluster_idx: dict[str, int] = {p: i for i, p in cluster_to_pick.items()}

    # anchor 초기 weight = 자기 cluster 의 own weight. 좌표 = 자기 좌표.
    anchor_targets: list[tuple[str, tuple[float, float, float]]] = []
    anchor_weights: dict[str, float] = {}
    for pick in surviving_picks:
        c_idx = pick_to_cluster_idx.get(pick)
        if c_idx is None:
            continue
        cluster = pixel_clusters[c_idx]
        anchor_weights[pick] = _share_to_weight(
            cluster.share, obj_pixel_count, frame_area,
        )
        anchor_targets.append((pick, cluster.lab))

    # R2 — non-anchor cluster 분류
    r2_solos: list[WeightedCluster] = []
    etc_weight: float = 0.0
    for c_idx in non_anchor_idx:
        cluster = pixel_clusters[c_idx]
        c_chroma = _chroma(cluster.lab)
        is_vivid_solo = (
            c_chroma >= chroma_vivid and cluster.share >= r2_min_share
        )
        is_highlight_solo = (
            cluster.share >= r2_min_share
            and _is_bright_highlight(
                cluster, anchor_targets, hue_near_deg, chroma_ratio_min,
            )
        )
        if is_vivid_solo or is_highlight_solo:
            r2_solos.append(
                WeightedCluster(
                    hex=cluster.hex,
                    rgb=cluster.rgb,
                    lab=cluster.lab,
                    weight=_share_to_weight(
                        cluster.share, obj_pixel_count, frame_area,
                    ),
                    is_anchor=False,
                ),
            )
            continue
        target_anchor = _resolve_merge_target(
            cluster, anchor_targets, r2_merge_deltae76, hue_near_deg,
            chroma_ratio_min,
        )
        cluster_weight = _share_to_weight(
            cluster.share, obj_pixel_count, frame_area,
        )
        if target_anchor is not None:
            anchor_weights[target_anchor] += cluster_weight
        else:
            etc_weight += cluster_weight

    # 최종 anchor WeightedCluster 빌드 — picks 입력 순서 유지
    merged: list[WeightedCluster] = []
    for pick in surviving_picks:
        c_idx = pick_to_cluster_idx.get(pick)
        if c_idx is None:
            continue
        cluster = pixel_clusters[c_idx]
        merged.append(
            WeightedCluster(
                hex=cluster.hex, rgb=cluster.rgb, lab=cluster.lab,
                weight=anchor_weights[pick],
                is_anchor=True,
            ),
        )

    return merged + r2_solos, etc_weight


__all__ = [
    "CHROMA_RATIO_MIN",
    "CHROMA_VIVID",
    "HUE_NEAR_DEG",
    "R2_MERGE_DELTAE76",
    "R2_MIN_SHARE",
    "R3_DROP_DELTAE76",
    "WEIGHT_EPS",
    "WEIGHT_SCALE",
    "WeightedCluster",
    "build_object_palette",
    "filter_picks_by_pixel_evidence",
]
