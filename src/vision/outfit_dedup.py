"""Phase 4.5 — intra-post outfit dedup (Pipeline B M3.A Step D).

같은 post 안 여러 이미지에 동일 의상이 재등장하면 vision LLM 이 각각 별개 outfit 으로
추출한다. 이 모듈은 (image_id, outfit_index) 쌍을 union-find 로 묶어 canonical 1 개로
축약한다. 병합된 member 들의 BBOX 는 Phase 3 crop union pool 로 살아남는다.

설계 (advisor 점검 후 확정, 2026-04-24):

1. per-signal binary match (weighted sum):
   - color_preset: |top3 ∩ top3| ≥ 2 → 1 (2개 이상 겹치면 "같은 팔레트")
   - color_family: dominant family (family_map 으로 매핑된 top3 의 카운트 최다)
     양쪽 존재 + 동일 → 1. dominant 은 node 생성 시 1회 계산 (pair loop 에서 재계산 X).
   - garment_type: upper 매칭 + dress_as_single 브리징 룰
   - technique: 양쪽 non-null + string 동일 → 1

2. None vs None = 0 (서로 모른다는 공통점은 신호로 치지 않음 — 과병합 방지)

3. similarity = Σ cfg.{signal}_weight * match_i. sum ≥ cfg.threshold 이면 병합 후보.

4. union-find 로 transitive 병합. 단, same_image_merge=False 이면 "같은 image 의 두
   outfit 이 같은 component 로 합쳐지는 union" 은 reject — A(img1) ~ B(img2) ~ C(img1)
   같은 transitive 경로도 차단 (component_images 교집합 검사).

5. determinism:
   - candidate pair 처리 순서: (similarity desc, i.image_id, i.outfit_index, j.image_id,
     j.outfit_index) lexicographic. 같은 sim 에서 재실행해도 동일 결과.
   - representative: component 내 person_bbox_area_ratio 최대, tiebreak (image_id asc,
     outfit_index asc).
   - canonical_index: component 를 rep.area_ratio desc + rep.image_id asc +
     rep.outfit_index asc 순으로 정렬 후 0부터.
   - members: 같은 component 내에서 (image_id, outfit_index) 오름차순.

내부 구조:
- `_DedupNode` 는 (image_id, outfit_index, outfit, dominant_family) 를 들고 있음. dominant
  pre-compute 로 C(n,2) pair similarity 루프에서 동일 outfit 에 대해 반복 계산을 피함.
- `dedup_post` 는 collect → compute_candidates → apply_unions → assemble 4 단계 orchestrator.

vision extras dependency 없음 (Pure python). core 에서도 import 가능하지만 Pipeline B
의존성으로 분류해 src/vision/ 에 배치.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from contracts.common import ColorFamily
from contracts.vision import (
    CanonicalOutfit,
    EthnicOutfit,
    GarmentAnalysis,
    OutfitMember,
)
from settings import OutfitDedupConfig


@dataclass(frozen=True)
class _DedupNode:
    """pair-loop 진입 전에 outfit 속성을 정돈해 둔 내부 node.

    dominant_family 를 한 번만 계산해 `_similarity` 호출마다 재계산하지 않게 한다.
    """
    image_id: str
    outfit_index: int
    outfit: EthnicOutfit
    dominant_family: ColorFamily | None


def _dominant_family(
    outfit: EthnicOutfit, family_map: dict[str, ColorFamily]
) -> ColorFamily | None:
    """top3 preset → family 리스트 → 최다 count family.

    count 동률 시 family enum value alphabetical asc 로 결정론 tiebreak.
    `Counter.most_common` 은 insertion order 의존 — 같은 outfit 이라도 LLM 이 preset
    순서를 섞어 내면 dominant 가 바뀌어 color_family match 가 실패할 수 있어 명시적
    total order 로 안정화.
    """
    families = [
        family_map[name]
        for name in outfit.color_preset_picks_top3
        if name in family_map
    ]
    if not families:
        return None
    counts = Counter(families)
    return min(counts.items(), key=lambda kv: (-kv[1], kv[0].value))[0]


def _color_preset_match(a: EthnicOutfit, b: EthnicOutfit) -> bool:
    """top3 preset 중 2개 이상 공통이면 매치. 1 개만 겹치면 우연 — 배제."""
    shared = set(a.color_preset_picks_top3) & set(b.color_preset_picks_top3)
    return len(shared) >= 2


def _color_family_match(
    a: ColorFamily | None, b: ColorFamily | None,
) -> bool:
    if a is None or b is None:
        return False
    return a == b


def _garment_type_match(a: EthnicOutfit, b: EthnicOutfit) -> bool:
    """upper 매칭 + dress_as_single 브리징.

    - 둘 다 two-piece: upper AND lower 모두 동일 (salwar-kameez 가 churidar-kameez 와 다르면 no)
    - 둘 다 single: upper 만 동일 (saree=saree 면 OK, lower 개념 X)
    - 한쪽 single / 한쪽 two-piece: upper 만 동일이면 매치 (예: lehenga-as-single ↔ lehenga-choli)
    """
    ua, ub = a.upper_garment_type, b.upper_garment_type
    if ua is None or ub is None or ua != ub:
        return False
    if not a.dress_as_single and not b.dress_as_single:
        la, lb = a.lower_garment_type, b.lower_garment_type
        return la is not None and lb is not None and la == lb
    return True


def _technique_match(a: EthnicOutfit, b: EthnicOutfit) -> bool:
    if a.technique is None or b.technique is None:
        return False
    return a.technique == b.technique


def _similarity(
    a: _DedupNode, b: _DedupNode, cfg: OutfitDedupConfig,
) -> float:
    total = 0.0
    if _color_preset_match(a.outfit, b.outfit):
        total += cfg.color_preset_weight
    if _color_family_match(a.dominant_family, b.dominant_family):
        total += cfg.color_family_weight
    if _garment_type_match(a.outfit, b.outfit):
        total += cfg.garment_type_weight
    if _technique_match(a.outfit, b.outfit):
        total += cfg.technique_weight
    return total


def _collect_outfit_nodes(
    post_items: list[tuple[str, GarmentAnalysis]],
    family_map: dict[str, ColorFamily],
) -> list[_DedupNode]:
    """ethnic post 에서 outfit 을 평탄화 + dominant_family pre-compute."""
    nodes: list[_DedupNode] = []
    for image_id, analysis in post_items:
        if not analysis.is_india_ethnic_wear:
            continue
        for outfit_index, outfit in enumerate(analysis.outfits):
            nodes.append(
                _DedupNode(
                    image_id=image_id,
                    outfit_index=outfit_index,
                    outfit=outfit,
                    dominant_family=_dominant_family(outfit, family_map),
                )
            )
    return nodes


def _compute_dedup_candidates(
    nodes: list[_DedupNode], cfg: OutfitDedupConfig,
) -> list[tuple[float, int, int]]:
    """threshold 이상 pair 수집 후 (sim desc, i tiebreak, j tiebreak) 로 정렬."""
    n = len(nodes)
    candidates: list[tuple[float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = _similarity(nodes[i], nodes[j], cfg)
            if sim >= cfg.threshold:
                candidates.append((sim, i, j))
    candidates.sort(
        key=lambda t: (
            -t[0],
            nodes[t[1]].image_id, nodes[t[1]].outfit_index,
            nodes[t[2]].image_id, nodes[t[2]].outfit_index,
        )
    )
    return candidates


def _apply_unions(
    nodes: list[_DedupNode],
    candidates: list[tuple[float, int, int]],
    same_image_merge: bool,
) -> dict[int, list[int]]:
    """union-find 적용 — same-image 충돌 발생 시 union skip. {root: [i, ...]} 반환."""
    n = len(nodes)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def component_images(root: int) -> set[str]:
        return {nodes[i].image_id for i in range(n) if find(i) == root}

    for _sim, i, j in candidates:
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        if not same_image_merge and (component_images(ri) & component_images(rj)):
            continue
        parent[ri] = rj

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return groups


def _rep_sort_key(node: _DedupNode) -> tuple[float, str, int]:
    """representative 선택 / canonical_index 정렬 공용 key — area desc, id asc, idx asc."""
    return (-node.outfit.person_bbox_area_ratio, node.image_id, node.outfit_index)


def _pick_representative(nodes: list[_DedupNode], members: list[int]) -> int:
    """component 의 representative index — area desc, id asc, idx asc."""
    return min(members, key=lambda i: _rep_sort_key(nodes[i]))


def _assemble_canonicals(
    nodes: list[_DedupNode], groups: dict[int, list[int]],
) -> list[CanonicalOutfit]:
    """component 별 representative 결정 + canonical_index 부여."""
    ordered = sorted(
        groups.values(),
        key=lambda members: _rep_sort_key(nodes[_pick_representative(nodes, members)]),
    )
    result: list[CanonicalOutfit] = []
    for canonical_index, members in enumerate(ordered):
        rep_i = _pick_representative(nodes, members)
        sorted_members = sorted(
            members, key=lambda i: (nodes[i].image_id, nodes[i].outfit_index),
        )
        result.append(
            CanonicalOutfit(
                canonical_index=canonical_index,
                representative=nodes[rep_i].outfit,
                members=[
                    OutfitMember(
                        image_id=nodes[i].image_id,
                        outfit_index=nodes[i].outfit_index,
                        person_bbox=nodes[i].outfit.person_bbox,
                    )
                    for i in sorted_members
                ],
            )
        )
    return result


def dedup_post(
    post_items: list[tuple[str, GarmentAnalysis]],
    cfg: OutfitDedupConfig,
    family_map: dict[str, ColorFamily],
) -> list[CanonicalOutfit]:
    """post 단위 outfit dedup — collect → compute_candidates → apply_unions → assemble.

    Args:
      post_items: [(image_id, GarmentAnalysis), ...]. 순서는 caller (보통 carousel index).
      cfg: weight / threshold / same_image_merge 정책.
      family_map: preset name → ColorFamily (color_family_preset.load_preset_family_map).

    Returns:
      canonical_index 오름차순 CanonicalOutfit 리스트. is_india_ethnic_wear=False 인
      analysis 는 skip. 전부 skip 이면 빈 리스트.
    """
    nodes = _collect_outfit_nodes(post_items, family_map)
    if not nodes:
        return []
    candidates = _compute_dedup_candidates(nodes, cfg)
    groups = _apply_unions(nodes, candidates, cfg.same_image_merge)
    return _assemble_canonicals(nodes, groups)
