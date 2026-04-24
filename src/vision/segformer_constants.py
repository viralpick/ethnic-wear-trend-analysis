"""ATR 18-class segformer label 매핑 + wear/skin 클래스 집합.

원래 `pipeline_b_extractor.py` 내부에 `_WEAR_CLASS_IDS` / `_SKIN_CLASS_IDS` 로 private
선언돼 있었으나, Phase 3 canonical_extractor 가 동일 ID set 을 사용 — 다른 모듈의 private
심볼 import smell 을 제거하기 위해 공용 모듈로 이관.

Phase 5 에서 legacy `pipeline_b_extractor` 의 `extract_instances` 경로가 canonical path
로 교체되면 이 모듈만 남고 pipeline_b_extractor 가 사라질 수 있다.

labels ↔ class id 는 ATR dataset (동료 PoC 에서 인용) 기준. spec §4.1 ① GarmentType 과
직접 매칭 X — ATR 은 서양 복식 분류라 garment_type 분류는 텍스트/LLM 담당.
"""
from __future__ import annotations

ATR_LABELS: dict[int, str] = {
    0: "background",
    1: "hat",
    2: "hair",
    3: "sunglasses",
    4: "upper-clothes",
    5: "skirt",
    6: "pants",
    7: "dress",
    8: "belt",
    9: "left-shoe",
    10: "right-shoe",
    11: "bag",
    12: "skin-face",
    13: "skin-face",
    14: "skin-left-arm",
    15: "skin-right-arm",
    16: "skin-left-leg",
    17: "skin-right-leg",
}

# M3.A Step D — LLM-centric 전환으로 Pipeline B 역할이 "상/하의 색상 추출" 로 좁혀졌다.
# hat / shoe / bag / belt 는 LLM 이 텍스트 속성으로 처리. segformer 가 얼굴/머리/발 주변
# skin 영역을 hat/shoe 로 오분류하는 edge case 도 이 축소로 감소.
WEAR_KEEP: frozenset[str] = frozenset({
    "upper-clothes", "pants", "skirt", "dress",
})

# ATR 의 피부 클래스 라벨 (bbox false positive 필터용). "진짜 사람" 검증에 사용 —
# 동상/마네킹/제품샷은 skin class pixel 이 거의 0.
SKIN_LABELS: frozenset[str] = frozenset({
    "skin-face", "skin-left-arm", "skin-right-arm", "skin-left-leg", "skin-right-leg",
})

WEAR_CLASS_IDS: frozenset[int] = frozenset(
    {cid for cid, lbl in ATR_LABELS.items() if lbl in WEAR_KEEP}
)
SKIN_CLASS_IDS: frozenset[int] = frozenset(
    {cid for cid, lbl in ATR_LABELS.items() if lbl in SKIN_LABELS}
)
