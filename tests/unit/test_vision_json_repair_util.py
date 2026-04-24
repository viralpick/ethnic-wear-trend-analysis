"""vision.json_repair_util.parse_json_with_repair 단위 테스트.

Phase 0 v0.2 재파일럿 (2026-04-24) 에서 관측된 JSON 형식 버그 (1/20 확률,
outfit 중간 닫는 `}` 누락) 를 코드 기반 json-repair 로 방어하는지 검증.

vision extras 필요 — `uv sync --extra vision`. 미설치 환경에서는 모듈 수집 건너뜀.
"""
from __future__ import annotations

import pytest

pytest.importorskip("json_repair", reason="vision extras required")

from pydantic import ValidationError  # noqa: E402

from contracts.vision import GarmentAnalysis  # noqa: E402
from vision.json_repair_util import parse_json_with_repair  # noqa: E402

# Phase 0 v0.2 실측: post 01KPT4X4P0S2FVF64CGGTQ5SV7 의 Gemini raw.
# 첫 outfit 의 닫는 `}` 가 누락돼 JSONDecodeError. repair 가 구조를 복원해야 함.
OBSERVED_BROKEN_RAW = """{
  "is_india_ethnic_wear": true,
  "outfits": [
    {
      "person_bbox": [0.35, 0.05, 0.6, 0.95],
      "person_bbox_area_ratio": 0.57,
      "upper_garment_type": "kurta",
      "lower_garment_type": "salwar",
      "dress_as_single": false,
      "silhouette": "straight",
      "color_preset_picks_top3": ["mint_green", "pool_08", "cream_ivory"]
    ,
    {
      "person_bbox": [0.05, 0.05, 0.5, 0.95],
      "person_bbox_area_ratio": 0.475,
      "upper_garment_type": "suit",
      "lower_garment_type": "trousers",
      "dress_as_single": false,
      "silhouette": "straight",
      "color_preset_picks_top3": ["charcoal_grey", "mustard_olive", "cream_ivory"]
    }
  ]
}"""


def test_clean_json_parses_without_repair():
    payload = parse_json_with_repair('{"a": 1, "b": [2, 3]}')
    assert payload == {"a": 1, "b": [2, 3]}


def test_observed_broken_json_is_repaired_structurally():
    """관측 버그: 값은 그대로, 구조만 복원."""
    payload = parse_json_with_repair(OBSERVED_BROKEN_RAW)
    assert payload["is_india_ethnic_wear"] is True
    assert len(payload["outfits"]) == 2
    assert payload["outfits"][0]["upper_garment_type"] == "kurta"
    assert payload["outfits"][0]["lower_garment_type"] == "salwar"
    assert payload["outfits"][0]["color_preset_picks_top3"] == [
        "mint_green", "pool_08", "cream_ivory",
    ]
    assert payload["outfits"][1]["upper_garment_type"] == "suit"
    assert payload["outfits"][1]["lower_garment_type"] == "trousers"


def test_repaired_payload_passes_pydantic_validation():
    """repair 된 dict 이 GarmentAnalysis.model_validate 를 통과 — 최종 gate 검증."""
    payload = parse_json_with_repair(OBSERVED_BROKEN_RAW)
    analysis = GarmentAnalysis.model_validate(payload)
    assert analysis.is_india_ethnic_wear is True
    assert len(analysis.outfits) == 2
    assert analysis.outfits[0].upper_garment_type == "kurta"
    assert analysis.outfits[0].person_bbox == (0.35, 0.05, 0.6, 0.95)


def test_trailing_comma_is_repaired():
    raw = '{"x": 1, "y": [1, 2, 3,],}'
    payload = parse_json_with_repair(raw)
    assert payload == {"x": 1, "y": [1, 2, 3]}


def test_markdown_fence_wrapped_json_is_repaired():
    raw = '```json\n{"is_india_ethnic_wear": false, "outfits": []}\n```'
    payload = parse_json_with_repair(raw)
    assert payload == {"is_india_ethnic_wear": False, "outfits": []}


def test_prose_response_raises_value_error():
    """json-repair 가 dict 를 못 만들면 ValueError — 상위 retry 루프로 escalate."""
    with pytest.raises(ValueError, match="non-dict"):
        parse_json_with_repair("Sorry, I cannot process this image.")


def test_non_schema_valid_json_passes_util_but_fails_pydantic():
    """util 은 **구조적** 수리만 담당. schema 위반은 Pydantic 이 최종 gate."""
    # 필드 이름만 맞고 값이 schema 위반 (person_bbox_area_ratio > 1)
    raw = (
        '{"is_india_ethnic_wear": true, "outfits": [{'
        '"person_bbox": [0.0, 0.0, 1.0, 1.0], '
        '"person_bbox_area_ratio": 1.5, '
        '"upper_garment_type": "kurta", "lower_garment_type": null, '
        '"dress_as_single": false, "silhouette": "straight", '
        '"color_preset_picks_top3": []}]}'
    )
    payload = parse_json_with_repair(raw)
    assert payload["outfits"][0]["person_bbox_area_ratio"] == 1.5
    with pytest.raises(ValidationError):
        GarmentAnalysis.model_validate(payload)


def test_truncated_silhouette_repaired_then_fails_schema():
    """잘린 silhouette 문자열 — util 이 구조 복원해도 Silhouette enum 에 없어 Pydantic 탈락."""
    raw = (
        '{"is_india_ethnic_wear": true, "outfits": [{'
        '"person_bbox": [0.0, 0.0, 0.5, 0.5], '
        '"person_bbox_area_ratio": 0.25, '
        '"upper_garment_type": "kurta", "lower_garment_type": "churidar", '
        '"dress_as_single": false, "silhouette": "strai'  # 잘림
    )
    payload = parse_json_with_repair(raw)
    assert isinstance(payload, dict)
    # Pydantic 에서 Silhouette enum 위반으로 탈락 (ValidationError 로 retry 경로 escalate)
    with pytest.raises(ValidationError):
        GarmentAnalysis.model_validate(payload)
