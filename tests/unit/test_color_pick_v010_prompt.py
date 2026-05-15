"""color.B v0.10 Pass 2 prompt builder unit tests.

`build_color_pick_v010_user_payload` 는 LLM user message 직렬화 함수.
Pass 1 분류 + KMeans cluster top-N 을 JSON 으로 직렬화. cluster_index 는 입력
범위 안으로 closed-set anchoring — system_prompt 가 그 규칙을 명시.
"""
from __future__ import annotations

import json

from vision.prompts import (
    COLOR_PICK_V010_PROMPT_VERSION,
    COLOR_PICK_V010_SYSTEM_PROMPT,
    build_color_pick_v010_user_payload,
)


def _classification() -> dict[str, object]:
    return {
        "upper_garment_type": "kurta",
        "lower_garment_type": "palazzo",
        "upper_is_ethnic": True,
        "lower_is_ethnic": True,
        "dress_as_single": False,
    }


def _clusters() -> list[dict[str, object]]:
    return [
        {"index": 0, "hex": "#f5e7c4", "share": 0.42},
        {"index": 1, "hex": "#c11a4a", "share": 0.31},
        {"index": 2, "hex": "#2a6b3f", "share": 0.18},
    ]


def test_prompt_version_constant() -> None:
    assert COLOR_PICK_V010_PROMPT_VERSION == "v0.10"


def test_system_prompt_mentions_closed_set_rule() -> None:
    # closed-set anchoring 규칙이 prompt 에 명시되어야 환각 차단 가능
    assert "closed-set" in COLOR_PICK_V010_SYSTEM_PROMPT.lower()
    assert "cluster_index" in COLOR_PICK_V010_SYSTEM_PROMPT


def test_system_prompt_mentions_1_to_3_picks() -> None:
    # padding 금지 + 1~3 정책
    assert "1 to 3" in COLOR_PICK_V010_SYSTEM_PROMPT
    assert "DO NOT pad" in COLOR_PICK_V010_SYSTEM_PROMPT


def test_user_payload_contains_classification_keys() -> None:
    payload = build_color_pick_v010_user_payload(
        garment_classification=_classification(),
        kmeans_clusters=_clusters(),
    )
    assert "garment_classification" in payload
    assert "kurta" in payload
    assert "palazzo" in payload


def test_user_payload_contains_clusters() -> None:
    payload = build_color_pick_v010_user_payload(
        garment_classification=_classification(),
        kmeans_clusters=_clusters(),
    )
    assert "kmeans_clusters" in payload
    assert "#f5e7c4" in payload
    assert "#c11a4a" in payload


def test_user_payload_is_parseable_json_for_each_block() -> None:
    payload = build_color_pick_v010_user_payload(
        garment_classification=_classification(),
        kmeans_clusters=_clusters(),
    )
    # 직렬화한 두 블록이 JSON 으로 round-trip 가능해야 한다.
    classification_part = payload.split("garment_classification:\n", 1)[1].split(
        "\n\nkmeans_clusters", 1
    )[0]
    clusters_part = payload.split("only):\n", 1)[1]

    parsed_cls = json.loads(classification_part)
    parsed_clusters = json.loads(clusters_part)

    assert parsed_cls["upper_garment_type"] == "kurta"
    assert len(parsed_clusters) == 3
    assert parsed_clusters[0]["hex"] == "#f5e7c4"


def test_user_payload_empty_clusters_serialized() -> None:
    # edge case — caller 가 빈 cluster 를 넘기는 경우. prompt 자체는 거부 안 함
    # (adapter 가 상위에서 가드). LLM 이 0 pick 반환하면 contract validation 실패.
    payload = build_color_pick_v010_user_payload(
        garment_classification=_classification(),
        kmeans_clusters=[],
    )
    assert "kmeans_clusters" in payload
    assert "[]" in payload
