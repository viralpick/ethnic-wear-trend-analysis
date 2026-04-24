"""LLM 응답 JSON 파싱 + 코드 기반 repair 유틸.

Gemini 관측 버그 (1/20 확률, outfit 중간 닫는 `}` 누락) 대응:
1차 `json.loads` 실패 시 `json_repair.repair_json` 으로 구조 복원. LLM 호출 0회.

`vision` extras 필요 — core 에서 import 하면 `json_repair` ImportError. 절대 금지.
`GeminiVisionLLMClient` 등 vision 내부 구현만 사용한다.

Pydantic validate 가 최종 gate 이므로 repair 가 값을 변질시켜도 schema 위반은 걸러진다.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from json_repair import repair_json

logger = logging.getLogger(__name__)


def parse_json_with_repair(raw: str) -> dict[str, Any]:
    """1차 json.loads. 실패 시 json_repair 로 코드 기반 수리 (LLM 호출 없음).

    repair 결과가 dict 가 아니면 ValueError — Pydantic 에 넘기기 전에 차단.
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.info(
            "llm_json_malformed attempting_repair reason=%s raw_head=%r",
            exc.msg, raw[:80],
        )
        repaired = repair_json(raw, return_objects=True)
        if not isinstance(repaired, dict):
            raise ValueError(
                f"json-repair produced non-dict: {type(repaired).__name__}"
            ) from exc
        logger.info("llm_json_repaired payload_keys=%s", list(repaired.keys()))
        return repaired
