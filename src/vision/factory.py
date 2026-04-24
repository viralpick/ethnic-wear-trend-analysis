"""VisionLLMClient factory — `VisionLLMConfig` → 구체 구현 분기.

core 에서 직접 `GeminiVisionLLMClient` 를 import 하면 vision extras 강제되므로 DI 경계가
깨진다. factory 는 **vision extras 내부 모듈** 이라 top-level import 자유롭게 사용.

현재 provider:
- `gemini` (default): GeminiVisionLLMClient + LocalJSONCache
- `azure-openai`: pilot 용 (scripts/pilot_llm_bbox.py) — production client 미구현.
  설정이 들어오면 NotImplementedError 로 명시 실패 (실패 숨김 X).

prompt_version drift 체크: `VisionLLMConfig.prompt_version` vs `prompts.PROMPT_VERSION`.
불일치 시 warning — cache 는 어차피 key 에 편입돼 자동 invalidation 되지만, yaml 을
갱신 안 한 실수는 drift 조기 경고.
"""
from __future__ import annotations

import logging
from pathlib import Path

from settings import VisionLLMConfig
from vision.gemini_client import GeminiVisionLLMClient
from vision.llm_cache import LocalJSONCache, VisionLLMCacheBackend
from vision.prompts import PROMPT_VERSION as CODE_PROMPT_VERSION

logger = logging.getLogger(__name__)


def build_vision_llm_client(
    cfg: VisionLLMConfig,
    *,
    cache: VisionLLMCacheBackend | None = None,
) -> GeminiVisionLLMClient:
    """Config → VisionLLMClient 인스턴스.

    cache=None 이면 cfg.cache_dir 기반 LocalJSONCache 를 자동 생성.
    명시적으로 캐시 끄려면 cache=_NoCache() 같은 no-op 을 주입할 것 — None 은 "default" 의미.
    """
    if cfg.prompt_version != CODE_PROMPT_VERSION:
        logger.warning(
            "vision_llm_prompt_drift yaml=%s code=%s — yaml 의 prompt_version 을 "
            "코드의 PROMPT_VERSION 에 맞추면 drift 경고 해소.",
            cfg.prompt_version, CODE_PROMPT_VERSION,
        )

    if cfg.provider != "gemini":
        raise NotImplementedError(
            f"VisionLLMClient provider={cfg.provider!r} not yet implemented. "
            "Only 'gemini' is supported in production. "
            "Use scripts/pilot_llm_bbox.py for azure-openai pilot runs."
        )

    if cache is None:
        cache = LocalJSONCache(
            base_dir=Path(cfg.cache_dir),
            model_id=cfg.model_id,
            prompt_version=cfg.prompt_version,
        )

    return GeminiVisionLLMClient(
        model_id=cfg.model_id,
        prompt_version=cfg.prompt_version,
        cache=cache,
    )
