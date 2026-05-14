"""OpenAI 직호출 backup LLM 클라이언트.

`AzureOpenAILLMClient` 와 dispatch sibling — Azure deployment 가 막힌 기간용. parent 의
extract_attributes / _extract_batch 그대로 재사용하고 SDK 초기화만 override. parent 가
사용하는 `self._deployment` 는 OpenAI model id 로 alias (호환 유지).

환경변수 (.env):
  OPENAI_API_KEY (필수)
  OPENAI_MODEL (선택, default: gpt-5-mini)

top-level import 로 openai / python-dotenv 사용. `[llm]` extras 미설치 환경에서는
이 모듈 import 자체가 ImportError — core 는 절대 top-level import 하지 말 것.
"""
from __future__ import annotations

import os

import openai
from dotenv import load_dotenv

from attributes.azure_openai_llm_client import AzureOpenAILLMClient

_DEFAULT_MODEL = "gpt-5-mini"


class OpenAILLMClient(AzureOpenAILLMClient):
    """OpenAI (direct API) backup provider. AzureOpenAILLMClient 와 동일 정책 —
    SDK init 만 override.
    """

    def __init__(
        self, batch_size: int = 10, seed: int = 42, max_workers: int = 1,
    ) -> None:
        # parent __init__ 우회 — Azure SDK init 시도 막기 위해 super() 호출 안 함.
        load_dotenv()
        self._client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        # parent `_extract_batch` 가 `self._deployment` 를 model id 로 사용 — alias.
        self._deployment = os.environ.get("OPENAI_MODEL", _DEFAULT_MODEL)
        self._batch_size = batch_size
        self._seed = seed
        self._max_workers = max(1, max_workers)

    @classmethod
    def from_env(
        cls, batch_size: int = 10, max_workers: int = 1,
    ) -> "OpenAILLMClient":
        return cls(batch_size=batch_size, max_workers=max_workers)
