"""Vision LLM 캐시 Protocol + LocalJSONCache.

Phase 2 목적:
- Gemini 호출 1회당 ~$0.0009 / 12s — 131 post × 재실행 자주 하면 비용/시간 부담.
- 동일 이미지 + 동일 prompt_version + 동일 model_id 면 같은 JSON 을 반환 → 디스크 캐시.

Cache key = sha256(image_bytes + prompt_version + model_id). 프롬프트 / 모델 / 이미지
하나라도 바뀌면 자동 miss — drift 숨김 방지 (필수).

BE 연동 / 데모 시연 시점에 `AzureBlobCache` 로 교체 예정 (동일 Protocol). POC 기간엔
`outputs/llm_cache/` (gitignore 대상) 에 JSON 저장.

디스크 포맷:
{
  "cache_key": "<hex>",
  "model_id": "gemini-2.5-flash",
  "prompt_version": "v0.1",
  "image_sha256": "<hex>",
  "stored_at": "<ISO8601>",
  "garment_analysis": { ... GarmentAnalysis 직렬화 ... }
}

파일명은 `{model_id}/{cache_key}.json` — model 별 디렉토리 분리로 육안 검수 편의.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, runtime_checkable

from contracts.vision import GarmentAnalysis, KMeansAnchoredPickResponse


def compute_cache_key(
    image_bytes: bytes, *, prompt_version: str, model_id: str
) -> str:
    """Cache key — image + prompt_version + model_id 에 sha256."""
    h = hashlib.sha256()
    h.update(image_bytes)
    h.update(b"\x1f")  # separator, 값 경계 모호성 방지
    h.update(prompt_version.encode("utf-8"))
    h.update(b"\x1f")
    h.update(model_id.encode("utf-8"))
    return h.hexdigest()



def compute_v010_cache_key(
    image_bytes: bytes,
    *,
    prompt_version: str,
    model_id: str,
    cluster_hexes: tuple[str, ...],
) -> str:
    """color.B v0.10 — Pass 2 cache key.

    Pass 1 cache key + cluster_top_n hex 시퀀스 (share desc 정렬 후 top-N) 까지 포함.
    같은 image 라도 cluster top-N 이 다르면 (KMeans 결과 변동, e.g. dyn_cfg 변경) 별
    cache.
    """
    h = hashlib.sha256()
    h.update(image_bytes)
    h.update(b"\x1f")
    h.update(prompt_version.encode("utf-8"))
    h.update(b"\x1f")
    h.update(model_id.encode("utf-8"))
    h.update(b"\x1f")
    h.update("|".join(cluster_hexes).encode("utf-8"))
    return h.hexdigest()


@runtime_checkable
class VisionLLMCacheBackend(Protocol):
    """get/put 만 있는 간단한 KV Protocol.

    결정론 / 멱등 — 같은 key 로 put 한 뒤 get 하면 put 한 값이 그대로 나옴.
    삭제는 Protocol 스펙에 넣지 않음 (invalidation 은 key 자체 변경으로).
    """

    def get(self, cache_key: str) -> GarmentAnalysis | None: ...
    def put(self, cache_key: str, analysis: GarmentAnalysis) -> None: ...


class LocalJSONCache:
    """`outputs/llm_cache/{model_id}/{cache_key}.json` 에 저장하는 로컬 캐시."""

    def __init__(self, base_dir: Path, *, model_id: str, prompt_version: str) -> None:
        self._model_dir = base_dir / model_id
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._model_id = model_id
        self._prompt_version = prompt_version

    def get(self, cache_key: str) -> GarmentAnalysis | None:
        path = self._path(cache_key)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        payload = data.get("garment_analysis")
        if payload is None:
            return None
        return GarmentAnalysis.model_validate(payload)

    def put(self, cache_key: str, analysis: GarmentAnalysis) -> None:
        path = self._path(cache_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        envelope = {
            "cache_key": cache_key,
            "model_id": self._model_id,
            "prompt_version": self._prompt_version,
            "stored_at": datetime.now(timezone.utc).isoformat(),
            "garment_analysis": analysis.model_dump(mode="json"),
        }
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(envelope, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        tmp.replace(path)  # atomic — 반쯤 쓰인 파일 남기지 않음

    def _path(self, cache_key: str) -> Path:
        return self._model_dir / f"{cache_key}.json"



@runtime_checkable
class KMeansAnchoredPickCacheBackend(Protocol):
    """color.B v0.10 — Pass 2 결과용 KV cache Protocol.

    `VisionLLMCacheBackend` 와 동형이나 value 가 `KMeansAnchoredPickResponse`. Pass 1
    cache 와 직교 (cache_key 가 cluster_hexes 포함이라 자연 분리).
    """

    def get(self, cache_key: str) -> KMeansAnchoredPickResponse | None: ...
    def put(self, cache_key: str, response: KMeansAnchoredPickResponse) -> None: ...


class LocalJSONCacheV010:
    """color.B v0.10 — `outputs/llm_cache/{model_id}/v010/{cache_key}.json` 저장.

    Pass 1 (`LocalJSONCache`) 와 동일한 base_dir 를 공유하되 v010 subdir 에 분리. envelope
    key 가 `pick_response` 라 Pass 1 의 `garment_analysis` 와 자연 분리.
    """

    def __init__(self, base_dir: Path, *, model_id: str, prompt_version: str) -> None:
        self._model_dir = base_dir / model_id / "v010"
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._model_id = model_id
        self._prompt_version = prompt_version

    def get(self, cache_key: str) -> KMeansAnchoredPickResponse | None:
        path = self._path(cache_key)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        payload = data.get("pick_response")
        if payload is None:
            return None
        return KMeansAnchoredPickResponse.model_validate(payload)

    def put(self, cache_key: str, response: KMeansAnchoredPickResponse) -> None:
        path = self._path(cache_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        envelope = {
            "cache_key": cache_key,
            "model_id": self._model_id,
            "prompt_version": self._prompt_version,
            "stored_at": datetime.now(timezone.utc).isoformat(),
            "pick_response": response.model_dump(mode="json"),
        }
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(envelope, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        tmp.replace(path)

    def _path(self, cache_key: str) -> Path:
        return self._model_dir / f"{cache_key}.json"
