"""VisionLLMClient Protocol + FakeVisionLLMClient (결정론 테스트용).

Phase 2 — Pipeline B LLM-centric 재설계의 허리. 상위 (pipeline_b_extractor / Phase 4.5
dedup) 는 이 Protocol 에만 의존하고, 실제 모델 (gemini-2.5-flash / gpt-5-mini) swap 은
`--vision-llm` CLI 플래그 / config 로 분기.

top-level import 정책:
- 이 파일은 Protocol + Fake 만 — google.genai / openai / pillow 같은 vision extras 의존 없음.
- 실 구현은 `src/vision/gemini_client.py` 에 둬서 extras 없는 환경에서도 import 가능.

FakeVisionLLMClient 결정론:
- key = sha256(image_bytes + prompt_version) 의 8-byte prefix → random.Random seed
- is_india_ethnic_wear / outfit 수 / bbox / silhouette 를 seed 로부터 generate
- 동일 입력 2회 호출 시 byte-level identical 출력 보장 (cache 테스트 가능)
"""
from __future__ import annotations

import hashlib
import random
from typing import Protocol, runtime_checkable

from contracts.common import Silhouette
from contracts.vision import EthnicOutfit, GarmentAnalysis


@runtime_checkable
class VisionLLMClient(Protocol):
    """image bytes → GarmentAnalysis.

    구현체는 반드시 결정론 (temperature=0 / seed 고정) 이어야 함. 실패는 raise —
    retry 는 구현체 내부 1회 한정 (Phase 0 Gemini JSON 형식 버그 대응), 그 이상은 조용히
    성공한 척하지 않고 예외를 그대로 위로 보낸다.
    """

    def extract_garment(
        self,
        image_bytes: bytes,
        *,
        preset: list[dict[str, str]],
    ) -> GarmentAnalysis: ...


_ETHNIC_WORDS_UPPER = ["kurta", "anarkali", "saree", "ethnic_shirt", "tunic"]
_ETHNIC_WORDS_LOWER = ["palazzo", "churidar", "salwar", "sharara"]
_FABRIC_WORDS = ["cotton", "linen", "silk", "chiffon", "georgette", "rayon", "khadi"]
_TECHNIQUE_WORDS = [
    "chikankari", "block_print", "bandhani", "zardosi",
    "mirror_work", "embroidery", "plain",
]


class FakeVisionLLMClient:
    """테스트 / CI / extras-free 환경용 결정론 vision LLM.

    입력 bytes 의 sha256 에서 seed 를 유도해 GarmentAnalysis 를 합성. 실제 모델 응답과
    의미적으로 유사하진 않지만, 파이프라인 downstream (Phase 4.5 dedup / Phase 3 BBOX
    crop) 이 제대로 돌아가는지 검증하기엔 충분.
    """

    def __init__(self, *, prompt_version: str = "fake-v1") -> None:
        self._prompt_version = prompt_version

    def extract_garment(
        self,
        image_bytes: bytes,
        *,
        preset: list[dict[str, str]],
    ) -> GarmentAnalysis:
        seed = self._seed(image_bytes)
        rng = random.Random(seed)
        is_ethnic = rng.random() > 0.15  # 85% ethnic (실 데이터 분포 근사)
        if not is_ethnic:
            return GarmentAnalysis(is_india_ethnic_wear=False, outfits=[])
        n_outfits = rng.choices([1, 2], weights=[0.8, 0.2], k=1)[0]
        outfits = [self._fake_outfit(rng, preset) for _ in range(n_outfits)]
        return GarmentAnalysis(is_india_ethnic_wear=True, outfits=outfits)

    def _seed(self, image_bytes: bytes) -> int:
        digest = hashlib.sha256(image_bytes + self._prompt_version.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big", signed=False)

    def _fake_outfit(
        self, rng: random.Random, preset: list[dict[str, str]]
    ) -> EthnicOutfit:
        # bbox: 좌상단 [0.1..0.4], 크기 [0.3..0.7], 이미지 밖 안 벗어나게 clamp
        x = round(rng.uniform(0.1, 0.4), 3)
        y = round(rng.uniform(0.05, 0.3), 3)
        w = round(min(1.0 - x, rng.uniform(0.3, 0.7)), 3)
        h = round(min(1.0 - y, rng.uniform(0.4, 0.9)), 3)
        area = round(w * h, 3)
        single = rng.random() < 0.25
        upper = rng.choice(_ETHNIC_WORDS_UPPER)
        lower = None if single else rng.choice(_ETHNIC_WORDS_LOWER)
        silhouette = rng.choice(list(Silhouette) + [None])
        # fabric 은 실 LLM 이 null 많이 낼 걸 반영해 20% null. technique 은 "plain" 포함.
        fabric = None if rng.random() < 0.20 else rng.choice(_FABRIC_WORDS)
        technique = None if rng.random() < 0.15 else rng.choice(_TECHNIQUE_WORDS)
        picks_pool = [p["name"] for p in preset] if preset else ["pool_00", "pool_01", "pool_02"]
        picks = rng.sample(picks_pool, k=min(3, len(picks_pool)))
        return EthnicOutfit(
            person_bbox=(x, y, w, h),
            person_bbox_area_ratio=area,
            upper_garment_type=upper,
            lower_garment_type=lower,
            dress_as_single=single,
            silhouette=silhouette,
            fabric=fabric,
            technique=technique,
            color_preset_picks_top3=picks,
        )
