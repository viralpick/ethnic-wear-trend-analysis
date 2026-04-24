"""Vision LLM (Gemini / gpt-5-mini) 추출 결과 contract — Phase 2.

`VisionLLMClient.extract_garment(image_bytes) -> GarmentAnalysis` 반환 타입.
pilot v0.1 (scripts/pilot_llm_bbox.py) 의 JSON schema 와 1:1 매칭.

설계 원칙:
- silhouette 는 Silhouette enum 으로 강제 (LLM 이 enum 밖 값 내면 Pydantic ValidationError).
- upper/lower garment_type / fabric / technique 은 free-form single lowercase word.
  (kurta/saree/salwar, cotton/silk/linen, chikankari/block_print 등). Phase 4.5 dedup 에서
  별도 매핑. GarmentType / 기타 post-level text attribute enum 과 어휘가 달라 직접 바인드
  하지 않는다.
- fabric / technique 는 Phase 0 dedup 기준에 포함된 필드. prompts v0.3 (2026-04-24) 에서
  schema 추가. Phase 2 scaffolding 당시 누락돼 있던 drift 복구.
- person_bbox 는 [x, y, w, h] normalized [0..1] 좌표, 면적 ratio = w*h.
- color_preset_picks_top3 는 50-color preset 이름만 허용 (free-form hex 금지).
  validate 는 상위 layer (preset 로드 시점) 에서 — contract 에는 단순 list[str].

LLM 이 JSON 형식 자체를 깨뜨리면 (Phase 0 에서 관측한 trailing `}` 누락 등) 상위에서
ValidationError 를 raise 시켜 명시 실패. retry 는 client 1회 한정 (spec §4 timeout 가드).
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from contracts.common import Silhouette


class EthnicOutfit(BaseModel):
    """단일 person BBOX 에 대한 outfit 분석 결과.

    two-piece (upper+lower) / single-piece (saree, lehenga as single, ethnic_dress)
    를 모두 커버. single-piece 시 `dress_as_single=true`, `lower_garment_type=None`.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    person_bbox: tuple[float, float, float, float] = Field(
        description="[x, y, w, h] normalized [0..1], top-left origin"
    )
    person_bbox_area_ratio: float = Field(
        ge=0.0, le=1.0,
        description="w * h, [0..1]. Phase 3 size drop threshold 대상.",
    )
    upper_garment_type: str | None = Field(
        default=None,
        description="single lowercase word (kurta/saree/anarkali/sherwani 등)",
    )
    lower_garment_type: str | None = Field(
        default=None,
        description="single lowercase word (palazzo/churidar/salwar 등). "
        "dress_as_single=True 면 None.",
    )
    dress_as_single: bool = Field(
        default=False,
        description="saree drape / lehenga-choli-as-single / ethnic_dress 여부.",
    )
    silhouette: Silhouette | None = Field(
        default=None,
        description="Silhouette enum 12종 중 하나 또는 null.",
    )
    fabric: str | None = Field(
        default=None,
        description="single lowercase word (cotton/linen/silk/chiffon/georgette 등). "
        "prompts v0.3+ 에서 LLM 이 채움. dedup 은 참고용 (가중치 0, log only).",
    )
    technique: str | None = Field(
        default=None,
        description="single lowercase word (chikankari/block_print/bandhani/zardosi 등). "
        "'plain' 은 장식 없음을 명시. unknown 은 null. dedup 가중치 有.",
    )
    color_preset_picks_top3: list[str] = Field(
        default_factory=list,
        min_length=0, max_length=3,
        description="50-color preset 이름 (pool_NN 또는 self name). free-form hex 금지.",
    )

    @field_validator("person_bbox")
    @classmethod
    def _validate_bbox(
        cls, v: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        x, y, w, h = v
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError(f"bbox origin out of [0,1]: x={x} y={y}")
        if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            raise ValueError(f"bbox size out of (0,1]: w={w} h={h}")
        if x + w > 1.0001 or y + h > 1.0001:
            raise ValueError(f"bbox exceeds image: x+w={x+w} y+h={y+h}")
        return v


class GarmentAnalysis(BaseModel):
    """이미지 1 장에 대한 vision LLM 전체 응답.

    is_india_ethnic_wear=False 면 outfits 는 빈 list 허용.
    True 면 최소 1 outfit, 최대 2 (top-2 by size, 배경 인물 컷).
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    is_india_ethnic_wear: bool
    outfits: list[EthnicOutfit] = Field(
        default_factory=list,
        max_length=2,
        description="top-2 outfits by person_bbox_area_ratio.",
    )


class OutfitMember(BaseModel):
    """Phase 4.5 dedup trace — canonical 1개로 병합된 원본 outfit 의 위치 정보.

    image_id 는 post 안의 이미지 식별자 (IG carousel index 또는 URL) — 같은 image 내
    outfit 은 절대 병합되지 않으므로 (image_id, outfit_index) 쌍은 canonical 내에서 unique.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    image_id: str
    outfit_index: int = Field(ge=0, description="GarmentAnalysis.outfits 내 원본 index")
    person_bbox: tuple[float, float, float, float]


class CanonicalOutfit(BaseModel):
    """Phase 4.5 dedup 결과 — 같은 post 안에서 "동일 의상" 으로 판정된 outfit 군의 대표.

    representative 는 members 중 가장 큰 person_bbox_area_ratio 를 가진 원본. members 는
    Phase 3 BBOX crop 에서 모두 사용 (union pool → segformer → hex). dedup key 는 post +
    canonical_index 로 aggregation 계층에서 식별.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    canonical_index: int = Field(ge=0, description="post 내 canonical 순번 (0부터).")
    representative: EthnicOutfit = Field(
        description="가장 큰 person_bbox_area_ratio 를 가진 member 의 EthnicOutfit."
    )
    members: list[OutfitMember] = Field(
        min_length=1,
        description="병합된 원본 outfit 들의 trace. 최소 1 (자기 자신).",
    )
