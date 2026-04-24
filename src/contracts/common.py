"""공유 enum과 contract 상수.

이 파일은 모든 contract가 참조하는 closed-set enum과 버전 상수를 소유한다.
LLM/VLM 추출기는 이 enum에 **정확히** 일치하지 않는 값을 절대 저장하지 않는다:
Pydantic이 ValidationError 를 내면 호출자가 해당 속성을 None 으로 떨어뜨린다.
절대 fuzzy matching 이나 값 보정(coercion)을 하지 않는다 (llm-feature-check §4 참조).

StrEnum을 사용하는 이유:
- JSON 직렬화 시 그대로 문자열. Pydantic 이 자동으로 검증.
- 값이 곧 식별자이므로 매핑 테이블과 어휘 일치.
- 새로운 값 추가 시 코드에서 한 곳(enum)만 바꾸면 Pydantic 이 전 contract에 반영.

enum 멤버 순서는 spec §4.1 / §8 / §9 의 원본 리스트 순서와 일치시킨다.
"""
from __future__ import annotations

from enum import StrEnum
from typing import TypeAlias

from pydantic import BaseModel, ConfigDict, Field

# v2 에서 스키마가 깨지는 변경을 넣을 때만 bump. POC 기간엔 0.1.x 유지.
CONTRACT_VERSION: str = "0.1.0"


# --------------------------------------------------------------------------- #
# 8 extracted attributes + embellishment intensity + color family
# --------------------------------------------------------------------------- #

class GarmentType(StrEnum):
    # spec §4.1 ① — 의류 종류
    KURTA_SET = "kurta_set"
    KURTA_DRESS = "kurta_dress"
    CO_ORD = "co_ord"
    ANARKALI = "anarkali"
    STRAIGHT_KURTA = "straight_kurta"
    A_LINE_KURTA = "a_line_kurta"
    TUNIC = "tunic"
    ETHNIC_DRESS = "ethnic_dress"
    CASUAL_SAREE = "casual_saree"
    FUSION_TOP = "fusion_top"
    ETHNIC_SHIRT = "ethnic_shirt"


class Fabric(StrEnum):
    # spec §4.1 ② — 소재
    COTTON = "cotton"
    COTTON_BLEND = "cotton_blend"
    LINEN = "linen"
    LINEN_BLEND = "linen_blend"
    RAYON = "rayon"
    MODAL = "modal"
    CHANDERI = "chanderi"
    GEORGETTE = "georgette"
    CREPE = "crepe"
    CHIFFON = "chiffon"
    KHADI = "khadi"
    POLYESTER_BLEND = "polyester_blend"
    JACQUARD = "jacquard"


class Technique(StrEnum):
    # spec §4.1 ③ — 기법/장식
    SOLID = "solid"
    SELF_TEXTURE = "self_texture"
    CHIKANKARI = "chikankari"
    BLOCK_PRINT = "block_print"
    FLORAL_PRINT = "floral_print"
    GEOMETRIC_PRINT = "geometric_print"
    ETHNIC_MOTIF = "ethnic_motif"
    DIGITAL_PRINT = "digital_print"
    THREAD_EMBROIDERY = "thread_embroidery"
    MIRROR_WORK = "mirror_work"
    SCHIFFLI = "schiffli"
    PINTUCK = "pintuck"
    LACE_CUTWORK = "lace_cutwork"
    GOTA_PATTI = "gota_patti"


class EmbellishmentIntensity(StrEnum):
    # spec §4.1 ③ 보조 플래그 — 장식 강도
    EVERYDAY = "everyday"
    FESTIVE_LITE = "festive_lite"
    HEAVY = "heavy"


class ColorFamily(StrEnum):
    # spec §4.1 ④ — 색 계열
    PASTEL = "pastel"
    EARTH = "earth"
    NEUTRAL = "neutral"
    WHITE_ON_WHITE = "white_on_white"
    JEWEL = "jewel"
    BRIGHT = "bright"
    DUAL_TONE = "dual_tone"
    MULTICOLOR = "multicolor"


class Silhouette(StrEnum):
    # spec §4.1 ⑤ — 실루엣. ANARKALI 가 GarmentType 과 의도적으로 중복됨 (독립 enum).
    STRAIGHT = "straight"
    A_LINE = "a_line"
    FLARED = "flared"
    ANARKALI = "anarkali"
    FIT_AND_FLARE = "fit_and_flare"
    TIERED = "tiered"
    HIGH_LOW = "high_low"
    BOXY = "boxy"
    KAFTAN = "kaftan"
    SHIRT_STYLE = "shirt_style"
    ANGRAKHA = "angrakha"
    EMPIRE = "empire"


class Occasion(StrEnum):
    # spec §4.1 ⑥ — 맥락
    OFFICE = "office"
    CASUAL = "casual"
    CAMPUS = "campus"
    WEEKEND = "weekend"
    FESTIVE_LITE = "festive_lite"
    TRAVEL = "travel"


class StylingCombo(StrEnum):
    # spec §4.1 ⑦ — 스타일링 조합
    WITH_PALAZZO = "with_palazzo"
    WITH_PANTS = "with_pants"
    WITH_CHURIDAR = "with_churidar"
    WITH_DUPATTA = "with_dupatta"
    STANDALONE = "standalone"
    WITH_JACKET = "with_jacket"
    WITH_JEANS = "with_jeans"
    CO_ORD_SET = "co_ord_set"


class BrandTier(StrEnum):
    # spec §4.1 ⑧ — 브랜드 티어 (자유 텍스트 → 후처리 매핑)
    VALUE = "value"
    MID = "mid"
    PREMIUM_EVERYDAY = "premium_everyday"


# --------------------------------------------------------------------------- #
# Extraction / scoring / content source enums
# --------------------------------------------------------------------------- #

class ClassificationMethod(StrEnum):
    # 어떤 방법으로 속성을 추출했는지
    RULE = "rule"
    LLM = "llm"
    VLM = "vlm"


class Direction(StrEnum):
    # spec §9.3 — ▲▼→
    UP = "up"
    DOWN = "down"
    FLAT = "flat"


class LifecycleStage(StrEnum):
    # spec §9.4 — 라이프사이클 단계
    EARLY = "early"
    GROWTH = "growth"
    MATURITY = "maturity"
    DECLINE = "decline"


class DataMaturity(StrEnum):
    """
    purpose: 클러스터 요약 한 장의 베이스라인 데이터 양을 나타내는 플래그
    stage: output
    ownership: analysis-owned
    stability: locked

    BOOTSTRAP: 사용 가능한 히스토리가 3일 미만. weekly_direction / momentum 은 신뢰하지 않는다.
    PARTIAL:   3~6일. weekly 가 약하게 의미를 가지기 시작한다.
    FULL:      7일 이상. 모든 필드가 의미 있는 값.
    """
    BOOTSTRAP = "bootstrap"
    PARTIAL = "partial"
    FULL = "full"


class ContentSource(StrEnum):
    # NormalizedContentItem 의 source tag
    INSTAGRAM = "instagram"
    YOUTUBE = "youtube"


class InstagramSourceType(StrEnum):
    # spec §3.1 A/B/C — Instagram 수집 채널 구분
    INFLUENCER_FIXED = "influencer_fixed"
    HASHTAG_TRACKING = "hashtag_tracking"
    BOLLYWOOD_DECODE = "bollywood_decode"


# --------------------------------------------------------------------------- #
# Shared shapes
# --------------------------------------------------------------------------- #

DistributionMap: TypeAlias = dict[str, float]
"""속성 분포 맵. key 는 enum 문자열, value 는 0~1 비율."""


class ColorPaletteItem(BaseModel):
    """
    purpose: 드릴다운 팔레트 한 칸. RGB 대표값을 HEX 로 표시 (spec §4.1 ④, §8.2)
    stage: output
    ownership: analysis-owned
    stability: locked
    """
    model_config = ConfigDict(frozen=True)

    r: int
    g: int
    b: int
    hex_display: str  # 예: "#B8D4C3"
    name: str
    family: ColorFamily
    pct: float


class PaletteCluster(BaseModel):
    """
    purpose: 3층 palette (canonical / post / cluster) 의 한 칸. contracts 경계용
    pydantic — numpy-free. rgb/lab 내부 CV 값은 src/vision/dynamic_palette.py 의
    dataclass 에만 보존.
    stage: output
    ownership: analysis-owned
    stability: evolving (Color 파이프라인 3층 재설계, 2026-04-24)

    share: 같은 레벨 내 합 = 1.0 (canonical/post/cluster 모두).
    family: dynamic_palette 가 pixel 기반이라 family 미매핑 가능 (B 단계 adapter 에서
        color_preset_picks_top3 / LAB rule fallback 으로 채움).
    """
    model_config = ConfigDict(frozen=True)

    hex: str
    share: float = Field(ge=0.0, le=1.0)
    family: ColorFamily | None = None
