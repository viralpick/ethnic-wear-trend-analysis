"""Pydantic-settings 기반 로컬 설정.

이 모듈은 editable 설치 시 hatchling의 force-include 때문에 site-packages로 복사될 수 있다.
따라서 __file__ 기반 루트 탐색은 신뢰할 수 없으므로, 실행 시점 CWD에서 위로 올라가며
configs/local.yaml 이 있는 디렉토리를 리포 루트로 간주한다 (pytest·pyproject 같은 도구들과
동일한 관례).
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


def _find_repo_root(start: Path | None = None) -> Path:
    origin = (start or Path.cwd()).resolve()
    for candidate in [origin, *origin.parents]:
        if (candidate / "configs" / "local.yaml").exists():
            return candidate
    raise FileNotFoundError(
        "configs/local.yaml not found. Run from the repo root (or an ancestor)."
    )


class Paths(BaseModel):
    sample_data: Path
    outputs: Path


class UnknownSignalsConfig(BaseModel):
    window_days: int = 3
    threshold: int = 10


class LoggingConfig(BaseModel):
    level: str = "INFO"


# --------------------------------------------------------------------------- #
# Step 3-B configs
# --------------------------------------------------------------------------- #

class PipelineConfig(BaseModel):
    collection_start_date: date
    target_date: date | None = None


class ScoringWeights(BaseModel):
    social: float
    youtube: float
    cultural: float
    momentum: float


class LifecycleConfig(BaseModel):
    early_below: float
    growth_until: float
    early_post_count_threshold: int


class InfluencerWeights(BaseModel):
    mega: float
    macro: float
    mid: float
    micro: float


class InfluencerTierThresholds(BaseModel):
    mega: int
    macro: int
    mid: int


class YouTubeFactorWeights(BaseModel):
    video_count: float
    views: float
    view_growth: float


class CulturalFactorWeights(BaseModel):
    festival: float
    bollywood: float


class CulturalFestival(BaseModel):
    name: str
    window_start: date
    window_end: date
    tags: list[str]


class MomentumFactorWeights(BaseModel):
    post_growth: float
    hashtag_velocity: float
    new_account_ratio: float


class DataMaturityConfig(BaseModel):
    bootstrap_below_days: int
    full_from_days: int


class ScoringConfig(BaseModel):
    weights: ScoringWeights
    normalization_method: str = "minmax_same_run"
    direction_threshold_pct: float
    lifecycle: LifecycleConfig
    influencer_weights: InfluencerWeights
    influencer_tier_thresholds: InfluencerTierThresholds
    youtube_factor_weights: YouTubeFactorWeights
    cultural_factor_weights: CulturalFactorWeights
    cultural_festival_boost: float
    cultural_bollywood_bonus: float
    cultural_festivals: list[CulturalFestival] = Field(default_factory=list)
    momentum_factor_weights: MomentumFactorWeights
    momentum_window_days: int
    data_maturity: DataMaturityConfig


class VLMConfig(BaseModel):
    seed: int = 42
    temperature: float = 0.0
    case2_per_cluster_cap: int = 10
    case1_daily_cap: int = 150


class PaletteConfig(BaseModel):
    top_k: int = 5


class ExportConfig(BaseModel):
    summaries_filename: str = "summaries.json"
    enriched_filename: str = "enriched.json"


# --------------------------------------------------------------------------- #
# Vision (color_space Pipeline B — spec §4.1 ④)
# --------------------------------------------------------------------------- #

class SkinLabBox(BaseModel):
    """Skin LAB box 경계 (drop_skin / hex_skin_leak 공용).

    방법론 실험용: segformer 없이 bbox crop 만 쓰면 drop_skin 이 주 방어선이 되어 box 넓이가
    결과에 큰 영향. 인도 skin tone 스펙트럼 전체를 잡으려면 L_min 을 30 까지 내릴지 검토 (TODO).
    """
    min: list[float] = Field(min_length=3, max_length=3)
    max: list[float] = Field(min_length=3, max_length=3)


class ExtractColorsConfig(BaseModel):
    """KMeans 팔레트 추출 파라미터 (extract_colors 호출용)."""
    k: int = 5
    min_pixels: int = 150


class InstanceConfig(BaseModel):
    """phase 3 — (frame × person × garment_class) instance 기반 palette 설정."""
    single_color_max_delta_e: float = 8.0   # instance 내 top chip 간 ΔE 미만이면 단색 판정
    duplicate_max_delta_e: float = 15.0     # instance 간 top-1 chip ΔE 미만이면 같은 옷
    # Literal 로 로드 시점 검증 — 오타 시 runtime ValueError 가 아니라 Settings 초기화 실패.
    weight_formula: Literal["log", "linear", "sqrt"] = "log"


class SceneFilterConfig(BaseModel):
    """M4.I — CLIP zero-shot pre-filter (scene + gender + age).

    enabled=False 이면 NoopSceneFilter 로 매 frame pass. True 이면 CLIPSceneFilter 로드.
    prompts / pass_index / min_confidence 모두 tunable — 인도 ethnic wear bias 대응용.

    pass 조건: (scene argmax == scene_pass_index) AND (argmax confidence >= min_confidence)
    AND (gender argmax == gender_pass_index) AND (age argmax == age_pass_index).
    """
    enabled: bool = False
    model_id: str = "openai/clip-vit-base-patch32"
    # scene argmax 가 이 index 여야 pass. "product only" 제외 — fashion shot 이 옷 중심
    # 구도라 CLIP 이 product 로 오분류 빈발. statue / landscape 만 negative class.
    scene_prompts: list[str] = Field(default_factory=lambda: [
        "a fashion photo of a person wearing clothing",
        "a photo of a statue or mannequin",
        "a photo of a landscape or outdoor scenery",
    ])
    scene_pass_index: int = 0
    # gender argmax 가 이 index 여야 pass. 기본 0 = "woman"
    gender_prompts: list[str] = Field(default_factory=lambda: [
        "a photo of a woman",
        "a photo of a man",
    ])
    gender_pass_index: int = 0
    # age argmax 가 이 index 여야 pass. 10대는 adult 쪽으로 흡수하려 "teenager or adult" 명시.
    age_prompts: list[str] = Field(default_factory=lambda: [
        "a photo of a young child",
        "a photo of a teenager or adult",
    ])
    age_pass_index: int = 1
    # argmax confidence 가 이 값 미만이면 scene_low_confidence drop. 0.3 = 경계 case 는 살림.
    min_confidence: float = 0.3


class VisionConfig(BaseModel):
    skin_lab_box: SkinLabBox
    extract_colors: ExtractColorsConfig = ExtractColorsConfig()
    instance: InstanceConfig = InstanceConfig()
    scene_filter: SceneFilterConfig = SceneFilterConfig()
    # YOLO person detect 실패 시 전체 이미지를 bbox 로 간주. mirror selfie (거울 셀카) 처럼
    # YOLOv8n 이 OOD 로 놓치는 케이스 방어. segformer 가 skin/background/의류 자체적으로 분리하므로
    # 전체 이미지 → segformer 도 의류 pixel 을 제대로 뽑아낸다. IG 에 거울 셀카가 흔해 기본 True.
    # 부작용 주의: product-only shot (사람 없는 상품 단독 이미지) 에서 배경이 의류로 오분류될
    # 수 있음 — 이 경우 false 로 내려서 YOLO 필터 유지.
    fallback_full_image_on_no_person: bool = True
    # drop_skin_adaptive threshold. class pixel 중 skin LAB box 안 비율이 이 값 초과하면
    # "skin-tone 의류" 로 판정하고 원본 pixel 전체 유지 (베이지/탄 kurta 등 보호).
    # 그 미만이면 edge noise 로 간주해 box 안 pixel 만 제거. 0.0~1.0 범위.
    skin_drop_threshold_pct: float = 0.5
    # bbox-level false positive filter — 동상/마네킹/제품샷 오탐 방어.
    # - min_skin_ratio: segformer skin class pixel / crop area. 미만이면 "사람 아님" 판정
    #   drop. 동상은 돌/금속 색이라 skin 감지 거의 0. 0.005 = 0.5% 기본 (전신샷 얼굴/손만
    #   나와도 통과).
    # - max_garment_ratio: crop 면적 중 의류 class 비율. 초과하면 "배경까지 의류로 오분류"
    #   판정 drop. 실제 전신 의류샷도 배경 여유가 있어 보통 60% 이하. 0.90 으로 여유.
    min_skin_ratio_for_person: float = 0.005
    max_garment_ratio_for_person: float = 0.90


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="forbid")

    paths: Paths
    unknown_signals: UnknownSignalsConfig = UnknownSignalsConfig()
    logging: LoggingConfig = LoggingConfig()

    pipeline: PipelineConfig
    scoring: ScoringConfig
    vlm: VLMConfig = VLMConfig()
    palette: PaletteConfig = PaletteConfig()
    export: ExportConfig = ExportConfig()
    vision: VisionConfig

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        yaml_path = _find_repo_root() / "configs" / "local.yaml"
        return (YamlConfigSettingsSource(settings_cls, yaml_file=str(yaml_path)),)

    @model_validator(mode="after")
    def _resolve_paths(self) -> "Settings":
        repo_root = _find_repo_root()
        if not self.paths.sample_data.is_absolute():
            self.paths.sample_data = (repo_root / self.paths.sample_data).resolve()
        if not self.paths.outputs.is_absolute():
            self.paths.outputs = (repo_root / self.paths.outputs).resolve()
        return self


def load_settings() -> Settings:
    return Settings()
