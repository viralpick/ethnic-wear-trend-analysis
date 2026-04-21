"""Pydantic-settings 기반 로컬 설정.

이 모듈은 editable 설치 시 hatchling의 force-include 때문에 site-packages로 복사될 수 있다.
따라서 __file__ 기반 루트 탐색은 신뢰할 수 없으므로, 실행 시점 CWD에서 위로 올라가며
configs/local.yaml 이 있는 디렉토리를 리포 루트로 간주한다 (pytest·pyproject 같은 도구들과
동일한 관례).
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

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


class VisionConfig(BaseModel):
    skin_lab_box: SkinLabBox
    extract_colors: ExtractColorsConfig = ExtractColorsConfig()
    # YOLO person detect 실패 시 전체 이미지를 bbox 로 간주. mirror selfie (거울 셀카) 처럼
    # YOLOv8n 이 OOD 로 놓치는 케이스 방어. segformer 가 skin/background/의류 자체적으로 분리하므로
    # 전체 이미지 → segformer 도 의류 pixel 을 제대로 뽑아낸다. IG 에 거울 셀카가 흔해 기본 True.
    # 부작용 주의: product-only shot (사람 없는 상품 단독 이미지) 에서 배경이 의류로 오분류될
    # 수 있음 — 이 경우 false 로 내려서 YOLO 필터 유지.
    fallback_full_image_on_no_person: bool = True


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
