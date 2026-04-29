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
    # M3.F — brand registry JSON 경로. None 이면 brand 추출 skip (backwards-compat).
    brand_registry: Path | None = None


class UnknownSignalsConfig(BaseModel):
    window_days: int = 3
    threshold: int = 10


class NormalizationConfig(BaseModel):
    """Raw → NormalizedContentItem 단계의 파생 분류 규칙.

    M3.E 하울 파생: post.hashtags 중 이 리스트와 겹치는 게 있으면 HASHTAG_TRACKING →
    HASHTAG_HAUL 로 승격 (raw source_type 은 건드리지 않고 normalized 계층에서만 반영).
    모두 lowercase, `#` 없이 저장.
    """
    haul_tags: list[str] = Field(default_factory=list)


class LoggingConfig(BaseModel):
    level: str = "INFO"


# --------------------------------------------------------------------------- #
# Step 3-B configs
# --------------------------------------------------------------------------- #

class PipelineConfig(BaseModel):
    collection_start_date: date
    target_date: date | None = None
    # StarRocks 로드 모드: count(갯수 기준) | date(기간 기준)
    window_mode: str = "count"
    page_size: int = 200      # count 모드: 배치당 포스트 수
    window_days: int = 30     # date 모드: target_date 기준 N일 이내


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


class SourceTypeWeights(BaseModel):
    """M3.E — ig_source_type 별 Social score multiplier.

    Social weighted_engagement 계산: engagement_raw × influencer_weight × source_type_weight.
    haul > hashtag > bollywood > profile 순서 (roadmap M3.E, "Bollywood=seed /
    Haul=검증 / YouTube=설명" ver2 방향).
    """
    influencer_fixed: float = 1.0
    hashtag_tracking: float = 1.2
    hashtag_haul: float = 1.5
    bollywood_decode: float = 1.1


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
    source_type_weights: SourceTypeWeights = SourceTypeWeights()
    youtube_factor_weights: YouTubeFactorWeights
    cultural_factor_weights: CulturalFactorWeights
    cultural_festival_boost: float
    cultural_bollywood_bonus: float
    cultural_festivals: list[CulturalFestival] = Field(default_factory=list)
    momentum_factor_weights: MomentumFactorWeights
    momentum_window_days: int
    new_account_window_days: int = 90
    data_maturity: DataMaturityConfig


class VLMConfig(BaseModel):
    seed: int = 42
    temperature: float = 0.0
    # M3.G/H 이후 IG/YT 둘 다 vision 흐름에 진입. cap 은 source 별 분리축 — YT 단가가
    # 더 비쌈 (영상 1건 ≈ frame 20장 = Gemini 호출 20회, IG 1건 ≈ image 1~10장).
    # 분리 cap 으로 IG 처리량 보존하면서 YT 비용 상한 명시.
    case2_per_cluster_cap_ig: int = 10
    case2_per_cluster_cap_yt: int = 1
    case1_daily_cap_ig: int = 150
    case1_daily_cap_yt: int = 10
    # ζ (2026-04-28): Case2 picking share threshold. trend_cluster_shares 의 share 가
    # 이 값 미만이면 picking 후보에서 제외. 0.0 = 모든 fan-out cluster 후보 (cap 자연
    # cutoff), 0.10 = G(0.6,0.4)×T(0.5,0.5)×F(0.5,0.5) 분포의 자연 lower bound 대응.
    case2_picking_min_share: float = 0.10


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


class DynamicPaletteConfig(BaseModel):
    """Phase 4 canonical outfit 동적 k palette (roadmap §80).

    pipeline: initial_k KMeans(LAB) → greedy ΔE76 pair-merge (threshold=10) →
    share<min_cluster_share drop. 최종 k ∈ {0..initial_k}.

    roadmap spec 과 default 고정: 튜닝은 Phase 5 smoke (131 post) 이후. min_pixels 는
    기존 extract_colors 와 동일 기준 — pool 이 너무 작으면 K-means 의미 없음.
    """
    initial_k: int = 5
    merge_deltae76_threshold: float = 10.0
    min_cluster_share: float = 0.05
    min_pixels: int = 150


class HybridPaletteConfig(BaseModel):
    """Phase 1 per-object β-hybrid + Phase 3 통합 weighted KMeans 의 임계 묶음.

    Phase 1 (vision/hybrid_palette.py) 의 R3 drop / R1 anchor merge 양쪽이 같은 ΔE76
    임계를 공유 — pick 과 cluster LAB 의 매칭 임계 ("pick_match" 네이밍).

    R2 재설계 (2026-04-26):
      - 원색 + share ≥ r2_min_share 면 독립 보존 (Gemini 가 놓친 진짜 색).
      - 원색 아닌 cluster 는 anchor 로 머지 — Δh ≤ hue_near_deg 면 양방향 OK,
        그 외에는 "음영의 정의" 강제 (anchor.L > cluster.L AND anchor.chroma > cluster.chroma).
      - 머지 후보 중 ΔE76 가 r2_merge_deltae76 이내 만 살린다.
      - 머지 못한 잔여는 etc bucket → aggregator 가 cut_off_share 에 합산.

    default 는 vision/hybrid_palette.py 의 모듈 상수 (R3_DROP_DELTAE76=28.0,
    R2_MIN_SHARE=0.10, CHROMA_VIVID=15.0, HUE_NEAR_DEG=30.0, R2_MERGE_DELTAE76=40.0) 와
    일치. settings 가 vision 을 import 하면 격리 규칙 위반이라 hardcode. drift 방지는
    pinning 테스트 (`test_hybrid_config_default_matches_module_const`).

    2026-04-27: pick_match 25→28 (rani_pink ΔE76=26.2 보더라인 살리려고 사용자 결정).
    """
    pick_match_deltae76: float = 28.0
    r2_min_share: float = 0.10
    chroma_vivid: float = 15.0
    hue_near_deg: float = 30.0
    r2_merge_deltae76: float = 40.0
    top_n: int = 3


class InstanceConfig(BaseModel):
    """phase 3 — (frame × person × garment_class) instance 기반 palette 설정."""
    single_color_max_delta_e: float = 8.0   # instance 내 top chip 간 ΔE 미만이면 단색 판정
    duplicate_max_delta_e: float = 15.0     # instance 간 top-1 chip ΔE 미만이면 같은 옷
    # Literal 로 로드 시점 검증 — 오타 시 runtime ValueError 가 아니라 Settings 초기화 실패.
    weight_formula: Literal["log", "linear", "sqrt"] = "log"


class SceneFilterConfig(BaseModel):
    """M3.A Step D Phase 1 — CLIP zero-shot 2-stage pre-filter.

    enabled=False 이면 NoopSceneFilter 로 매 frame pass. True 이면 CLIPSceneFilter 로드.
    prompts / threshold 전부 tunable — 인도 ethnic wear bias 대응.

    **Stage 1 pass 조건 (argmax 완화, 2026-04-24 Phase 1 재설계)**:
      scene[pass_index] >= scene_min_pass_score AND
      woman_score >= stage1_female_min AND
      adult_score >= stage1_adult_min.
    여성이 frame 에 있어도 전체 argmax 가 남성 쪽으로 몰리는 케이스 (남성 비중이 더 큰
    mix shot) 를 살리기 위해 argmax 기준을 버리고 softmax threshold 로 전환.

    **Stage 2 trigger**: Stage 1 통과 후 man_score >= stage2_mix_threshold OR
    child_score >= stage2_mix_threshold 면 pipeline_b_extractor 가 classify_persons 호출 —
    YOLO person BBOX 별 CLIP forward 로 female+adult BBOX 만 keep.
    """
    enabled: bool = False
    model_id: str = "openai/clip-vit-base-patch32"
    # scene 판정: pass prompt (index=scene_pass_index) 의 softmax score 가 min_pass_score
    # 이상이면 pass. argmax 조건 X — 여성+옷 야외샷이 "landscape" 쪽에 argmax 이어도 fashion
    # 쪽 점수가 임계값 이상이면 살린다. "product only" 는 negative class 에서 제외 — fashion
    # shot 이 옷 중심 구도라 product 와 혼동 빈발.
    scene_prompts: list[str] = Field(default_factory=lambda: [
        "a fashion photo of a person wearing clothing",
        "a photo of a statue or mannequin",
        "a photo of a landscape or outdoor scenery",
    ])
    scene_pass_index: int = 0
    # gender_prompts[female_index] 가 woman. stage1 / stage2 공통 재사용.
    gender_prompts: list[str] = Field(default_factory=lambda: [
        "a photo of a woman",
        "a photo of a man",
    ])
    female_index: int = 0  # stage1_female_min / stage2_female_min 검사 대상 index
    male_index: int = 1    # stage2_mix_threshold 검사 대상 index
    # age_prompts[adult_index] 가 adult. "teenager or adult" 로 10대 흡수.
    age_prompts: list[str] = Field(default_factory=lambda: [
        "a photo of a young child",
        "a photo of a teenager or adult",
    ])
    child_index: int = 0  # stage2_mix_threshold 검사 대상 index
    adult_index: int = 1  # stage1_adult_min / stage2_adult_min 검사 대상 index
    # scene pass prompt 의 softmax score 가 이 값 이상이면 pass. 낮을수록 drop 감소.
    # 0.20 = fashion 쪽이 20% 이상이면 살림 — 야외 배경 / product-like 구도에서도 fashion
    # 신호가 조금이라도 있으면 통과 (sample 131 post 기반 튜닝).
    scene_min_pass_score: float = 0.20
    # Stage 1 여성/성인 signal softmax 최소값. argmax 완화 기준 (2026-04-24 Phase 1).
    # 0.30 = 여성이 30% 이상으로 잡히면 존재 간주. 낮출수록 false-pass 증가.
    stage1_female_min: float = 0.30
    stage1_adult_min: float = 0.30
    # Stage 2 trigger: man 또는 child signal 이 이 값 이상이면 BBOX 재판정 필요.
    # 0.30 = 남성/아동 signal 이 30% 이상일 때만 Stage 2 활성.
    stage2_mix_threshold: float = 0.30
    # Stage 2 에서 BBOX 판정 시 통과 임계값. stage1 과 동일 기준 default.
    stage2_female_min: float = 0.50  # BBOX 단일 인물 전제 → stage1 보다 높게
    stage2_adult_min: float = 0.50
    # Stage 2 전체 토글. False 면 stage=stage1_mix_needs_stage2 여도 classify_persons 호출 X
    # → bbox 전부 keep (Stage 1 만으로 운영).
    stage2_enabled: bool = True
    # Stage 2 BBOX crop 의 짧은 변이 이 값 미만이면 CLIP embedding noise 우려로 skip (drop).
    # CLIP ViT-B/32 는 224 resize 내부적으로 수행하지만 너무 작은 crop 은 embedding 불안정.
    min_person_bbox_side_px: int = 80


class VideoFrameConfig(BaseModel):
    """영상 frame sampling/scoring 파라미터 (IG Reel / IG carousel video / YT 영상).

    `vision/video_frame_selector.py::VideoFrameSelectorConfig` 와 1:1 매핑. 영상 1건 당
    n_candidate 균등 sampling → quality score (Laplacian + brightness gate) → scene
    diversity NMS (HSV H+S corr) → top n_final.
    """
    n_candidate: int = 50
    n_final: int = 20
    blur_min: float = 100.0
    brightness_range: tuple[float, float] = (30.0, 225.0)
    # scene change 임계 (cv2.HISTCMP_CORREL ∈ [-1,1]). 0.85 = 강한 컷 차단 + 같은 의상
    # 다른 각도는 통과 (fashion 영상 typical).
    scene_corr_max: float = 0.85
    histogram_bins: int = 32


class VisionConfig(BaseModel):
    skin_lab_box: SkinLabBox
    extract_colors: ExtractColorsConfig = ExtractColorsConfig()
    dynamic_palette: DynamicPaletteConfig = DynamicPaletteConfig()
    hybrid_palette: HybridPaletteConfig = HybridPaletteConfig()
    instance: InstanceConfig = InstanceConfig()
    scene_filter: SceneFilterConfig = SceneFilterConfig()
    video_frame: VideoFrameConfig = VideoFrameConfig()
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
    # drop_skin_adaptive upper ceiling. 이 비율 초과 시 segment 통째 drop (segformer 가
    # 팔/다리를 upper-clothes/pants 로 오분류한 케이스 방어). 실제 베이지 kurta 는 shadow/
    # 주름으로 97% 를 넘기기 어려워 false positive 위험 낮음. smoke 결과 보며 0.95~0.98
    # 범위에서 튜닝.
    skin_drop_upper_ceiling: float = 0.97
    # drop_skin_adaptive_spatial — skin class mask 를 N pixel dilate 한 "인접 zone" 만
    # drop 후보로. 옷 내부 패턴(skin-tone 자수/음영)은 보존, 경계 skin leak 만 제거.
    # iterations 0 이면 spatial 방어 비활성 = pixel-list 분기와 동일. 기본 4 (크롭 scale 기준).
    skin_dilate_iterations: int = 4
    # bbox-level false positive filter — 동상/마네킹/제품샷 오탐 방어.
    # - min_skin_ratio: segformer skin class pixel / crop area. 미만이면 "사람 아님" 판정
    #   drop. 동상은 돌/금속 색이라 skin 감지 거의 0. 0.005 = 0.5% 기본 (전신샷 얼굴/손만
    #   나와도 통과).
    # - max_garment_ratio: crop 면적 중 의류 class 비율. 초과하면 "배경까지 의류로 오분류"
    #   판정 drop. 실제 전신 의류샷도 배경 여유가 있어 보통 60% 이하. 0.90 으로 여유.
    min_skin_ratio_for_person: float = 0.005
    max_garment_ratio_for_person: float = 0.90
    # Phase 3 canonical_extractor — person_bbox_area_ratio 가 이 값 미만인 outfit drop.
    # LLM 이 배경에 작게 잡힌 "남성 착장 + 여성 배경" 같은 컷에서 배경 인물 제거용.
    # 0.10 = 이미지 면적의 10% 미만이면 주 피사체 아님 — Phase 4.5 dedup 전에 걸러냄.
    min_person_bbox_area_ratio: float = 0.10


class VisionLLMConfig(BaseModel):
    """Phase 2 VisionLLMClient (Gemini / gpt-5-mini) 설정.

    yaml 의 prompt_version 은 `src/vision/prompts.py::PROMPT_VERSION` 과 일치해야 함
    (캐시 키에 편입 — drift 시 자동 invalidation).
    """
    provider: Literal["gemini", "azure-openai"] = "gemini"
    model_id: str = "gemini-2.5-flash"
    prompt_version: str = "v0.3"
    cache_dir: Path = Path("outputs/llm_cache")


class OutfitDedupConfig(BaseModel):
    """Phase 4.5 intra-post outfit dedup 설정.

    similarity = Σ weight_i * match_i (match 는 signal 별 0/1 binary). 합 ≥ threshold 면 병합.
    Phase 0 결정 (project_phase0_gemini_adopted.md) 따라 우선순위:
    color_preset → color_family → garment_type → technique. silhouette / fabric 은 참고용.

    실 weight/threshold 튜닝은 Phase 5 full smoke (131 post) 후 진행 — 초기값은 경험 가정.
    """
    color_preset_weight: float = 0.40
    color_family_weight: float = 0.25
    garment_type_weight: float = 0.25
    technique_weight: float = 0.10
    threshold: float = 0.60
    # 같은 image 안 2-outfit 은 보통 다른 사람 — merge 금지 (안전 가드).
    same_image_merge: bool = False
    preset_path: Path = Path("outputs/color_preset/color_preset.json")

    @model_validator(mode="after")
    def _weights_sum_to_one(self) -> "OutfitDedupConfig":
        total = (
            self.color_preset_weight + self.color_family_weight
            + self.garment_type_weight + self.technique_weight
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"outfit_dedup weights 합은 1.0 이어야 함 — 현재 sum={total:.6f}. "
                "yaml 이나 기본값 확인."
            )
        return self


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="forbid")

    paths: Paths
    unknown_signals: UnknownSignalsConfig = UnknownSignalsConfig()
    logging: LoggingConfig = LoggingConfig()
    normalization: NormalizationConfig = NormalizationConfig()

    pipeline: PipelineConfig
    scoring: ScoringConfig
    vlm: VLMConfig = VLMConfig()
    palette: PaletteConfig = PaletteConfig()
    export: ExportConfig = ExportConfig()
    vision: VisionConfig
    vision_llm: VisionLLMConfig = VisionLLMConfig()
    outfit_dedup: OutfitDedupConfig = OutfitDedupConfig()

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
        if self.paths.brand_registry is not None and not self.paths.brand_registry.is_absolute():
            self.paths.brand_registry = (repo_root / self.paths.brand_registry).resolve()
        return self


def load_settings() -> Settings:
    return Settings()
