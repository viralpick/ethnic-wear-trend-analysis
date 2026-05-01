"""PipelineBColorExtractor — canonical path 기반 ColorExtractor 구현체 (Phase 5).

흐름:
  post.image_urls
    → _resolve_local_paths / _download_blob_paths (파일 시스템 / Azure Blob)
    → _load_images (path → (image_id, bytes, rgb))
    → VisionLLMClient.extract_garment(bytes, preset=llm_preset)
    → extract_canonical_pixels_per_object → list[(CanonicalOutfit, list[ObjectPool])]
    → canonical 별 β-hybrid: per-object build_object_palette (Phase 1+2)
      → aggregate_canonical_palette (Phase 3 통합 weighted KMeans + cut_off_share)
      → palette + cut_off_share 동시 주입
    → ColorExtractionResult(canonicals=[...], post_palette=[...])

설계 결정 (project_phase5_adapter_design.md 참조):
  - `_load_images` 를 module-level 로 분리 — 테스트가 monkeypatch 로 Path/PIL 의존 없이 stub.
  - preset.json 은 factory 에서 1회 로드 (`load_preset_views`) — 3 view (llm_preset /
    matcher_entries / family_map) 로 adapter 에 주입. adapter 안에서 I/O 재호출 X.
  - SceneFilter 는 `__init__` 별도 인자 (Phase 2, 2026-04-25). v2 (adult-woman-only):
    stage1_reject → image skip / stage1_mix_needs_stage2 → YOLO + classify_persons gate /
    stage1_pass + disabled → 풀 이미지 Gemini 1회. `bundle.scene_filter` 는 legacy
    `extract_instances` 경로 전용 — canonical path 는 `self._scene_filter` 만 본다.
  - per-image LLM 실패는 log-and-skip, post 내 모든 이미지가 실패하면 빈 결과.

top-level 로 vision extras (torch/transformers/ultralytics) 를 필요로 하는
pipeline_b_extractor 를 import 한다 — core 는 `import vision.pipeline_b_adapter` 를 절대
top-level 하지 말 것 (run_daily_pipeline 의 `_select_extractor` 가 lazy import 로 격리).

이미지 소스 우선순위:
1. absolute local path 가 존재하면 그대로 사용
2. image_root 가 주어지면 URL basename 으로 로컬 스캔
3. blob_downloader + blob_cache_dir 가 주어지면 Azure Blob 에서 다운로드
4. 위 모두 실패 시 해당 post skip (빈 결과)
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from contracts.common import ColorFamily
from contracts.normalized import NormalizedContentItem
from contracts.vision import CanonicalOutfit, GarmentAnalysis, OutfitMember
from settings import (
    HybridPaletteConfig,
    OutfitDedupConfig,
    VideoFrameConfig,
    VisionConfig,
)
from utils.logging import get_logger
from vision.canonical_extractor import (
    ObjectPool,
    extract_canonical_pixels_per_object,
)
from vision.canonical_palette_aggregator import (
    aggregate_canonical_palette,
    finalize_object_palette,
)
from vision.color_extractor import ColorExtractionResult
from vision.color_family_preset import MatcherEntry
from vision.frame_source import VideoFrameSource
from vision.hybrid_palette import build_object_palette
from vision.llm_client import VisionLLMClient
from vision.pipeline_b_extractor import SegBundle, detect_people
from vision.post_palette import build_post_palette
from vision.scene_filter import NoopSceneFilter, SceneFilter
from vision.video_frame_selector import VideoFrameSelectorConfig

if TYPE_CHECKING:
    from loaders.blob_downloader import BlobDownloader

logger = get_logger(__name__)

_IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"})
_VIDEO_EXTS = frozenset({".mp4", ".mov", ".webm", ".m4v"})


def _to_selector_cfg(cfg: VideoFrameConfig) -> VideoFrameSelectorConfig:
    """Pydantic VideoFrameConfig → frozen dataclass VideoFrameSelectorConfig.

    Pydantic v3 migration 안전 — `model_dump()` 보다 명시적 속성 mapping.
    """
    return VideoFrameSelectorConfig(
        n_candidate=cfg.n_candidate,
        n_final=cfg.n_final,
        blur_min=cfg.blur_min,
        brightness_range=cfg.brightness_range,
        scene_corr_max=cfg.scene_corr_max,
        histogram_bins=cfg.histogram_bins,
    )


def _resolve_local_paths(
    item: NormalizedContentItem, image_root: Path | None
) -> list[Path]:
    """image_urls 를 로컬 Path 리스트로 변환."""
    out: list[Path] = []
    for url in item.image_urls:
        if Path(url).suffix.lower() not in _IMAGE_EXTS:
            continue
        as_path = Path(url)
        if as_path.is_absolute() and as_path.exists():
            out.append(as_path)
            continue
        if image_root is not None:
            candidate = image_root / Path(url).name
            if candidate.exists():
                out.append(candidate)
    return out


def _download_blob_paths(
    item: NormalizedContentItem,
    downloader: BlobDownloader,
    cache_dir: Path,
) -> list[Path]:
    """image_urls 를 Azure Blob 에서 cache_dir 로 다운로드."""
    out: list[Path] = []
    for url in item.image_urls:
        if Path(url).suffix.lower() not in _IMAGE_EXTS:
            continue
        result = downloader.download(url, cache_dir)
        if result is not None:
            out.append(result)
    return out


def _resolve_local_video_paths(
    item: NormalizedContentItem, image_root: Path | None
) -> list[Path]:
    """video_urls 를 로컬 Path 리스트로 변환. SAS query string 제거 후 확장자 검사."""
    out: list[Path] = []
    for url in item.video_urls:
        path_only = url.split("?", 1)[0]
        if Path(path_only).suffix.lower() not in _VIDEO_EXTS:
            continue
        as_path = Path(path_only)
        if as_path.is_absolute() and as_path.exists():
            out.append(as_path)
            continue
        if image_root is not None:
            candidate = image_root / as_path.name
            if candidate.exists():
                out.append(candidate)
    return out


def _download_blob_video_paths(
    item: NormalizedContentItem,
    downloader: BlobDownloader,
    cache_dir: Path,
) -> list[Path]:
    """video_urls 를 Azure Blob 에서 cache_dir 로 다운로드."""
    out: list[Path] = []
    for url in item.video_urls:
        path_only = url.split("?", 1)[0]
        if Path(path_only).suffix.lower() not in _VIDEO_EXTS:
            continue
        result = downloader.download(url, cache_dir)
        if result is not None:
            out.append(result)
    return out


def _encode_jpeg_deterministic(rgb: np.ndarray) -> bytes:
    """RGB ndarray → JPEG bytes (결정론).

    VisionLLMClient `LocalJSONCache` key = sha256(image_bytes + ...). 같은 RGB →
    같은 bytes 보장 필수. PIL 기본값이 라이브러리 버전마다 달라질 수 있으므로 5개
    파라미터 모두 명시 핀:
    - quality=95: 충분한 화질, 4:4:4 sampling 과 함께 LLM input 으로 안정적
    - optimize=False: Huffman 테이블 재계산 → bit-level 비결정 차단
    - subsampling=0: 4:4:4 (chroma 보존), 2:0 보다 ΔE 안정
    - progressive=False: baseline DCT → encoder 분기 일관성
    - dpi/exif/icc_profile 미전달 → 라이브러리 기본 metadata 주입 차단
    """
    from PIL import Image  # lazy

    img = Image.fromarray(rgb, mode="RGB")
    buf = BytesIO()
    img.save(
        buf,
        "JPEG",
        quality=95,
        optimize=False,
        subsampling=0,
        progressive=False,
    )
    return buf.getvalue()


def _load_video_frames(
    paths: list[Path], cfg: VideoFrameConfig
) -> list[tuple[str, bytes, np.ndarray]]:
    """video Path 리스트 → [(image_id={stem}_f{idx}, bytes, rgb), ...].

    VideoFrameSource 가 영상 1건에서 quality+diversity 통과한 top n_final Frame
    yield. 각 frame 의 RGB → 결정론 JPEG encode → image flow 와 동일 tuple shape
    로 반환. 영상 열기 실패는 log-and-skip (image flow 와 parity).
    """
    selector_cfg = _to_selector_cfg(cfg)
    out: list[tuple[str, bytes, np.ndarray]] = []
    for path in paths:
        try:
            source = VideoFrameSource(path, selector_cfg)
            for frame in source.iter_frames():
                data = _encode_jpeg_deterministic(frame.rgb)
                out.append((frame.id, data, frame.rgb))
        except Exception as exc:
            logger.warning(
                "pipeline_b_video_skip path=%s reason=%r", path, exc,
            )
            continue
    return out


def _load_images(paths: list[Path]) -> list[tuple[str, bytes, np.ndarray]]:
    """Path 리스트 → [(image_id, bytes, rgb_uint8), ...].

    image_id = path.name (post carousel 내 고유). bytes 는 disk 원본 그대로 — 캐시 키에
    쓰는 쪽이 re-encode 로 해시가 바뀌지 않게 raw bytes 유지. rgb 는 PIL convert("RGB").

    Per-path 실패는 log-and-skip — 131-post run 에서 한 장 깨진 JPEG 이 전체 post 를
    crash 시키지 않도록. `_analyze_images` 의 per-image LLM skip 과 parity.
    """
    from PIL import Image  # lazy — pillow 는 vision extras

    out: list[tuple[str, bytes, np.ndarray]] = []
    for path in paths:
        try:
            data = path.read_bytes()
            rgb = np.array(Image.open(BytesIO(data)).convert("RGB"))
        except Exception as exc:
            logger.warning(
                "pipeline_b_decode_skip path=%s reason=%r", path, exc,
            )
            continue
        out.append((path.name, data, rgb))
    return out


def _analyze_images(
    source_post_id: str,
    images: list[tuple[str, bytes, np.ndarray]],
    vision_llm: VisionLLMClient,
    llm_preset: list[dict[str, str]],
    scene_filter: SceneFilter,
    yolo,
) -> list[tuple[str, np.ndarray, GarmentAnalysis]]:
    """이미지 N 개에 대해 SceneFilter gate → LLM 호출 → post_items.

    Phase 2 (2026-04-25): SceneFilter v2 (adult-woman-only) 통합.
    - stage1_reject: scene off-fashion / 여성 또는 성인 signal 부재 → image skip
      (Gemini 호출 0). reason 은 stage1_female_low / stage1_adult_low / scene_reject
      중 하나.
    - stage1_mix_needs_stage2: 4-way mix (adult 여성 + 성인 남성 + 아동 모두 켜짐)
      → YOLO person bbox → classify_persons → female+adult bbox 한 개라도 살면 통과,
      0개면 frame drop. crop 은 안 보냄 — 풀 이미지를 Gemini v0.6 으로 그대로 보냄
      (v0.6 prompt 가 비-adult-female 제외 강제).
    - stage1_pass / disabled: 풀 이미지 그대로 Gemini.

    per-image LLM 실패는 log-and-skip — 한 이미지의 LLM 예외로 post 전체를 버리지 않음.
    """
    post_items: list[tuple[str, np.ndarray, GarmentAnalysis]] = []
    for image_id, data, rgb in images:
        verdict = scene_filter.accept(rgb, image_id)
        if verdict.stage == "stage1_reject":
            logger.info(
                "scene_filter_skip post_id=%s image=%s reason=%s",
                source_post_id, image_id, verdict.reason,
            )
            continue
        if verdict.stage == "stage1_mix_needs_stage2":
            boxes = detect_people(yolo, rgb)
            if not boxes:
                logger.info(
                    "scene_filter_skip post_id=%s image=%s reason=mix_no_yolo",
                    source_post_id, image_id,
                )
                continue
            person_verdicts = scene_filter.classify_persons(rgb, boxes)
            if not any(pv.passed for pv in person_verdicts):
                logger.info(
                    "scene_filter_skip post_id=%s image=%s reason=mix_all_bbox_drop",
                    source_post_id, image_id,
                )
                continue
        try:
            analysis = vision_llm.extract_garment(data, preset=llm_preset)
        except Exception as exc:
            logger.warning(
                "pipeline_b_llm_skip post_id=%s image=%s reason=%r",
                source_post_id, image_id, exc,
            )
            continue
        post_items.append((image_id, rgb, analysis))
    return post_items


class PipelineBColorExtractor:
    """Pipeline B (LLM BBOX → canonical pool → 동적 k palette) ColorExtractor.

    blob_downloader + blob_cache_dir 를 주면 Azure Blob 이미지를 로컬 캐시로 받아서 처리.
    둘 다 None 이면 로컬 Path 접근만 시도 (sample_data 모드).

    preset 은 factory 단계에서 `load_preset_views` 로 1회 로드 후 3 view 를 직접 주입한다.
    """

    def __init__(
        self,
        bundle: SegBundle,
        cfg: VisionConfig,
        vision_llm: VisionLLMClient,
        llm_preset: list[dict[str, str]],
        matcher_entries: list[MatcherEntry],
        family_map: dict[str, ColorFamily],
        dedup_cfg: OutfitDedupConfig,
        image_root: Path | None = None,
        blob_downloader: BlobDownloader | None = None,
        blob_cache_dir: Path | None = None,
        scene_filter: SceneFilter | None = None,
        hybrid_cfg: HybridPaletteConfig | None = None,
        max_workers: int = 1,
    ) -> None:
        self._bundle = bundle
        self._cfg = cfg
        self._dyn_cfg = cfg.dynamic_palette
        self._hybrid_cfg = hybrid_cfg or cfg.hybrid_palette
        self._post_cfg = cfg.post_palette
        self._vision_llm = vision_llm
        self._llm_preset = llm_preset
        self._matcher_entries = matcher_entries
        self._family_map = family_map
        self._dedup_cfg = dedup_cfg
        self._image_root = image_root
        self._blob_downloader = blob_downloader
        self._blob_cache_dir = blob_cache_dir
        # canonical path 전용 SceneFilter — bundle.scene_filter 는 legacy
        # extract_instances 경로에서만 사용. 디폴트 NoopSceneFilter 면 v2 게이트 우회.
        self._scene_filter: SceneFilter = scene_filter or NoopSceneFilter()
        self._max_workers = max(1, max_workers)

    def extract_visual(
        self, items: list[NormalizedContentItem]
    ) -> list[ColorExtractionResult]:
        if self._max_workers <= 1 or len(items) <= 1:
            return [self._extract_one(item) for item in items]
        # 공유 자원 thread-safety:
        # - PyTorch model.forward (segformer/CLIP/YOLO): inference 모드에서 thread-safe
        # - GeminiVisionLLMClient: google-genai SDK thread-safe
        # - LocalJSONCache: tmp.replace atomic — 동시 put 안전 (last-wins)
        # - BlobDownloader: Azure SDK thread-safe per-client
        # IO-bound 단계 (blob 다운로드 + Gemini 호출) 가 dominant 라 8~16 workers 권장.
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            return list(executor.map(self._extract_one, items))

    def _extract_one(self, item: NormalizedContentItem) -> ColorExtractionResult:
        image_paths, video_paths = self._resolve_paths(item)
        if not image_paths and not video_paths:
            logger.info(
                "pipeline_b_skip post_id=%s reason=no_images", item.source_post_id,
            )
            return ColorExtractionResult(source_post_id=item.source_post_id)

        # image 와 video frame 을 같은 tuple shape ((image_id, bytes, rgb)) 으로 합쳐
        # _analyze_images 한 번에 흘려보낸다 (Option 1 tuple concat).
        images = _load_images(image_paths)
        video_frames = _load_video_frames(video_paths, self._cfg.video_frame)
        all_frames = images + video_frames
        if not all_frames:
            return ColorExtractionResult(source_post_id=item.source_post_id)

        post_items = _analyze_images(
            item.source_post_id, all_frames, self._vision_llm, self._llm_preset,
            self._scene_filter, self._bundle.yolo,
        )
        if not post_items:
            return ColorExtractionResult(source_post_id=item.source_post_id)

        pairs = extract_canonical_pixels_per_object(
            post_items, self._bundle, self._cfg,
            self._dedup_cfg, self._family_map,
        )
        canonicals = [
            self._attach_palette(canonical, pools) for canonical, pools in pairs
        ]
        return ColorExtractionResult(
            source_post_id=item.source_post_id,
            canonicals=canonicals,
            post_palette=build_post_palette(
                canonicals,
                merge_deltae76_threshold=self._post_cfg.merge_deltae76_threshold,
                min_cluster_share=self._post_cfg.min_cluster_share,
                max_clusters=self._post_cfg.max_clusters,
            ),
        )

    def _attach_palette(
        self, canonical: CanonicalOutfit, pools: list[ObjectPool],
    ) -> CanonicalOutfit:
        """per-object β-hybrid → canonical 통합 palette + 멤버별 palette 부착.

        per-object β-hybrid (Phase 1+2) 가 멤버별 list[WeightedCluster] 생산 →
        - canonical 단위: aggregator (Phase 3) 가 통합 weighted KMeans 로 top_n palette
          + cut_off_share 반환.
        - 멤버 단위 (spec §6.5): 같은 per-object 결과를 `finalize_object_palette` 로
          object 별 top_n cap → OutfitMember.palette + cut_off_share 채움.

        pools 가 비면 (non-ethnic / 전 멤버 background-only) 라벨만 보존 (B3a invariant).
        pool 이 없는 멤버는 default 빈 palette 유지 (canonical_extractor 의 skip 정책 따름).
        """
        if not pools:
            return canonical
        per_object_results = [
            build_object_palette(
                pool.rgb_pixels, pool.picks, self._dyn_cfg,
                self._matcher_entries, frame_area=pool.frame_area,
                drop_threshold=self._hybrid_cfg.pick_match_deltae76,
                r2_min_share=self._hybrid_cfg.r2_min_share,
                chroma_vivid=self._hybrid_cfg.chroma_vivid,
                hue_near_deg=self._hybrid_cfg.hue_near_deg,
                r2_merge_deltae76=self._hybrid_cfg.r2_merge_deltae76,
                chroma_ratio_min=self._hybrid_cfg.chroma_ratio_min,
            )
            for pool in pools
        ]
        member_palette_map: dict[
            tuple[str, int], tuple[list, float]
        ] = {}
        for pool, (clusters, etc_weight) in zip(pools, per_object_results):
            obj_palette, obj_cut_off = finalize_object_palette(
                clusters, etc_weight, self._matcher_entries, self._hybrid_cfg,
            )
            member_palette_map[(pool.member.image_id, pool.member.outfit_index)] = (
                obj_palette, obj_cut_off,
            )
        new_members = [
            self._maybe_update_member(member, member_palette_map)
            for member in canonical.members
        ]
        palette, cut_off = aggregate_canonical_palette(
            per_object_results, self._matcher_entries, self._hybrid_cfg,
        )
        return canonical.model_copy(
            update={
                "palette": palette,
                "cut_off_share": cut_off,
                "members": new_members,
            },
        )

    @staticmethod
    def _maybe_update_member(
        member: OutfitMember,
        member_palette_map: dict[tuple[str, int], tuple[list, float]],
    ) -> OutfitMember:
        """pool 이 있던 멤버만 palette/cut_off 주입. 없으면 원본 그대로."""
        key = (member.image_id, member.outfit_index)
        entry = member_palette_map.get(key)
        if entry is None:
            return member
        palette, cut_off = entry
        return member.model_copy(
            update={"palette": palette, "cut_off_share": cut_off},
        )

    def _resolve_paths(
        self, item: NormalizedContentItem
    ) -> tuple[list[Path], list[Path]]:
        """(image_paths, video_paths) 반환. local → blob 순, 둘 다 실패 시 ([], []).

        image / video 는 독립 — local image 가 있어도 video 는 blob 에서 받을 수
        있고 vice versa. carousel 이 image+video 혼입이라 한쪽 누락이 다른쪽 fallback
        억제하지 않도록 별도 평가.
        """
        image_paths = _resolve_local_paths(item, self._image_root)
        video_paths = _resolve_local_video_paths(item, self._image_root)
        if (
            self._blob_downloader is not None
            and self._blob_cache_dir is not None
        ):
            if not image_paths:
                image_paths = _download_blob_paths(
                    item, self._blob_downloader, self._blob_cache_dir,
                )
            if not video_paths:
                video_paths = _download_blob_video_paths(
                    item, self._blob_downloader, self._blob_cache_dir,
                )
        return image_paths, video_paths


__all__ = ["PipelineBColorExtractor"]
