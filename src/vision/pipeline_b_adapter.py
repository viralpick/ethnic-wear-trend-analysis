"""PipelineBColorExtractor — canonical path 기반 ColorExtractor 구현체 (Phase 5).

흐름:
  post.image_urls
    → _resolve_local_paths / _download_blob_paths (파일 시스템 / Azure Blob)
    → _load_images (path → (image_id, bytes, rgb))
    → VisionLLMClient.extract_garment(bytes, preset=llm_preset)
    → extract_canonical_pixels → list[(CanonicalOutfit, CanonicalOutfitPixels | None)]
    → canonical 별 build_canonical_palette (pixels != None 에 한해) → palette 주입
    → ColorExtractionResult(canonicals=[...], post_palette=[...])

설계 결정 (project_phase5_adapter_design.md 참조):
  - `_load_images` 를 module-level 로 분리 — 테스트가 monkeypatch 로 Path/PIL 의존 없이 stub.
  - preset.json 은 factory 에서 1회 로드 (`load_preset_views`) — 3 view (llm_preset /
    matcher_entries / family_map) 로 adapter 에 주입. adapter 안에서 I/O 재호출 X.
  - scene filter 는 canonical path 에서 생략 — LLM 의 is_india_ethnic_wear=False 가 대체.
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
from contracts.vision import GarmentAnalysis
from settings import OutfitDedupConfig, VisionConfig
from utils.logging import get_logger
from vision.canonical_extractor import extract_canonical_pixels
from vision.canonical_palette import build_canonical_palette
from vision.color_extractor import ColorExtractionResult
from vision.color_family_preset import MatcherEntry
from vision.llm_client import VisionLLMClient
from vision.pipeline_b_extractor import SegBundle
from vision.post_palette import build_post_palette

if TYPE_CHECKING:
    from loaders.blob_downloader import BlobDownloader

logger = get_logger(__name__)

_IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"})


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
) -> list[tuple[str, np.ndarray, GarmentAnalysis]]:
    """이미지 N 개에 대해 LLM 호출 → post_items (image_id, rgb, analysis).

    per-image 실패는 log-and-skip — 한 이미지의 LLM 예외로 post 전체를 버리지 않음.
    """
    post_items: list[tuple[str, np.ndarray, GarmentAnalysis]] = []
    for image_id, data, rgb in images:
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
    ) -> None:
        self._bundle = bundle
        self._cfg = cfg
        self._dyn_cfg = cfg.dynamic_palette
        self._vision_llm = vision_llm
        self._llm_preset = llm_preset
        self._matcher_entries = matcher_entries
        self._family_map = family_map
        self._dedup_cfg = dedup_cfg
        self._image_root = image_root
        self._blob_downloader = blob_downloader
        self._blob_cache_dir = blob_cache_dir

    def extract_visual(
        self, items: list[NormalizedContentItem]
    ) -> list[ColorExtractionResult]:
        return [self._extract_one(item) for item in items]

    def _extract_one(self, item: NormalizedContentItem) -> ColorExtractionResult:
        paths = self._resolve_paths(item)
        if not paths:
            logger.info(
                "pipeline_b_skip post_id=%s reason=no_images", item.source_post_id,
            )
            return ColorExtractionResult(source_post_id=item.source_post_id)

        images = _load_images(paths)
        post_items = _analyze_images(
            item.source_post_id, images, self._vision_llm, self._llm_preset,
        )
        if not post_items:
            return ColorExtractionResult(source_post_id=item.source_post_id)

        pairs = extract_canonical_pixels(
            post_items, self._bundle, self._cfg,
            self._dedup_cfg, self._family_map,
        )
        canonicals = [
            self._attach_palette(canonical, pixels) for canonical, pixels in pairs
        ]
        return ColorExtractionResult(
            source_post_id=item.source_post_id,
            canonicals=canonicals,
            post_palette=build_post_palette(canonicals),
        )

    def _attach_palette(self, canonical, pixels):
        """pixel pool 이 있으면 canonical palette 계산 후 model_copy, 없으면 원본 보존.

        B3a: canonical 라벨은 pool 이 None 이어도 enriched.canonicals 로 전달 (label
        preservation invariant, project_color_pipeline_redesign advisor 피드백).
        """
        if pixels is None:
            return canonical
        palette = build_canonical_palette(
            pixels.pooled_pixels, self._dyn_cfg, self._matcher_entries,
        )
        return canonical.model_copy(update={"palette": palette})

    def _resolve_paths(self, item: NormalizedContentItem) -> list[Path]:
        """local → blob 순. 셋 다 실패 시 빈 리스트."""
        paths = _resolve_local_paths(item, self._image_root)
        if paths:
            return paths
        if self._blob_downloader is not None and self._blob_cache_dir is not None:
            return _download_blob_paths(
                item, self._blob_downloader, self._blob_cache_dir,
            )
        return []


__all__ = ["PipelineBColorExtractor"]
