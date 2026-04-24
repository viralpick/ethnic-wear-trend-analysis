"""PipelineBColorExtractor — ColorExtractor Protocol 의 Pipeline B 구현체.

ColorExtractor 는 post 단위 추출 (→ ColorExtractionResult) 지만, Pipeline B 는 pixel-level
aggregation. 이 adapter 가 post → ImageFrameSource → extract_palette → top-1 dominant 로
bridge.

Top-level 로 vision extras (torch/transformers/ultralytics) 를 필요로 하는 pipeline_b_extractor
를 import 한다 — 이 모듈 자체도 vision extras 없이는 import 불가. core 는 `import vision.
pipeline_b_adapter` 를 절대 top-level 하지 말 것 (run_daily_pipeline 의 `_select_extractor`
가 lazy import 로 격리).

이미지 소스 우선순위:
1. absolute local path 가 존재하면 그대로 사용
2. image_root 가 주어지면 URL basename 으로 로컬 스캔
3. blob_downloader + blob_cache_dir 가 주어지면 Azure Blob 에서 다운로드
4. 위 모두 실패 시 해당 post skip (빈 결과)

silhouette: Pipeline B 는 silhouette=None (ATR class → Silhouette enum 매핑 미완 — M4.D).
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from contracts.normalized import NormalizedContentItem
from settings import VisionConfig
from utils.logging import get_logger
from vision.color_extractor import ColorExtractionResult
from vision.frame_source import ImageFrameSource
from vision.pipeline_b_extractor import SegBundle, extract_palette

if TYPE_CHECKING:
    from loaders.blob_downloader import BlobDownloader

logger = get_logger(__name__)

_IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"})


def _resolve_local_paths(item: NormalizedContentItem, image_root: Path | None) -> list[Path]:
    """image_urls 를 로컬 Path 리스트로 변환.

    규칙:
    - 이미 absolute path 이고 존재하면 그대로.
    - image_root 이 주어지면 URL basename 으로 {image_root}/basename 조회.
    - 둘 다 실패하면 빈 리스트 (해당 post skip).
    """
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
    """image_urls 를 Azure Blob 에서 cache_dir 로 다운로드. MP4 등 비이미지 및 실패 URL 은 skip."""
    out: list[Path] = []
    for url in item.image_urls:
        if Path(url).suffix.lower() not in _IMAGE_EXTS:
            continue
        result = downloader.download(url, cache_dir)
        if result is not None:
            out.append(result)
    return out


class PipelineBColorExtractor:
    """Pipeline B (YOLO+segformer+LAB KMeans) 기반 ColorExtractor 구현체.

    blob_downloader + blob_cache_dir 를 주면 Azure Blob 이미지를 로컬 캐시로 받아서 처리.
    둘 다 None 이면 로컬 Path 접근만 시도 (sample_data 모드).
    """

    def __init__(
        self,
        bundle: SegBundle,
        cfg: VisionConfig,
        image_root: Path | None = None,
        blob_downloader: BlobDownloader | None = None,
        blob_cache_dir: Path | None = None,
    ) -> None:
        self._bundle = bundle
        self._cfg = cfg
        self._image_root = image_root
        self._blob_downloader = blob_downloader
        self._blob_cache_dir = blob_cache_dir

    def extract_visual(
        self, items: list[NormalizedContentItem]
    ) -> list[ColorExtractionResult]:
        results: list[ColorExtractionResult] = []
        for item in items:
            paths = _resolve_local_paths(item, self._image_root)

            if not paths and self._blob_downloader and self._blob_cache_dir:
                paths = _download_blob_paths(item, self._blob_downloader, self._blob_cache_dir)

            if not paths:
                logger.info(
                    "pipeline_b_skip post_id=%s reason=no_images",
                    item.source_post_id,
                )
                results.append(ColorExtractionResult(source_post_id=item.source_post_id))
                continue

            source = ImageFrameSource(paths)
            palette = extract_palette(source, self._bundle, self._cfg)
            if not palette:
                results.append(ColorExtractionResult(source_post_id=item.source_post_id))
                continue

            top = palette[0]
            results.append(
                ColorExtractionResult(
                    source_post_id=item.source_post_id,
                    r=top.r, g=top.g, b=top.b,
                    name=top.name,
                    family=top.family,
                    silhouette=None,
                )
            )
        return results
