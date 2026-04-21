"""PipelineBColorExtractor — ColorExtractor Protocol 의 Pipeline B 구현체.

ColorExtractor 는 post 단위 추출 (→ ColorExtractionResult) 지만, Pipeline B 는 pixel-level
aggregation. 이 adapter 가 post → ImageFrameSource → extract_palette → top-1 dominant 로
bridge.

Top-level 로 vision extras (torch/transformers/ultralytics) 를 필요로 하는 pipeline_b_extractor
를 import 한다 — 이 모듈 자체도 vision extras 없이는 import 불가. core 는 `import vision.
pipeline_b_adapter` 를 절대 top-level 하지 말 것 (run_daily_pipeline 의 `_select_extractor`
가 lazy import 로 격리).

M3 제한 (현재 스코프):
- NormalizedContentItem.image_urls 는 URL 또는 Path. 현재는 **로컬 Path 로 변환 가능한 것만**
  처리 (sample_data/image/ 스캔 or 직접 Path 주입). URL download 는 M3 BlobRawLoader 에서.
- image_urls 가 처리 불가면 해당 post 는 skip (빈 결과) — Fake 와 달리 모든 post 에 결과가
  나오지 않을 수 있음.
"""
from __future__ import annotations

from pathlib import Path

from contracts.normalized import NormalizedContentItem
from settings import VisionConfig
from utils.logging import get_logger
from vision.color_extractor import ColorExtractionResult
from vision.frame_source import ImageFrameSource
from vision.pipeline_b_extractor import SegBundle, extract_palette

logger = get_logger(__name__)


def _resolve_local_paths(item: NormalizedContentItem, image_root: Path | None) -> list[Path]:
    """image_urls 를 로컬 Path 리스트로 변환.

    규칙:
    - 이미 absolute path 이고 존재하면 그대로.
    - image_root 이 주어지면 URL basename 으로 {image_root}/basename 조회.
    - 둘 다 실패하면 빈 리스트 (해당 post skip).
    """
    out: list[Path] = []
    for url in item.image_urls:
        as_path = Path(url)
        if as_path.is_absolute() and as_path.exists():
            out.append(as_path)
            continue
        if image_root is not None:
            candidate = image_root / Path(url).name
            if candidate.exists():
                out.append(candidate)
    return out


class PipelineBColorExtractor:
    """Pipeline B (YOLO+segformer+LAB KMeans) 기반 ColorExtractor 구현체.

    M3 완성 전까지는 로컬 Path 접근만 지원. image_root 가 None 이면 absolute path 만 시도.
    """

    def __init__(
        self,
        bundle: SegBundle,
        cfg: VisionConfig,
        image_root: Path | None = None,
    ) -> None:
        self._bundle = bundle
        self._cfg = cfg
        self._image_root = image_root

    def extract_visual(
        self, items: list[NormalizedContentItem]
    ) -> list[ColorExtractionResult]:
        results: list[ColorExtractionResult] = []
        for item in items:
            paths = _resolve_local_paths(item, self._image_root)
            if not paths:
                logger.info(
                    "pipeline_b_skip post_id=%s reason=no_local_images",
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
            # Pipeline B 는 silhouette 추출 안 함 (ATR class 는 있으나 Silhouette enum 과
            # 매핑 미완 — M4.D 세분 속성 과제). silhouette=None.
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
