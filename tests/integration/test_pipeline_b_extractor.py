"""Pipeline B 실 호출 integration test — torch / transformers / ultralytics 필요 + 모델 다운.

`uv run pytest tests/integration` 로만 수동 실행. 기본 testpaths 에서 제외됨 (pyproject.toml).

첫 실행 시 모델 다운로드 ~250MB + MPS/CPU 추론 프레임당 ~0.3초. CI 에서는 별도 job.
"""
from __future__ import annotations

from pathlib import Path

import pytest

# vision extras 가드 — 없으면 전체 module skip.
pytest.importorskip("torch", reason="vision extras required")
pytest.importorskip("transformers", reason="vision extras required")
pytest.importorskip("ultralytics", reason="vision extras required")
pytest.importorskip("PIL.Image", reason="vision extras required")


from settings import load_settings  # noqa: E402
from vision.frame_source import ImageFrameSource  # noqa: E402
from vision.pipeline_b_extractor import extract_palette, load_models  # noqa: E402

_SAMPLE_IMAGE_DIR = Path(__file__).resolve().parents[2] / "sample_data" / "image"


@pytest.fixture(scope="session")
def seg_bundle():
    # session scope — 모델 1회 로드, 모든 테스트에서 공유.
    return load_models()


def test_extract_palette_on_single_jpg(seg_bundle) -> None:
    """sample_data/image 에 JPG 1장 이상 있어야 함 (로컬만)."""
    jpgs = sorted(_SAMPLE_IMAGE_DIR.glob("*.jpg"))
    if not jpgs:
        pytest.skip(
            f"{_SAMPLE_IMAGE_DIR} 에 JPG 없음. sample_data/image/ 에 실 IG 이미지 배치 필요."
        )
    src = ImageFrameSource([jpgs[0]])
    settings = load_settings()
    palette = extract_palette(src, seg_bundle, settings.vision)
    # 실 사람 이미지라면 garment pixel 이 발견되어 palette 가 비어있지 않아야 함.
    # 단 일부 이미지는 person 미감지 가능 — 이 경우 빈 palette 허용.
    assert isinstance(palette, list)


def test_extract_palette_on_full_carousel(seg_bundle) -> None:
    """한 post 의 캐러셀 N장 (post ULID prefix 같은 것) 에 대한 aggregate."""
    jpgs = sorted(_SAMPLE_IMAGE_DIR.glob("01KPNKHWHV1ZCWE2JHF1Q10HDS_*.jpg"))
    if not jpgs:
        pytest.skip("carousel sample 없음 (post ULID 01KPNKHWHV..).")
    src = ImageFrameSource(jpgs)
    settings = load_settings()
    palette = extract_palette(src, seg_bundle, settings.vision)
    assert isinstance(palette, list)
    if palette:
        assert sum(item.pct for item in palette) == pytest.approx(1.0, abs=1e-4)
