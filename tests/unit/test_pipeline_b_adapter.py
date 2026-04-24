"""Phase 5 — PipelineBColorExtractor canonical path e2e pinning.

`vision.pipeline_b_adapter` 는 canonical path (LLM BBOX → extract_canonical_pixels →
canonical 별 build_canonical_palette) 로 재작성됨. B3a/B3b/B3d: ColorExtractionResult 는
`canonicals: list[CanonicalOutfit]` + `post_palette` 두 필드만 채운다 (silhouette/단일 hex
모두 B3d 까지 제거).

본 테스트는:
  - 이미지 디스크 I/O 없이 `_load_images` / `_resolve_local_paths` 를 monkeypatch 로 주입
  - `run_segformer` 를 stub 해 상단 절반을 upper-clothes 로 반환
  - Fake VisionLLMClient 대신 `_StubLLM` 으로 결정적 analysis 주입

vision extras (torch/transformers/ultralytics) 없는 환경에서는 skip — adapter 가
pipeline_b_extractor 를 top-level import.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch", reason="vision extras required")
pytest.importorskip("transformers", reason="vision extras required")
pytest.importorskip("ultralytics", reason="vision extras required")


from contracts.common import ColorFamily, ContentSource  # noqa: E402
from contracts.normalized import NormalizedContentItem  # noqa: E402
from contracts.vision import EthnicOutfit, GarmentAnalysis  # noqa: E402
from settings import OutfitDedupConfig, load_settings  # noqa: E402
from vision.color_family_preset import (  # noqa: E402
    PresetViews,
    load_preset_views,
)
from vision.pipeline_b_adapter import (  # noqa: E402
    PipelineBColorExtractor,
    _load_images,
)
from vision.pipeline_b_extractor import SegBundle  # noqa: E402
from vision.scene_filter import NoopSceneFilter  # noqa: E402

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

VIBRANT_RED_RGB: tuple[int, int, int] = (230, 30, 40)


def _make_item(
    post_id: str = "post_A",
    image_urls: list[str] | None = None,
) -> NormalizedContentItem:
    return NormalizedContentItem(
        source=ContentSource.INSTAGRAM,
        source_post_id=post_id,
        text_blob="test caption",
        hashtags=[],
        image_urls=image_urls or ["image_A.jpg"],
        post_date=datetime(2026, 4, 24, tzinfo=timezone.utc),
        engagement_raw=100,
    )


def _make_bundle() -> SegBundle:
    return SegBundle(
        yolo=object(), seg_processor=object(), seg_model=object(),
        device="cpu", scene_filter=NoopSceneFilter(),
    )


def _make_outfit(
    area_ratio: float = 0.25,
    silhouette=None,
    picks: list[str] | None = None,
) -> EthnicOutfit:
    side = area_ratio ** 0.5
    return EthnicOutfit(
        person_bbox=(0.0, 0.0, side, side),
        person_bbox_area_ratio=area_ratio,
        upper_garment_type="kurta",
        upper_is_ethnic=True,
        dress_as_single=False,
        silhouette=silhouette,
        color_preset_picks_top3=picks or ["pool_00"],
    )


def _write_preset(tmp_path: Path) -> Path:
    """red 매칭 entry + 반대쪽 green entry. red 는 LAB(53.24, 80.09, 67.20) 근처."""
    preset_path = tmp_path / "preset.json"
    entries = [
        {"name": "pool_00", "hex": "#FF0000", "lab": [53.24, 80.09, 67.20]},
        {"name": "pool_01", "hex": "#00FF00", "lab": [87.73, -86.18, 83.18]},
    ]
    preset_path.write_text(json.dumps(entries), encoding="utf-8")
    return preset_path


def _seg_stub_top_half_upper(_bundle, crop_rgb):
    h, w = crop_rgb.shape[:2]
    seg = np.zeros((h, w), dtype=np.int32)
    seg[: h // 2] = 4
    return seg


def _fake_loader(paths: list[Path]) -> list[tuple[str, bytes, np.ndarray]]:
    """각 path → (image_id=path.name, bytes=path.name.encode(), 200x200 RED RGB)."""
    out: list[tuple[str, bytes, np.ndarray]] = []
    for p in paths:
        rgb = np.full((200, 200, 3), VIBRANT_RED_RGB, dtype=np.uint8)
        out.append((p.name, p.name.encode(), rgb))
    return out


def _build_adapter(
    tmp_path: Path,
    vision_llm,
    *,
    views: PresetViews | None = None,
) -> PipelineBColorExtractor:
    if views is None:
        views = load_preset_views(_write_preset(tmp_path))
    settings = load_settings()
    return PipelineBColorExtractor(
        bundle=_make_bundle(),
        cfg=settings.vision,
        vision_llm=vision_llm,
        llm_preset=views.llm_preset,
        matcher_entries=views.matcher_entries,
        family_map=views.family_map,
        dedup_cfg=OutfitDedupConfig(preset_path=_write_preset(tmp_path)),
    )


def _install_stubs(
    monkeypatch,
    adapter: PipelineBColorExtractor,
    paths: list[Path],
    loader=_fake_loader,
) -> None:
    monkeypatch.setattr(
        "vision.canonical_extractor.run_segformer", _seg_stub_top_half_upper,
    )
    monkeypatch.setattr(
        "vision.pipeline_b_adapter._resolve_local_paths",
        lambda _item, _root: paths,
    )
    monkeypatch.setattr(
        "vision.pipeline_b_adapter._load_images", loader,
    )


class _StubLLM:
    """고정 analysis 반환. FakeVisionLLMClient 는 이미지 bytes 로 분기 → 본 테스트는 결정성
    단순화 위해 image-invariant stub 사용."""

    def __init__(self, analysis: GarmentAnalysis) -> None:
        self._analysis = analysis
        self.calls: list[bytes] = []

    def extract_garment(self, image_bytes, *, preset):  # noqa: ARG002
        self.calls.append(image_bytes)
        return self._analysis


class _FailingLLM:
    """매 호출 예외 raise — all-fail 케이스용."""

    def extract_garment(self, image_bytes, *, preset):  # noqa: ARG002
        raise RuntimeError("simulated llm failure")


class _SelectiveFailLLM:
    """image_A 는 실패, image_B 는 성공 — per-image skip 테스트."""

    def __init__(self, analysis: GarmentAnalysis) -> None:
        self._analysis = analysis

    def extract_garment(self, image_bytes, *, preset):  # noqa: ARG002
        if b"image_A" in image_bytes:
            raise RuntimeError("selective fail")
        return self._analysis


# --------------------------------------------------------------------------- #
# e2e — 성공 경로 (preset 매칭 성공)
# --------------------------------------------------------------------------- #

def test_extract_visual_returns_populated_result(monkeypatch, tmp_path) -> None:
    """e2e — RED pixel → canonical 1개 + palette non-empty. LLM 1회 호출 pinning."""
    analysis = GarmentAnalysis(
        is_india_ethnic_wear=True,
        outfits=[_make_outfit(area_ratio=0.25, picks=["pool_00"])],
    )
    stub_llm = _StubLLM(analysis)
    adapter = _build_adapter(tmp_path, stub_llm)
    _install_stubs(monkeypatch, adapter, [Path("image_A.jpg")])

    results = adapter.extract_visual([_make_item()])

    assert len(results) == 1
    r = results[0]
    assert r.source_post_id == "post_A"
    assert len(r.canonicals) == 1
    assert len(r.canonicals[0].palette) >= 1
    first = r.canonicals[0].palette[0]
    assert first.hex.startswith("#")
    assert first.family is not None
    # B3b: post_palette 는 canonical palette 를 area_ratio × share 가중으로 merge.
    # 단일 canonical × 단일 색 → post_palette 도 1 cluster, share=1.0 로 떨어짐.
    assert len(r.post_palette) == 1
    assert r.post_palette[0].share == 1.0
    assert r.post_palette[0].hex.startswith("#")
    # LLM 1회 호출 — 단일 이미지.
    assert len(stub_llm.calls) == 1


# --------------------------------------------------------------------------- #
# ΔE76 ≤ 15 → preset 매칭 / > 15 → fallback
# --------------------------------------------------------------------------- #

def test_hybrid_preset_match_when_within_threshold(monkeypatch, tmp_path) -> None:
    """preset 에 실제 RED cluster (LAB ≈ 52.66, 70.90, 47.98) 과 가까운 entry 를 넣고 ΔE76
    ≤ 15 분기 → preset family (BRIGHT) 태우는지 확인. B3a: preset name 은 PaletteCluster 에
    노출 안 되므로 family 만 pin."""
    preset_path = tmp_path / "preset_close.json"
    preset_path.write_text(json.dumps([
        {"name": "close_red", "hex": "#E62024", "lab": [52.66, 70.90, 47.98]},
        {"name": "pool_01", "hex": "#00FF00", "lab": [87.73, -86.18, 83.18]},
    ]), encoding="utf-8")
    views = load_preset_views(preset_path)

    analysis = GarmentAnalysis(
        is_india_ethnic_wear=True,
        outfits=[_make_outfit(picks=["close_red"])],
    )
    settings = load_settings()
    adapter = PipelineBColorExtractor(
        bundle=_make_bundle(),
        cfg=settings.vision,
        vision_llm=_StubLLM(analysis),
        llm_preset=views.llm_preset,
        matcher_entries=views.matcher_entries,
        family_map=views.family_map,
        dedup_cfg=OutfitDedupConfig(preset_path=preset_path),
    )
    _install_stubs(monkeypatch, adapter, [Path("image_A.jpg")])

    results = adapter.extract_visual([_make_item()])
    palette = results[0].canonicals[0].palette
    assert len(palette) >= 1
    # preset close_red 의 family = lab_to_family(52.66, 70.90, 47.98): chroma > 55 → BRIGHT
    assert palette[0].family == ColorFamily.BRIGHT


def test_hybrid_lab_to_family_fallback_when_over_threshold(monkeypatch, tmp_path) -> None:
    """preset 전부 먼 색일 때 fallback — canonical_palette 가 lab_to_family 경로로 family 결정."""
    adapter = _build_adapter(tmp_path, _StubLLM(GarmentAnalysis(
        is_india_ethnic_wear=True,
        outfits=[_make_outfit(picks=["pool_00"])],
    )))
    _install_stubs(monkeypatch, adapter, [Path("image_A.jpg")])

    results = adapter.extract_visual([_make_item()])
    palette = results[0].canonicals[0].palette
    assert len(palette) >= 1
    # RED cluster → chroma > 55 → BRIGHT (lab_to_family fallback)
    assert palette[0].family == ColorFamily.BRIGHT


# --------------------------------------------------------------------------- #
# silhouette 전달 — representative.silhouette
# --------------------------------------------------------------------------- #

def test_silhouette_preserved_on_canonical_representative(monkeypatch, tmp_path) -> None:
    """B3d: post-level silhouette 은 ColorExtractionResult 에서 아예 제거됐고,
    silhouette 정보는 오직 `canonicals[*].representative.silhouette` 로만 보존
    (feedback_post_level_single_value)."""
    from contracts.common import Silhouette
    analysis = GarmentAnalysis(
        is_india_ethnic_wear=True,
        outfits=[_make_outfit(silhouette=Silhouette.ANARKALI)],
    )
    adapter = _build_adapter(tmp_path, _StubLLM(analysis))
    _install_stubs(monkeypatch, adapter, [Path("image_A.jpg")])

    results = adapter.extract_visual([_make_item()])
    assert results[0].canonicals[0].representative.silhouette == Silhouette.ANARKALI


# --------------------------------------------------------------------------- #
# 빈 결과 케이스
# --------------------------------------------------------------------------- #

def test_empty_when_no_paths_resolved(monkeypatch, tmp_path) -> None:
    adapter = _build_adapter(tmp_path, _StubLLM(GarmentAnalysis(
        is_india_ethnic_wear=True, outfits=[_make_outfit()],
    )))
    # paths 빈 리스트 → LLM 호출 없음, source_post_id 만 있는 빈 결과.
    _install_stubs(monkeypatch, adapter, [])
    results = adapter.extract_visual([_make_item()])
    assert len(results) == 1
    assert results[0].source_post_id == "post_A"
    assert results[0].canonicals == []
    assert results[0].post_palette == []


def test_empty_when_llm_returns_not_ethnic(monkeypatch, tmp_path) -> None:
    analysis = GarmentAnalysis(is_india_ethnic_wear=False, outfits=[])
    adapter = _build_adapter(tmp_path, _StubLLM(analysis))
    _install_stubs(monkeypatch, adapter, [Path("image_A.jpg")])

    results = adapter.extract_visual([_make_item()])
    assert results[0].canonicals == []


def test_empty_when_all_llm_calls_fail(monkeypatch, tmp_path) -> None:
    adapter = _build_adapter(tmp_path, _FailingLLM())
    _install_stubs(monkeypatch, adapter, [Path("image_A.jpg"), Path("image_B.jpg")])
    results = adapter.extract_visual([_make_item()])
    assert results[0].canonicals == []


# --------------------------------------------------------------------------- #
# Per-image LLM 실패는 log-and-skip, 나머지 이미지로 진행
# --------------------------------------------------------------------------- #

def test_partial_llm_failure_continues_with_remaining_images(monkeypatch, tmp_path) -> None:
    analysis = GarmentAnalysis(
        is_india_ethnic_wear=True,
        outfits=[_make_outfit()],
    )
    adapter = _build_adapter(tmp_path, _SelectiveFailLLM(analysis))
    # image_A 는 실패, image_B 는 성공 → 결과 canonical 1개 + palette 채움.
    _install_stubs(
        monkeypatch, adapter,
        [Path("image_A.jpg"), Path("image_B.jpg")],
    )
    results = adapter.extract_visual([_make_item()])
    assert len(results[0].canonicals) == 1
    assert len(results[0].canonicals[0].palette) >= 1


# --------------------------------------------------------------------------- #
# 여러 post 순차 처리
# --------------------------------------------------------------------------- #

def test_load_images_skips_corrupt_file(tmp_path) -> None:
    """깨진 JPEG 1개가 post 전체를 crash 시키지 않도록 per-path 복구."""
    from PIL import Image

    good = tmp_path / "good.jpg"
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(good, format="JPEG")
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not a real jpeg")

    out = _load_images([good, bad])

    assert len(out) == 1
    assert out[0][0] == "good.jpg"


# --------------------------------------------------------------------------- #
# 여러 post 순차 처리
# --------------------------------------------------------------------------- #

def test_multiple_posts_independent_results(monkeypatch, tmp_path) -> None:
    analysis = GarmentAnalysis(
        is_india_ethnic_wear=True, outfits=[_make_outfit()],
    )
    adapter = _build_adapter(tmp_path, _StubLLM(analysis))
    _install_stubs(monkeypatch, adapter, [Path("image_A.jpg")])

    items = [_make_item(post_id="p1"), _make_item(post_id="p2")]
    results = adapter.extract_visual(items)
    assert [r.source_post_id for r in results] == ["p1", "p2"]
    assert all(len(r.canonicals) == 1 for r in results)
    assert all(len(r.canonicals[0].palette) >= 1 for r in results)
