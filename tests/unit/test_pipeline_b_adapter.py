"""Phase 5 — PipelineBColorExtractor canonical path e2e pinning.

`vision.pipeline_b_adapter` 는 canonical path (LLM BBOX → extract_canonical_pixels_per_object
→ canonical 별 β-hybrid (per-object build_object_palette + aggregate_canonical_palette)) 로
재작성됨. B3a/B3b/B3d: ColorExtractionResult 는 `canonicals: list[CanonicalOutfit]` +
`post_palette` 두 필드만 채운다 (silhouette/단일 hex 모두 B3d 까지 제거).

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
from settings import VideoFrameConfig  # noqa: E402
from vision.pipeline_b_adapter import (  # noqa: E402
    PipelineBColorExtractor,
    _encode_jpeg_deterministic,
    _load_images,
    _load_video_frames,
    _to_selector_cfg,
)
from vision.video_frame_selector import VideoFrameSelectorConfig  # noqa: E402
from vision.pipeline_b_extractor import SegBundle  # noqa: E402
from vision.scene_filter import FakeSceneFilter, NoopSceneFilter  # noqa: E402

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
        engagement_raw_count=100,
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
    lower_garment_type: str | None = None,
    lower_is_ethnic: bool | None = None,
    fabric: str | None = None,
    technique: str | None = None,
) -> EthnicOutfit:
    side = area_ratio ** 0.5
    return EthnicOutfit(
        person_bbox=(0.0, 0.0, side, side),
        person_bbox_area_ratio=area_ratio,
        upper_garment_type="kurta",
        upper_is_ethnic=True,
        lower_garment_type=lower_garment_type,
        lower_is_ethnic=lower_is_ethnic,
        dress_as_single=False,
        silhouette=silhouette,
        fabric=fabric,
        technique=technique,
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
    scene_filter=None,
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
        scene_filter=scene_filter,
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


# --------------------------------------------------------------------------- #
# Phase 2 (2026-04-25) — SceneFilter v2 (adult-woman-only) gate
# --------------------------------------------------------------------------- #

def test_scene_filter_stage1_reject_skips_image(monkeypatch, tmp_path) -> None:
    """drop_rule reason → stage1_reject → LLM 호출 0 + 빈 결과."""
    analysis = GarmentAnalysis(
        is_india_ethnic_wear=True, outfits=[_make_outfit()],
    )
    stub_llm = _StubLLM(analysis)
    scene_filter = FakeSceneFilter(drop_rule=lambda _fid: "stage1_female_low")
    adapter = _build_adapter(tmp_path, stub_llm, scene_filter=scene_filter)
    _install_stubs(monkeypatch, adapter, [Path("image_A.jpg")])

    results = adapter.extract_visual([_make_item()])
    assert results[0].canonicals == []
    assert stub_llm.calls == []


def test_scene_filter_mix_no_yolo_skips_image(monkeypatch, tmp_path) -> None:
    """stage1_mix_needs_stage2 + detect_people 빈 결과 → image skip + LLM 호출 0."""
    analysis = GarmentAnalysis(
        is_india_ethnic_wear=True, outfits=[_make_outfit()],
    )
    stub_llm = _StubLLM(analysis)
    scene_filter = FakeSceneFilter(forced_stage="stage1_mix_needs_stage2")
    adapter = _build_adapter(tmp_path, stub_llm, scene_filter=scene_filter)
    _install_stubs(monkeypatch, adapter, [Path("image_A.jpg")])
    monkeypatch.setattr(
        "vision.pipeline_b_adapter.detect_people",
        lambda _yolo, _rgb: [],
    )

    results = adapter.extract_visual([_make_item()])
    assert results[0].canonicals == []
    assert stub_llm.calls == []


def test_scene_filter_mix_all_bbox_drop_skips_image(monkeypatch, tmp_path) -> None:
    """mix + classify_persons 모두 fail → frame drop + LLM 호출 0."""
    analysis = GarmentAnalysis(
        is_india_ethnic_wear=True, outfits=[_make_outfit()],
    )
    stub_llm = _StubLLM(analysis)
    scene_filter = FakeSceneFilter(
        forced_stage="stage1_mix_needs_stage2",
        bbox_drop_rule=lambda _fid, _idx, _bb: "stage2_female_low",
    )
    adapter = _build_adapter(tmp_path, stub_llm, scene_filter=scene_filter)
    _install_stubs(monkeypatch, adapter, [Path("image_A.jpg")])
    monkeypatch.setattr(
        "vision.pipeline_b_adapter.detect_people",
        lambda _yolo, _rgb: [(10, 10, 50, 50), (60, 60, 100, 100)],
    )

    results = adapter.extract_visual([_make_item()])
    assert results[0].canonicals == []
    assert stub_llm.calls == []


def test_scene_filter_mix_at_least_one_bbox_pass_keeps_image(
    monkeypatch, tmp_path,
) -> None:
    """mix + classify_persons 한 개라도 pass → 풀 이미지 LLM 1회 호출 (crop X)."""
    analysis = GarmentAnalysis(
        is_india_ethnic_wear=True, outfits=[_make_outfit(picks=["pool_00"])],
    )
    stub_llm = _StubLLM(analysis)
    # idx 0 만 fail, idx 1 pass → kept ≥ 1 → frame pass.
    scene_filter = FakeSceneFilter(
        forced_stage="stage1_mix_needs_stage2",
        bbox_drop_rule=lambda _fid, idx, _bb: "stage2_adult_low" if idx == 0 else None,
    )
    adapter = _build_adapter(tmp_path, stub_llm, scene_filter=scene_filter)
    _install_stubs(monkeypatch, adapter, [Path("image_A.jpg")])
    monkeypatch.setattr(
        "vision.pipeline_b_adapter.detect_people",
        lambda _yolo, _rgb: [(10, 10, 50, 50), (60, 60, 100, 100)],
    )

    results = adapter.extract_visual([_make_item()])
    assert len(results[0].canonicals) == 1
    assert len(stub_llm.calls) == 1


def test_outfit_member_palette_filled_per_object(monkeypatch, tmp_path) -> None:
    """spec §6.5: pool 이 있는 멤버는 OutfitMember.palette + cut_off_share 채워진다.

    canonical 1개 + 멤버 1개 (단일 이미지) → 빨강 픽셀 → 멤버 palette 1+ cluster.
    canonical_object 행 (row_builder) 이 NULL 적재 갭을 해소했음을 검증."""
    analysis = GarmentAnalysis(
        is_india_ethnic_wear=True,
        outfits=[_make_outfit(picks=["pool_00"])],
    )
    adapter = _build_adapter(tmp_path, _StubLLM(analysis))
    _install_stubs(monkeypatch, adapter, [Path("image_A.jpg")])

    results = adapter.extract_visual([_make_item()])
    canonical = results[0].canonicals[0]
    assert len(canonical.members) >= 1
    member = canonical.members[0]
    # B1 §6.5: pool 이 있는 멤버는 palette 채워진다.
    assert len(member.palette) >= 1
    assert all(c.hex.startswith("#") for c in member.palette)
    # 합 invariant — contracts.PaletteCluster 정의.
    assert abs(sum(c.share for c in member.palette) - 1.0) < 1e-6
    # cut_off_share ∈ [0, 1].
    assert 0.0 <= member.cut_off_share <= 1.0


# --------------------------------------------------------------------------- #
# M3.G (2026-04-28) — video frame wiring (JPEG 결정론 + selector cfg + mixed flow)
# --------------------------------------------------------------------------- #


def test_encode_jpeg_deterministic_byte_identical() -> None:
    """VisionLLMClient LocalJSONCache key = sha256(bytes). 같은 RGB → 같은 bytes 필수.

    PIL 기본 파라미터 (optimize / subsampling / progressive) 가 라이브러리 버전마다
    달라지면 cache miss 폭주 → 핀: quality=95/optimize=False/subsampling=0/
    progressive=False/metadata 미주입.
    """
    rng = np.random.default_rng(seed=42)
    rgb = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)

    b1 = _encode_jpeg_deterministic(rgb)
    b2 = _encode_jpeg_deterministic(rgb)

    assert b1 == b2
    # JPEG SOI marker — encode 가 실제 JPEG 를 만든다 boilerplate 검증.
    assert b1[:2] == b"\xff\xd8"


def test_encode_jpeg_deterministic_distinct_inputs_distinct_bytes() -> None:
    """같은 bytes 보장이 모든 입력에 대해 동일 결과를 의미하지 않는다 (해시 충돌 방어)."""
    rgb_red = np.full((32, 32, 3), (230, 30, 40), dtype=np.uint8)
    rgb_green = np.full((32, 32, 3), (30, 230, 40), dtype=np.uint8)
    assert _encode_jpeg_deterministic(rgb_red) != _encode_jpeg_deterministic(rgb_green)


def test_to_selector_cfg_field_by_field_mapping() -> None:
    """VideoFrameConfig (Pydantic) → VideoFrameSelectorConfig (frozen dataclass) 6 필드 1:1.

    Pydantic v3 migration 시 `model_dump()` semantics 가 깨질 수 있어 명시적 속성 매핑.
    """
    cfg = VideoFrameConfig(
        n_candidate=33,
        n_final=11,
        blur_min=77.5,
        brightness_range=(20.0, 200.0),
        scene_corr_max=0.9,
        histogram_bins=16,
    )
    selector = _to_selector_cfg(cfg)

    assert isinstance(selector, VideoFrameSelectorConfig)
    assert selector.n_candidate == 33
    assert selector.n_final == 11
    assert selector.blur_min == 77.5
    assert selector.brightness_range == (20.0, 200.0)
    assert selector.scene_corr_max == 0.9
    assert selector.histogram_bins == 16


def test_load_video_frames_skips_unopenable_path(tmp_path) -> None:
    """존재하지 않거나 cv2 가 못 여는 영상은 log-and-skip — 한 영상 깨져도 post 진행."""
    bogus = tmp_path / "nonexistent.mp4"
    out = _load_video_frames([bogus], VideoFrameConfig())
    assert out == []


def test_extract_one_mixes_image_and_video_frames(monkeypatch, tmp_path) -> None:
    """image + video 동시 입력 → `_analyze_images` 가 양쪽 모두 보고 LLM 호출 횟수 합계.

    Option 1 tuple concat 핀: image (`_load_images`) 결과와 video frame
    (`_load_video_frames`) 결과 모두 같은 (image_id, bytes, rgb) shape 으로 합쳐
    `_analyze_images` 한 번 흐름. carousel image+video 혼입 IG 포스트 케이스.
    """
    # outfit_dedup 통과 production-grade stub: picks 2+ (color_preset 0.40)
    # + lower=palazzo (garment_type 0.25 + dress bridging) + technique=block_print (0.10).
    # similarity = 0.40 + 0.25 + 0.10 = 0.75 ≥ 0.60 threshold.
    # color_family 0.25 는 dominant family 매칭 시 추가로 들어감 (LAB 동일색이라 자연 통과).
    analysis = GarmentAnalysis(
        is_india_ethnic_wear=True,
        outfits=[_make_outfit(
            picks=["pool_00", "pool_01"],
            lower_garment_type="palazzo",
            lower_is_ethnic=True,
            fabric="cotton",
            technique="block_print",
        )],
    )
    stub_llm = _StubLLM(analysis)
    adapter = _build_adapter(tmp_path, stub_llm)

    rgb_red = np.full((100, 100, 3), VIBRANT_RED_RGB, dtype=np.uint8)

    # _resolve_paths 자체는 (image_paths, video_paths) tuple 반환 — 둘 다 채운다.
    monkeypatch.setattr(
        "vision.canonical_extractor.run_segformer", _seg_stub_top_half_upper,
    )
    monkeypatch.setattr(
        "vision.pipeline_b_adapter._resolve_local_paths",
        lambda _item, _root: [Path("img_A.jpg")],
    )
    monkeypatch.setattr(
        "vision.pipeline_b_adapter._resolve_local_video_paths",
        lambda _item, _root: [Path("clip.mp4")],
    )
    monkeypatch.setattr(
        "vision.pipeline_b_adapter._load_images",
        lambda paths: [(p.name, p.name.encode(), rgb_red) for p in paths],
    )
    # video frame stub — 영상 1건이 frame 2개 yield 한 척.
    monkeypatch.setattr(
        "vision.pipeline_b_adapter._load_video_frames",
        lambda paths, cfg: [
            ("clip_f10", b"clip_f10_bytes", rgb_red),
            ("clip_f25", b"clip_f25_bytes", rgb_red),
        ],
    )

    item = _make_item(image_urls=["img_A.jpg"])
    results = adapter.extract_visual([item])

    # image 1 + video frame 2 = LLM 3회 호출.
    assert len(stub_llm.calls) == 3
    # bytes payload 분기 — image 와 video frame 이 모두 흐름.
    assert b"img_A.jpg" in stub_llm.calls
    assert b"clip_f10_bytes" in stub_llm.calls
    assert b"clip_f25_bytes" in stub_llm.calls
    # 동일 stub 3 frame → outfit_dedup 가 attribute 기반 (image_id 무관) 으로 1 canonical 병합.
    # video Reel 20-frame 이 20 canonical 으로 폭발하지 않는 것을 핀.
    assert len(results[0].canonicals) == 1
    assert len(results[0].post_palette) >= 1


def test_extract_one_video_only_when_no_images(monkeypatch, tmp_path) -> None:
    """image_urls=[] 인 IG Reel/YT 영상 → video frame 만으로 분석 진행."""
    analysis = GarmentAnalysis(
        is_india_ethnic_wear=True,
        outfits=[_make_outfit(
            picks=["pool_00", "pool_01"],
            lower_garment_type="palazzo",
            lower_is_ethnic=True,
            technique="block_print",
        )],
    )
    stub_llm = _StubLLM(analysis)
    adapter = _build_adapter(tmp_path, stub_llm)

    rgb_red = np.full((100, 100, 3), VIBRANT_RED_RGB, dtype=np.uint8)

    monkeypatch.setattr(
        "vision.canonical_extractor.run_segformer", _seg_stub_top_half_upper,
    )
    monkeypatch.setattr(
        "vision.pipeline_b_adapter._resolve_local_paths",
        lambda _item, _root: [],
    )
    monkeypatch.setattr(
        "vision.pipeline_b_adapter._resolve_local_video_paths",
        lambda _item, _root: [Path("reel.mp4")],
    )
    monkeypatch.setattr(
        "vision.pipeline_b_adapter._load_images",
        lambda _paths: [],
    )
    monkeypatch.setattr(
        "vision.pipeline_b_adapter._load_video_frames",
        lambda paths, cfg: [("reel_f5", b"reel_f5_bytes", rgb_red)],
    )

    item = _make_item(image_urls=[])
    results = adapter.extract_visual([item])

    assert len(stub_llm.calls) == 1
    assert len(results[0].canonicals) == 1


def test_scene_filter_default_noop_passes_through(monkeypatch, tmp_path) -> None:
    """scene_filter 인자 None → NoopSceneFilter 디폴트 → stage=disabled, 게이트 우회."""
    analysis = GarmentAnalysis(
        is_india_ethnic_wear=True, outfits=[_make_outfit(picks=["pool_00"])],
    )
    stub_llm = _StubLLM(analysis)
    adapter = _build_adapter(tmp_path, stub_llm)  # scene_filter=None
    _install_stubs(monkeypatch, adapter, [Path("image_A.jpg")])

    results = adapter.extract_visual([_make_item()])
    assert len(results[0].canonicals) == 1
    assert len(stub_llm.calls) == 1
