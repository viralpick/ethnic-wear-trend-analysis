"""LocalJSONCache + compute_cache_key 단위 테스트."""
from __future__ import annotations

import json

import pytest

from contracts.common import Silhouette
from contracts.vision import (
    EthnicOutfit,
    GarmentAnalysis,
    KMeansAnchoredPick,
    KMeansAnchoredPickResponse,
)
from vision.llm_cache import (
    LocalJSONCache,
    LocalJSONCacheV010,
    compute_cache_key,
    compute_v010_cache_key,
)


def _analysis() -> GarmentAnalysis:
    return GarmentAnalysis(
        is_india_ethnic_wear=True,
        outfits=[
            EthnicOutfit(
                person_bbox=(0.2, 0.1, 0.5, 0.7),
                person_bbox_area_ratio=0.35,
                upper_garment_type="kurta",
                lower_garment_type="palazzo",
                dress_as_single=False,
                silhouette=Silhouette.A_LINE,
                color_preset_picks_top3=["pool_00", "saffron", "pool_12"],
            )
        ],
    )


def test_cache_key_deterministic_same_inputs() -> None:
    k1 = compute_cache_key(b"img", prompt_version="v0.1", model_id="gemini-2.5-flash")
    k2 = compute_cache_key(b"img", prompt_version="v0.1", model_id="gemini-2.5-flash")
    assert k1 == k2
    assert len(k1) == 64  # sha256 hex


def test_cache_key_image_change_bumps_key() -> None:
    k1 = compute_cache_key(b"img-A", prompt_version="v0.1", model_id="gemini-2.5-flash")
    k2 = compute_cache_key(b"img-B", prompt_version="v0.1", model_id="gemini-2.5-flash")
    assert k1 != k2


def test_cache_key_prompt_version_bumps_key() -> None:
    k1 = compute_cache_key(b"img", prompt_version="v0.1", model_id="gemini-2.5-flash")
    k2 = compute_cache_key(b"img", prompt_version="v0.2", model_id="gemini-2.5-flash")
    assert k1 != k2


def test_cache_key_model_id_bumps_key() -> None:
    k1 = compute_cache_key(b"img", prompt_version="v0.1", model_id="gemini-2.5-flash")
    k2 = compute_cache_key(b"img", prompt_version="v0.1", model_id="gpt-5-mini")
    assert k1 != k2


def test_cache_key_separator_prevents_concat_collision() -> None:
    # 분리자 없으면 ("a"+"bc") 와 ("ab"+"c") 가 같은 키로 충돌 — 방지 확인
    k1 = compute_cache_key(b"img", prompt_version="a", model_id="bc")
    k2 = compute_cache_key(b"img", prompt_version="ab", model_id="c")
    assert k1 != k2


def test_local_cache_miss_returns_none(tmp_path) -> None:
    cache = LocalJSONCache(
        tmp_path, model_id="gemini-2.5-flash", prompt_version="v0.1"
    )
    assert cache.get("nonexistent-key") is None


def test_local_cache_put_then_get_roundtrip(tmp_path) -> None:
    cache = LocalJSONCache(
        tmp_path, model_id="gemini-2.5-flash", prompt_version="v0.1"
    )
    analysis = _analysis()
    cache.put("abc123", analysis)
    restored = cache.get("abc123")
    assert restored == analysis


def test_local_cache_writes_envelope_metadata(tmp_path) -> None:
    cache = LocalJSONCache(
        tmp_path, model_id="gemini-2.5-flash", prompt_version="v0.1"
    )
    cache.put("abc123", _analysis())
    path = tmp_path / "gemini-2.5-flash" / "abc123.json"
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["cache_key"] == "abc123"
    assert data["model_id"] == "gemini-2.5-flash"
    assert data["prompt_version"] == "v0.1"
    assert "stored_at" in data
    assert "garment_analysis" in data


def test_local_cache_model_id_isolates_directory(tmp_path) -> None:
    c_gemini = LocalJSONCache(tmp_path, model_id="gemini-2.5-flash", prompt_version="v0.1")
    c_gpt = LocalJSONCache(tmp_path, model_id="gpt-5-mini", prompt_version="v0.1")
    c_gemini.put("shared-key", _analysis())
    # 다른 모델 캐시에선 miss
    assert c_gpt.get("shared-key") is None


def test_local_cache_atomic_write_leaves_no_tmp(tmp_path) -> None:
    cache = LocalJSONCache(tmp_path, model_id="gemini-2.5-flash", prompt_version="v0.1")
    cache.put("abc", _analysis())
    leftovers = list(tmp_path.rglob("*.tmp"))
    assert leftovers == []


def test_local_cache_corrupt_file_returns_none_gracefully(tmp_path) -> None:
    cache = LocalJSONCache(tmp_path, model_id="gemini-2.5-flash", prompt_version="v0.1")
    path = tmp_path / "gemini-2.5-flash" / "corrupt.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"cache_key": "corrupt"}', encoding="utf-8")  # garment_analysis 누락
    assert cache.get("corrupt") is None


def test_local_cache_corrupt_json_raises(tmp_path) -> None:
    cache = LocalJSONCache(tmp_path, model_id="gemini-2.5-flash", prompt_version="v0.1")
    path = tmp_path / "gemini-2.5-flash" / "badjson.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not valid json", encoding="utf-8")
    # 파싱 실패는 raise — silent 로 숨기지 않음 (실패 숨김 금지)
    with pytest.raises(json.JSONDecodeError):
        cache.get("badjson")



# ---- color.B v0.10 — compute_v010_cache_key + LocalJSONCacheV010 ----


def _v010_response() -> KMeansAnchoredPickResponse:
    return KMeansAnchoredPickResponse(
        picks=[
            KMeansAnchoredPick(cluster_index=0, preset_label="cream_ivory"),
            KMeansAnchoredPick(cluster_index=2, preset_label="bottle_green"),
        ]
    )


def test_v010_cache_key_deterministic_same_inputs() -> None:
    a = compute_v010_cache_key(
        b"img", prompt_version="v0.10", model_id="gemini-2.5-flash",
        cluster_hexes=("#aa0000", "#00aa00", "#0000aa"),
    )
    b = compute_v010_cache_key(
        b"img", prompt_version="v0.10", model_id="gemini-2.5-flash",
        cluster_hexes=("#aa0000", "#00aa00", "#0000aa"),
    )
    assert a == b


def test_v010_cache_key_image_change_bumps_key() -> None:
    a = compute_v010_cache_key(
        b"img-A", prompt_version="v0.10", model_id="gemini-2.5-flash",
        cluster_hexes=("#aa0000",),
    )
    b = compute_v010_cache_key(
        b"img-B", prompt_version="v0.10", model_id="gemini-2.5-flash",
        cluster_hexes=("#aa0000",),
    )
    assert a != b


def test_v010_cache_key_cluster_hexes_change_bumps_key() -> None:
    a = compute_v010_cache_key(
        b"img", prompt_version="v0.10", model_id="gemini-2.5-flash",
        cluster_hexes=("#aa0000", "#00aa00"),
    )
    b = compute_v010_cache_key(
        b"img", prompt_version="v0.10", model_id="gemini-2.5-flash",
        cluster_hexes=("#aa0000", "#0000aa"),
    )
    assert a != b


def test_v010_cache_key_cluster_order_matters() -> None:
    # share desc 정렬 가정 — 순서 다르면 별 cache. KMeans 결과 변동을 자동 무효화.
    a = compute_v010_cache_key(
        b"img", prompt_version="v0.10", model_id="gemini-2.5-flash",
        cluster_hexes=("#aa0000", "#00aa00"),
    )
    b = compute_v010_cache_key(
        b"img", prompt_version="v0.10", model_id="gemini-2.5-flash",
        cluster_hexes=("#00aa00", "#aa0000"),
    )
    assert a != b


def test_v010_cache_key_prompt_version_bumps_key() -> None:
    a = compute_v010_cache_key(
        b"img", prompt_version="v0.10", model_id="gemini-2.5-flash",
        cluster_hexes=("#aa0000",),
    )
    b = compute_v010_cache_key(
        b"img", prompt_version="v0.11", model_id="gemini-2.5-flash",
        cluster_hexes=("#aa0000",),
    )
    assert a != b


def test_local_cache_v010_miss_returns_none(tmp_path) -> None:
    cache = LocalJSONCacheV010(
        tmp_path, model_id="gemini-2.5-flash", prompt_version="v0.10",
    )
    assert cache.get("missing-key") is None


def test_local_cache_v010_put_then_get_roundtrip(tmp_path) -> None:
    cache = LocalJSONCacheV010(
        tmp_path, model_id="gemini-2.5-flash", prompt_version="v0.10",
    )
    resp = _v010_response()
    cache.put("k1", resp)
    got = cache.get("k1")
    assert got is not None
    assert got.model_dump() == resp.model_dump()


def test_local_cache_v010_isolated_from_v09(tmp_path) -> None:
    # 같은 base_dir + 같은 model_id 라도 LocalJSONCache (v0.9) 와 LocalJSONCacheV010 가
    # 같은 cache_key 를 다른 파일에 저장 (v010/ subdir 분리). 충돌 없음.
    v09 = LocalJSONCache(
        tmp_path, model_id="gemini-2.5-flash", prompt_version="v0.9",
    )
    v010 = LocalJSONCacheV010(
        tmp_path, model_id="gemini-2.5-flash", prompt_version="v0.10",
    )
    v09.put("shared-key", _analysis())
    v010.put("shared-key", _v010_response())
    # 각자 자기 envelope 만 보임
    assert v09.get("shared-key") is not None
    assert v010.get("shared-key") is not None
    # cross-read miss
    assert v010.get("non-existing-key") is None


def test_local_cache_v010_writes_envelope_metadata(tmp_path) -> None:
    cache = LocalJSONCacheV010(
        tmp_path, model_id="gemini-2.5-flash", prompt_version="v0.10",
    )
    cache.put("k1", _v010_response())
    file_path = tmp_path / "gemini-2.5-flash" / "v010" / "k1.json"
    assert file_path.exists()
    envelope = json.loads(file_path.read_text(encoding="utf-8"))
    assert envelope["model_id"] == "gemini-2.5-flash"
    assert envelope["prompt_version"] == "v0.10"
    assert envelope["cache_key"] == "k1"
    assert "stored_at" in envelope
    assert envelope["pick_response"]["picks"][0]["cluster_index"] == 0
