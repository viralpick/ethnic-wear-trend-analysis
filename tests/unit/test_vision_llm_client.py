"""VisionLLMClient Protocol + FakeVisionLLMClient 결정론 테스트."""
from __future__ import annotations

import pytest

from contracts.vision import GarmentAnalysis
from vision.llm_client import FakeVisionLLMClient, VisionLLMClient


def _preset() -> list[dict[str, str]]:
    return [
        {"name": "pool_00", "hex": "#384F9A"},
        {"name": "pool_02", "hex": "#141314"},
        {"name": "saffron", "hex": "#DB633B"},
        {"name": "cream_ivory", "hex": "#E3CEB4"},
    ]


def test_fake_client_satisfies_protocol() -> None:
    client = FakeVisionLLMClient()
    assert isinstance(client, VisionLLMClient)


def test_fake_client_deterministic_same_input() -> None:
    client = FakeVisionLLMClient()
    a = client.extract_garment(b"image-bytes-1", preset=_preset())
    b = client.extract_garment(b"image-bytes-1", preset=_preset())
    # byte-level identical 직렬화
    assert a.model_dump_json() == b.model_dump_json()


def test_fake_client_differs_across_inputs() -> None:
    client = FakeVisionLLMClient()
    a = client.extract_garment(b"image-A", preset=_preset())
    b = client.extract_garment(b"image-B", preset=_preset())
    assert a.model_dump_json() != b.model_dump_json()


def test_fake_client_prompt_version_salts_seed() -> None:
    c1 = FakeVisionLLMClient(prompt_version="fake-v1")
    c2 = FakeVisionLLMClient(prompt_version="fake-v2")
    a = c1.extract_garment(b"img", preset=_preset())
    b = c2.extract_garment(b"img", preset=_preset())
    # prompt_version 바뀌면 결과도 달라짐 — drift invalidation 가드
    assert a.model_dump_json() != b.model_dump_json()


def test_fake_client_returns_garment_analysis() -> None:
    client = FakeVisionLLMClient()
    result = client.extract_garment(b"some-image", preset=_preset())
    assert isinstance(result, GarmentAnalysis)
    # binary=True 면 outfit 1~2, False 면 0
    if result.is_india_ethnic_wear:
        assert 1 <= len(result.outfits) <= 2
    else:
        assert result.outfits == []


def test_fake_client_outfit_shape() -> None:
    client = FakeVisionLLMClient()
    # 여러 입력 돌려서 적어도 1건은 ethnic=True 나와야 — 85% 확률
    preset = _preset()
    found_ethnic = False
    for i in range(10):
        result = client.extract_garment(f"img-{i}".encode(), preset=preset)
        if not result.is_india_ethnic_wear:
            continue
        found_ethnic = True
        for outfit in result.outfits:
            # bbox 유효성은 Pydantic 이 이미 걸렀고, here 에선 dress_as_single 논리만
            if outfit.dress_as_single:
                assert outfit.lower_garment_type is None
            # picks 는 preset 이름에서만
            for name in outfit.color_preset_picks_top3:
                assert name in {p["name"] for p in preset}
            # fabric / technique 은 None 허용, 값이 있으면 lowercase single-word
            if outfit.fabric is not None:
                assert outfit.fabric.islower() and " " not in outfit.fabric
            if outfit.technique is not None:
                assert outfit.technique.islower() and " " not in outfit.technique
    assert found_ethnic, "10 sample 중 ethnic=True 단 한 번도 안 나옴 — 분포 오류"


def test_fake_client_empty_preset_ok() -> None:
    # preset 이 비어있어도 raise 하지 않음 (defensive)
    client = FakeVisionLLMClient()
    result = client.extract_garment(b"img", preset=[])
    assert isinstance(result, GarmentAnalysis)



# ---- color.B v0.10 — FakeVisionLLMClient.pick_colors_from_kmeans ----


def _v010_classification() -> dict[str, object]:
    return {
        "upper_garment_type": "kurta",
        "lower_garment_type": "palazzo",
        "upper_is_ethnic": True,
        "lower_is_ethnic": True,
        "dress_as_single": False,
    }


def _v010_clusters() -> list[dict[str, object]]:
    return [
        {"index": 0, "hex": "#f5e7c4", "share": 0.42},
        {"index": 1, "hex": "#c11a4a", "share": 0.31},
        {"index": 2, "hex": "#2a6b3f", "share": 0.18},
        {"index": 3, "hex": "#1a1a1a", "share": 0.09},
    ]


def test_fake_pick_colors_from_kmeans_deterministic_same_input() -> None:
    client = FakeVisionLLMClient(prompt_version="v0.10-fake")
    img = b"image-bytes-v010"
    out1 = client.pick_colors_from_kmeans(
        img,
        garment_classification=_v010_classification(),
        kmeans_clusters=_v010_clusters(),
    )
    out2 = client.pick_colors_from_kmeans(
        img,
        garment_classification=_v010_classification(),
        kmeans_clusters=_v010_clusters(),
    )
    assert out1.model_dump() == out2.model_dump()


def test_fake_pick_colors_from_kmeans_cluster_index_within_range() -> None:
    client = FakeVisionLLMClient(prompt_version="v0.10-fake")
    clusters = _v010_clusters()
    out = client.pick_colors_from_kmeans(
        b"img",
        garment_classification=_v010_classification(),
        kmeans_clusters=clusters,
    )
    assert 1 <= len(out.picks) <= 3
    for pick in out.picks:
        assert 0 <= pick.cluster_index < len(clusters)


def test_fake_pick_colors_from_kmeans_empty_clusters_raises() -> None:
    client = FakeVisionLLMClient(prompt_version="v0.10-fake")
    with pytest.raises(ValueError, match="must not be empty"):
        client.pick_colors_from_kmeans(
            b"img",
            garment_classification=_v010_classification(),
            kmeans_clusters=[],
        )


def test_fake_pick_colors_from_kmeans_preset_label_nonempty() -> None:
    client = FakeVisionLLMClient(prompt_version="v0.10-fake")
    out = client.pick_colors_from_kmeans(
        b"img",
        garment_classification=_v010_classification(),
        kmeans_clusters=_v010_clusters(),
    )
    for pick in out.picks:
        assert pick.preset_label


def test_fake_pick_colors_from_kmeans_picks_no_duplicate_index() -> None:
    client = FakeVisionLLMClient(prompt_version="v0.10-fake")
    out = client.pick_colors_from_kmeans(
        b"img",
        garment_classification=_v010_classification(),
        kmeans_clusters=_v010_clusters(),
    )
    indices = [p.cluster_index for p in out.picks]
    assert len(indices) == len(set(indices))


def test_fake_pick_colors_from_kmeans_differs_across_inputs() -> None:
    client = FakeVisionLLMClient(prompt_version="v0.10-fake")
    a = client.pick_colors_from_kmeans(
        b"image-A",
        garment_classification=_v010_classification(),
        kmeans_clusters=_v010_clusters(),
    )
    b = client.pick_colors_from_kmeans(
        b"image-B",
        garment_classification=_v010_classification(),
        kmeans_clusters=_v010_clusters(),
    )
    # 다른 입력은 거의 항상 다른 출력 (확률 1/N^3 으로 동일 가능하지만 N=4 — 충분히 분리)
    assert a.model_dump() != b.model_dump()
