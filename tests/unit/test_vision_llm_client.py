"""VisionLLMClient Protocol + FakeVisionLLMClient 결정론 테스트."""
from __future__ import annotations

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
