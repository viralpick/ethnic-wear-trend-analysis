"""Vision raw word → GarmentType/Fabric/Technique enum 정규화 (2026-04-29).

배경: Gemini prompt 가 garment_type/fabric/technique 를 free-form single-word 로
받음 (silhouette 만 enum 강제). cluster_key 가 자유 문자열 그대로 흘러가서
- ethnic 외 단어 (`t_shirt`, `jeans`, `palazzo`) 가 cluster 에 등장
- spec §5.1 의 enum 가정과 어긋남
- normalize 단계에서 enum 매핑 + ethnic 측 garment 만 사용 (case D drop) 적용

정책 결정 (2026-04-29 user 위임):
- (a) `embroidery` (889건) → THREAD_EMBROIDERY (general 자수의 default)
- (b) `print` (172건) → DIGITAL_PRINT (modern print, IG 우세)
- (c) `kurti` → STRAIGHT_KURTA (kurti 는 작은/캐주얼 kurta)
- (d) `choli` + ethnic `blouse` → CASUAL_SAREE (saree blouse 가 다수)
- (e) `lehenga` upper → ETHNIC_DRESS (lehenga choli 통합 단위)
- (f) case D (upper 비-ethnic, lower 만 ethnic) → drop from cluster fan-out

매핑 외 단어는 None 반환 → cluster fan-out 에서 자연 제외.
"""
from __future__ import annotations

from contracts.common import Fabric, GarmentType, Technique
from contracts.vision import EthnicOutfit


# ─────────────── GarmentType (upper-side, ethnic) ───────────────
# upper_garment_type raw → enum 매핑. 키는 lowercase + 공백/하이픈 → 언더스코어 정규화 후 비교.
_GARMENT_MAP: dict[str, GarmentType] = {
    # straight kurta family
    "kurta": GarmentType.STRAIGHT_KURTA,
    "kurti": GarmentType.STRAIGHT_KURTA,  # 작은 kurta — 같은 카테고리 (정책 c)
    "straight_kurta": GarmentType.STRAIGHT_KURTA,
    # a-line kurta
    "a_line_kurta": GarmentType.A_LINE_KURTA,
    "a-line_kurta": GarmentType.A_LINE_KURTA,
    # kurta_set
    "kurta_set": GarmentType.KURTA_SET,
    "kameez": GarmentType.KURTA_SET,  # salwar kameez 의 kameez = kurta_set
    # kurta dress
    "kurta_dress": GarmentType.KURTA_DRESS,
    # co-ord
    "co_ord": GarmentType.CO_ORD,
    "co-ord": GarmentType.CO_ORD,
    "coord": GarmentType.CO_ORD,
    # anarkali
    "anarkali": GarmentType.ANARKALI,
    # tunic
    "tunic": GarmentType.TUNIC,
    # casual saree (saree drape + saree blouse 흡수)
    "saree": GarmentType.CASUAL_SAREE,
    "casual_saree": GarmentType.CASUAL_SAREE,
    "choli": GarmentType.CASUAL_SAREE,  # saree blouse (정책 d)
    "blouse": GarmentType.CASUAL_SAREE,  # ethnic only — 호출 측 가드 (정책 d)
    # ethnic dress (lehenga choli 통합 + kaftan + dress + jumpsuit)
    "ethnic_dress": GarmentType.ETHNIC_DRESS,
    "dress": GarmentType.ETHNIC_DRESS,  # ethnic only — 호출 측 가드
    "kaftan": GarmentType.ETHNIC_DRESS,
    "lehenga": GarmentType.ETHNIC_DRESS,  # 정책 e (lehenga choli unit)
    "jumpsuit": GarmentType.ETHNIC_DRESS,  # ethnic only
    # ethnic shirt (sherwani, kurta-style shirt)
    "ethnic_shirt": GarmentType.ETHNIC_SHIRT,
    "sherwani": GarmentType.ETHNIC_SHIRT,
    # fusion top
    "fusion_top": GarmentType.FUSION_TOP,
}


# ─────────────── Fabric ───────────────
_FABRIC_MAP: dict[str, Fabric] = {
    "cotton": Fabric.COTTON,
    "cotton_blend": Fabric.COTTON_BLEND,
    "linen": Fabric.LINEN,
    "linen_blend": Fabric.LINEN_BLEND,
    "rayon": Fabric.RAYON,
    "modal": Fabric.MODAL,
    "chanderi": Fabric.CHANDERI,
    "georgette": Fabric.GEORGETTE,
    "crepe": Fabric.CREPE,
    "chiffon": Fabric.CHIFFON,
    "khadi": Fabric.KHADI,
    "polyester_blend": Fabric.POLYESTER_BLEND,
    "jacquard": Fabric.JACQUARD,
    "silk": Fabric.SILK,
    "organza": Fabric.ORGANZA,
    "satin": Fabric.SATIN,
    "net": Fabric.NET,
    "velvet": Fabric.VELVET,
    # 비-ethnic 또는 너무 적음 → drop (None)
    # knit, denim, lace, leather, polyester
}


# ─────────────── Technique ───────────────
_TECHNIQUE_MAP: dict[str, Technique] = {
    # 기존 enum 직접
    "solid": Technique.SOLID,
    "plain": Technique.SOLID,  # plain == solid
    "self_texture": Technique.SELF_TEXTURE,
    "woven": Technique.SELF_TEXTURE,  # 직조 무늬
    "weaving": Technique.SELF_TEXTURE,
    "chikankari": Technique.CHIKANKARI,
    "block_print": Technique.BLOCK_PRINT,
    "floral_print": Technique.FLORAL_PRINT,
    "geometric_print": Technique.GEOMETRIC_PRINT,
    "ethnic_motif": Technique.ETHNIC_MOTIF,
    "digital_print": Technique.DIGITAL_PRINT,
    "print": Technique.DIGITAL_PRINT,  # 정책 b — generic print 의 default
    "printed": Technique.DIGITAL_PRINT,
    "thread_embroidery": Technique.THREAD_EMBROIDERY,
    "embroidery": Technique.THREAD_EMBROIDERY,  # 정책 a — generic 자수
    "mirror_work": Technique.MIRROR_WORK,
    "schiffli": Technique.SCHIFFLI,
    "pintuck": Technique.PINTUCK,
    "lace_cutwork": Technique.LACE_CUTWORK,
    "lace": Technique.LACE_CUTWORK,
    "cutwork": Technique.LACE_CUTWORK,
    "gota_patti": Technique.GOTA_PATTI,
    "gotapatti": Technique.GOTA_PATTI,
    # 신규 enum
    "sequin_work": Technique.SEQUIN_WORK,
    "sequins": Technique.SEQUIN_WORK,
    "sequin": Technique.SEQUIN_WORK,
    "ikat": Technique.IKAT,
    "brocade": Technique.BROCADE,
    "beadwork": Technique.BEADWORK,
    "bandhani": Technique.BANDHANI,
    "zari_zardosi": Technique.ZARI_ZARDOSI,
    "zari": Technique.ZARI_ZARDOSI,
    "zardosi": Technique.ZARI_ZARDOSI,
    "zardozi": Technique.ZARI_ZARDOSI,
    "kalamkari": Technique.KALAMKARI,
    # drop: crochet, batik, tie_dye, pleating (각 1건, 너무 적음)
}


def _normalize_word(raw: str | None) -> str | None:
    """공백/하이픈을 언더스코어로 통일 후 lowercase."""
    if raw is None:
        return None
    return raw.strip().lower().replace(" ", "_").replace("-", "_")


def normalize_garment_for_cluster(rep: EthnicOutfit) -> GarmentType | None:
    """case D drop + ethnic-side garment 만 enum 매핑.

    - dress_as_single=True 또는 upper_is_ethnic=True 면 upper_garment_type 매핑 시도
    - upper_is_ethnic=False (case D) 면 None 반환 → cluster 미참여
    - 매핑 외 단어 (top, shirt, kurti 외 등) 면 None
    """
    if rep.dress_as_single:
        if not rep.upper_is_ethnic:
            return None
        word = _normalize_word(rep.upper_garment_type)
        return _GARMENT_MAP.get(word) if word else None
    # 2-piece: ethnic 측만 사용. upper_is_ethnic 우선
    if rep.upper_is_ethnic:
        word = _normalize_word(rep.upper_garment_type)
        return _GARMENT_MAP.get(word) if word else None
    # case D — upper 만 비-ethnic 이면 lower 가 ethnic 이어도 cluster 미참여
    return None


def normalize_fabric(rep: EthnicOutfit) -> Fabric | None:
    """fabric raw → Fabric enum. 매핑 외 단어는 None."""
    word = _normalize_word(rep.fabric)
    return _FABRIC_MAP.get(word) if word else None


def normalize_technique(rep: EthnicOutfit) -> Technique | None:
    """technique raw → Technique enum. 매핑 외 단어는 None."""
    word = _normalize_word(rep.technique)
    return _TECHNIQUE_MAP.get(word) if word else None
