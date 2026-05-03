"""spec §6.2 해시태그/키워드 → 속성 매핑 테이블 (verbatim).

- forward index: enum → AttributeMapping(hashtags, keywords) — 문서화/반복 용도
- reverse index: tag(or keyword) → enum — 추출기 빠른 lookup 용도

해시태그는 '#' 제외한 lowercase 로 저장된다. 키워드는 lowercase 기준.
silhouette, color, brand_mentioned 는 VLM 또는 외부 registry 영역이라 §6.2 에 없음 —
이 파일은 rule 기반으로 잡히는 5개 속성만 다룬다.
"""
from __future__ import annotations

from dataclasses import dataclass

from contracts.common import Fabric, GarmentType, Occasion, StylingCombo, Technique


@dataclass(frozen=True)
class AttributeMapping:
    hashtags: tuple[str, ...]   # '#' 제외 lowercase
    keywords: tuple[str, ...]   # lowercase, 공백 포함 가능


# --------------------------------------------------------------------------- #
# garment_type (spec §6.2)
# --------------------------------------------------------------------------- #
GARMENT_TYPE_MAPPINGS: dict[GarmentType, AttributeMapping] = {
    GarmentType.KURTA_SET: AttributeMapping(
        hashtags=(
            "kurtaset", "kurtasets", "kurtiset", "kurtapalazzoset", "kurtasetsonline", "kurtastyle",
            # 2026-05-02 — 12w eval 자명 누락 (suits 변형 / salwar kameez)
            "salwarkameez", "salwarsuit", "suits", "suit", "punjabisuit",
            "partywearsuits", "partywearsuit", "designersuits", "designersuit",
            "cottonsuits", "cottonsuit", "weddingsuits",
        ),
        keywords=(
            "kurta set", "kurta palazzo set", "kurta pant set", "3 piece set",
            "salwar kameez", "salwar suit", "punjabi suit",
        ),
    ),
    GarmentType.ANARKALI: AttributeMapping(
        hashtags=("anarkali", "anarkalisuit", "anarkalikurta", "anarkalidress"),
        keywords=("anarkali",),
    ),
    GarmentType.CO_ORD: AttributeMapping(
        hashtags=("coordset", "coordsets", "twinningset", "matchingset"),
        keywords=("co-ord", "coord set", "matching set"),
    ),
    GarmentType.KURTA_DRESS: AttributeMapping(
        hashtags=("kurtadress", "ethnicdress", "indiandress"),
        keywords=("kurta dress", "ethnic dress"),
    ),
    GarmentType.CASUAL_SAREE: AttributeMapping(
        hashtags=(
            "saree", "sareelove", "readytowearsaree", "casualsaree", "officesaree",
            "sareeindia", "sareesofinstagram", "sareeswag", "sari",
        ),
        keywords=("saree", "sari", "ready to wear saree"),
    ),
    GarmentType.STRAIGHT_KURTA: AttributeMapping(
        hashtags=("straightkurta", "straightkurti", "straightcut"),
        keywords=("straight kurta", "straight cut"),
    ),
    GarmentType.TUNIC: AttributeMapping(
        hashtags=("kurti", "tunic", "ethnictunic", "shortkurti", "longkurti", "kurties"),
        keywords=("kurti", "tunic"),
    ),
    GarmentType.FUSION_TOP: AttributeMapping(
        hashtags=("fusiontop", "croptop", "peplumtop", "fusionwear", "peplumkurta"),
        keywords=("fusion top", "peplum", "crop top"),
    ),
    GarmentType.ETHNIC_SHIRT: AttributeMapping(
        hashtags=("ethnicshirt", "indianshirt", "bandcollarshirt"),
        keywords=("ethnic shirt", "band collar"),
    ),
    GarmentType.A_LINE_KURTA: AttributeMapping(
        hashtags=("alinekurta", "alinekurti"),
        keywords=("a-line kurta", "a line kurta"),
    ),
    GarmentType.ETHNIC_DRESS: AttributeMapping(
        hashtags=("indowesterndress",),
        keywords=("indo western dress", "tiered dress", "shirt dress ethnic"),
    ),
}


# --------------------------------------------------------------------------- #
# technique (spec §6.2)
# --------------------------------------------------------------------------- #
TECHNIQUE_MAPPINGS: dict[Technique, AttributeMapping] = {
    Technique.CHIKANKARI: AttributeMapping(
        hashtags=(
            "chikankari", "chikan", "chikanwork", "lucknowi", "lucknowichikankari",
            "chikankarisuit", "chikankarikurta", "chikankaricollection",
            "chickenkurta",
        ),
        keywords=("chikankari", "chikan", "lucknowi"),
    ),
    Technique.BLOCK_PRINT: AttributeMapping(
        hashtags=("blockprint", "handblockprint", "jaipuriprint", "ajrakh", "bagru", "handblock", "blockprintkurta"),
        keywords=("block print", "hand block", "ajrakh", "bagru", "jaipur print"),
    ),
    Technique.SOLID: AttributeMapping(
        hashtags=("solid", "solidkurta", "plain", "minimal"),
        keywords=("solid", "plain"),
    ),
    Technique.FLORAL_PRINT: AttributeMapping(
        hashtags=("floralprint", "floralkurta", "floralsuit", "ditsy", "botanical"),
        keywords=("floral", "ditsy", "botanical"),
    ),
    Technique.GEOMETRIC_PRINT: AttributeMapping(
        hashtags=("stripes", "checks", "ikat", "geometric", "chevron", "abstract"),
        keywords=("stripe", "check", "ikat", "geometric"),
    ),
    Technique.ETHNIC_MOTIF: AttributeMapping(
        hashtags=("paisley", "kalamkari", "bandhani", "bandhej", "buta"),
        keywords=("paisley", "kalamkari", "bandhani", "bandhej"),
    ),
    Technique.THREAD_EMBROIDERY: AttributeMapping(
        hashtags=("embroidery", "threadwork", "threadembroidery", "embroidered"),
        keywords=("embroidery", "thread work", "embroidered"),
    ),
    Technique.MIRROR_WORK: AttributeMapping(
        hashtags=("mirrorwork", "shisha", "mirrorembroidery"),
        keywords=("mirror work", "shisha"),
    ),
    Technique.SELF_TEXTURE: AttributeMapping(
        hashtags=("selfdesign", "selftexture", "selfstripe", "dobby", "jacquard"),
        keywords=("self design", "self texture", "jacquard", "dobby"),
    ),
    Technique.DIGITAL_PRINT: AttributeMapping(
        hashtags=("digitalprint", "digitalprintkurta"),
        keywords=("digital print",),
    ),
    Technique.PINTUCK: AttributeMapping(
        hashtags=("pintuck", "pintuckkurta"),
        keywords=("pintuck",),
    ),
    Technique.LACE_CUTWORK: AttributeMapping(
        hashtags=("lace", "cutwork", "crochet", "schiffli"),
        keywords=("lace", "cutwork", "schiffli"),
    ),
    Technique.GOTA_PATTI: AttributeMapping(
        hashtags=("gotapatti", "gota", "gotawork"),
        keywords=("gota patti", "gota work"),
    ),
}


# --------------------------------------------------------------------------- #
# fabric (spec §6.2)
# --------------------------------------------------------------------------- #
FABRIC_MAPPINGS: dict[Fabric, AttributeMapping] = {
    Fabric.COTTON: AttributeMapping(
        hashtags=(
            "cotton", "cottonkurta", "cottonsuit", "purecotton", "mulmul", "cambric",
            "cottonkurtaset", "summercotton", "mulmulkurta",
        ),
        keywords=("cotton", "mulmul", "muslin", "cambric"),
    ),
    Fabric.LINEN: AttributeMapping(
        hashtags=("linen", "linenkurta", "linenkurti", "purelinen", "linenblend", "linencotton"),
        keywords=("linen",),
    ),
    Fabric.RAYON: AttributeMapping(
        hashtags=("rayon", "viscose", "rayonkurta"),
        keywords=("rayon", "viscose"),
    ),
    Fabric.MODAL: AttributeMapping(
        hashtags=("modal", "modalsatin", "modalcotton"),
        keywords=("modal",),
    ),
    Fabric.CHANDERI: AttributeMapping(
        hashtags=("chanderi", "chanderikurta", "chandericotton"),
        keywords=("chanderi",),
    ),
    Fabric.GEORGETTE: AttributeMapping(
        hashtags=("georgette", "georgettekurta"),
        keywords=("georgette",),
    ),
    Fabric.KHADI: AttributeMapping(
        hashtags=("khadi", "khadicotton", "handspun", "khadikurta"),
        keywords=("khadi", "handspun"),
    ),
}


# --------------------------------------------------------------------------- #
# occasion (spec §6.2)
# --------------------------------------------------------------------------- #
OCCASION_MAPPINGS: dict[Occasion, AttributeMapping] = {
    Occasion.OFFICE: AttributeMapping(
        hashtags=(
            "officewear", "officekurta", "workwear", "workwearethnic",
            "indianofficewear", "officelook", "corporateethnic", "indowesternoffice",
            "officestyle", "worklook", "workoutfit", "businesscasual",
        ),
        keywords=(
            "office", "workwear", "work wear", "corporate",
            "desk to dinner", "professional", "9 to 5", "business casual",
        ),
    ),
    Occasion.CASUAL: AttributeMapping(
        hashtags=(
            "casualethnic", "everydayethnic", "dailywear", "casualwear",
            "casuallook", "indowestern", "casualootd", "relaxedlook",
            "homewear", "lounge",
        ),
        keywords=("casual", "everyday", "daily wear", "easy wear", "indo western"),
    ),
    Occasion.CAMPUS: AttributeMapping(
        hashtags=(
            "collegewear", "campuslook", "collegefashion",
            "studentfashion", "collegegirl", "campusstyle",
        ),
        keywords=("college", "campus", "university", "student"),
    ),
    Occasion.WEEKEND: AttributeMapping(
        hashtags=(
            "weekendlook", "brunch", "weekendoutfit", "sundaylook",
            "partywear", "partyoutfit", "eveningwear", "nightout",
            "datenight", "sundowner", "cocktailparty",
        ),
        keywords=("weekend", "brunch", "outing", "day out", "party", "evening wear", "date night"),
    ),
    Occasion.FESTIVE_LITE: AttributeMapping(
        hashtags=(
            "festivevibes", "pujalook", "festivelook", "akshayatritiya",
            "indianwedding", "wedding", "weddingguest", "weddingseason",
            "lehengacholi", "indianweddingdress", "designerlehenga",
            "bridalwear", "bridallook", "mehendi", "mehendicerumony",
            "sangeet", "haldi", "reception", "engagement",
            "navratri", "diwali", "eid", "durgapuja", "ganeshchaturthi",
            "festivewear", "functionwear", "familyfunction",
            # 2026-05-02 — 12w eval 자명 누락
            "weddingfashion", "weddingcollection", "weddingedit", "weddinginspiration",
        ),
        keywords=(
            "festive", "puja", "small function", "get together", "akshaya tritiya",
            "wedding", "mehendi", "sangeet", "haldi", "reception", "engagement",
            "navratri", "diwali", "eid", "festival", "function",
        ),
    ),
    Occasion.TRAVEL: AttributeMapping(
        hashtags=(
            "travelootd", "travelstyle", "travelwear", "vacationoutfit",
            "holidaylook", "traveldiaries", "tripoutfit",
        ),
        keywords=("travel", "trip", "vacation", "holiday", "sightseeing"),
    ),
}


# --------------------------------------------------------------------------- #
# styling_combo (spec §6.2)
# --------------------------------------------------------------------------- #
STYLING_COMBO_MAPPINGS: dict[StylingCombo, AttributeMapping] = {
    StylingCombo.WITH_PALAZZO: AttributeMapping(
        hashtags=("palazzo", "palazzoset", "kurtapalazzo"),
        keywords=("palazzo",),
    ),
    StylingCombo.WITH_PANTS: AttributeMapping(
        hashtags=("kurtapants", "cigarettepants", "straightpants", "trousers"),
        keywords=("pants", "cigarette pants", "straight pants", "trousers"),
    ),
    StylingCombo.WITH_CHURIDAR: AttributeMapping(
        hashtags=("churidar", "churidarset"),
        keywords=("churidar",),
    ),
    StylingCombo.WITH_DUPATTA: AttributeMapping(
        hashtags=("dupatta", "3pieceset", "suitset"),
        keywords=("dupatta", "3 piece", "suit set"),
    ),
    StylingCombo.STANDALONE: AttributeMapping(
        hashtags=("kurtadress", "ethnicdress", "onepieceethnic", "maxidress"),
        keywords=("dress", "one piece", "standalone"),
    ),
    StylingCombo.WITH_JACKET: AttributeMapping(
        hashtags=("jacket", "shrug", "overlay", "layered"),
        keywords=("jacket", "shrug", "overlay"),
    ),
    StylingCombo.WITH_JEANS: AttributeMapping(
        hashtags=("kurtawithjeans", "ethnicwithjeans", "fusionlook"),
        keywords=("with jeans", "denim"),
    ),
}


# --------------------------------------------------------------------------- #
# Reverse indexes — tag/keyword → enum. 추출기가 O(1) lookup 으로 사용.
# --------------------------------------------------------------------------- #
def _build_tag_index(forward: dict) -> dict[str, object]:
    return {tag: key for key, mapping in forward.items() for tag in mapping.hashtags}


def _build_keyword_index(forward: dict) -> dict[str, object]:
    return {kw: key for key, mapping in forward.items() for kw in mapping.keywords}


GARMENT_TAG_INDEX: dict[str, GarmentType] = _build_tag_index(GARMENT_TYPE_MAPPINGS)
GARMENT_KEYWORD_INDEX: dict[str, GarmentType] = _build_keyword_index(GARMENT_TYPE_MAPPINGS)
TECHNIQUE_TAG_INDEX: dict[str, Technique] = _build_tag_index(TECHNIQUE_MAPPINGS)
TECHNIQUE_KEYWORD_INDEX: dict[str, Technique] = _build_keyword_index(TECHNIQUE_MAPPINGS)
FABRIC_TAG_INDEX: dict[str, Fabric] = _build_tag_index(FABRIC_MAPPINGS)
FABRIC_KEYWORD_INDEX: dict[str, Fabric] = _build_keyword_index(FABRIC_MAPPINGS)
OCCASION_TAG_INDEX: dict[str, Occasion] = _build_tag_index(OCCASION_MAPPINGS)
OCCASION_KEYWORD_INDEX: dict[str, Occasion] = _build_keyword_index(OCCASION_MAPPINGS)
STYLING_TAG_INDEX: dict[str, StylingCombo] = _build_tag_index(STYLING_COMBO_MAPPINGS)
STYLING_KEYWORD_INDEX: dict[str, StylingCombo] = _build_keyword_index(STYLING_COMBO_MAPPINGS)


# 5 dict union — module load 시 1회 build 후 frozen 재사용. unknown_signal_tracker 가
# `build_counters` 마다 호출 → 16w replay 32+ 호출. 매번 재빌드 회피 (P1-B 효율).
_ALL_KNOWN_HASHTAGS: frozenset[str] = frozenset(
    GARMENT_TAG_INDEX.keys()
    | TECHNIQUE_TAG_INDEX.keys()
    | FABRIC_TAG_INDEX.keys()
    | OCCASION_TAG_INDEX.keys()
    | STYLING_TAG_INDEX.keys()
)


def all_known_hashtags() -> frozenset[str]:
    """Unknown signal tracker 가 쓰는 전체 '알려진 해시태그' 집합 (hashtag side only).

    module-level frozen 캐시 — 매 호출 재빌드 0. mapping_tables 정의 후 5 dict union 1회.
    """
    return _ALL_KNOWN_HASHTAGS


# --------------------------------------------------------------------------- #
# IG meta stoplist (2026-05-02, Phase 2 Tier 1) — 12w eval 결과 noise 차단.
# stoplist 태그는 unknown_signal_tracker 가 hashtag list 자체에서 drop —
# buckets/co_occur 양쪽 미카운트. emergence rule 평가 대상도 아님.
# 기준:
#   - 보편 IG 메타 (universal across all post genres): love / instagood / viral / trending /
#     instadaily / photooftheday / aesthetic / lifestyle / vibes
#   - reels family (regex prefix 등가): reel*
#   - 일반 SMB / handmade: supportsmallbusiness / smallbusiness / handmade / shopnow
#   - 일반 marketing meta (fashion 비특정): designer / fashion / fashionblogger / ootd /
#     style / trends / luxurywear / traditional / indiantraditionalwear / menswear /
#     mensethnicwear (project scope = women's ethnic)
# 변경 시 신중 — 정규 fashion 단어 (kurta, saree 등) 와 구분되어야 함.
# --------------------------------------------------------------------------- #
IG_META_STOPLIST_EXACT: frozenset[str] = frozenset((
    # universal IG meta
    "love", "instagood", "viral", "trending", "instadaily", "photooftheday",
    "picoftheday", "aesthetic", "vibes", "lifestyle", "follow", "like", "explore",
    "explorepage", "foryou", "foryoupage", "fyp",
    # SMB / handmade
    "supportsmallbusiness", "smallbusiness", "handmade", "shopnow", "shoplocal",
    "supportlocal",
    # marketing meta — fashion 비특정
    "designer", "designers", "fashion", "fashionblogger", "fashionista",
    "fashiongram", "fashionstyle", "ootd", "style", "stylish", "styleblogger",
    "styleinspo", "trends",
    # luxury/category meta
    "luxurywear", "luxury", "traditional", "indiantraditionalwear",
    # 남성 wear (프로젝트 scope = 여성 ethnic)
    "menswear", "mensethnicwear", "menfashion", "mensfashion",
    # location/culture generic
    "india", "indian", "mumbai", "delhi", "bangalore", "bollywood",
    # 2026-05-02 v2.3 검수 — desi 류 marketing meta + saree slogan
    # (16w 백필 surface 결과 #desiaesthetic / #desilook / #sareenotsorry 부분 정리)
    "desiaesthetic", "desilook", "desistyle", "desivibes",
    "desigirl", "desiwear", "desifashion", "desibride", "desigram",
    "sareenotsorry", "sareelover", "sareeobsession",
))


# `reels` prefix family — `reels`, `reelsindia`, `reelsinstagram`, `reelitfeelit` 등
IG_META_STOPLIST_PREFIXES: tuple[str, ...] = (
    "reels", "reel",
)


def is_meta_hashtag(tag: str) -> bool:
    """IG meta / 일반 marketing noise 인지. tag 는 '#' 제외 lowercase 전제.

    spec §4.2 Tier 1 stoplist (2026-05-02). exact match + prefix family 양쪽 검사.
    """
    if tag in IG_META_STOPLIST_EXACT:
        return True
    return any(tag.startswith(p) for p in IG_META_STOPLIST_PREFIXES)
