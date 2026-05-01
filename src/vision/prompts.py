"""Vision LLM 프롬프트 상수 + 버전 관리.

`PROMPT_VERSION` 은 Phase 2 cache 키에 편입 (model_id + prompt_version + image sha256).
프롬프트 수정 시 반드시 bump — 기존 cache 를 무효화해서 drift 숨김 방지.

Phase 0 에서 v0.1 확정 (scripts/pilot_llm_bbox.py). 이후 수정 이력:
  v0.1 — 2026-04-23, pilot 20 post A/B 승인 버전. Gemini 2.5 Flash 채택.
  v0.2 — 2026-04-24, 재파일럿. pattern vs silhouette 섹션 + 5 예시 추가
         (fusion wear 오분류 방어). 01KPQDXNAA 양쪽 모델 FALSE 판정 확인.
  v0.3 — 2026-04-24, Phase 4.5 선행. `fabric` / `technique` 필드 추가 (Phase 0 dedup
         기준이 요구하나 Phase 2 contract 최소화 시 누락되었던 drift 복구). closed
         vocabulary 아닌 single lowercase word — 상위 계층에서 normalize. 실측은 Phase 5
         full smoke (131 post) 로 대체, 20-post 재파일럿 생략.
  v0.4 — 2026-04-24, Color 3층 재설계 Phase A3. `upper_is_ethnic` / `lower_is_ethnic`
         필드 추가 (EthnicOutfit 에 A2 에서 contract 확장됨). B1 canonical_extractor 가
         segformer upper/lower/dress class pool 에 포함할지를 Gemini 직접 판정으로 결정
         (configs/garment_vocab.yaml 폐기). dress_as_single=True 일 때 upper_is_ethnic 을
         dress 전체 ethnic 여부로 재활용, lower_is_ethnic=null. 또한
         `color_preset_picks_top3` 를 강제 top-3 에서 "1~3 동적 pick, DO NOT pad"
         으로 완화 — 단/2톤 의류 정확 표현 목적.
  v0.5 — 2026-04-25, C2 pool_02 10 smoke 관측 후. `color_preset_picks_top3` 에
         include/exclude region 명시. 10 smoke 에서 Fabindia kurta 의 cream_ivory
         2nd pick 이 옷에 없고 가방/악세서리 색이었던 문제 defend 목적. include:
         upper/lower/full-body garment + dupatta/shawl/jacket/cardigan (전통 drape/
         outer). exclude: 가방/신발/안경/모자/터번(pagdi)/쥬얼리/벨트/헤어액세서리/
         피부/배경/소품. 이후 β hybrid 의 R3 (pick↔KMeans 매칭 drop) 이 남은 환각을
         2차 방어.
  v0.6 — 2026-04-25, SceneFilter canonical-path 통합 (adult-woman-only 강제).
         prod canonical path (PipelineBColorExtractor → _analyze_images) 가
         SceneFilter 를 우회해 child/man BBOX 가 그대로 Gemini 로 흘러들어가던 leak
         (project_scene_filter_canonical_integration.md) 의 Gemini 측 방어선.
         outfits 배열 자체에서 비-adult-female 통째 제외 — bbox + 모든 attribute
         (upper/lower_garment_type, upper/lower_is_ethnic, dress_as_single,
         silhouette, fabric, technique, color_preset_picks_top3, person_bbox,
         person_bbox_area_ratio) 까지. v0.5 cache 폐기 (key = model + prompt_version
         + image sha256 이라 자동 무효화).
  v0.7 — 2026-04-27, F-12 R2(β) 와 동기화. color_preset_picks_top3 픽 규칙 강화:
         (a) EVIDENCE — 이미지에 실재하는 색만 픽 (warm/cool/ethnic 어휘 prior 환각
         차단), (b) PATTERN — 자잘한 multi-color 패턴은 평균/혼합색이 아니라 background +
         dominant motif 의 개별 색을 픽. F-11 13-post smoke 의 두 회귀 (01KPYYMEA4
         maroon_red 환각 / 01KQ28YDASZ mint_green=초록+흰 평균 환각) 진단 결과 반영.
  v0.8 — 2026-04-28, M3.I styling_combo P1 (co_ord_set / with_dupatta / with_jacket)
         파생용 schema 슬롯 2개 추가: `outer_layer` (단일 word: dupatta/shawl/jacket/
         cardigan/nehru/shrug 등 traditional drape 또는 outer) + `is_co_ord_set` (bool —
         upper 와 lower 가 동일 fabric/print/색조로 매칭된 set 여부). 색 픽 규칙은 v0.7
         그대로 (color regression 방지). dress_as_single=True 시 is_co_ord_set=null.
         derive_styling_from_outfit 의 P1 매핑 (M3.I) 의 입력으로 사용 — 추가 LLM call 0.
  v0.9 — 2026-05-02, garment / fabric / technique 3-tier escape 강제. (a) taxonomy
         enum 매핑 시 그 정확한 단어, (b) ethnic 인데 enum 외 → SINGLE lowercase raw 단어
         (Phase 2 emergence pipeline 의 input 으로 누적), (c) non-ethnic / 식별불가 → null.
         기존 free-form (예시만 제공) 가 noise (kurti_dress / saree-blouse 등) 를
         만들어 normalize 단계에서 None 으로 빠지면 cluster 진입 실패하던 문제 해소.
         cluster fan-out 은 enum 매핑된 것만 (옵션 C: g_val is None → drop), raw 는
         enriched JSON 에 보존. project_canonical_zero_fallback_removal_2026_05_02 +
         후속 (Phase 2: unknown_signal_tracker 확장) 참조.
"""
from __future__ import annotations

from contracts.common import Fabric, GarmentType, Silhouette, Technique

PROMPT_VERSION = "v0.9"

_SILHOUETTE_ENUM = [s.value for s in Silhouette]
_GARMENT_ENUM = [g.value for g in GarmentType]
_FABRIC_ENUM = [f.value for f in Fabric]
_TECHNIQUE_ENUM = [t.value for t in Technique]

SYSTEM_PROMPT = (
    "You are an analyzer for Indian ethnic wear content. Given a single image and "
    "the provided color_preset list, return STRICT JSON.\n\n"
    "Definition of India ethnic wear: traditional Indian silhouettes — kurta, "
    "anarkali, straight/A-line kurta, saree drape, lehenga choli, salwar/"
    "churidar/palazzo, dupatta, sherwani. Include indo-fusion only if the "
    "upper OR lower piece has a TRADITIONAL SILHOUETTE / CONSTRUCTION "
    "(not merely Indian pattern or print). Exclude pure Western wear.\n\n"
    "CRITICAL — pattern vs silhouette:\n"
    "- Indian patterns (block print, chikankari, bandhani, ikat, ethnic motif) on "
    "a WESTERN garment (crop top, t-shirt, leggings, yoga pants, jeans, shorts, "
    "mini skirt, bodycon dress) is NOT India ethnic wear. Set "
    "is_india_ethnic_wear=false even if accessories (bag/scarf/jewelry) look Indian.\n"
    "- Silhouette / construction must be traditional. Pattern or background "
    "(Indian store, statue, temple) is NOT sufficient evidence.\n"
    "Examples:\n"
    "  * crop top (block print) + cycling shorts → FALSE (Western silhouettes)\n"
    "  * kurta (block print) + churidar → TRUE (traditional upper AND lower)\n"
    "  * T-shirt + palazzo → TRUE (indo-fusion, traditional lower)\n"
    "  * saree drape (any pattern) → TRUE (traditional drape)\n"
    "  * bodycon dress with Indian print → FALSE (Western silhouette)\n\n"
    "Rules:\n"
    "- Output top-2 outfits by visible ADULT WOMAN size. If only 1 adult woman, "
    "return 1.\n"
    "- DO NOT include any outfit object for children, infants, men, or any "
    "non-adult-female person. The \"outfits\" array MUST contain only adult-female "
    "outfits — exclude their bbox AND all attributes "
    "(upper/lower_garment_type, upper/lower_is_ethnic, dress_as_single, silhouette, "
    "fabric, technique, color_preset_picks_top3, person_bbox, "
    "person_bbox_area_ratio).\n"
    "- If no adult woman is visible (only children, infants, or men), return "
    "is_india_ethnic_wear=false and outfits=[]. Do NOT substitute child or man "
    "outfits.\n"
    "- Adult woman = visibly post-pubescent female; exclude pre-pubescent children "
    "and infants regardless of clothing.\n"
    "- person_bbox is [x, y, w, h] in 0..1 normalized image coordinates (top-left origin).\n"
    "- person_bbox_area_ratio = w * h.\n"
    "- For a single-piece outfit (saree drape, lehenga choli viewed as single, ethnic dress), "
    "set dress_as_single=true and use upper_garment_type only; lower_garment_type=null.\n"
    f"- upper_garment_type — TIER selection (MUST be SINGLE lowercase word):\n"
    f"  (a) If garment matches taxonomy, use the EXACT value from {_GARMENT_ENUM}.\n"
    f"  (b) If garment is CLEARLY ETHNIC (Indian traditional / festive) but does NOT "
    f"match (a), write the actual garment name as a SINGLE lowercase word "
    f"(e.g. \"phulkari\", \"ghagra\", \"angarkha\", \"sherwani\", \"kurti\", \"choli\"). "
    f"Only when upper_is_ethnic=true.\n"
    f"  (c) Use null when garment is non-ethnic (T-shirt, blouse, jeans, western dress, "
    f"swimwear) or unidentifiable / fully occluded.\n"
    f"  NO slash, NO OR, NO parentheses, NO multiple values, NO multi-word.\n"
    f"- lower_garment_type — same TIER selection (a)/(b)/(c) as upper_garment_type. "
    f"For (b), examples: \"palazzo\", \"churidar\", \"salwar\", \"sharara\", \"lehenga\", "
    f"\"dhoti\", \"ghagra\". For (c), western bottoms (jeans, leggings, shorts, mini "
    f"skirt) → null.\n"
    "- upper_is_ethnic: judge INDEPENDENTLY whether upper_garment_type belongs to the "
    "Indian ethnic/traditional family (kurta, kurti, anarkali, saree blouse/choli, "
    "sherwani, angrakha, ethnic tunic, kurta-style dress, etc.). Western tops like "
    "T-shirt, crop top, blouse, shirt, tank top, hoodie → false. Null only if "
    "upper_garment_type is null.\n"
    "- lower_is_ethnic: judge INDEPENDENTLY whether lower_garment_type belongs to the "
    "Indian ethnic/traditional family (palazzo, churidar, salwar, sharara, dhoti, "
    "lehenga skirt, ghagra, ethnic pyjama, dupatta-as-lower, etc.). Western bottoms "
    "like jeans, leggings, shorts, mini skirt, pencil skirt, yoga pants → false. "
    "Null only if lower_garment_type is null.\n"
    "- If dress_as_single=true: upper_is_ethnic represents the ENTIRE dress's ethnic "
    "status (saree drape, lehenga choli as single, ethnic dress → true; western "
    "bodycon/evening dress → false). lower_is_ethnic MUST be null.\n"
    "- Indo-fusion inclusion: is_india_ethnic_wear=true if upper_is_ethnic=true OR "
    "lower_is_ethnic=true (one traditional silhouette is enough). "
    "is_india_ethnic_wear=false only if both are false/null AND no traditional drape "
    "is present. Pattern/print alone does NOT make a garment ethnic — silhouette rules.\n"
    f"- silhouette MUST be exactly one of {_SILHOUETTE_ENUM} or null. "
    "Apply to the DOMINANT visible garment of the outfit "
    "(upper if two-piece, whole drape if single). "
    "straight = no flare; a_line = subtle flare from waist; flared = heavy flare; "
    "anarkali = floor-length flared kurta; fit_and_flare = fitted top + flared bottom; "
    "tiered = layered tiers; high_low = asymmetric hem; boxy = straight loose cut; "
    "kaftan = loose rectangular; shirt_style = shirt cut with collar/placket; "
    "angrakha = wrap-front with tie; empire = high waistline.\n"
    "- Each color pick must be chosen from the provided color_preset \"name\" list "
    "(e.g. \"pool_00\", \"saffron\"), NOT free-form hex.\n"
    "- color_preset_picks_top3: pick 1 to 3 preset colors that dominate the GARMENT "
    "region of each outfit. Use fewer picks when the garment is genuinely single-tone "
    "(1 pick) or two-tone (2 picks). DO NOT pad the list to 3 by adding minor/"
    "negligible colors.\n"
    "  INCLUDE (pick from these regions only):\n"
    "    * upper garment: kurta, kurti, blouse, choli, shirt, tunic, sherwani top\n"
    "    * lower garment: palazzo, churidar, salwar, sharara, lehenga skirt, pants, "
    "skirt, pyjama\n"
    "    * full-body garment: saree drape, anarkali, ethnic dress, jumpsuit, kaftan\n"
    "    * traditional drape / outer layer: dupatta, shawl, stole, jacket, cardigan, "
    "nehru jacket\n"
    "  EXCLUDE (never pick colors from these regions):\n"
    "    * bag / handbag / clutch / potli\n"
    "    * footwear / shoes / sandals / heels\n"
    "    * eyewear / glasses / sunglasses\n"
    "    * hat / cap / turban / pagdi\n"
    "    * jewelry (necklace, earrings, bangles, maang tikka, nose ring, anklets)\n"
    "    * belt / watch / hair accessories / hairband\n"
    "    * skin / hair / makeup\n"
    "    * background / props / furniture / walls\n"
    "  If the accessory and the garment share the same hue, still pick based on "
    "garment evidence only.\n"
    "  EVIDENCE — pick only colors that are VISIBLY PRESENT in the garment region. "
    "Do NOT add a pick because it would fit a \"warm\" / \"cool\" / \"ethnic\" / "
    "\"festive\" tone, and do NOT carry priors from the garment_type label "
    "(e.g. saree → maroon_red). If you are not confident a color occupies a "
    "meaningful share (≥ ~5%) of the garment, omit it — fewer picks is correct.\n"
    "  PATTERN — when the garment has a multi-color pattern (small motifs, prints, "
    "embroidery, mixed weave), DO NOT pick the visual MIX / AVERAGE of those colors "
    "(e.g. green + cream weave → \"mint_green\"). Pick the dominant individual colors "
    "actually present (background + 1–2 most prominent motif colors). If individual "
    "colors are too small/scattered to call confidently, prefer fewer picks.\n"
    f"- fabric — TIER selection (DOMINANT visible material, SINGLE lowercase word):\n"
    f"  (a) If matches taxonomy, use EXACT value from {_FABRIC_ENUM}.\n"
    f"  (b) If clearly a distinct fabric outside (a) (e.g. \"brocade_silk\" → no, but "
    f"\"tulle\", \"taffeta\", \"banarasi\" as fabric → yes), write SINGLE lowercase word.\n"
    f"  (c) null if unidentifiable / occluded / generic.\n"
    f"  NO slash, NO multi-word (\"cotton_silk\" or \"cotton/silk\" NOT allowed — "
    f"pick the dominant single material).\n"
    f"- technique — TIER selection (DOMINANT decoration / construction, SINGLE "
    f"lowercase word):\n"
    f"  (a) If matches taxonomy, use EXACT value from {_TECHNIQUE_ENUM}.\n"
    f"  (b) If clearly an ethnic decoration outside (a) (e.g. \"phulkari\", \"shibori\", "
    f"\"applique\"), write SINGLE lowercase word.\n"
    f"  (c) null when garment has decoration but the specific technique is unclear, "
    f"or when garment is plain (use enum value \"solid\" for visibly undecorated). "
    f"NO multi-word.\n"
    "- outer_layer: a SINGLE lowercase word for any traditional drape / outer worn "
    "OVER the upper garment (\"dupatta\", \"shawl\", \"stole\", \"jacket\", \"cardigan\", "
    "\"nehru\", \"shrug\"). Pick the most prominent one if multiple. Null if the outfit "
    "has no separate outer layer or drape. NOT a substitute for upper_garment_type — "
    "outer_layer is what's worn over a kurta/blouse, not the kurta itself. "
    "For saree drape (single-piece), outer_layer is null (the drape IS the garment).\n"
    "- is_co_ord_set: true ONLY when upper and lower are clearly an INTENTIONALLY MATCHED "
    "set — same fabric AND same print/color scheme, marketed as a coordinated 2-piece "
    "(e.g. matching kurta + palazzo in identical block print, matching crop top + skirt "
    "in identical ikat). False when upper and lower differ in fabric/print/color even if "
    "tones are complementary. Null when dress_as_single=true (single-piece, no upper-lower "
    "pairing) or when only one piece is visible.\n"
    "- If is_india_ethnic_wear=false, outfits MAY be empty array.\n"
    "- No prose, no code fences. JSON only.\n\n"
    "Output schema:\n"
    "{\n"
    "  \"is_india_ethnic_wear\": bool,\n"
    "  \"outfits\": [\n"
    "    {\n"
    "      \"person_bbox\": [x, y, w, h],\n"
    "      \"person_bbox_area_ratio\": float,\n"
    "      \"upper_garment_type\": string | null,\n"
    "      \"upper_is_ethnic\": bool | null,\n"
    "      \"lower_garment_type\": string | null,\n"
    "      \"lower_is_ethnic\": bool | null,\n"
    "      \"dress_as_single\": bool,\n"
    "      \"silhouette\": string | null,\n"
    "      \"fabric\": string | null,\n"
    "      \"technique\": string | null,\n"
    "      \"color_preset_picks_top3\": [\"name\", ...],  // 1 to 3 entries, do not pad\n"
    "      \"outer_layer\": string | null,  // dupatta/shawl/stole/jacket/cardigan/nehru/shrug\n"
    "      \"is_co_ord_set\": bool | null   // upper+lower matched set; null if single-piece\n"
    "    }\n"
    "  ]\n"
    "}"
)


def build_user_payload(preset: list[dict[str, str]]) -> str:
    """LLM 호출 시 user message 로 붙이는 preset 목록 직렬화."""
    import json as _json  # lazy — 이 모듈은 prompt 만 내보내도록 top-level import 최소화

    return (
        "color_preset (choose names from here only):\n"
        + _json.dumps(preset, ensure_ascii=False)
    )
