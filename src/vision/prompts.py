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
"""
from __future__ import annotations

from contracts.common import Silhouette

PROMPT_VERSION = "v0.3"

_SILHOUETTE_ENUM = [s.value for s in Silhouette]

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
    "- Output top-2 outfits by visible person size. If only 1 person, return 1.\n"
    "- person_bbox is [x, y, w, h] in 0..1 normalized image coordinates (top-left origin).\n"
    "- person_bbox_area_ratio = w * h.\n"
    "- For a single-piece outfit (saree drape, lehenga choli viewed as single, ethnic dress), "
    "set dress_as_single=true and use upper_garment_type only; lower_garment_type=null.\n"
    "- upper_garment_type / lower_garment_type MUST be a SINGLE lowercase word "
    "(e.g. \"kurta\", \"saree\", \"anarkali\", \"sherwani\", \"palazzo\", \"churidar\"). "
    "NO slash, NO OR, NO parentheses, NO multiple values. "
    "If unsure, pick the best single guess or null.\n"
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
    "- Pick top-3 preset colors that dominate the ethnic-wear region of each outfit "
    "(skin excluded).\n"
    "- fabric MUST be a SINGLE lowercase word describing the DOMINANT visible material "
    "(e.g. \"cotton\", \"linen\", \"silk\", \"chiffon\", \"georgette\", \"rayon\", "
    "\"khadi\", \"chanderi\", \"organza\", \"velvet\", \"net\", \"satin\"). "
    "NO slash, NO multi-word (\"cotton_silk\" or \"cotton/silk\" NOT allowed — "
    "pick the dominant single material). If unsure, null.\n"
    "- technique MUST be a SINGLE lowercase word describing the DOMINANT decoration / "
    "construction technique (e.g. \"chikankari\", \"block_print\", \"bandhani\", "
    "\"zardosi\", \"kalamkari\", \"mirror_work\", \"gotapatti\", \"embroidery\", "
    "\"schiffli\", \"ikat\", \"pintuck\", \"plain\"). "
    "Use \"plain\" only when the garment is visibly undecorated. If the garment has "
    "decoration but the specific technique is unclear, null. NO multi-word.\n"
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
    "      \"lower_garment_type\": string | null,\n"
    "      \"dress_as_single\": bool,\n"
    "      \"silhouette\": string | null,\n"
    "      \"fabric\": string | null,\n"
    "      \"technique\": string | null,\n"
    "      \"color_preset_picks_top3\": [\"name\", \"name\", \"name\"]\n"
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
