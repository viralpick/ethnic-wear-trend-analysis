"""Phase 0 — LLM BBOX 파일럿 (gpt-5-mini vs Gemini 2.5 Flash).

목적: 동일 샘플에 대해 두 vision LLM 을 호출해서
`is_india_ethnic_wear` + outfit BBOX + color_preset picks 를 받고,
side-by-side HTML 로 사용자 수동 검수 가능한 형태로 출력.

사용:
    uv run python scripts/pilot_llm_bbox.py \
        --sample-dir outputs/blob_cache_preset \
        --preset outputs/color_preset/color_preset.json \
        --sample-size 20 --seed 42 \
        --output-dir outputs/pilot_llm_bbox

vision extras 필요 (google-genai, openai, python-dotenv, pillow).
.env 에 AZURE_OPENAI_* + GEMINI_API_KEY.
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import os.path
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("pilot_llm_bbox")

# 프롬프트 본문과 whitelist 는 `src/vision/` 에 canonical 로 유지 (production Phase 2
# GeminiVisionLLMClient 와 공유). 파일럿은 얇은 wrapper — A/B 비교 / HTML 렌더만 담당.
# repo root/src 를 path 에 넣어 `uv run python scripts/...` 로 직접 실행 가능하게.
_SRC_ROOT = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from vision.prompts import SYSTEM_PROMPT  # noqa: E402
from vision.traditional_filter import is_outfit_traditional  # noqa: E402

_SYSTEM_PROMPT = SYSTEM_PROMPT  # 기존 호출부 호환 (call_azure / call_gemini)


# --------------------------------------------------------------------------- #
# Post-filter — vision.traditional_filter 의 원시-인자 API 를 dict 에 어댑트.
# --------------------------------------------------------------------------- #
# pilot 은 parsed dict 에 `binary_raw` / `post_filter_reason` 을 직접 심어 HTML 에
# 표시. production 은 GarmentAnalysis.model_copy + logger.info 로 처리 (경로 분리).

def _apply_post_filter(parsed: dict[str, Any] | None) -> dict[str, Any] | None:
    """is_india_ethnic_wear 에 post-filter 적용. 원본 값은 `binary_raw` 로 보존.

    idempotent — 이미 필터된 parsed 에 재적용해도 `binary_raw` 기준으로 재계산.
    """
    if parsed is None:
        return None
    if "binary_raw" in parsed:
        raw_binary = bool(parsed["binary_raw"])
    else:
        raw_binary = bool(parsed.get("is_india_ethnic_wear"))
        parsed["binary_raw"] = raw_binary
    outfits_raw = parsed.get("outfits")
    outfits = outfits_raw if isinstance(outfits_raw, list) else []
    if not raw_binary:
        parsed["is_india_ethnic_wear"] = False
        parsed["post_filter_reason"] = None
        return parsed
    if not outfits:
        parsed["is_india_ethnic_wear"] = False
        parsed["post_filter_reason"] = "binary_true_but_no_outfits"
        return parsed
    has_traditional = any(
        is_outfit_traditional(
            o.get("upper_garment_type"),
            o.get("lower_garment_type"),
            bool(o.get("dress_as_single")),
        )
        for o in outfits
        if isinstance(o, dict)
    )
    if has_traditional:
        parsed["is_india_ethnic_wear"] = True
        parsed["post_filter_reason"] = None
        return parsed
    parsed["is_india_ethnic_wear"] = False
    parsed["post_filter_reason"] = "no_traditional_silhouette"
    return parsed


@dataclass
class PilotResult:
    post_id: str
    model: str
    latency_s: float
    raw_text: str
    parsed: dict[str, Any] | None
    parse_error: str | None
    prompt_tokens: int | None
    completion_tokens: int | None


# --------------------------------------------------------------------------- #
# Sample + preset 로딩
# --------------------------------------------------------------------------- #

def load_preset(preset_path: Path) -> list[dict[str, str]]:
    raw = json.loads(preset_path.read_text(encoding="utf-8"))
    return [{"name": e["name"], "hex": e["hex"]} for e in raw]


def pick_sample(sample_dir: Path, n: int, seed: int) -> list[Path]:
    files = sorted(sample_dir.glob("*.jpg"))
    if not files:
        raise FileNotFoundError(f"no .jpg in {sample_dir}")
    rng = random.Random(seed)
    return rng.sample(files, min(n, len(files)))


def build_user_payload(preset: list[dict[str, str]]) -> str:
    return (
        "color_preset (choose names from here only):\n"
        + json.dumps(preset, ensure_ascii=False)
    )


# --------------------------------------------------------------------------- #
# Azure OpenAI (gpt-5-mini) 호출
# --------------------------------------------------------------------------- #

def _azure_client():
    import openai
    from dotenv import load_dotenv

    load_dotenv()
    client = openai.AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    )
    deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]
    return client, deployment


def call_azure(image_bytes: bytes, user_payload: str) -> PilotResult:
    client, deployment = _azure_client()
    b64 = base64.b64encode(image_bytes).decode()
    start = time.perf_counter()
    resp = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_payload},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}",
                            "detail": "high",
                        },
                    },
                ],
            },
        ],
        response_format={"type": "json_object"},
        seed=42,
    )
    latency = time.perf_counter() - start
    raw = resp.choices[0].message.content or "{}"
    parsed, err = _safe_parse(raw)
    usage = resp.usage
    return PilotResult(
        post_id="",
        model="gpt-5-mini",
        latency_s=latency,
        raw_text=raw,
        parsed=parsed,
        parse_error=err,
        prompt_tokens=getattr(usage, "prompt_tokens", None),
        completion_tokens=getattr(usage, "completion_tokens", None),
    )


# --------------------------------------------------------------------------- #
# Gemini 2.5 Flash 호출
# --------------------------------------------------------------------------- #

def call_gemini(image_bytes: bytes, user_payload: str) -> PilotResult:
    from google import genai
    from google.genai import types
    from dotenv import load_dotenv

    load_dotenv()
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    contents = [
        types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
        _SYSTEM_PROMPT + "\n\n" + user_payload,
    ]
    start = time.perf_counter()
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0,
        ),
    )
    latency = time.perf_counter() - start
    raw = resp.text or "{}"
    parsed, err = _safe_parse(raw)
    usage = getattr(resp, "usage_metadata", None)
    return PilotResult(
        post_id="",
        model="gemini-2.5-flash",
        latency_s=latency,
        raw_text=raw,
        parsed=parsed,
        parse_error=err,
        prompt_tokens=getattr(usage, "prompt_token_count", None),
        completion_tokens=getattr(usage, "candidates_token_count", None),
    )


def _safe_parse(raw: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        return json.loads(raw), None
    except json.JSONDecodeError as exc:
        return None, f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------- #
# 실행 루프
# --------------------------------------------------------------------------- #

def run_pilot(
    sample_paths: list[Path],
    preset: list[dict[str, str]],
    output_dir: Path,
    models: list[str],
) -> dict[str, list[PilotResult]]:
    user_payload = build_user_payload(preset)
    results: dict[str, list[PilotResult]] = {m: [] for m in models}
    for idx, img_path in enumerate(sample_paths, 1):
        post_id = img_path.stem
        image_bytes = img_path.read_bytes()
        logger.info("[%d/%d] %s", idx, len(sample_paths), post_id)
        for model in models:
            cached = _load_cached(model, post_id, output_dir)
            if cached is not None:
                cached.parsed = _apply_post_filter(cached.parsed)
                results[model].append(cached)
                logger.info("  %-16s cached", model)
                continue
            result = _call_model_safe(model, image_bytes, user_payload)
            result.post_id = post_id
            result.parsed = _apply_post_filter(result.parsed)
            _dump_result(result, output_dir)
            results[model].append(result)
            logger.info(
                "  %-16s latency=%.2fs tokens=%s/%s parse=%s",
                model, result.latency_s,
                result.prompt_tokens, result.completion_tokens,
                "ok" if result.parsed else result.parse_error,
            )
    return results


def _call_model(model: str, image_bytes: bytes, user_payload: str) -> PilotResult:
    if model == "gpt-5-mini":
        return call_azure(image_bytes, user_payload)
    if model == "gemini-2.5-flash":
        return call_gemini(image_bytes, user_payload)
    raise ValueError(f"unknown model: {model}")


def _call_model_safe(model: str, image_bytes: bytes, user_payload: str) -> PilotResult:
    """API 호출 예외를 PilotResult.parse_error 로 기록 (실패 숨김 X, 명시 기록)."""
    try:
        return _call_model(model, image_bytes, user_payload)
    except Exception as exc:
        return PilotResult(
            post_id="",
            model=model,
            latency_s=0.0,
            raw_text="",
            parsed=None,
            parse_error=f"{type(exc).__name__}: {exc}",
            prompt_tokens=None,
            completion_tokens=None,
        )


def _load_cached(model: str, post_id: str, output_dir: Path) -> PilotResult | None:
    path = output_dir / model / f"{post_id}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("parsed") is None and data.get("parse_error"):
        # 이전 실패는 재시도 — cache miss 로 처리
        return None
    return PilotResult(
        post_id=data["post_id"],
        model=data["model"],
        latency_s=data.get("latency_s", 0.0),
        raw_text=data.get("raw_text") or "",
        parsed=data.get("parsed"),
        parse_error=data.get("parse_error"),
        prompt_tokens=data.get("prompt_tokens"),
        completion_tokens=data.get("completion_tokens"),
    )


def _dump_result(result: PilotResult, output_dir: Path) -> None:
    dest = output_dir / result.model / f"{result.post_id}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        json.dumps(
            {
                "post_id": result.post_id,
                "model": result.model,
                "latency_s": result.latency_s,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "parse_error": result.parse_error,
                "parsed": result.parsed,
                "raw_text": result.raw_text if result.parsed is None else None,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


# --------------------------------------------------------------------------- #
# HTML 렌더 (side-by-side, 수동 검수용)
# --------------------------------------------------------------------------- #

def render_html(
    sample_paths: list[Path],
    results: dict[str, list[PilotResult]],
    preset: list[dict[str, str]],
    output_path: Path,
) -> None:
    hex_by_name = {e["name"]: e["hex"] for e in preset}
    models = list(results.keys())
    by_post = {
        m: {r.post_id: r for r in results[m]} for m in models
    }
    rows = []
    for img_path in sample_paths:
        post_id = img_path.stem
        per_model = [by_post[m].get(post_id) for m in models]
        rows.append(_render_row(img_path, post_id, models, per_model, hex_by_name, output_path.parent))
    summary = _render_summary(models, results)
    html = _HTML_TEMPLATE.format(summary=summary, rows="\n".join(rows))
    output_path.write_text(html, encoding="utf-8")


_MODEL_COLORS = {
    "gpt-5-mini": "#e74c3c",
    "gemini-2.5-flash": "#3498db",
}


def _render_row(
    img_path: Path,
    post_id: str,
    models: list[str],
    per_model: list[PilotResult | None],
    hex_by_name: dict[str, str],
    html_dir: Path,
) -> str:
    rel_img = os.path.relpath(img_path.resolve(), html_dir.resolve())
    thumb = (
        f'<div class="thumb-wrap" id="post-{post_id}">'
        f'<img src="{rel_img}" alt="{post_id}" class="thumb" />'
        f'</div>'
    )
    cells = [f'<td class="post"><div class="pid">{post_id}</div>{thumb}</td>']
    for model, result in zip(models, per_model):
        cells.append(_render_model_cell(post_id, model, result, hex_by_name))
    return f'<tr>{"".join(cells)}</tr>'


def _render_model_cell(
    post_id: str,
    model: str,
    result: PilotResult | None,
    hex_by_name: dict[str, str],
) -> str:
    if result is None:
        return '<td class="model"><div class="na">no result</div></td>'
    header = (
        f'<div class="meta"><b>{model}</b> — {result.latency_s:.2f}s · '
        f'tokens {result.prompt_tokens}/{result.completion_tokens}</div>'
    )
    if result.parsed is None:
        return (
            f'<td class="model">{header}'
            f'<div class="err">parse error: {result.parse_error}</div>'
            f'<pre class="raw">{_escape(result.raw_text[:800])}</pre></td>'
        )
    parsed = result.parsed
    binary = parsed.get("is_india_ethnic_wear")
    raw_binary = parsed.get("binary_raw", binary)
    reason = parsed.get("post_filter_reason")
    binary_label = f"is_india_ethnic_wear: {binary}"
    if raw_binary != binary:
        binary_label += f" <span class='flip'>(raw={raw_binary} → filtered: {reason})</span>"
    binary_html = (
        f'<div class="binary {"yes" if binary else "no"}">{binary_label}</div>'
    )
    outfits_html = "".join(
        _render_outfit(post_id, model, idx, o, hex_by_name)
        for idx, o in enumerate(parsed.get("outfits") or [])
    )
    if not outfits_html:
        outfits_html = '<div class="na">no outfits</div>'
    return f'<td class="model">{header}{binary_html}{outfits_html}</td>'


def _render_outfit(
    post_id: str,
    model: str,
    outfit_idx: int,
    outfit: dict[str, Any],
    hex_by_name: dict[str, str],
) -> str:
    bbox = outfit.get("person_bbox") or []
    bbox_valid = len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox)
    bbox_txt = ", ".join(f"{v:.2f}" for v in bbox) if bbox_valid else "—"
    ratio = outfit.get("person_bbox_area_ratio")
    upper = outfit.get("upper_garment_type")
    lower = outfit.get("lower_garment_type")
    single = outfit.get("dress_as_single")
    silhouette = outfit.get("silhouette")
    # v0.4 신규 — upper/lower_is_ethnic 독립 판정. null 가능 (Gemini 미채움 또는
    # garment_type 이 null 일 때). dress_as_single=True 면 lower_is_ethnic=null 강제.
    upper_eth = outfit.get("upper_is_ethnic")
    lower_eth = outfit.get("lower_is_ethnic")
    picks = outfit.get("color_preset_picks_top3") or []
    chips = "".join(_color_chip(p, hex_by_name) for p in picks)
    bbox_btn = ""
    if bbox_valid:
        color = _MODEL_COLORS.get(model, "#333")
        bbox_btn = (
            f'<button class="bbox-btn" '
            f'data-post-id="{post_id}" data-model="{model}" '
            f'data-outfit-idx="{outfit_idx}" '
            f'data-bbox="{json.dumps(bbox)}" data-color="{color}" '
            f'onclick="toggleBbox(this)">BBOX {outfit_idx + 1} 보기</button>'
        )
    silhouette_html = (
        f'<div class="sil">silhouette=<b>{silhouette}</b></div>'
        if silhouette is not None else ""
    )
    upper_badge = _ethnic_badge(upper_eth)
    lower_badge = _ethnic_badge(lower_eth)
    picks_label = f"picks={len(picks)}"
    return (
        '<div class="outfit">'
        f'<div class="bbox">{bbox_btn} bbox=[{bbox_txt}] area={ratio}</div>'
        f'<div class="gt">upper={upper} {upper_badge} · '
        f'lower={lower} {lower_badge} · single={single}</div>'
        f'{silhouette_html}'
        f'<div class="chips"><span class="picks-label">{picks_label}</span>{chips}</div>'
        '</div>'
    )


def _ethnic_badge(value: bool | None) -> str:
    """upper_is_ethnic / lower_is_ethnic 값을 시각화. null / missing 은 회색."""
    if value is True:
        return '<span class="eth-badge yes">ethnic</span>'
    if value is False:
        return '<span class="eth-badge no">western</span>'
    return '<span class="eth-badge na">n/a</span>'


def _color_chip(name: str, hex_by_name: dict[str, str]) -> str:
    hx = hex_by_name.get(name, "#cccccc")
    return (
        f'<span class="chip" title="{name} {hx}">'
        f'<span class="swatch" style="background:{hx}"></span>'
        f'<span class="cname">{name}</span></span>'
    )


def _render_summary(models: list[str], results: dict[str, list[PilotResult]]) -> str:
    rows = []
    for m in models:
        rs = results[m]
        parsed_ok = sum(1 for r in rs if r.parsed is not None)
        total_latency = sum(r.latency_s for r in rs)
        total_prompt = sum((r.prompt_tokens or 0) for r in rs)
        total_out = sum((r.completion_tokens or 0) for r in rs)
        raw_true = sum(
            1 for r in rs
            if r.parsed is not None and r.parsed.get("binary_raw")
        )
        filtered_true = sum(
            1 for r in rs
            if r.parsed is not None and r.parsed.get("is_india_ethnic_wear")
        )
        flipped = raw_true - filtered_true
        # v0.4 신규 지표 — outfit 단위 aggregation.
        outfits_flat = [
            o
            for r in rs
            if r.parsed is not None
            for o in (r.parsed.get("outfits") or [])
            if isinstance(o, dict)
        ]
        total_outfits = len(outfits_flat)
        pick_dist = {1: 0, 2: 0, 3: 0, 0: 0}
        upper_fill = lower_fill = 0
        for o in outfits_flat:
            picks = o.get("color_preset_picks_top3") or []
            k = len(picks) if len(picks) in (0, 1, 2, 3) else 3
            pick_dist[k] = pick_dist.get(k, 0) + 1
            if o.get("upper_is_ethnic") is not None:
                upper_fill += 1
            if o.get("lower_is_ethnic") is not None:
                lower_fill += 1
        pick_txt = f"{pick_dist[1]}/{pick_dist[2]}/{pick_dist[3]}"
        if pick_dist[0]:
            pick_txt += f" (+{pick_dist[0]} empty)"
        fill_txt = (
            f"up {upper_fill}/{total_outfits} · "
            f"lo {lower_fill}/{total_outfits}"
        ) if total_outfits else "—"
        rows.append(
            f'<tr><td>{m}</td><td>{parsed_ok}/{len(rs)}</td>'
            f'<td>{total_latency:.1f}s</td><td>{total_prompt}</td>'
            f'<td>{total_out}</td>'
            f'<td>{raw_true}</td><td>{filtered_true}</td><td>{flipped}</td>'
            f'<td>{pick_txt}</td><td>{fill_txt}</td></tr>'
        )
    return (
        '<table class="summary">'
        '<thead><tr><th>model</th><th>parsed</th><th>total latency</th>'
        '<th>prompt tokens</th><th>completion tokens</th>'
        '<th>raw_true</th><th>filtered_true</th><th>flipped</th>'
        '<th>picks 1/2/3</th><th>is_ethnic fill</th></tr></thead>'
        '<tbody>' + "".join(rows) + '</tbody></table>'
    )


def _escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


_HTML_TEMPLATE = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<title>Phase 0 LLM BBOX pilot</title>
<style>
body {{ font-family: -apple-system, sans-serif; margin: 16px; font-size: 13px; }}
h1 {{ font-size: 18px; }}
.summary {{ border-collapse: collapse; margin-bottom: 16px; }}
.summary th, .summary td {{ border: 1px solid #ddd; padding: 4px 8px; }}
.grid {{ width: 100%; border-collapse: collapse; }}
.grid th, .grid td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
.grid th {{ background: #f5f5f5; text-align: left; }}
.post {{ width: 320px; }}
.pid {{ font-family: monospace; font-size: 11px; color: #666; margin-bottom: 4px; word-break: break-all; }}
.thumb-wrap {{ position: relative; display: inline-block; }}
.thumb {{ max-width: 300px; max-height: 400px; display: block; }}
.bbox-overlay {{ position: absolute; top: 0; left: 0; pointer-events: none; }}
.bbox-btn {{ font-size: 11px; padding: 2px 6px; margin-right: 6px; cursor: pointer; border: 1px solid #888; background: #fff; border-radius: 3px; }}
.bbox-btn.active {{ background: #333; color: #fff; }}
.model {{ width: 420px; }}
.meta {{ color: #666; font-size: 11px; margin-bottom: 6px; }}
.binary {{ font-weight: bold; padding: 4px 6px; border-radius: 3px; display: inline-block; margin-bottom: 8px; }}
.binary.yes {{ background: #d4edda; color: #155724; }}
.binary.no {{ background: #f8d7da; color: #721c24; }}
.flip {{ font-weight: normal; font-size: 11px; color: #555; font-style: italic; margin-left: 6px; }}
.outfit {{ border-left: 3px solid #888; padding: 4px 8px; margin-bottom: 6px; background: #fafafa; }}
.bbox {{ font-family: monospace; font-size: 11px; color: #666; }}
.gt {{ font-size: 12px; margin: 2px 0; }}
.sil {{ font-size: 12px; margin: 2px 0; color: #2c3e50; }}
.chips {{ display: flex; gap: 6px; flex-wrap: wrap; align-items: center; margin-top: 4px; }}
.picks-label {{ font-family: monospace; font-size: 11px; color: #666; margin-right: 4px; }}
.chip {{ display: inline-flex; align-items: center; gap: 4px; background: #fff; border: 1px solid #ccc; border-radius: 12px; padding: 2px 8px; font-size: 11px; }}
.swatch {{ width: 14px; height: 14px; border-radius: 50%; border: 1px solid rgba(0,0,0,0.1); }}
.eth-badge {{ display: inline-block; padding: 1px 6px; border-radius: 8px; font-size: 10px; font-weight: bold; margin-left: 2px; }}
.eth-badge.yes {{ background: #d4edda; color: #155724; }}
.eth-badge.no {{ background: #fde2e4; color: #6e2c2c; }}
.eth-badge.na {{ background: #ececec; color: #666; }}
.na {{ color: #999; font-style: italic; }}
.err {{ color: #c00; font-size: 11px; }}
.raw {{ background: #f5f5f5; padding: 4px; font-size: 11px; white-space: pre-wrap; word-break: break-all; max-height: 200px; overflow: auto; }}
</style>
<script>
function toggleBbox(btn) {{
  const postId = btn.dataset.postId;
  const model = btn.dataset.model;
  const outfitIdx = btn.dataset.outfitIdx;
  const container = document.getElementById('post-' + postId);
  if (!container) return;
  const overlayId = `bbox-${{postId}}-${{model}}-${{outfitIdx}}`;
  const existing = document.getElementById(overlayId);
  if (existing) {{
    existing.remove();
    btn.classList.remove('active');
    btn.textContent = `BBOX ${{Number(outfitIdx) + 1}} 보기`;
    return;
  }}
  const img = container.querySelector('img');
  if (!img.complete) {{
    img.addEventListener('load', () => drawBbox(btn, container, overlayId), {{ once: true }});
    return;
  }}
  drawBbox(btn, container, overlayId);
}}

function drawBbox(btn, container, overlayId) {{
  const img = container.querySelector('img');
  const bbox = JSON.parse(btn.dataset.bbox);
  const color = btn.dataset.color;
  const outfitIdx = btn.dataset.outfitIdx;
  const svgNs = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(svgNs, 'svg');
  svg.id = overlayId;
  svg.setAttribute('class', 'bbox-overlay');
  svg.setAttribute('viewBox', '0 0 1 1');
  svg.setAttribute('preserveAspectRatio', 'none');
  svg.style.width = img.clientWidth + 'px';
  svg.style.height = img.clientHeight + 'px';
  const rect = document.createElementNS(svgNs, 'rect');
  rect.setAttribute('x', bbox[0]);
  rect.setAttribute('y', bbox[1]);
  rect.setAttribute('width', bbox[2]);
  rect.setAttribute('height', bbox[3]);
  rect.setAttribute('fill', 'none');
  rect.setAttribute('stroke', color);
  rect.setAttribute('stroke-width', '0.008');
  svg.appendChild(rect);
  const label = document.createElementNS(svgNs, 'text');
  label.setAttribute('x', bbox[0] + 0.01);
  label.setAttribute('y', bbox[1] + 0.04);
  label.setAttribute('fill', color);
  label.setAttribute('font-size', '0.04');
  label.setAttribute('font-weight', 'bold');
  label.textContent = `${{btn.dataset.model.split('-')[0]}}#${{Number(outfitIdx) + 1}}`;
  svg.appendChild(label);
  container.appendChild(svg);
  btn.classList.add('active');
  btn.textContent = `BBOX ${{Number(outfitIdx) + 1}} 끄기`;
}}
</script>
</head>
<body>
<h1>Phase 0 LLM BBOX pilot — side-by-side 수동 검수</h1>
<p style="font-size:12px;color:#666">BBOX 색: <b style="color:#e74c3c">gpt-5-mini = 빨강</b> · <b style="color:#3498db">gemini-2.5-flash = 파랑</b> · 각 outfit 옆 "BBOX N 보기" 버튼으로 토글.</p>
{summary}
<table class="grid">
<thead><tr><th>post</th><th>gpt-5-mini</th><th>gemini-2.5-flash</th></tr></thead>
<tbody>
{rows}
</tbody></table>
</body></html>
"""


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sample-dir", type=Path, default=Path("outputs/blob_cache_preset"))
    p.add_argument("--preset", type=Path, default=Path("outputs/color_preset/color_preset.json"))
    p.add_argument("--sample-size", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/pilot_llm_bbox_v02"))
    p.add_argument(
        "--models", type=str, default="gpt-5-mini,gemini-2.5-flash",
        help="comma-separated",
    )
    return p.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = _parse_args()
    preset = load_preset(args.preset)
    sample_paths = pick_sample(args.sample_dir, args.sample_size, args.seed)
    logger.info("sample size = %d (seed=%d)", len(sample_paths), args.seed)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = run_pilot(sample_paths, preset, args.output_dir, models)
    html_path = args.output_dir / "review.html"
    render_html(sample_paths, results, preset, html_path)
    logger.info("wrote HTML: %s", html_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
