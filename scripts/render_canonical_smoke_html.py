"""Phase 5 Step C canonical path smoke 결과 관찰용 간이 HTML viewer.

입력: outputs/{date}/enriched.json (run_daily_pipeline 결과)
출력: outputs/{date}/smoke_stepC.html

용도:
  canonical path (LLM BBOX → canonical pool → dynamic k palette → ΔE76≤15 preset 매칭)
  결과를 post 단위 카드로 나열. ΔE76 threshold 15 / merge 10 / min_share 0.05 튜닝 감각
  잡기 위한 관찰 도구. Step E 풀스케일 comparison HTML 아님.

실행:
  uv run python scripts/render_canonical_smoke_html.py
  uv run python scripts/render_canonical_smoke_html.py --date 2026-04-24
  uv run python scripts/render_canonical_smoke_html.py \
      --enriched outputs/2026-04-24/enriched.json \
      --blob-cache sample_data/image_cache \
      --out outputs/2026-04-24/smoke_stepC.html
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
from collections import Counter
from datetime import date as date_cls
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_PROMPT_VERSION = "v0.3"  # run_daily_pipeline 현행 prompts
_MODEL_ID = "gemini-2.5-flash"


def _rgb_to_hex(r: int | None, g: int | None, b: int | None) -> str:
    if r is None or g is None or b is None:
        return "#cccccc"
    return f"#{r:02x}{g:02x}{b:02x}"


def _thumbnail_data_uri(path: Path, max_bytes: int = 400_000) -> str | None:
    """blob_cache 안의 jpg 를 base64 data URI 로 임베드. 용량 크면 None."""
    if not path.exists():
        return None
    raw = path.read_bytes()
    if len(raw) > max_bytes:
        # 크기 초과는 다음 이미지로. 이 간이 뷰어는 샘플 보기 용도라 리사이즈는 생략.
        return None
    return "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")


def _resolve_thumbnails(blob_urls: list[str], blob_cache: Path, limit: int = 4) -> list[str]:
    """blob_url 의 basename 으로 blob_cache 에서 찾기. 없으면 스킵."""
    thumbs: list[str] = []
    for url in blob_urls:
        if len(thumbs) >= limit:
            break
        name = Path(url).name
        data_uri = _thumbnail_data_uri(blob_cache / name)
        if data_uri is not None:
            thumbs.append(data_uri)
    return thumbs


def _compute_cache_key(image_path: Path) -> str | None:
    """src/vision/llm_cache.py 의 compute_cache_key 와 동일 로직.
    Cache key = sha256(image_bytes + 0x1f + prompt_version + 0x1f + model_id).
    """
    if not image_path.exists():
        return None
    h = hashlib.sha256()
    h.update(image_path.read_bytes())
    h.update(b"\x1f")
    h.update(_PROMPT_VERSION.encode("utf-8"))
    h.update(b"\x1f")
    h.update(_MODEL_ID.encode("utf-8"))
    return h.hexdigest()


def _load_gemini_for_image(image_path: Path, cache_dir: Path) -> dict | None:
    """blob_cache 의 이미지 → cache_key → llm_cache JSON 로드. 없으면 None."""
    key = _compute_cache_key(image_path)
    if key is None:
        return None
    p = cache_dir / _MODEL_ID / f"{key}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _render_gemini_block(blob_urls: list[str], blob_cache: Path, llm_cache: Path) -> str:
    """IG 이미지별 Gemini 원본 응답 (is_ethnic, preset picks, silhouette). canonical path 결과와
    육안 비교하기 위한 디버그 섹션."""
    rows: list[str] = []
    for url in blob_urls:
        name = Path(url).name
        img_path = blob_cache / name
        ga = _load_gemini_for_image(img_path, llm_cache)
        if ga is None:
            rows.append(
                f'<li class="gemini-miss"><code>{html.escape(name)}</code> — no cache entry</li>'
            )
            continue
        analysis = ga.get("garment_analysis") or {}
        is_ethnic = analysis.get("is_india_ethnic_wear")
        outfits = analysis.get("outfits") or []
        if not is_ethnic:
            rows.append(
                f'<li class="gemini-false"><code>{html.escape(name)}</code> — <b>is_ethnic=False</b></li>'
            )
            continue
        parts = [f'<code>{html.escape(name)}</code> — is_ethnic=True']
        for i, o in enumerate(outfits):
            up = o.get("upper_garment_type") or "-"
            lo = o.get("lower_garment_type") or "-"
            sil = o.get("silhouette") or "-"
            fab = o.get("fabric") or "-"
            tech = o.get("technique") or "-"
            picks = o.get("color_preset_picks_top3") or []
            picks_str = ", ".join(html.escape(p) for p in picks)
            area = o.get("person_bbox_area_ratio", 0)
            parts.append(
                f"<div class='outfit'>#{i} upper={html.escape(up)} lower={html.escape(lo)} "
                f"sil={html.escape(sil)} fab={html.escape(fab)} tech={html.escape(tech)} "
                f"area={area:.2f} <b>picks={picks_str}</b></div>"
            )
        rows.append(f'<li class="gemini-true">{"".join(parts)}</li>')
    return f'<details open><summary>Gemini raw ({len(blob_urls)} images)</summary><ul class="gemini">{"".join(rows)}</ul></details>'


def _render_post_card(row: dict, blob_cache: Path, llm_cache: Path) -> str:
    norm = row.get("normalized") or {}
    post_id = norm.get("source_post_id", "?")
    source = norm.get("source") or "?"
    text = (norm.get("text_blob") or "")[:200]
    image_urls = norm.get("image_urls") or []
    handle = norm.get("account_handle") or "?"
    engagement = norm.get("engagement_raw_count") or norm.get("engagement_raw") or 0
    post_date = norm.get("post_date") or ""
    source_type = norm.get("ig_source_type") or "-"

    color = row.get("color") or {}
    hex_code = _rgb_to_hex(color.get("r"), color.get("g"), color.get("b"))
    color_name = color.get("name") or "-"
    color_family = color.get("family") or "-"

    garment_type = row.get("garment_type") or "-"
    silhouette = row.get("silhouette") or "-"
    fabric = row.get("fabric") or "-"
    technique = row.get("technique") or "-"
    cluster_key = row.get("trend_cluster_key") or "-"
    cluster_shares = row.get("trend_cluster_shares") or {}
    # ζ (2026-04-28): shares dict 가 multi-entry 면 각 cluster:share 표시 (sorted desc).
    # single-entry 또는 빈 dict 는 winner key 와 동일 정보라 표시 생략.
    shares_html = ""
    if len(cluster_shares) > 1:
        sorted_shares = sorted(cluster_shares.items(), key=lambda kv: (-kv[1], kv[0]))
        shares_display = " · ".join(
            f"{html.escape(k)}={v:.2f}" for k, v in sorted_shares
        )
        shares_html = (
            '<div class="row"><span class="k">trend_cluster_shares</span>'
            f'<span class="v mono">{shares_display}</span></div>'
        )

    is_youtube = source == "youtube"
    source_badge_cls = "yt" if is_youtube else "ig"

    if is_youtube:
        thumb_imgs = '<div class="nothumb">youtube source — no image_urls (M3.G/H: VideoFrameSource 필요)</div>'
        gemini_block = ""
    else:
        thumbs = _resolve_thumbnails(image_urls, blob_cache)
        if thumbs:
            thumb_imgs = "".join(
                f'<img src="{src}" loading="lazy" alt="thumb"/>' for src in thumbs
            )
        else:
            thumb_imgs = f'<div class="nothumb">({len(image_urls)} blob_urls, cache miss)</div>'
        gemini_block = _render_gemini_block(image_urls, blob_cache, llm_cache)

    return f"""
<article class="card source-{source_badge_cls}">
  <header>
    <span class="badge {source_badge_cls}">{html.escape(source)}</span>
    <code class="pid">{html.escape(post_id)}</code>
    <span class="meta">@{html.escape(handle)} · {html.escape(source_type)} · {engagement:,} eng · {html.escape(post_date)}</span>
  </header>
  <div class="thumbs">{thumb_imgs}</div>
  <div class="attrs">
    <div class="row"><span class="k">garment_type</span><span class="v">{html.escape(garment_type)}</span></div>
    <div class="row"><span class="k">silhouette</span><span class="v">{html.escape(silhouette)}</span></div>
    <div class="row"><span class="k">fabric</span><span class="v">{html.escape(fabric)}</span></div>
    <div class="row"><span class="k">technique</span><span class="v">{html.escape(technique)}</span></div>
    <div class="row"><span class="k">trend_cluster_key</span><span class="v mono">{html.escape(cluster_key)}</span></div>
    {shares_html}
  </div>
  <div class="color">
    <div class="swatch" style="background:{hex_code}"></div>
    <div class="colortext">
      <div><code>{hex_code}</code></div>
      <div>preset: <b>{html.escape(color_name)}</b></div>
      <div>family: <b>{html.escape(color_family)}</b></div>
    </div>
  </div>
  {gemini_block}
  <details><summary>caption</summary><pre>{html.escape(text)}</pre></details>
</article>
"""


def _render_summary(rows: list[dict]) -> str:
    if not rows:
        return "<section class='summary'><h2>summary</h2><p>(no rows)</p></section>"
    n = len(rows)
    sil = Counter((r.get("silhouette") or "?") for r in rows)
    fam = Counter(((r.get("color") or {}).get("family") or "?") for r in rows)
    gt = Counter((r.get("garment_type") or "?") for r in rows)
    fab = Counter((r.get("fabric") or "?") for r in rows)
    tech = Counter((r.get("technique") or "?") for r in rows)
    preset_name = Counter(((r.get("color") or {}).get("name") or "?") for r in rows)

    def _fmt(counter: Counter) -> str:
        items = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
        return ", ".join(f"{html.escape(k)}={v}" for k, v in items)

    return f"""
<section class="summary">
  <h2>summary ({n} rows)</h2>
  <p><b>silhouette</b>: {_fmt(sil)}</p>
  <p><b>color_family</b>: {_fmt(fam)}</p>
  <p><b>garment_type</b>: {_fmt(gt)}</p>
  <p><b>fabric</b>: {_fmt(fab)}</p>
  <p><b>technique</b>: {_fmt(tech)}</p>
  <p><b>preset_name</b>: {_fmt(preset_name)}</p>
</section>
"""


_CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif; background:#f6f7f9; color:#1a1a1a; margin:0; padding:16px; }
h1 { margin:0 0 8px; font-size:20px; }
.summary { background:#fff; border:1px solid #e5e7eb; border-radius:8px; padding:12px 16px; margin-bottom:16px; }
.summary p { margin:4px 0; font-size:13px; }
.card { background:#fff; border:1px solid #e5e7eb; border-radius:8px; padding:12px; margin-bottom:12px; display:grid; grid-template-columns:1fr 280px 180px; gap:12px; align-items:start; }
.card.source-yt { background:#fafafa; opacity:0.7; }
.card header { grid-column:1/4; border-bottom:1px solid #eef; padding-bottom:6px; margin-bottom:4px; display:flex; gap:12px; align-items:baseline; }
.badge { font-size:10px; font-weight:600; padding:2px 8px; border-radius:10px; text-transform:uppercase; }
.badge.ig { background:#e1306c; color:#fff; }
.badge.yt { background:#ff0000; color:#fff; }
.pid { font-family: SFMono-Regular, Menlo, monospace; font-size:12px; background:#eef; padding:2px 6px; border-radius:4px; }
.meta { color:#666; font-size:12px; }
.thumbs { display:flex; gap:6px; flex-wrap:wrap; }
.thumbs img { width:130px; height:130px; object-fit:cover; border-radius:4px; border:1px solid #ddd; }
.nothumb { color:#888; font-size:12px; padding:8px; background:#fafafa; border:1px dashed #ddd; border-radius:4px; }
.attrs { font-size:13px; }
.attrs .row { display:flex; padding:3px 0; border-bottom:1px dashed #f0f0f0; }
.attrs .k { color:#666; width:90px; flex-shrink:0; }
.attrs .v { color:#111; }
.mono { font-family: SFMono-Regular, Menlo, monospace; font-size:12px; }
.color { display:flex; gap:10px; align-items:center; }
.swatch { width:80px; height:80px; border-radius:6px; border:1px solid #ccc; }
.colortext { font-size:12px; line-height:1.6; }
details { grid-column:1/4; }
details summary { cursor:pointer; color:#0a66c2; font-size:12px; padding:4px 0; }
pre { background:#fafafa; padding:8px; border-radius:4px; font-size:11px; overflow:auto; max-height:200px; margin:4px 0; }
ul.gemini { list-style:none; padding:6px 0; margin:0; font-size:12px; }
ul.gemini li { padding:4px 8px; margin:2px 0; border-radius:4px; }
ul.gemini .gemini-miss { background:#fef3c7; color:#92400e; }
ul.gemini .gemini-false { background:#fee2e2; color:#991b1b; }
ul.gemini .gemini-true { background:#d1fae5; color:#064e3b; }
ul.gemini .outfit { margin:2px 0 2px 12px; font-family:SFMono-Regular,Menlo,monospace; font-size:11px; }
"""


def _render_html(rows: list[dict], blob_cache: Path, llm_cache: Path, title: str) -> str:
    cards = "\n".join(_render_post_card(r, blob_cache, llm_cache) for r in rows)
    summary = _render_summary(rows)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>{_CSS}</style>
</head><body>
<h1>{html.escape(title)}</h1>
{summary}
{cards}
</body></html>"""


def _parse_args() -> argparse.Namespace:
    today = date_cls.today().isoformat()
    p = argparse.ArgumentParser(description="Canonical path smoke → HTML viewer.")
    p.add_argument("--date", default=today, help="outputs/{date} (default: today)")
    p.add_argument("--enriched", type=Path, default=None,
                   help="enriched.json path override (default: outputs/{date}/enriched.json)")
    p.add_argument("--blob-cache", type=Path, default=_REPO / "sample_data" / "image_cache",
                   help="thumbnail source (default: sample_data/image_cache)")
    p.add_argument("--llm-cache", type=Path, default=_REPO / "outputs" / "llm_cache",
                   help="LLM cache base dir (default: outputs/llm_cache)")
    p.add_argument("--out", type=Path, default=None,
                   help="output html path (default: outputs/{date}/smoke_stepC.html)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    enriched_path = args.enriched or (_REPO / "outputs" / args.date / "enriched.json")
    out_path = args.out or (_REPO / "outputs" / args.date / "smoke_stepC.html")

    if not enriched_path.exists():
        print(f"[html] enriched not found: {enriched_path}")
        return

    rows: list[dict] = json.loads(enriched_path.read_text(encoding="utf-8"))
    title = f"Phase 5 Step C smoke — {args.date} ({len(rows)} rows)"
    html_doc = _render_html(rows, args.blob_cache, args.llm_cache, title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_doc, encoding="utf-8")
    print(f"[html] wrote {out_path} ({len(rows)} rows, {len(html_doc):,} bytes)")


if __name__ == "__main__":
    main()
