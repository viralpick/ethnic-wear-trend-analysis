"""정적 검수 HTML 생성 — outputs/enriched.json + outputs/summaries.json 기반.

M3.A Step E 의 1단계 (정적 HTML). BE/FE 검수 대시보드 초석.

사용:
    uv run python scripts/build_review_html.py
    uv run python scripts/build_review_html.py --enriched outputs/2026-04-29/enriched.json --summaries outputs/2026-04-29/summaries.json --output outputs/review.html

이미지 해결 순서:
1. sample_data/image_cache/<basename> (블롭 캐시)
2. 원본 image_url (CDN, SAS 만료 시 깨짐)
"""
from __future__ import annotations

import argparse
import html as html_mod
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

_REPO = Path(__file__).resolve().parents[1]
_IMG_CACHE = _REPO / "sample_data" / "image_cache"


def _esc(s: Any) -> str:
    return html_mod.escape(str(s)) if s is not None else ""


def _resolve_image_src(url: str, html_dir: Path) -> str:
    """blob 캐시에 있으면 상대 경로, 아니면 원본 URL 반환."""
    if not url:
        return ""
    path_only = url.split("?", 1)[0]
    basename = Path(urlparse(path_only).path).name
    if not basename:
        return url
    cached = _IMG_CACHE / basename
    if cached.exists():
        # html_dir 기준 상대 경로
        try:
            return str(cached.resolve().relative_to(html_dir.resolve())).replace("\\", "/")
        except ValueError:
            return cached.as_uri()
    return url


def _color_chips(palette: list[dict[str, Any]]) -> str:
    if not palette:
        return '<span class="muted">no palette</span>'
    chips = []
    for c in palette:
        hex_v = c.get("hex", "#000")
        share = c.get("share", 0.0)
        family = c.get("family", "")
        chips.append(
            f'<span class="chip" style="background:{_esc(hex_v)}" '
            f'title="{_esc(hex_v)} share={share:.2f} family={_esc(family)}">'
            f'<span class="chip-label">{_esc(hex_v)} <small>{share*100:.0f}%</small></span>'
            f'</span>'
        )
    return '<div class="chips">' + "".join(chips) + "</div>"


def _dist_table(dist: dict[str, float], top_n: int = 5) -> str:
    if not dist:
        return '<span class="muted">∅</span>'
    sorted_items = sorted(dist.items(), key=lambda kv: -kv[1])[:top_n]
    rows = "".join(
        f'<tr><td>{_esc(k)}</td><td class="num">{v*100:.1f}%</td></tr>'
        for k, v in sorted_items
    )
    return f'<table class="dist">{rows}</table>'


def _account_url(source: str, handle: str | None) -> str:
    """source_post_id 는 내부 ID 라 외부 링크 못 만듦. 계정 프로필 URL 만 링크."""
    if not handle:
        return ""
    if source == "instagram":
        return f"https://www.instagram.com/{handle}/"
    if source == "youtube":
        return f"https://www.youtube.com/@{handle}"
    return ""


def _render_canonical(canonical: dict[str, Any]) -> str:
    rep = canonical.get("representative", {})
    members = canonical.get("members", [])
    palette = canonical.get("palette", [])
    rows: list[str] = []
    rows.append(
        f'<div class="canon-row"><b>upper</b>: '
        f'garment={_esc(rep.get("upper_garment_type"))} '
        f'<small>(ethnic={_esc(rep.get("upper_is_ethnic"))})</small></div>'
    )
    if rep.get("lower_garment_type"):
        rows.append(
            f'<div class="canon-row"><b>lower</b>: '
            f'garment={_esc(rep.get("lower_garment_type"))} '
            f'<small>(ethnic={_esc(rep.get("lower_is_ethnic"))})</small></div>'
        )
    if rep.get("outer_layer"):
        rows.append(
            f'<div class="canon-row"><b>outer</b>: {_esc(rep.get("outer_layer"))}</div>'
        )
    rows.append(
        f'<div class="canon-row">'
        f'silhouette={_esc(rep.get("silhouette"))} / '
        f'fabric={_esc(rep.get("fabric"))} / '
        f'technique={_esc(rep.get("technique"))} / '
        f'co_ord={_esc(rep.get("is_co_ord_set"))} / '
        f'dress_as_single={_esc(rep.get("dress_as_single"))}'
        f'</div>'
    )
    picks = rep.get("color_preset_picks_top3") or []
    rows.append(
        f'<div class="canon-row">'
        f'members={len(members)} / area_ratio={_esc(rep.get("person_bbox_area_ratio"))} / '
        f'preset_picks=[{_esc(", ".join(picks))}]'
        f'</div>'
    )
    rows.append(f'<div class="canon-row">palette: {_color_chips(palette)}</div>')
    return f'<div class="canonical"><div class="canon-head">canonical[{canonical.get("canonical_index", 0)}]</div>{"".join(rows)}</div>'


def _render_post_card(item: dict[str, Any], html_dir: Path) -> str:
    n = item["normalized"]
    pid = n.get("source_post_id", "")
    src = n.get("source", "?")
    handle = n.get("account_handle") or "—"
    date = n.get("post_date", "")[:10] if n.get("post_date") else ""
    eng = n.get("engagement_raw", 0)
    src_type = n.get("ig_source_type") or ""
    image_urls = n.get("image_urls") or []
    video_urls = n.get("video_urls") or []
    hashtags = n.get("hashtags") or []
    canonicals = item.get("canonicals", [])
    palette = item.get("post_palette", [])
    brands = item.get("brands") or item.get("brand")  # 호환
    if isinstance(brands, dict):
        brands = [brands]
    elif brands is None:
        brands = []
    cluster_shares = item.get("trend_cluster_shares") or {}
    cluster_key = item.get("trend_cluster_key") or "—"

    method = item.get("classification_method_per_attribute") or {}

    img_html = ""
    for url in image_urls[:5]:
        src_path = _resolve_image_src(url, html_dir)
        img_html += f'<img src="{_esc(src_path)}" loading="lazy" tabindex="0" alt="" />'
    if video_urls:
        img_html += f'<div class="video-tag">▶ video × {len(video_urls)}</div>'

    canon_html = "".join(_render_canonical(c) for c in canonicals) or '<span class="muted">no canonical</span>'

    brands_html = ", ".join(
        f'{_esc(b.get("name"))} <small>({_esc(b.get("source_method") or b.get("source"))})</small>'
        for b in brands
    ) or "—"

    text_attrs = (
        f'garment_type: {_esc(item.get("garment_type"))} <small>({_esc(method.get("garment_type"))})</small><br/>'
        f'fabric: {_esc(item.get("fabric"))} <small>({_esc(method.get("fabric"))})</small><br/>'
        f'technique: {_esc(item.get("technique"))} <small>({_esc(method.get("technique"))})</small><br/>'
        f'occasion: {_esc(item.get("occasion"))} <small>({_esc(method.get("occasion"))})</small><br/>'
        f'styling_combo: {_esc(item.get("styling_combo"))} <small>({_esc(method.get("styling_combo"))})</small><br/>'
        f'embellishment: {_esc(item.get("embellishment_intensity"))}'
    )

    shares_html = _dist_table(cluster_shares, top_n=5)

    acct_url = _account_url(src, handle if handle != "—" else None)
    handle_html = (
        f'<a href="{_esc(acct_url)}" target="_blank">@{_esc(handle)}</a>'
        if acct_url else f'<span>@{_esc(handle)}</span>'
    )
    return f'''
<article class="post" data-cluster="{_esc(cluster_key)}" data-source="{_esc(src)}" id="post-{_esc(pid)}">
  <header>
    <span class="badge src-{_esc(src)}">{_esc(src)}</span>
    {handle_html}
    <code class="post-id">{_esc(pid)}</code>
    <span class="muted">{_esc(date)}</span>
    <span class="muted">eng={eng:,}</span>
    {f'<span class="muted">type={_esc(src_type)}</span>' if src_type else ''}
  </header>
  <div class="post-body">
    <div class="post-images">{img_html}</div>
    <div class="post-meta">
      <section>
        <h4>text attributes</h4>
        <div>{text_attrs}</div>
      </section>
      <section>
        <h4>brands</h4>
        <div>{brands_html}</div>
      </section>
      <section>
        <h4>post_palette</h4>
        {_color_chips(palette)}
      </section>
      <section>
        <h4>cluster_shares (winner: <code>{_esc(cluster_key)}</code>)</h4>
        {shares_html}
      </section>
      <section>
        <h4>canonicals ({len(canonicals)})</h4>
        {canon_html}
      </section>
      <section>
        <h4>hashtags</h4>
        <div class="muted">{_esc(" ".join(f"#{h}" for h in hashtags[:20]))}</div>
      </section>
    </div>
  </div>
</article>'''


def _render_cluster_card(s: dict[str, Any]) -> str:
    drill = s.get("drilldown") or {}
    breakdown = s.get("score_breakdown") or {}
    breakdown_html = "".join(
        f'<tr><td>{_esc(k)}</td><td class="num">{v:.2f}</td></tr>'
        for k, v in breakdown.items()
    )
    palette_html = _color_chips(drill.get("color_palette") or [])
    top_posts = drill.get("top_posts") or []
    top_videos = drill.get("top_videos") or []
    top_inf = drill.get("top_influencers") or []
    top_posts_html = "".join(
        f'<li><a href="https://www.instagram.com/p/{_esc(p)}/" target="_blank">'
        f'<code>{_esc(p)}</code></a></li>'
        for p in top_posts[:5]
    ) or '<li class="muted">∅</li>'
    top_videos_html = "".join(
        f'<li><a href="https://www.youtube.com/watch?v={_esc(v)}" target="_blank">'
        f'<code>{_esc(v)}</code></a></li>'
        for v in top_videos[:5]
    ) or '<li class="muted">∅</li>'
    top_inf_html = "".join(
        f'<li><a href="https://www.instagram.com/{_esc(h)}/" target="_blank">@{_esc(h)}</a></li>'
        for h in top_inf[:5]
    ) or '<li class="muted">∅</li>'
    return f'''
<article class="cluster">
  <header>
    <span class="cluster-key"><code>{_esc(s.get("cluster_key"))}</code></span>
    <span class="score">{s.get("score", 0):.1f}</span>
    <span class="muted">{_esc(s.get("daily_direction"))}/{_esc(s.get("weekly_direction"))} · {_esc(s.get("lifecycle_stage"))}</span>
  </header>
  <div class="cluster-body">
    <section>
      <h4>display_name</h4>
      <div>{_esc(s.get("display_name"))}</div>
    </section>
    <section>
      <h4>breakdown</h4>
      <table class="dist">{breakdown_html}</table>
    </section>
    <section>
      <h4>counts</h4>
      <div>total={s.get("post_count_total", 0):.1f} · today={s.get("post_count_today", 0):.1f} · views={s.get("total_video_views", 0):,}</div>
    </section>
    <section>
      <h4>color_palette</h4>
      {palette_html}
    </section>
    <section>
      <h4>top_posts (IG)</h4>
      <ul>{top_posts_html}</ul>
    </section>
    <section>
      <h4>top_videos (YT)</h4>
      <ul>{top_videos_html}</ul>
    </section>
    <section>
      <h4>top_influencers</h4>
      <ul>{top_inf_html}</ul>
    </section>
    <section>
      <h4>distributions</h4>
      <div class="dist-grid">
        <div><b>silhouette</b>{_dist_table(drill.get("silhouette_distribution") or {})}</div>
        <div><b>occasion</b>{_dist_table(drill.get("occasion_distribution") or {})}</div>
        <div><b>styling</b>{_dist_table(drill.get("styling_distribution") or {})}</div>
        <div><b>brand</b>{_dist_table(drill.get("brand_distribution") or {})}</div>
      </div>
    </section>
  </div>
</article>'''


_CSS = """
* { box-sizing: border-box; }
body { font: 13px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       margin: 0; padding: 16px 24px; background: #f7f7f7; color: #222; }
h1 { font-size: 18px; }
h2 { font-size: 15px; border-bottom: 2px solid #333; padding-bottom: 4px; margin-top: 24px; }
h4 { font-size: 12px; color: #666; margin: 8px 0 4px; text-transform: uppercase; }
.muted { color: #888; }
.num { text-align: right; font-variant-numeric: tabular-nums; }
code { background: #eef; padding: 1px 4px; border-radius: 3px; font-size: 11px; }
small { color: #888; font-size: 0.85em; }

.summary-bar { background: #fff; padding: 8px 12px; border-radius: 6px; margin-bottom: 8px;
               display: flex; gap: 16px; flex-wrap: wrap; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
.summary-bar b { color: #2a5db8; }

.filter-bar { background: #fff; padding: 8px 12px; border-radius: 6px; margin-bottom: 16px;
              display: flex; gap: 12px; flex-wrap: wrap; align-items: center;
              box-shadow: 0 1px 3px rgba(0,0,0,0.05); font-size: 12px; }
.filter-bar select, .filter-bar input { font-size: 12px; padding: 3px 6px;
              border: 1px solid #ccc; border-radius: 3px; }
.filter-bar input { width: 200px; }

.cluster-list { display: grid; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); gap: 12px; }

.post-id { font-size: 10px; color: #999; font-family: ui-monospace, monospace;
           background: transparent; padding: 0; }

.cluster, .post { background: #fff; border-radius: 6px; padding: 12px 16px;
                  margin-bottom: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
.cluster header, .post header { display: flex; gap: 12px; align-items: baseline;
                                 margin-bottom: 8px; padding-bottom: 6px;
                                 border-bottom: 1px solid #eee; }
.cluster .score { font-size: 16px; font-weight: bold; color: #d35400; }
.cluster .cluster-key { flex: 1; }

.cluster-body { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; }
.cluster-body section { font-size: 12px; }
.dist-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }

.post-body { display: grid; grid-template-columns: 320px 1fr; gap: 16px; }
.post-images img { width: 60px; height: 60px; object-fit: cover; margin-right: 4px;
                   border-radius: 4px; vertical-align: top; cursor: zoom-in;
                   transition: transform 0.2s; }
.post-images img:hover { transform: scale(1.1); }
.post-images img:focus, .post-images img:active {
                   position: fixed; top: 50%; left: 50%; transform: translate(-50%,-50%);
                   width: 90vw; height: 90vh; object-fit: contain; z-index: 1000;
                   background: rgba(0,0,0,0.9); cursor: zoom-out; }
.post-images { display: flex; flex-wrap: wrap; gap: 4px; }
.video-tag { font-size: 11px; background: #333; color: #fff; padding: 2px 6px;
             border-radius: 3px; margin-top: 4px; display: inline-block; }
.post-meta section { font-size: 12px; margin-bottom: 6px; }

.badge { font-size: 10px; padding: 1px 6px; border-radius: 8px;
         font-weight: bold; text-transform: uppercase; }
.badge.src-instagram { background: #e1306c; color: #fff; }
.badge.src-youtube { background: #ff0000; color: #fff; }

.chips { display: flex; flex-wrap: wrap; gap: 4px; }
.chip { display: inline-block; height: 28px; min-width: 60px; padding: 4px 8px;
        border-radius: 4px; border: 1px solid rgba(0,0,0,0.15); font-size: 10px;
        font-family: ui-monospace, monospace; color: #fff;
        text-shadow: 0 0 2px rgba(0,0,0,0.7); cursor: help; }
.chip-label small { color: #fff; opacity: 0.85; margin-left: 4px; }

table.dist { font-size: 11px; border-collapse: collapse; width: 100%; }
table.dist td { padding: 1px 4px; border-bottom: 1px dotted #eee; }
table.dist td:first-child { font-family: ui-monospace, monospace; }

.canonical { background: #fafafa; border-left: 3px solid #2a5db8;
             padding: 6px 10px; margin: 4px 0; border-radius: 4px; }
.canon-head { font-size: 11px; color: #2a5db8; font-weight: bold; margin-bottom: 4px; }
.canon-row { font-size: 11px; line-height: 1.6; }

a { color: #2a5db8; text-decoration: none; }
a:hover { text-decoration: underline; }
"""


def build_html(
    enriched: list[dict[str, Any]], summaries: list[dict[str, Any]], html_dir: Path
) -> str:
    # cluster_key 기준 그룹 정렬 (검수 시 "같은 cluster post 모아보기" 편의)
    sorted_enriched = sorted(
        enriched,
        key=lambda it: (it.get("trend_cluster_key") or "~~~", it["normalized"].get("source_post_id", "")),
    )
    posts_html = "\n".join(_render_post_card(item, html_dir) for item in sorted_enriched)
    sorted_summaries = sorted(summaries, key=lambda s: -s.get("score", 0))
    clusters_html = "\n".join(_render_cluster_card(s) for s in sorted_summaries)

    n_items = len(enriched)
    n_clusters = len(summaries)
    n_canonicals = sum(len(it.get("canonicals", [])) for it in enriched)
    n_with_palette = sum(1 for it in enriched if it.get("post_palette"))
    n_ig = sum(1 for it in enriched if it["normalized"].get("source") == "instagram")
    n_yt = sum(1 for it in enriched if it["normalized"].get("source") == "youtube")
    n_with_brand = sum(1 for it in enriched if (it.get("brands") or it.get("brand")))

    distinct_clusters = sorted({it.get("trend_cluster_key") or "—" for it in enriched})
    cluster_options = "".join(
        f'<option value="{_esc(c)}">{_esc(c)}</option>' for c in distinct_clusters
    )

    return f"""<!DOCTYPE html>
<html lang="ko"><head>
<meta charset="utf-8">
<title>분석 검수 — {n_items} items / {n_clusters} clusters</title>
<style>{_CSS}</style>
</head><body>

<h1>분석 검수 (M3.A Step E 1단계 정적 HTML)</h1>

<div class="summary-bar">
  <span><b>{n_items}</b> items</span>
  <span>(IG <b>{n_ig}</b> / YT <b>{n_yt}</b>)</span>
  <span><b>{n_clusters}</b> trend clusters</span>
  <span><b>{n_canonicals}</b> canonicals</span>
  <span><b>{n_with_palette}</b> with palette</span>
  <span><b>{n_with_brand}</b> with brand</span>
</div>

<div class="filter-bar">
  <label>cluster filter:
    <select id="cluster-filter" onchange="filterPosts()">
      <option value="">(all)</option>
      {cluster_options}
    </select>
  </label>
  <label>source:
    <select id="source-filter" onchange="filterPosts()">
      <option value="">(all)</option>
      <option value="instagram">instagram</option>
      <option value="youtube">youtube</option>
    </select>
  </label>
  <label>id 검색: <input id="id-search" type="search" placeholder="post_id 일부" oninput="filterPosts()"></label>
  <span id="filter-count" class="muted"></span>
</div>

<h2>Trend Clusters (score desc)</h2>
<div class="cluster-list">{clusters_html}</div>

<h2>Items ({n_items})</h2>
<div id="post-list">{posts_html}</div>

<script>
function filterPosts() {{
  const ck = document.getElementById('cluster-filter').value;
  const src = document.getElementById('source-filter').value;
  const q = document.getElementById('id-search').value.trim().toLowerCase();
  let shown = 0;
  document.querySelectorAll('article.post').forEach(el => {{
    const okCk = !ck || el.dataset.cluster === ck;
    const okSrc = !src || el.dataset.source === src;
    const okQ = !q || el.id.toLowerCase().includes(q);
    const visible = okCk && okSrc && okQ;
    el.style.display = visible ? '' : 'none';
    if (visible) shown++;
  }});
  document.getElementById('filter-count').textContent = `(${{shown}} 표시)`;
}}
filterPosts();
</script>

</body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enriched", type=Path, default=_REPO / "outputs" / "enriched.json")
    parser.add_argument("--summaries", type=Path, default=_REPO / "outputs" / "summaries.json")
    parser.add_argument("--output", type=Path, default=_REPO / "outputs" / "review.html")
    args = parser.parse_args()

    if not args.enriched.exists():
        raise SystemExit(f"enriched not found: {args.enriched}")
    if not args.summaries.exists():
        raise SystemExit(f"summaries not found: {args.summaries}")

    with args.enriched.open() as f:
        enriched = json.load(f)
    with args.summaries.open() as f:
        summaries = json.load(f)

    html_dir = args.output.parent
    html_dir.mkdir(parents=True, exist_ok=True)
    html = build_html(enriched, summaries, html_dir)
    args.output.write_text(html, encoding="utf-8")
    print(f"Wrote {args.output} — {len(enriched)} items / {len(summaries)} clusters")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
