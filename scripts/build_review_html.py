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

# src/ import — enriched.trend_cluster_shares 가 옛 raw 키 (PR #31 vision_normalize 전)
# 일 수 있어 HTML 빌드 시 enriched_to_item_distribution + item_cluster_shares 로
# 신 normalize 적용한 share 를 재계산. summaries cluster_key 와 일치 보장.
import sys as _sys
_sys.path.insert(0, str(_REPO / "src"))


_POST_URL_CACHE: dict[str, str] | None = None


def _load_post_urls() -> dict[str, str]:
    """raw DB 에서 post_id → 외부 URL 매핑 build time 1회 조회. 캐시.

    IG: source_post_id (raw id) → instagram.com/p/{shortcode} URL
    YT: source_post_id (raw id) → youtube.com/watch?v={vid} URL
    실패 시 빈 dict 반환 (HTML 은 fallback).
    """
    global _POST_URL_CACHE
    if _POST_URL_CACHE is not None:
        return _POST_URL_CACHE
    out: dict[str, str] = {}
    try:
        import os
        import pymysql
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=_REPO / ".env")
        if not os.environ.get("STARROCKS_HOST"):
            _POST_URL_CACHE = out
            return out
        conn = pymysql.connect(
            host=os.environ["STARROCKS_HOST"],
            port=int(os.environ.get("STARROCKS_PORT", "9030")),
            user=os.environ["STARROCKS_USER"],
            password=os.environ["STARROCKS_PASSWORD"],
            database=os.environ.get("STARROCKS_RAW_DATABASE", "png"),
            connect_timeout=15,
            cursorclass=pymysql.cursors.DictCursor,
        )
        with conn.cursor() as cur:
            cur.execute("SELECT id, url FROM india_ai_fashion_inatagram_posting WHERE url IS NOT NULL AND url != ''")
            for r in cur.fetchall():
                out[r["id"]] = r["url"]
            cur.execute("SELECT id, url FROM india_ai_fashion_youtube_posting WHERE url IS NOT NULL AND url != ''")
            for r in cur.fetchall():
                out[r["id"]] = r["url"]
        conn.close()
    except Exception as e:
        print(f"WARN: _load_post_urls failed (fallback to account profile only): {e}")
    _POST_URL_CACHE = out
    return out


def _esc(s: Any) -> str:
    return html_mod.escape(str(s)) if s is not None else ""


def _resolve_media_src(url: str, html_dir: Path) -> str:
    """blob 캐시 (이미지/영상) 있으면 상대 경로, 아니면 원본 URL 반환."""
    if not url:
        return ""
    path_only = url.split("?", 1)[0]
    basename = Path(urlparse(path_only).path).name
    if not basename:
        return url
    cached = _IMG_CACHE / basename
    if cached.exists():
        try:
            return str(cached.resolve().relative_to(html_dir.resolve())).replace("\\", "/")
        except ValueError:
            return cached.as_uri()
    return url


# 호환 alias
_resolve_image_src = _resolve_media_src


_VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".webm"}


def _is_video_url(url: str) -> bool:
    path_only = url.split("?", 1)[0].lower()
    return any(path_only.endswith(ext) for ext in _VIDEO_EXTS)


def _ffprobe_duration(video_path: Path) -> float | None:
    """ffprobe 로 영상 duration (초) 반환. 실패 시 None."""
    import subprocess
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
            capture_output=True, text=True, timeout=10,
        )
        return float(out.stdout.strip()) if out.returncode == 0 else None
    except Exception:
        return None


def _extract_video_thumbs(
    video_path: Path, thumbs_dir: Path, *, n_frames: int = 3,
) -> list[Path]:
    """영상에서 중간 frame n 개 추출 (25%/50%/75% 등 균등 분포). 캐시 히트 시 재추출 안 함."""
    import subprocess
    if not video_path.exists():
        return []
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem
    out_paths: list[Path] = []

    duration = _ffprobe_duration(video_path)
    if duration is None or duration <= 0:
        return []

    # 균등 분포 fraction (n=3 → 0.25/0.5/0.75)
    fractions = [(i + 1) / (n_frames + 1) for i in range(n_frames)]
    for idx, frac in enumerate(fractions):
        ts = duration * frac
        thumb_path = thumbs_dir / f"{stem}_f{idx}.jpg"
        if thumb_path.exists() and thumb_path.stat().st_size > 0:
            out_paths.append(thumb_path)
            continue
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-ss", f"{ts:.2f}", "-i", str(video_path),
                 "-frames:v", "1", "-q:v", "3", str(thumb_path)],
                capture_output=True, timeout=15,
            )
            if thumb_path.exists() and thumb_path.stat().st_size > 0:
                out_paths.append(thumb_path)
        except Exception:
            continue
    return out_paths


def _resolve_video_local_path(url: str) -> Path | None:
    """video URL → blob 캐시 로컬 Path. 없으면 None."""
    if not url:
        return None
    path_only = url.split("?", 1)[0]
    basename = Path(urlparse(path_only).path).name
    if not basename:
        return None
    cached = _IMG_CACHE / basename
    return cached if cached.exists() else None


def _hex_text_color(hex_v: str) -> str:
    """배경 hex 의 luminance 기준 contrast 색상 (white/black) 결정."""
    if not hex_v or not hex_v.startswith("#") or len(hex_v) < 7:
        return "#000"
    try:
        r = int(hex_v[1:3], 16); g = int(hex_v[3:5], 16); b = int(hex_v[5:7], 16)
    except ValueError:
        return "#000"
    # ITU-R BT.601 luminance
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return "#fff" if lum < 130 else "#000"


def _color_bar(palette: list[dict[str, Any]]) -> str:
    """비율 기반 horizontal bar — 각 segment 너비 = share, hex + % 표기."""
    if not palette:
        return '<span class="muted">no palette</span>'
    total = sum(max(c.get("share", 0.0), 0.0) for c in palette)
    if total <= 0:
        return '<span class="muted">no palette</span>'
    segs = []
    for c in palette:
        share = max(c.get("share", 0.0), 0.0)
        if share <= 0:
            continue
        hex_v = c.get("hex", "#000")
        family = c.get("family", "")
        pct = share / total * 100
        text_color = _hex_text_color(hex_v)
        segs.append(
            f'<div class="palette-seg" '
            f'style="width:{pct:.2f}%;background:{_esc(hex_v)};color:{text_color}" '
            f'title="{_esc(hex_v)} share={share*100:.1f}% family={_esc(family)}">'
            f'<span class="palette-hex">{_esc(hex_v)}</span>'
            f'<span class="palette-pct">{share*100:.0f}%</span>'
            f'</div>'
        )
    return f'<div class="palette-bar">{"".join(segs)}</div>'


def _color_chips(palette: list[dict[str, Any]]) -> str:
    """legacy chip 표시 — canonical detail 등 좁은 영역용. 메인 palette 는 _color_bar 사용."""
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
    rows.append(f'<div class="canon-row">palette: {_color_bar(palette)}</div>')
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

    media_html = ""
    for url in image_urls:  # 모두 표시
        src_path = _resolve_media_src(url, html_dir)
        media_html += (
            f'<img src="{_esc(src_path)}" loading="lazy" tabindex="0" alt="" '
            f'class="media-img" />'
        )
    # 영상은 ffmpeg 으로 중간 frame 3 개 추출해 thumbnail 로 표시 (player 무거움 회피)
    thumbs_dir = html_dir / "_video_thumbs"
    for url in video_urls:
        local_path = _resolve_video_local_path(url)
        if local_path is None:
            # blob cache 없으면 fallback <video> tag
            src_path = _resolve_media_src(url, html_dir)
            media_html += (
                f'<video src="{_esc(src_path)}" controls muted preload="metadata" '
                f'class="media-video"></video>'
            )
            continue
        thumbs = _extract_video_thumbs(local_path, thumbs_dir, n_frames=3)
        if not thumbs:
            src_path = _resolve_media_src(url, html_dir)
            media_html += (
                f'<video src="{_esc(src_path)}" controls muted preload="metadata" '
                f'class="media-video"></video>'
            )
            continue
        for thumb in thumbs:
            try:
                rel = str(thumb.resolve().relative_to(html_dir.resolve())).replace("\\", "/")
            except ValueError:
                rel = thumb.as_uri()
            media_html += (
                f'<img src="{_esc(rel)}" loading="lazy" tabindex="0" '
                f'alt="video frame" class="media-img video-thumb" />'
            )

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
    # post 외부 URL — raw DB lookup (build time)
    post_url = _load_post_urls().get(pid, "")
    post_link_html = (
        f'<a href="{_esc(post_url)}" target="_blank" class="post-link">'
        f'<code class="post-id">{_esc(pid)}</code> ↗</a>'
        if post_url
        else f'<code class="post-id">{_esc(pid)}</code>'
    )
    return f'''
<article class="post" data-cluster="{_esc(cluster_key)}" data-source="{_esc(src)}" id="post-{_esc(pid)}">
  <header>
    <span class="badge src-{_esc(src)}">{_esc(src)}</span>
    {handle_html}
    {post_link_html}
    <span class="muted">{_esc(date)}</span>
    <span class="muted">eng={eng:,}</span>
    {f'<span class="muted">type={_esc(src_type)}</span>' if src_type else ''}
  </header>
  <div class="post-body">
    <div class="post-images">{media_html}</div>
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
        {_color_bar(palette)}
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


def _item_thumbnail_src(item: dict[str, Any], html_dir: Path) -> tuple[str, str] | None:
    """item 의 첫 thumbnail (IG=image[0], YT=video frame[0]) 와 source 라벨 반환.
    None 이면 thumbnail 생성 실패 (blob cache 없음 등).
    """
    n = item["normalized"]
    src = n.get("source", "")
    image_urls = n.get("image_urls") or []
    video_urls = n.get("video_urls") or []
    # IG: image 우선
    for url in image_urls:
        path_only = url.split("?", 1)[0]
        basename = Path(urlparse(path_only).path).name
        if not basename:
            continue
        cached = _IMG_CACHE / basename
        if cached.exists():
            try:
                rel = str(cached.resolve().relative_to(html_dir.resolve())).replace("\\", "/")
                return rel, src
            except ValueError:
                return cached.as_uri(), src
    # YT: video → frame thumb
    thumbs_dir = html_dir / "_video_thumbs"
    for url in video_urls:
        local = _resolve_video_local_path(url)
        if local is None:
            continue
        thumbs = _extract_video_thumbs(local, thumbs_dir, n_frames=3)
        if thumbs:
            try:
                rel = str(thumbs[0].resolve().relative_to(html_dir.resolve())).replace("\\", "/")
                return rel, src
            except ValueError:
                return thumbs[0].as_uri(), src
    return None


def _render_contributor_thumb(
    item: dict[str, Any], share: float, html_dir: Path,
) -> str:
    """contributor item 1 개를 thumbnail + share % badge 로 렌더 (cluster card 안)."""
    pid = item["normalized"].get("source_post_id", "")
    handle = item["normalized"].get("account_handle") or "—"
    src = item["normalized"].get("source", "?")
    pct = share * 100
    thumb = _item_thumbnail_src(item, html_dir)
    if thumb is None:
        # thumbnail 없으면 placeholder
        return (
            f'<a href="#post-{_esc(pid)}" class="contrib-thumb no-thumb" '
            f'title="@{_esc(handle)} · {pct:.1f}% contribution · {_esc(pid)}">'
            f'<div class="thumb-placeholder">{_esc(src)[:2].upper()}</div>'
            f'<span class="contrib-badge">{pct:.1f}%</span>'
            f'</a>'
        )
    src_path, src_label = thumb
    return (
        f'<a href="#post-{_esc(pid)}" class="contrib-thumb" '
        f'title="@{_esc(handle)} · {pct:.1f}% contribution · {_esc(pid)}">'
        f'<img src="{_esc(src_path)}" loading="lazy" alt="" />'
        f'<span class="contrib-badge src-{_esc(src_label)}">{pct:.1f}%</span>'
        f'</a>'
    )


def _cluster_summary_body(s: dict[str, Any]) -> str:
    """cluster summary 공통 body — 카드/디테일 양쪽 재사용."""
    drill = s.get("drilldown") or {}
    breakdown = s.get("score_breakdown") or {}
    breakdown_html = "".join(
        f'<tr><td>{_esc(k)}</td><td class="num">{v:.2f}</td></tr>'
        for k, v in breakdown.items()
    )
    palette_html = _color_bar(drill.get("color_palette") or [])
    top_inf = drill.get("top_influencers") or []
    top_inf_html = "".join(
        f'<li><a href="https://www.instagram.com/{_esc(h)}/" target="_blank">@{_esc(h)}</a></li>'
        for h in top_inf[:5]
    ) or '<li class="muted">∅</li>'
    return f'''
  <div class="cluster-body">
    <section class="full-row">
      <h4>color_palette</h4>
      {palette_html}
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
      <h4>top_influencers</h4>
      <ul>{top_inf_html}</ul>
    </section>
    <section class="full-row">
      <h4>distributions</h4>
      <div class="dist-grid">
        <div><b>silhouette</b>{_dist_table(drill.get("silhouette_distribution") or {})}</div>
        <div><b>occasion</b>{_dist_table(drill.get("occasion_distribution") or {})}</div>
        <div><b>styling</b>{_dist_table(drill.get("styling_distribution") or {})}</div>
        <div><b>brand</b>{_dist_table(drill.get("brand_distribution") or {})}</div>
      </div>
    </section>
  </div>'''


def _render_cluster_card(
    s: dict[str, Any],
    contributors: list[tuple[dict[str, Any], float]] | None,
    html_dir: Path,
    *,
    top_n_thumb: int = 5,
) -> str:
    """cluster summary card (클릭하면 detail 로 이동) — list view 용."""
    contributors = contributors or []
    n_contrib = len(contributors)
    cluster_key = s.get("cluster_key", "")
    return f'''
<article class="cluster cluster-summary" data-cluster-key="{_esc(cluster_key)}"
         onclick="showClusterDetail(this.dataset.weekIdx, this.dataset.clusterKey)"
         data-week-idx="{{week_idx}}">
  <header>
    <span class="cluster-key"><code>{_esc(cluster_key)}</code></span>
    <span class="score">{s.get("score", 0):.1f}</span>
    <span class="muted">{_esc(s.get("daily_direction"))}/{_esc(s.get("weekly_direction"))} · {_esc(s.get("lifecycle_stage"))}</span>
    <span class="muted">{n_contrib} contributors</span>
  </header>
  {_cluster_summary_body(s)}
</article>'''


def _render_compact_contributor(
    item: dict[str, Any], share: float, html_dir: Path,
) -> str:
    """cluster detail 안에 들어갈 compact contributor card — 이미지 + 핵심 정보 + palette bar."""
    n = item["normalized"]
    pid = n.get("source_post_id", "")
    src = n.get("source", "?")
    handle = n.get("account_handle") or "—"
    date = n.get("post_date", "")[:10] if n.get("post_date") else ""
    eng = n.get("engagement_raw", 0)
    pct = share * 100
    palette = item.get("post_palette") or []
    canonicals = item.get("canonicals") or []

    # thumbnail
    thumb_info = _item_thumbnail_src(item, html_dir)
    if thumb_info:
        src_path, _ = thumb_info
        thumb_html = f'<img src="{_esc(src_path)}" loading="lazy" tabindex="0" class="contrib-card-img" alt="" />'
    else:
        thumb_html = f'<div class="contrib-card-img-placeholder">{_esc(src[:2].upper())}</div>'

    acct_url = _account_url(src, handle if handle != "—" else None)
    handle_html = (
        f'<a href="{_esc(acct_url)}" target="_blank">@{_esc(handle)}</a>'
        if acct_url else f'<span>@{_esc(handle)}</span>'
    )
    post_url = _load_post_urls().get(pid, "")
    post_link_html = (
        f'<a href="{_esc(post_url)}" target="_blank"><code class="post-id">{_esc(pid)}</code> ↗</a>'
        if post_url
        else f'<code class="post-id">{_esc(pid)}</code>'
    )

    # canonical 의 주요 attributes 한 줄로 (첫 canonical 기준)
    canon_attr = ""
    if canonicals:
        rep = canonicals[0].get("representative", {})
        canon_attr = (
            f'upper={_esc(rep.get("upper_garment_type") or "—")} / '
            f'lower={_esc(rep.get("lower_garment_type") or "—")} / '
            f'fabric={_esc(rep.get("fabric") or "—")} / '
            f'technique={_esc(rep.get("technique") or "—")} / '
            f'silhouette={_esc(rep.get("silhouette") or "—")}'
        )

    return f'''
<div class="contrib-card">
  <div class="contrib-card-thumb">{thumb_html}</div>
  <div class="contrib-card-body">
    <div class="contrib-card-header">
      <span class="badge src-{_esc(src)}">{_esc(src)}</span>
      {handle_html}
      {post_link_html}
      <span class="muted">{_esc(date)}</span>
      <span class="muted">eng={eng:,}</span>
      <span class="contrib-share">{pct:.2f}%</span>
    </div>
    {f'<div class="contrib-attrs">{canon_attr}</div>' if canon_attr else ''}
    <div class="contrib-card-palette">{_color_bar(palette)}</div>
  </div>
</div>'''


def _render_cluster_detail(
    s: dict[str, Any],
    contributors: list[tuple[dict[str, Any], float]] | None,
    html_dir: Path,
    week_idx: int,
) -> str:
    """cluster detail panel — list 에서 cluster 카드 클릭 시 노출. 상단 cluster info,
    하단 contributor 인라인 카드 (share desc).
    """
    cluster_key = s.get("cluster_key", "")
    contributors = sorted(contributors or [], key=lambda t: -t[1])
    n_contrib = len(contributors)
    contrib_html = "".join(
        _render_compact_contributor(it, sh, html_dir) for it, sh in contributors
    ) or '<div class="muted">contributor 없음</div>'
    return f'''
<article class="cluster cluster-detail" data-cluster-key="{_esc(cluster_key)}" style="display:none">
  <div class="detail-nav">
    <button class="back-btn" onclick="showClusterList({week_idx})">← cluster 목록으로</button>
  </div>
  <header>
    <span class="cluster-key"><code>{_esc(cluster_key)}</code></span>
    <span class="score">{s.get("score", 0):.1f}</span>
    <span class="muted">{_esc(s.get("daily_direction"))}/{_esc(s.get("weekly_direction"))} · {_esc(s.get("lifecycle_stage"))}</span>
    <span class="muted">{n_contrib} contributors</span>
  </header>
  {_cluster_summary_body(s)}
  <h3 class="contrib-section-title">Contributors ({n_contrib}, share desc)</h3>
  <div class="contrib-cards-list">{contrib_html}</div>
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

.post-body { display: grid; grid-template-columns: 540px 1fr; gap: 16px; }
.post-images { display: flex; flex-wrap: wrap; gap: 6px; }
.media-img { width: 250px; height: 250px; object-fit: cover;
             border-radius: 4px; vertical-align: top; cursor: zoom-in;
             transition: transform 0.2s; }
.media-img:hover { transform: scale(1.05); }
.media-img:focus, .media-img:active {
             position: fixed; top: 50%; left: 50%; transform: translate(-50%,-50%);
             width: 90vw; height: 90vh; object-fit: contain; z-index: 1000;
             background: rgba(0,0,0,0.9); cursor: zoom-out; outline: none; }
.media-video { width: 250px; max-height: 320px; border-radius: 4px;
               background: #000; vertical-align: top; }
.video-thumb { border: 2px solid #ff0000; }
.video-thumb::after { content: '▶'; position: absolute; }
.post-meta section { font-size: 12px; margin-bottom: 6px; }
.post-meta { min-width: 0; }

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

/* week selector */
.week-selector-bar { background: #2a5db8; color: #fff; padding: 8px 12px;
                     border-radius: 6px; margin-bottom: 12px;
                     position: sticky; top: 0; z-index: 100; }
.week-selector-bar select { font-size: 13px; padding: 4px 8px;
                            border: 1px solid #fff; border-radius: 3px;
                            background: #fff; color: #222; }

.week-block { margin-top: 12px; }

.cluster-body section.full-row { grid-column: 1 / -1; }

/* palette horizontal bar */
.palette-bar { display: flex; height: 36px; border-radius: 4px; overflow: hidden;
               border: 1px solid rgba(0,0,0,0.1); width: 100%; }
.palette-seg { display: flex; flex-direction: column; align-items: center;
               justify-content: center; font-family: ui-monospace, monospace;
               font-size: 10px; line-height: 1.2; min-width: 0; padding: 2px;
               overflow: hidden; cursor: help; }
.palette-hex { white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
               max-width: 100%; }
.palette-pct { font-weight: bold; }

/* contributor thumbnail strip */
.contrib-strip { display: flex; flex-wrap: wrap; gap: 6px; }
.contrib-thumb { position: relative; display: inline-block; width: 90px;
                 height: 90px; border-radius: 6px; overflow: hidden;
                 border: 2px solid transparent; transition: transform 0.15s; }
.contrib-thumb:hover { transform: scale(1.05); border-color: #2a5db8; }
.contrib-thumb img { width: 100%; height: 100%; object-fit: cover; }
.contrib-thumb.no-thumb .thumb-placeholder { display: flex;
                 width: 100%; height: 100%; align-items: center;
                 justify-content: center; background: #ddd; color: #666;
                 font-weight: bold; font-size: 14px; }
.contrib-badge { position: absolute; bottom: 2px; right: 2px;
                 background: rgba(0,0,0,0.75); color: #fff;
                 padding: 1px 4px; border-radius: 3px; font-size: 10px;
                 font-weight: bold; }
.contrib-badge.src-instagram { border-bottom: 2px solid #e1306c; }
.contrib-badge.src-youtube { border-bottom: 2px solid #ff0000; }

/* tab bar */
.tab-bar { display: flex; gap: 4px; margin-bottom: 12px;
           border-bottom: 2px solid #ddd; }
.tab-btn { background: transparent; border: none; padding: 8px 16px;
           font-size: 13px; cursor: pointer; color: #666;
           border-radius: 6px 6px 0 0; transition: background 0.15s; }
.tab-btn:hover { background: #f0f0f0; }
.tab-btn.active { background: #2a5db8; color: #fff; font-weight: bold; }
.tab-content { min-height: 200px; }

/* cluster summary card (clickable) */
.cluster-summary { cursor: pointer; transition: transform 0.15s, box-shadow 0.15s; }
.cluster-summary:hover { transform: translateY(-2px);
                         box-shadow: 0 4px 12px rgba(42,93,184,0.15); }
.cluster-summary:active { transform: translateY(0); }

/* cluster detail nav */
.cluster-detail { background: #fff; border-radius: 6px; padding: 16px 20px;
                  margin-bottom: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
.detail-nav { margin-bottom: 12px; }
.back-btn { background: #fff; border: 1px solid #2a5db8; color: #2a5db8;
            padding: 6px 12px; border-radius: 4px; cursor: pointer;
            font-size: 12px; }
.back-btn:hover { background: #2a5db8; color: #fff; }
.contrib-section-title { margin-top: 20px; font-size: 14px;
                         border-bottom: 1px solid #ddd; padding-bottom: 6px; }

/* compact contributor card (cluster detail 안) */
.contrib-cards-list { display: grid;
                      grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
                      gap: 12px; margin-top: 12px; }
.contrib-card { background: #fafafa; border-radius: 6px;
                padding: 10px; display: grid;
                grid-template-columns: 160px 1fr; gap: 12px;
                border-left: 3px solid #2a5db8; }
.contrib-card-thumb { width: 160px; height: 160px; }
.contrib-card-img { width: 100%; height: 100%; object-fit: cover;
                    border-radius: 4px; cursor: zoom-in; transition: transform 0.15s; }
.contrib-card-img:hover { transform: scale(1.03); }
.contrib-card-img:focus { position: fixed; top: 50%; left: 50%;
                          transform: translate(-50%,-50%); width: 90vw;
                          height: 90vh; object-fit: contain; z-index: 1000;
                          background: rgba(0,0,0,0.9); cursor: zoom-out;
                          outline: none; }
.contrib-card-img-placeholder { display: flex; width: 100%; height: 100%;
                                align-items: center; justify-content: center;
                                background: #ddd; color: #666; font-weight: bold; }
.contrib-card-body { display: flex; flex-direction: column; gap: 4px;
                     min-width: 0; }
.contrib-card-header { display: flex; gap: 6px; flex-wrap: wrap;
                       align-items: baseline; font-size: 11px; }
.contrib-share { font-weight: bold; color: #d35400;
                 background: rgba(211,84,0,0.1); padding: 1px 6px;
                 border-radius: 3px; }
.contrib-attrs { font-size: 11px; color: #555; line-height: 1.4; }
.contrib-card-palette { margin-top: 4px; }
.contrib-card-palette .palette-bar { height: 24px; }

/* post-link icon */
.post-link { color: #2a5db8; }
"""


def _build_cluster_contributors(
    enriched: list[dict[str, Any]],
) -> dict[str, list[tuple[dict[str, Any], float]]]:
    """cluster_key → [(item, share), ...] (share desc 정렬).

    enriched.json 의 `trend_cluster_shares` 가 PR #31 (vision_normalize) 이전 raw 키
    (`blouse__embroidery__...`) 일 수 있어 신 normalize 로 재계산. src/ 의
    enriched_to_item_distribution + item_cluster_shares 호출. summaries 의
    cluster_key (정규화된 enum 값) 와 일치 보장.
    """
    from collections import defaultdict
    from aggregation.item_distribution_builder import enriched_to_item_distribution
    from aggregation.representative_builder import item_cluster_shares
    from contracts.enriched import EnrichedContentItem

    out: dict[str, list[tuple[dict[str, Any], float]]] = defaultdict(list)
    for it in enriched:
        try:
            model = EnrichedContentItem.model_validate(it)
        except Exception:
            continue
        item_dist = enriched_to_item_distribution(model)
        shares = item_cluster_shares(item_dist)
        for ck, sh in shares.items():
            if sh > 0:
                out[ck].append((it, sh))
    for ck in out:
        out[ck].sort(key=lambda t: -t[1])
    return dict(out)


def _build_week_section(
    week_idx: int,
    label: str,
    enriched: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    html_dir: Path,
) -> str:
    """주차별 section — Clusters tab + Items tab 구조. id prefix `w<idx>-` 로 anchor 충돌 방지."""
    sorted_enriched = sorted(
        enriched,
        key=lambda it: (it.get("trend_cluster_key") or "~~~",
                        it["normalized"].get("source_post_id", "")),
    )
    contributors_map = _build_cluster_contributors(enriched)

    # post id prefix 처리
    posts_html_raw = "\n".join(_render_post_card(item, html_dir) for item in sorted_enriched)
    posts_html = posts_html_raw.replace('id="post-', f'id="w{week_idx}-post-')

    # cluster summary cards (list view) — onclick 의 week_idx 채워넣기
    sorted_summaries = sorted(summaries, key=lambda s: -s.get("score", 0))
    cluster_cards_raw = "\n".join(
        _render_cluster_card(s, contributors_map.get(s.get("cluster_key", "")), html_dir)
        for s in sorted_summaries
    )
    cluster_cards = cluster_cards_raw.replace("{week_idx}", str(week_idx))

    # cluster detail panels (모두 hidden, 클릭 시 JS 가 표시)
    cluster_details = "\n".join(
        _render_cluster_detail(
            s, contributors_map.get(s.get("cluster_key", "")), html_dir, week_idx,
        )
        for s in sorted_summaries
    )

    n_items = len(enriched)
    n_clusters = len(summaries)
    n_canonicals = sum(len(it.get("canonicals", [])) for it in enriched)
    n_with_palette = sum(1 for it in enriched if it.get("post_palette"))
    n_ig = sum(1 for it in enriched if it["normalized"].get("source") == "instagram")
    n_yt = sum(1 for it in enriched if it["normalized"].get("source") == "youtube")

    distinct_clusters = sorted({it.get("trend_cluster_key") or "—" for it in enriched})
    cluster_options = "".join(
        f'<option value="{_esc(c)}">{_esc(c)}</option>' for c in distinct_clusters
    )

    return f"""
<section class="week-block" data-week="{week_idx}">
  <div class="summary-bar">
    <span><b>{label}</b></span>
    <span><b>{n_items}</b> items (IG <b>{n_ig}</b> / YT <b>{n_yt}</b>)</span>
    <span><b>{n_clusters}</b> clusters</span>
    <span><b>{n_canonicals}</b> canonicals</span>
    <span><b>{n_with_palette}</b> with palette</span>
  </div>

  <div class="tab-bar">
    <button class="tab-btn active" data-tab="clusters" onclick="showTab({week_idx},'clusters')">📊 Clusters ({n_clusters})</button>
    <button class="tab-btn" data-tab="items" onclick="showTab({week_idx},'items')">📷 Items ({n_items})</button>
  </div>

  <div class="tab-content tab-clusters">
    <div class="cluster-list">{cluster_cards}</div>
    <div class="cluster-details-container">{cluster_details}</div>
  </div>

  <div class="tab-content tab-items" style="display:none">
    <div class="filter-bar">
      <label>cluster filter:
        <select class="cluster-filter" data-week="{week_idx}" onchange="filterPosts({week_idx})">
          <option value="">(all)</option>
          {cluster_options}
        </select>
      </label>
      <label>source:
        <select class="source-filter" data-week="{week_idx}" onchange="filterPosts({week_idx})">
          <option value="">(all)</option>
          <option value="instagram">instagram</option>
          <option value="youtube">youtube</option>
        </select>
      </label>
      <label>id 검색: <input class="id-search" data-week="{week_idx}" type="search" placeholder="post_id 일부" oninput="filterPosts({week_idx})"></label>
      <span class="filter-count muted" data-week="{week_idx}"></span>
    </div>
    <div class="post-list">{posts_html}</div>
  </div>
</section>"""


def build_multi_week_html(
    weeks: list[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]],
    html_dir: Path,
) -> str:
    """여러 주차 single HTML — selector 로 토글.

    weeks: [(label, enriched, summaries), ...] — label 은 화면 표시용 ("2026-04-20 ~ 2026-04-26").
    """
    if not weeks:
        raise ValueError("weeks must be non-empty")
    sections = "\n".join(
        _build_week_section(idx, label, enr, summ, html_dir)
        for idx, (label, enr, summ) in enumerate(weeks)
    )
    options = "".join(
        f'<option value="{idx}">{_esc(label)}</option>'
        for idx, (label, _, _) in enumerate(weeks)
    )
    return f"""<!DOCTYPE html>
<html lang="ko"><head>
<meta charset="utf-8">
<title>분석 검수 — {len(weeks)} weeks</title>
<style>{_CSS}</style>
</head><body>

<h1>분석 검수 (M3.A Step E — multi-week)</h1>

<div class="week-selector-bar">
  <label>주차 선택:
    <select id="week-selector" onchange="showWeek(this.value)">
      {options}
    </select>
  </label>
</div>

{sections}

<script>
function showWeek(idx) {{
  document.querySelectorAll('section.week-block').forEach(el => {{
    el.style.display = (String(el.dataset.week) === String(idx)) ? '' : 'none';
  }});
}}
function showTab(weekIdx, tabName) {{
  const root = document.querySelector(`section.week-block[data-week="${{weekIdx}}"]`);
  if (!root) return;
  root.querySelectorAll('.tab-btn').forEach(b => {{
    b.classList.toggle('active', b.dataset.tab === tabName);
  }});
  root.querySelectorAll('.tab-content').forEach(c => {{
    const active = c.classList.contains('tab-' + tabName);
    c.style.display = active ? '' : 'none';
  }});
  // tab 전환 시 cluster detail 도 list view 로 reset
  if (tabName === 'clusters') showClusterList(weekIdx);
}}
function showClusterDetail(weekIdx, clusterKey) {{
  const root = document.querySelector(`section.week-block[data-week="${{weekIdx}}"]`);
  if (!root) return;
  root.querySelector('.cluster-list').style.display = 'none';
  root.querySelectorAll('.cluster-detail').forEach(el => {{
    el.style.display = (el.dataset.clusterKey === clusterKey) ? '' : 'none';
  }});
  window.scrollTo({{ top: 0, behavior: 'smooth' }});
}}
function showClusterList(weekIdx) {{
  const root = document.querySelector(`section.week-block[data-week="${{weekIdx}}"]`);
  if (!root) return;
  root.querySelector('.cluster-list').style.display = '';
  root.querySelectorAll('.cluster-detail').forEach(el => {{
    el.style.display = 'none';
  }});
}}
function filterPosts(weekIdx) {{
  const root = document.querySelector(`section.week-block[data-week="${{weekIdx}}"]`);
  if (!root) return;
  const ck = root.querySelector('.cluster-filter')?.value || '';
  const src = root.querySelector('.source-filter')?.value || '';
  const q = root.querySelector('.id-search')?.value.trim().toLowerCase() || '';
  let shown = 0;
  root.querySelectorAll('article.post').forEach(el => {{
    const okCk = !ck || el.dataset.cluster === ck;
    const okSrc = !src || el.dataset.source === src;
    const okQ = !q || el.id.toLowerCase().includes(q);
    const visible = okCk && okSrc && okQ;
    el.style.display = visible ? '' : 'none';
    if (visible) shown++;
  }});
  const counter = root.querySelector('.filter-count');
  if (counter) counter.textContent = `(${{shown}} 표시)`;
}}
// 초기: 첫 주만 표시 + 모든 주의 filter 한 번씩 호출
showWeek(0);
document.querySelectorAll('section.week-block').forEach(el => {{
  filterPosts(el.dataset.week);
}});
</script>

</body></html>"""


def build_html(
    enriched: list[dict[str, Any]], summaries: list[dict[str, Any]], html_dir: Path
) -> str:
    """단일 주차 HTML (backwards compat). 내부적으로 multi-week 1 entry 로 빌드."""
    return build_multi_week_html([("(single)", enriched, summaries)], html_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enriched", type=Path, default=None,
                        help="단일 주차 모드: enriched.json 경로")
    parser.add_argument("--summaries", type=Path, default=None,
                        help="단일 주차 모드: summaries.json 경로")
    parser.add_argument("--weeks", type=str, default=None,
                        help="multi-week 모드: end_date 콤마 구분 "
                             "(예: 2026-04-26,2026-04-19,2026-04-12). 각 date 의 "
                             "outputs/<DATE>/{enriched,summaries}.json 사용.")
    parser.add_argument("--output", type=Path, default=_REPO / "outputs" / "review.html")
    args = parser.parse_args()

    html_dir = args.output.parent
    html_dir.mkdir(parents=True, exist_ok=True)

    if args.weeks:
        from datetime import date as Date, timedelta
        weeks: list[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]] = []
        for end_str in [s.strip() for s in args.weeks.split(",") if s.strip()]:
            end_date = Date.fromisoformat(end_str)
            start_date = end_date - timedelta(days=6)
            label = f"{start_date} ~ {end_date}"
            enr_path = _REPO / "outputs" / end_str / "enriched.json"
            sum_path = _REPO / "outputs" / end_str / "summaries.json"
            if not enr_path.exists() or not sum_path.exists():
                print(f"WARN: skip {label} — {enr_path} or {sum_path} missing")
                continue
            with enr_path.open() as f: enr = json.load(f)
            with sum_path.open() as f: summ = json.load(f)
            weeks.append((label, enr, summ))
        if not weeks:
            raise SystemExit("--weeks 결과 빈 list. 각 date 의 enriched/summaries 확인")
        html = build_multi_week_html(weeks, html_dir)
        args.output.write_text(html, encoding="utf-8")
        total_items = sum(len(e) for _, e, _ in weeks)
        total_clusters = sum(len(s) for _, _, s in weeks)
        print(f"Wrote {args.output} — {len(weeks)} weeks / {total_items} items / {total_clusters} clusters")
        return 0

    # 단일 주차 모드 (하위 호환)
    enriched_path = args.enriched or (_REPO / "outputs" / "enriched.json")
    summaries_path = args.summaries or (_REPO / "outputs" / "summaries.json")
    if not enriched_path.exists():
        raise SystemExit(f"enriched not found: {enriched_path}")
    if not summaries_path.exists():
        raise SystemExit(f"summaries not found: {summaries_path}")

    with enriched_path.open() as f: enriched = json.load(f)
    with summaries_path.open() as f: summaries = json.load(f)
    html = build_html(enriched, summaries, html_dir)
    args.output.write_text(html, encoding="utf-8")
    print(f"Wrote {args.output} — {len(enriched)} items / {len(summaries)} clusters")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
