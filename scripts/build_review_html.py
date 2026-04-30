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


def _rel_path(target: Path, base: Path) -> str:
    """target 을 base 기준 상대 경로로 (`..` 사용 가능). HTTP 서버 호환.

    Path.relative_to 는 descendant 만 지원 (sample_data/image_cache 가
    outputs/weekly_review 의 자손이 아니라 ValueError → file:// fallback 시
    ngrok HTTP 접근 깨짐). os.path.relpath 는 `..` 으로 cross-tree 처리 가능.
    """
    import os as _os
    return _os.path.relpath(str(target.resolve()), str(base.resolve())).replace("\\", "/")


def _resolve_media_src(url: str, html_dir: Path) -> str:
    """blob 캐시 (이미지/영상) 있으면 상대 경로 (`..` 포함 OK), 아니면 원본 URL 반환."""
    if not url:
        return ""
    path_only = url.split("?", 1)[0]
    basename = Path(urlparse(path_only).path).name
    if not basename:
        return url
    cached = _IMG_CACHE / basename
    if cached.exists():
        return _rel_path(cached, html_dir)
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


def _ffprobe_fps(video_path: Path) -> float | None:
    """ffprobe 로 영상 fps (frame rate) 반환. r_frame_rate "30/1" 형태 → 30.0."""
    import subprocess
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0:
            return None
        s = out.stdout.strip()
        if "/" in s:
            num, den = s.split("/")
            den_f = float(den)
            return float(num) / den_f if den_f != 0 else None
        return float(s)
    except Exception:
        return None


def _selected_video_frames(
    item: dict[str, Any], video_path: Path, thumbs_dir: Path, *, max_n: int = 5,
    match_image_ids: set[str] | None = None,
) -> list[Path]:
    """item.canonicals[*].members[*].image_id 에서 video_path.stem 매칭하는
    global_idx 추출 → 균등 분포 max_n 개 선택 → ffmpeg 으로 idx/fps 시점 frame 추출.

    canonical 에 등장한 frame 이 0 개면 빈 list (호출자가 fallback 처리).

    match_image_ids: 지정 시 그 image_id set 에 속하는 frame (= cluster 매칭 canonical
    의 frame) 만 후보. None = 모든 canonical (옛 동작).
    """
    import subprocess
    stem = video_path.stem
    indices: set[int] = set()
    for c in item.get("canonicals", []):
        for m in (c.get("members") or []):
            iid = m.get("image_id") or ""
            if match_image_ids is not None and iid not in match_image_ids:
                continue
            prefix = f"{stem}_f"
            if iid.startswith(prefix):
                tail = iid[len(prefix):]
                try:
                    indices.add(int(tail))
                except ValueError:
                    continue
    if not indices:
        return []
    sorted_idx = sorted(indices)
    if len(sorted_idx) <= max_n:
        chosen = sorted_idx
    else:
        # 균등 분포 — 첫/끝 포함, 중간 균등
        step = (len(sorted_idx) - 1) / (max_n - 1) if max_n > 1 else 0
        chosen = [sorted_idx[round(i * step)] for i in range(max_n)]

    fps = _ffprobe_fps(video_path)
    if fps is None or fps <= 0:
        return []
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for idx in chosen:
        ts = idx / fps
        thumb_path = thumbs_dir / f"{stem}_sel{idx}.jpg"
        if thumb_path.exists() and thumb_path.stat().st_size > 0:
            out.append(thumb_path)
            continue
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-ss", f"{ts:.3f}", "-i", str(video_path),
                 "-frames:v", "1", "-q:v", "3", str(thumb_path)],
                capture_output=True, timeout=15,
            )
            if thumb_path.exists() and thumb_path.stat().st_size > 0:
                out.append(thumb_path)
        except Exception:
            continue
    return out


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
    eng = n.get("engagement_raw_count") or n.get("engagement_raw") or 0
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
            rel = _rel_path(thumb, html_dir)
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
    """item 의 첫 thumbnail 1개. _item_media_srcs[0] 의 alias."""
    media = _item_media_srcs(item, html_dir, max_n=1)
    return media[0] if media else None


def _item_media_srcs(
    item: dict[str, Any], html_dir: Path, *, max_n: int = 5,
    match_image_ids: set[str] | None = None,
) -> list[tuple[str, str]]:
    """item 의 media 최대 max_n 개 — IG 는 image_urls 첫 N 장, YT 는 영상 frame
    균등 분포 N 개 (1/(N+1) ~ N/(N+1) 위치). image+video mix 면 image 우선.

    각 entry: (src_path, src_label "instagram"|"youtube"). blob 캐시 없는 URL 은 skip.

    match_image_ids: 특정 canonical 에 매칭된 image_id 집합. 지정 시 그 image_id 와
    basename 일치하는 image_url / video frame 만 표시 — cluster matching 카드에서
    "이 cluster 에 매칭된 frame 만" 보여주기 위함. None = 전체 (옛 동작).
    """
    n = item["normalized"]
    src = n.get("source", "")
    image_urls = n.get("image_urls") or []
    video_urls = n.get("video_urls") or []
    out: list[tuple[str, str]] = []

    # IG: image 우선 (post 내 등장 순서). match_image_ids 지정 시 그 안 image 만.
    for url in image_urls:
        if len(out) >= max_n:
            break
        path_only = url.split("?", 1)[0]
        basename = Path(urlparse(path_only).path).name
        if not basename:
            continue
        if match_image_ids is not None and basename not in match_image_ids:
            continue
        cached = _IMG_CACHE / basename
        if cached.exists():
            out.append((_rel_path(cached, html_dir), src))

    # YT/Reel: 분석 시 선택된 frame 우선 → fallback 으로 균등 분포
    if len(out) < max_n:
        thumbs_dir = html_dir / "_video_thumbs"
        for url in video_urls:
            if len(out) >= max_n:
                break
            local = _resolve_video_local_path(url)
            if local is None:
                continue
            remaining = max_n - len(out)
            # 1차: canonical 에 등장한 selected frame (분석 파이프라인이 실제 사용한 frame).
            # match_image_ids 지정 시 그 set 만 (cluster 매칭 frame 한정).
            selected = _selected_video_frames(
                item, local, thumbs_dir, max_n=remaining,
                match_image_ids=match_image_ids,
            )
            # 2차 fallback: selected 0 개 + match_image_ids 가 None 이면 균등 분포.
            # match_image_ids 가 지정됐는데 0 개면 균등 fallback 안 함 (cluster 와 무관한
            # frame 노출 차단). YT CDN fallback 이 아래에서 동작.
            if selected:
                thumbs = selected
            elif match_image_ids is None:
                thumbs = _extract_video_thumbs(local, thumbs_dir, n_frames=remaining)
            else:
                thumbs = []
            for thumb in thumbs:
                if len(out) >= max_n:
                    break
                out.append((_rel_path(thumb, html_dir), src))

    # 3차 fallback: YT video 가 blob 캐시에 없거나 ffmpeg 추출 실패 → YouTube CDN
    # thumbnail (`https://img.youtube.com/vi/{video_id}/{kind}.jpg`) 사용. video_id
    # = source_post_id. frame 보단 informativeness 떨어지지만 placeholder 보단 나음.
    # 균등 frame 효과를 위해 0/1/2/3.jpg (썸네일 슬라이드) 와 hqdefault 조합.
    if src == "youtube" and len(out) < max_n:
        vid = n.get("source_post_id") or n.get("url_short_tag") or ""
        if vid:
            cdn_kinds = ["hqdefault", "0", "1", "2", "3"]
            for kind in cdn_kinds:
                if len(out) >= max_n:
                    break
                out.append((f"https://img.youtube.com/vi/{vid}/{kind}.jpg", src))
    return out


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


def _cluster_summary_body(
    s: dict[str, Any], *, extra_html: str = "", extra_html_bottom: str = "",
) -> str:
    """cluster summary 공통 body — 카드/디테일 양쪽 재사용.
    extra_html: color_palette 직후 삽입. extra_html_bottom: 가장 마지막 (distributions 후)."""
    drill = s.get("drilldown") or {}
    breakdown = s.get("score_breakdown") or {}

    def _fmt_breakdown_row(key: str, value: Any) -> str:
        if isinstance(value, dict):
            sub = ", ".join(
                f"{_esc(str(sk))}={sv:.2f}" if isinstance(sv, (int, float)) else f"{_esc(str(sk))}={_esc(str(sv))}"
                for sk, sv in value.items()
            )
            return f'<tr><td>{_esc(key)}</td><td class="num">{sub}</td></tr>'
        if isinstance(value, (int, float)):
            return f'<tr><td>{_esc(key)}</td><td class="num">{value:.2f}</td></tr>'
        return f'<tr><td>{_esc(key)}</td><td class="num">{_esc(str(value))}</td></tr>'

    breakdown_html = "".join(_fmt_breakdown_row(k, v) for k, v in breakdown.items())
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
    {extra_html}
    <section>
      <h4>breakdown</h4>
      <table class="dist">{breakdown_html}</table>
    </section>
    <section>
      <h4>counts</h4>
      <div class="counts-line">
        <span>total={s.get("post_count_total", 0):.1f}</span>
        <span>today={s.get("post_count_today", 0):.1f}</span>
        <span>views={s.get("total_video_views", 0):,}</span>
      </div>
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
    {extra_html_bottom}
  </div>'''


def _render_cluster_card(
    s: dict[str, Any],
    contributors: list[tuple[dict[str, Any], float]] | None,
    html_dir: Path,
    *,
    top_n_thumb: int = 3,
) -> str:
    """cluster summary card (클릭하면 detail 로 이동) — list view 용.
    상단에 IG/YT 각 top N 의 thumbnail 1장씩 표시 (검수 시 cluster 성격 빠른 파악).
    """
    contributors = contributors or []
    n_contrib = len(contributors)
    cluster_key = s.get("cluster_key", "")

    # IG/YT contributor 분리 + top N 의 1장 thumbnail
    contributors_sorted = sorted(contributors, key=lambda t: -t[1])
    ig = [(it, sh) for it, sh in contributors_sorted
          if it["normalized"].get("source") == "instagram"][:top_n_thumb]
    yt = [(it, sh) for it, sh in contributors_sorted
          if it["normalized"].get("source") == "youtube"][:top_n_thumb]

    def _mini_thumb(item: dict[str, Any], share: float, label: str) -> str:
        thumb_info = _item_thumbnail_src(item, html_dir)
        pct = share * 100
        if thumb_info:
            src_path, src = thumb_info
            return (
                f'<div class="cluster-mini-thumb" '
                f'title="{_esc(label)} · {pct:.1f}%">'
                f'<img src="{_esc(src_path)}" loading="lazy" alt="" />'
                f'<span class="cluster-mini-badge src-{_esc(src)}">{pct:.1f}%</span>'
                f'</div>'
            )
        return (
            f'<div class="cluster-mini-thumb no-thumb" title="{_esc(label)}">'
            f'<div class="cluster-mini-placeholder">·</div></div>'
        )

    ig_thumbs = "".join(
        _mini_thumb(it, sh, f'@{it["normalized"].get("account_handle") or "—"}')
        for it, sh in ig
    ) or '<span class="muted small">∅</span>'
    yt_thumbs = "".join(
        _mini_thumb(it, sh, f'@{it["normalized"].get("account_handle") or "—"}')
        for it, sh in yt
    ) or '<span class="muted small">∅</span>'

    contrib_strip_html = (
        f'<section class="full-row cluster-mini-strip">'
        f'<h4>Top Contributors ({n_contrib} contributors · IG / YT × top 3 · 클릭 상세)</h4>'
        f'<div class="mini-strip-row"><span class="mini-label">IG</span>{ig_thumbs}</div>'
        f'<div class="mini-strip-row"><span class="mini-label">YT</span>{yt_thumbs}</div>'
        f'</section>'
    )

    return f'''
<article class="cluster cluster-summary" data-cluster-key="{_esc(cluster_key)}"
         onclick="showClusterDetail(this.dataset.weekIdx, this.dataset.clusterKey)"
         data-week-idx="{{week_idx}}">
  <header>
    <div class="cluster-title-row">
      <span class="cluster-key"><code>{_esc(cluster_key)}</code></span>
    </div>
    <div class="cluster-meta-row">
      <span class="score">{s.get("score", 0):.1f}</span>
      <span class="meta-pill direction">{_esc(s.get("daily_direction"))}/{_esc(s.get("weekly_direction"))}</span>
      <span class="meta-pill lifecycle">{_esc(s.get("lifecycle_stage"))}</span>
    </div>
  </header>
  {_cluster_summary_body(s, extra_html=contrib_strip_html)}
</article>'''


def _render_compact_contributor(
    item: dict[str, Any], share: float, html_dir: Path,
    *, cluster_key: str | None = None,
) -> str:
    """cluster detail 안에 들어갈 compact contributor card — 이미지 + 핵심 정보 + palette bar.

    contributor 단위는 Item (= 1 post). post 안에 multi-outfit (multi-canonical) 인 경우
    각 canonical 의 (G, F, T, silhouette) 를 한 줄씩 펼침 + 현재 cluster_key 와 매칭되는
    canonical (G__F) 는 굵은 파란색으로 highlight. 옆에는 post-level 합산 (eng, palette,
    brand mention) 표시 — Item 단위 정보 함께 노출.
    """
    n = item["normalized"]
    pid = n.get("source_post_id", "")
    src = n.get("source", "?")
    handle = n.get("account_handle") or "—"
    date = n.get("post_date", "")[:10] if n.get("post_date") else ""
    eng = n.get("engagement_raw_count") or n.get("engagement_raw") or 0
    pct = share * 100
    palette = item.get("post_palette") or []
    canonicals = item.get("canonicals") or []
    brands = item.get("brands") or []

    # cluster_key 매칭되는 canonical 의 members image_id 집합 — 매칭 frame/image 만
    # thumbnail 표시 (다른 pose / 사람 없는 image 제외). cluster_key=None 또는 매칭
    # canonical 없으면 전체 fallback (옛 동작).
    match_image_ids: set[str] | None = None
    if cluster_key is not None and canonicals:
        from aggregation.vision_normalize import (
            normalize_garment_for_cluster, normalize_fabric,
        )
        from contracts.vision import EthnicOutfit
        ids: set[str] = set()
        for c in canonicals:
            rep_dict = c.get("representative") or {}
            try:
                rep_model = EthnicOutfit.model_validate(rep_dict)
                g = normalize_garment_for_cluster(rep_model)
                f = normalize_fabric(rep_model)
                g_val = g.value if g is not None else None
                f_val = f.value if f is not None else None
            except Exception:
                continue
            canon_ck = f"{g_val}__{f_val}" if g_val and f_val else None
            if canon_ck == cluster_key:
                for m in (c.get("members") or []):
                    iid = m.get("image_id")
                    if iid:
                        ids.add(iid)
        if ids:
            match_image_ids = ids

    # thumbnail grid (최대 5장 — IG image / YT frame 균등 분포)
    media = _item_media_srcs(item, html_dir, max_n=5, match_image_ids=match_image_ids)
    if media:
        thumb_html = "".join(
            f'<img src="{_esc(p)}" loading="lazy" tabindex="0" class="contrib-card-img" alt="" />'
            for p, _ in media
        )
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

    # canonical 별 한 줄 펼침 — multi-outfit post 의 모든 group 노출, 현재 cluster 매칭 highlight
    canon_lines = []
    if canonicals:
        from aggregation.vision_normalize import (
            normalize_garment_for_cluster, normalize_fabric,
        )
        from contracts.vision import EthnicOutfit
        for idx, c in enumerate(canonicals):
            rep_dict = c.get("representative") or {}
            try:
                rep_model = EthnicOutfit.model_validate(rep_dict)
                g = normalize_garment_for_cluster(rep_model)
                f = normalize_fabric(rep_model)
                g_val = g.value if g is not None else None
                f_val = f.value if f is not None else None
            except Exception:
                g_val = rep_dict.get("upper_garment_type")
                f_val = rep_dict.get("fabric")
            canon_ck = f"{g_val}__{f_val}" if g_val and f_val else None
            is_match = cluster_key is not None and canon_ck == cluster_key
            klass = "canon-line match" if is_match else "canon-line"
            badge = '<span class="canon-match-badge">★ this cluster</span>' if is_match else ''
            canon_lines.append(
                f'<div class="{klass}">'
                f'<span class="canon-idx">#{idx}</span> '
                f'upper={_esc(rep_dict.get("upper_garment_type") or "—")} / '
                f'lower={_esc(rep_dict.get("lower_garment_type") or "—")} / '
                f'fabric={_esc(rep_dict.get("fabric") or "—")} / '
                f'technique={_esc(rep_dict.get("technique") or "—")} / '
                f'silhouette={_esc(rep_dict.get("silhouette") or "—")}'
                f' {badge}</div>'
            )
    canon_html = "\n".join(canon_lines)

    brand_names = [b.get("name") if isinstance(b, dict) else str(b) for b in brands]
    brand_names = [b for b in brand_names if b]
    brands_html = (
        f'<span class="muted">brands: {_esc(", ".join(brand_names[:3]))}</span>'
        if brand_names else ''
    )
    item_summary = (
        f'<div class="contrib-item-summary">'
        f'<span class="muted">canonicals: {len(canonicals)}</span>'
        f' {brands_html}'
        f'</div>' if canonicals else ''
    )

    # Item full detail expand (post 전체 사진 + 모든 outfit + post-level 속성).
    # 기본 hidden, contributor 카드 click 시 toggle.
    item_detail_html = _render_item_full_detail(item, html_dir, cluster_key=cluster_key)
    detail_id = f"itemdetail-{_esc(pid)}-{cluster_key or 'na'}"

    return f'''
<div class="contrib-card-wrap">
  <div class="contrib-card" onclick="toggleItemDetail('{_esc(detail_id)}')" title="클릭 → 이 item 의 전체 사진 / 모든 outfit / post-level 속성 보기">
    <div class="contrib-card-thumb">{thumb_html}</div>
    <div class="contrib-card-body">
      <div class="contrib-card-header">
        <span class="badge src-{_esc(src)}">{_esc(src)}</span>
        {handle_html}
        {post_link_html}
        <span class="muted">{_esc(date)}</span>
        <span class="muted">eng={eng:,}</span>
        <span class="contrib-share">{pct:.2f}%</span>
        <span class="contrib-expand-hint">▾ click for item detail</span>
      </div>
      {item_summary}
      {f'<div class="contrib-attrs">{canon_html}</div>' if canon_html else ''}
      <div class="contrib-card-palette">{_color_bar(palette)}</div>
    </div>
  </div>
  <div class="item-detail" id="{_esc(detail_id)}" style="display:none">{item_detail_html}</div>
</div>'''


def _render_item_full_detail(
    item: dict[str, Any], html_dir: Path, *, cluster_key: str | None = None,
) -> str:
    """Item 전체 상세 — contributor 카드 click 시 expand. 8단계 신규 (2026-04-30).

    구성:
    - 전체 carousel/frame 사진 (cluster 매칭 outfit 만이 아니라 모든 image/frame)
    - post-level 속성 (occasion, brands, post_palette, engagement raw count)
    - 모든 canonical (outfit) 의 attribute + palette + 매칭 cluster highlight
    """
    n = item["normalized"]
    canonicals = item.get("canonicals") or []
    brands = item.get("brands") or []
    occasion = item.get("occasion") or "—"
    palette = item.get("post_palette") or []
    eng_raw = n.get("engagement_raw_count") or n.get("engagement_raw") or 0
    eng_score = n.get("engagement_score") or 0.0
    growth = n.get("growth_metric") or 0
    text_blob = (n.get("text_blob") or "")[:300]

    # 전체 사진 — match_image_ids=None 으로 모든 이미지 (carousel 전체 / 모든 video frame)
    full_media = _item_media_srcs(item, html_dir, max_n=12, match_image_ids=None)
    if full_media:
        all_thumbs = "".join(
            f'<img src="{_esc(p)}" loading="lazy" class="item-detail-img" alt="" />'
            for p, _ in full_media
        )
    else:
        all_thumbs = '<span class="muted">no media available</span>'

    # 모든 canonical 펼침 — palette 도 같이
    from aggregation.vision_normalize import (
        normalize_garment_for_cluster, normalize_fabric,
    )
    from contracts.vision import EthnicOutfit
    canon_blocks = []
    for idx, c in enumerate(canonicals):
        rep = c.get("representative") or {}
        try:
            rm = EthnicOutfit.model_validate(rep)
            g = normalize_garment_for_cluster(rm); f = normalize_fabric(rm)
            ck_canon = f"{g.value if g else 'unknown'}__{f.value if f else 'unknown'}"
        except Exception:
            ck_canon = None
        is_match = cluster_key is not None and ck_canon == cluster_key
        klass = "item-canon-block match" if is_match else "item-canon-block"
        match_badge = '<span class="canon-match-badge">★ this cluster</span>' if is_match else ''
        canon_palette_html = _color_bar(c.get("palette") or [])
        members_count = len(c.get("members") or [])
        canon_blocks.append(
            f'<div class="{klass}">'
            f'<div class="item-canon-header">'
            f'<span class="canon-idx">#{idx}</span> '
            f'<code>{_esc(ck_canon or "?")}</code> '
            f'<span class="muted">({members_count} members)</span> {match_badge}'
            f'</div>'
            f'<div class="item-canon-attrs">'
            f'upper={_esc(rep.get("upper_garment_type") or "—")} / '
            f'lower={_esc(rep.get("lower_garment_type") or "—")} / '
            f'fabric={_esc(rep.get("fabric") or "—")} / '
            f'technique={_esc(rep.get("technique") or "—")} / '
            f'silhouette={_esc(rep.get("silhouette") or "—")}'
            f'</div>'
            f'<div class="item-canon-palette">{canon_palette_html}</div>'
            f'</div>'
        )
    canon_blocks_html = "\n".join(canon_blocks) or '<span class="muted">no canonical</span>'

    brand_names = [b.get("name") if isinstance(b, dict) else str(b) for b in brands]
    brand_names = [b for b in brand_names if b]
    brand_str = ", ".join(brand_names) if brand_names else "—"
    return f'''
<div class="item-detail-inner">
  <h4 class="item-detail-title">📌 Item full detail</h4>
  <section class="item-detail-section">
    <h5>전체 사진 / Frame ({len(full_media)})</h5>
    <div class="item-detail-media-grid">{all_thumbs}</div>
  </section>
  <section class="item-detail-section">
    <h5>Post-level 속성</h5>
    <div class="item-detail-meta">
      <div><b>occasion</b>: {_esc(occasion)}</div>
      <div><b>brands</b>: {_esc(brand_str)}</div>
      <div><b>engagement_raw_count</b>: {eng_raw:,} / <b>engagement_score</b>: {eng_score:.4f}</div>
      <div><b>growth_metric</b>: {growth:,} (IG=likes / YT=view_count)</div>
      <div class="muted item-detail-text"><b>text</b>: {_esc(text_blob)}{'…' if text_blob and len(text_blob) >= 300 else ''}</div>
    </div>
    <div class="item-detail-section-palette">
      <h5>post_palette (item-level)</h5>
      {_color_bar(palette)}
    </div>
  </section>
  <section class="item-detail-section">
    <h5>모든 outfit ({len(canonicals)} canonicals) — cluster 매칭 ★ 표시</h5>
    {canon_blocks_html}
  </section>
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
        _render_compact_contributor(it, sh, html_dir, cluster_key=cluster_key)
        for it, sh in contributors
    ) or '<div class="muted">contributor 없음</div>'
    return f'''
<article class="cluster cluster-detail" data-cluster-key="{_esc(cluster_key)}" style="display:none">
  <div class="detail-nav">
    <button class="back-btn" onclick="showClusterList({week_idx})">← cluster 목록으로</button>
  </div>
  <header>
    <div class="cluster-title-row">
      <span class="cluster-key"><code>{_esc(cluster_key)}</code></span>
    </div>
    <div class="cluster-meta-row">
      <span class="score">{s.get("score", 0):.1f}</span>
      <span class="meta-pill direction">{_esc(s.get("daily_direction"))}/{_esc(s.get("weekly_direction"))}</span>
      <span class="meta-pill lifecycle">{_esc(s.get("lifecycle_stage"))}</span>
      <span class="muted">{n_contrib} contributors</span>
    </div>
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

.cluster-list { display: grid; grid-template-columns: repeat(auto-fit, minmax(440px, 1fr)); gap: 12px; }

.post-id { font-size: 10px; color: #999; font-family: ui-monospace, monospace;
           background: transparent; padding: 0; }

.cluster, .post { background: #fff; border-radius: 6px; padding: 12px 16px;
                  margin-bottom: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
.post header { display: flex; gap: 12px; align-items: baseline;
               margin-bottom: 8px; padding-bottom: 6px;
               border-bottom: 1px solid #eee; }
.cluster header { margin-bottom: 10px; padding-bottom: 8px;
                  border-bottom: 1px solid #eee; }
.cluster-title-row { margin-bottom: 8px; }
.cluster-meta-row { display: flex; gap: 8px; font-size: 12px;
                    flex-wrap: wrap; align-items: center; }
.cluster .score { font-size: 20px; font-weight: bold; color: #d35400;
                  background: rgba(211,84,0,0.08); padding: 2px 10px;
                  border-radius: 4px; }
.cluster .cluster-key { display: block; }
.cluster .cluster-key code { display: inline-block; font-size: 14px;
                              font-weight: bold; background: #fff3cd;
                              color: #1f3a68; padding: 4px 10px;
                              border-radius: 4px; border: 1px solid #ffd966;
                              word-break: break-all; line-height: 1.4; }
.meta-pill { padding: 2px 8px; border-radius: 10px; font-size: 11px;
             font-weight: 500; border: 1px solid #ddd; background: #fafafa; }
.meta-pill.direction { color: #2a5db8; border-color: #c5d6f0; background: #eff4fc; }
.meta-pill.lifecycle { color: #6b3aa0; border-color: #d8c5f0; background: #f4eefc; }

.cluster-body { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; }
.cluster-body section { font-size: 12px; min-width: 0; overflow-wrap: anywhere; }
.counts-line { display: flex; flex-wrap: wrap; gap: 4px 12px; font-variant-numeric: tabular-nums; }
.counts-line > span { white-space: nowrap; }
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

/* cluster summary 의 mini thumbnail strip (IG/YT top N) — 카드 하단 별도 섹션 */
.cluster-mini-strip { background: #fafafa; padding: 8px 10px;
                      border-radius: 4px; margin-top: 4px; }
.cluster-mini-strip h4 { margin-top: 0; }
.mini-strip-row { display: flex; gap: 8px; align-items: center;
                  margin-bottom: 6px; }
.mini-strip-row:last-child { margin-bottom: 0; }
.mini-label { font-size: 11px; font-weight: bold; color: #555;
              min-width: 30px; text-align: center;
              border: 1px solid #ccc; border-radius: 4px; padding: 2px 6px;
              background: #fff; flex-shrink: 0; }
.mini-label[data-src="ig"] { border-color: #e1306c; color: #e1306c; }
.cluster-mini-thumb { position: relative; width: 100px; height: 100px;
                      border-radius: 6px; overflow: hidden; flex-shrink: 0;
                      box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
.cluster-mini-thumb img { width: 100%; height: 100%; object-fit: cover; }
.cluster-mini-badge { position: absolute; bottom: 2px; right: 2px;
                      background: rgba(0,0,0,0.75); color: #fff;
                      font-size: 11px; font-weight: bold;
                      padding: 1px 5px; border-radius: 3px; }
.cluster-mini-badge.src-instagram { border-bottom: 2px solid #e1306c; }
.cluster-mini-badge.src-youtube { border-bottom: 2px solid #ff0000; }
.cluster-mini-placeholder { display: flex; width: 100%; height: 100%;
                             align-items: center; justify-content: center;
                             background: #ddd; color: #aaa; font-size: 18px; }
.muted.small { font-size: 10px; }

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
                      grid-template-columns: repeat(auto-fill, minmax(440px, 1fr));
                      gap: 12px; margin-top: 12px; }
.contrib-card { background: #fafafa; border-radius: 6px;
                padding: 10px; display: flex; flex-direction: column;
                gap: 8px; border-left: 3px solid #2a5db8; }
/* thumb area — 최대 5장 grid */
.contrib-card-thumb { display: grid;
                      grid-template-columns: repeat(5, 1fr);
                      gap: 4px; }
.contrib-card-img { width: 100%; aspect-ratio: 1; object-fit: cover;
                    border-radius: 4px; cursor: zoom-in;
                    transition: transform 0.15s; }
.contrib-card-img:hover { transform: scale(1.05); z-index: 5;
                          position: relative; }
.contrib-card-img:focus { position: fixed; top: 50%; left: 50%;
                          transform: translate(-50%,-50%); width: 90vw;
                          height: 90vh; aspect-ratio: auto; object-fit: contain;
                          z-index: 1000; background: rgba(0,0,0,0.9);
                          cursor: zoom-out; outline: none; }
.contrib-card-img-placeholder { grid-column: 1 / -1;
                                display: flex; height: 80px;
                                align-items: center; justify-content: center;
                                background: #ddd; color: #666; font-weight: bold;
                                border-radius: 4px; }
.contrib-card-body { display: flex; flex-direction: column; gap: 4px;
                     min-width: 0; }
.contrib-card-header { display: flex; gap: 6px; flex-wrap: wrap;
                       align-items: baseline; font-size: 11px; }
.contrib-share { font-weight: bold; color: #d35400;
                 background: rgba(211,84,0,0.1); padding: 1px 6px;
                 border-radius: 3px; }
.contrib-attrs { font-size: 11px; color: #555; line-height: 1.4; }
.contrib-item-summary { font-size: 11px; color: #888; }
.canon-line { padding: 2px 6px; border-radius: 3px; margin: 2px 0;
              border-left: 2px solid transparent; }
.canon-line.match { background: #eff4fc; border-left-color: #2a5db8;
                    color: #1a3a78; font-weight: 500; }
.canon-idx { font-family: ui-monospace, monospace; color: #999;
             font-size: 10px; margin-right: 4px; }
.canon-line.match .canon-idx { color: #2a5db8; }
.canon-match-badge { font-size: 9px; color: #2a5db8;
                     background: rgba(42,93,184,0.1); padding: 1px 4px;
                     border-radius: 3px; margin-left: 4px;
                     font-weight: bold; text-transform: uppercase; }
.contrib-card-palette { margin-top: 4px; }

/* 8단계 — contributor click → item full detail expand */
.contrib-card { cursor: pointer; transition: background 0.15s; }
.contrib-card:hover { background: #f0f4fc; }
.contrib-card-wrap { margin-bottom: 8px; }
.contrib-expand-hint { color: #2a5db8; font-size: 10px; margin-left: auto; opacity: 0.6; }
.contrib-card:hover .contrib-expand-hint { opacity: 1; }
.item-detail { background: #fffbe6; border-left: 3px solid #f5c518;
               border-radius: 0 6px 6px 0; padding: 10px 14px;
               margin: 6px 0 12px 24px; }
.item-detail-title { font-size: 12px; color: #c08400; margin: 0 0 8px;
                     text-transform: none; }
.item-detail-section { margin-bottom: 12px; }
.item-detail-section h5 { font-size: 11px; color: #555; margin: 6px 0 4px;
                          text-transform: uppercase; }
.item-detail-media-grid { display: flex; flex-wrap: wrap; gap: 4px; }
.item-detail-img { width: 120px; height: 120px; object-fit: cover; border-radius: 4px; }
.item-detail-meta { font-size: 11px; line-height: 1.6; }
.item-detail-meta b { color: #555; }
.item-detail-text { margin-top: 4px; font-style: italic; max-width: 100%;
                    overflow-wrap: anywhere; }
.item-detail-section-palette { margin-top: 8px; }
.item-detail-section-palette .palette-bar { height: 24px; }
.item-canon-block { background: #fff; border-radius: 4px; padding: 6px 10px;
                    margin: 4px 0; border-left: 2px solid transparent; }
.item-canon-block.match { border-left-color: #2a5db8; background: #eff4fc; }
.item-canon-header { font-size: 11px; margin-bottom: 4px; }
.item-canon-attrs { font-size: 11px; color: #555; line-height: 1.5; }
.item-canon-palette { margin-top: 4px; }
.item-canon-palette .palette-bar { height: 18px; }
.contrib-card-palette .palette-bar { height: 24px; }

/* post-link icon */
.post-link { color: #2a5db8; }

/* unknown signals panel (spec §4.2) */
.unknown-signals-panel { background: #fffbe6; border: 1px solid #ffd966;
                          border-radius: 6px; padding: 10px 14px;
                          margin-bottom: 12px; }
.unknown-signals-panel summary { font-size: 13px; font-weight: bold;
                                  color: #8b6f00; cursor: pointer; }
.unknown-signals-panel[open] summary { margin-bottom: 8px; }
.unknown-signals-panel p { margin: 6px 0; }
.unknown-signals-table { width: 100%; font-size: 12px; border-collapse: collapse; }
.unknown-signals-table th { text-align: left; padding: 4px 8px;
                            border-bottom: 1px solid #ffd966; background: #fff8d4; }
.unknown-signals-table td { padding: 3px 8px; border-bottom: 1px dotted #ffd966;
                            vertical-align: middle; }
.unknown-signals-table tr:hover { background: #fff3cd; }
"""


def _dedup_enriched_by_url(
    enriched: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """raw DB url 기준 dedup — 같은 url 가진 item 들 중 engagement_raw 최대값 1건만 유지.

    view 단 dedup 만 적용 (rep summaries 는 이미 산출됨, 별도 재계산 안 함).
    url 미존재 (raw DB lookup 실패) item 은 전부 keep — 가짜 dedup 방지.
    """
    from collections import defaultdict
    post_urls = _load_post_urls()
    by_url: dict[str, list[dict[str, Any]]] = defaultdict(list)
    no_url: list[dict[str, Any]] = []
    for it in enriched:
        pid = it["normalized"].get("source_post_id", "")
        url = post_urls.get(pid)
        if url:
            by_url[url].append(it)
        else:
            no_url.append(it)
    out: list[dict[str, Any]] = list(no_url)
    n_dups = 0
    for url, items in by_url.items():
        if len(items) > 1:
            n_dups += len(items) - 1
        items.sort(
            key=lambda it: -(it["normalized"].get("engagement_raw_count") or it["normalized"].get("engagement_raw") or 0)
        )
        out.append(items[0])
    print(
        f"  dedup_by_url: {len(enriched)} → {len(out)} "
        f"(dropped {n_dups} url-duplicates, {len(no_url)} kept w/o url lookup)"
    )
    return out


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
    # view-단 dedup: raw url 기준 동일 post 중 engagement 최대 1건만 keep.
    # rep summaries 는 dedup 적용 X (이미 산출, 영향 평가 후 별도 재계산 결정).
    print(f"[week {week_idx} {label}]")
    enriched = _dedup_enriched_by_url(enriched)
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


def _load_unknown_signals() -> list[dict[str, Any]]:
    """outputs/unknown_signals.json (spec §4.2 hashtag tracker 결과) 로드.
    파일 없거나 빈 dict 면 [] 반환.
    """
    path = _REPO / "outputs" / "unknown_signals.json"
    if not path.exists():
        return []
    try:
        with path.open() as f:
            state = json.load(f)
    except json.JSONDecodeError:
        return []
    if not isinstance(state, dict):
        return []
    rows: list[dict[str, Any]] = []
    for tag, entry in state.items():
        if not isinstance(entry, dict):
            continue
        buckets = entry.get("buckets") or {}
        if not isinstance(buckets, dict):
            continue
        count_3day = sum(buckets.values())
        if count_3day < 10:
            continue  # spec §4.2 threshold
        first_seen = min(buckets.keys()) if buckets else ""
        rows.append({
            "tag": f"#{tag}",
            "count_3day": count_3day,
            "first_seen": first_seen,
            "likely_category": entry.get("likely_category"),
            "reviewed": bool(entry.get("reviewed", False)),
        })
    rows.sort(key=lambda r: -r["count_3day"])
    return rows


def _render_unknown_signals_panel(signals: list[dict[str, Any]]) -> str:
    """spec §4.2 — 매핑 외 신규 해시태그 시그널 패널 (검수용)."""
    if not signals:
        return ""
    rows_html = "".join(
        f'<tr>'
        f'<td><code>{_esc(s["tag"])}</code></td>'
        f'<td class="num">{s["count_3day"]}</td>'
        f'<td>{_esc(s["first_seen"])}</td>'
        f'<td>{_esc(s.get("likely_category") or "—")}</td>'
        f'<td>{"✓" if s["reviewed"] else "—"}</td>'
        f'</tr>'
        for s in signals[:30]  # 상위 30 개
    )
    return f'''
<details class="unknown-signals-panel">
  <summary>🆕 신규 시그널 감지 — 매핑 외 해시태그 ({len(signals)}건, 3일 ≥10) ▼</summary>
  <p class="muted">spec §4.2 — 자동 감지된 해시태그. 매핑에 추가 또는 noise 무시 검토.</p>
  <table class="unknown-signals-table">
    <thead><tr><th>tag</th><th>count_3day</th><th>first_seen</th><th>likely_category</th><th>reviewed</th></tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
</details>'''


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
    unknown_signals_html = _render_unknown_signals_panel(_load_unknown_signals())
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

{unknown_signals_html}

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
function toggleItemDetail(detailId) {{
  // contributor 카드 click 시 그 item 의 full detail panel toggle (8단계, 2026-04-30)
  const el = document.getElementById(detailId);
  if (!el) return;
  el.style.display = el.style.display === 'none' ? '' : 'none';
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
