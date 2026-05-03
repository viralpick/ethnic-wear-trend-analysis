"""M3.H — 1-post live smoke (YT video frame phase).

특정 YT video_id (StarRocks `id`) 를 1건 pull 한 뒤
RawYouTubeVideo → normalize → PipelineBColorExtractor 로 흐름을 돌려
video frame 분석 (cv2 → JPEG 결정론 → Gemini live → canonical/post_palette) 을 검증.

목적:
1. download_urls → RawYouTubeVideo.video_urls 매핑이 e2e 까지 흐르는지
2. cv2 가 YT mp4 (보통 IG 보다 큼) 를 frame 으로 추출
3. canonical/post_palette 결과가 채워지는지 (text-only 환각 리스크 해소 검증)

비용: Gemini 2.5 Flash live 호출 N회. 1-post 한정. `--html` 시 frame 재디코딩.

사용:
    uv run python scripts/m3h_youtube_smoke.py --post-id 01KPZ41GH8THB0QZH2N2YMTMMW
    uv run python scripts/m3h_youtube_smoke.py --post-id 01KPZ41GH8THB0QZH2N2YMTMMW \\
        --html outputs/m3h_smoke/01KPZ41GH8THB0QZH2N2YMTMMW.html

전제조건:
    .env: STARROCKS_*, GEMINI_API_KEY, AZURE_STORAGE_CONNECTION_STRING, AZURE_STORAGE_CONTAINER
    Pritunl VPN ON (Azure Storage IP allow-list)
    uv sync --extra vision --extra blob
"""
from __future__ import annotations

import argparse
import base64
import html as html_lib
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

load_dotenv()

_FRAME_RE = re.compile(r"^(.+)_f(\d+)$")
_PRESET_PATH = ROOT / "outputs" / "color_preset" / "color_preset.json"


def _load_preset_hex_map() -> dict[str, str]:
    """preset name → hex 매핑 로드 (PM 전송 HTML 의 preset_picks 시각화용)."""
    if not _PRESET_PATH.exists():
        return {}
    data = json.loads(_PRESET_PATH.read_text(encoding="utf-8"))
    return {entry["name"]: entry["hex"] for entry in data if "name" in entry and "hex" in entry}


def _fetch_one_yt(post_id: str) -> dict:
    from loaders.starrocks_connect import connect_raw
    conn = connect_raw()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, url, channel, title, description, tags,
                   thumbnail_url, upload_date, view_count, like_count,
                   comment_count, comments, download_urls, created_at
            FROM india_ai_fashion_youtube_posting
            WHERE id = %s
            """,
            (post_id,),
        )
        row = cur.fetchone()
        if row is None:
            raise SystemExit(f"YT post_id={post_id} not found")
        return row
    finally:
        conn.close()


def _decode_frames(mp4_path: Path, frame_indices: set[int], max_dim: int = 480) -> dict[int, str]:
    """frame_idx 별로 mp4 에서 1 frame 재디코딩 → JPEG base64 data URI 반환.

    렌더용 thumbnail 이라 max_dim 으로 다운스케일.
    """
    import cv2  # noqa: PLC0415
    import io
    from PIL import Image

    out: dict[int, str] = {}
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"cv2 cannot open {mp4_path}")
    try:
        for idx in sorted(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, bgr = cap.read()
            if not ok or bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            w, h = img.size
            scale = max_dim / max(w, h)
            if scale < 1.0:
                img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=80)
            out[idx] = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    finally:
        cap.release()
    return out


def _swatch(hex_color: str, label: str = "", width: int = 80) -> str:
    text = html_lib.escape(label) if label else ""
    return (
        f'<div style="display:inline-block;text-align:center;margin:2px;">'
        f'<div style="background:{hex_color};width:{width}px;height:40px;border:1px solid #aaa;"></div>'
        f'<div style="font-size:11px;font-family:monospace;color:#555;">{hex_color}</div>'
        f'<div style="font-size:11px;color:#333;">{text}</div>'
        f'</div>'
    )


def _render_html(
    raw, item, res, frame_data: dict[int, str], stem: str, mp4_size_mb: float,
    preset_map: dict[str, str],
) -> str:
    """ColorExtractionResult + 재디코딩된 frame 으로 self-contained HTML 생성 (PM 전송용)."""
    total_frames = sum(len(can.members) for can in res.canonicals)
    multi_frame_canonicals = sum(1 for can in res.canonicals if len(can.members) > 1)

    parts: list[str] = []
    parts.append("<!DOCTYPE html><html lang='ko'><head><meta charset='utf-8'>")
    parts.append(f"<title>YouTube 영상 분석 데모 — {html_lib.escape(raw.title)}</title>")
    parts.append("""
<style>
body { font-family: -apple-system, BlinkMacSystemFont, 'Helvetica Neue', sans-serif; max-width: 1280px; margin: 20px auto; padding: 0 24px; color: #222; line-height: 1.5; }
h1 { font-size: 24px; margin: 0 0 4px 0; }
h2 { font-size: 18px; margin-top: 28px; border-bottom: 2px solid #eee; padding-bottom: 6px; }
h3 { margin: 0 0 8px 0; font-size: 15px; }
.meta { color: #666; font-size: 13px; margin-bottom: 8px; }
.lead { color: #444; font-size: 14px; margin: 8px 0 16px 0; }
.guide { background: #f4f8ff; border-left: 4px solid #4a90e2; padding: 12px 16px; margin: 16px 0; font-size: 13px; line-height: 1.7; }
.guide b { color: #2c5282; }
.stats { display: flex; gap: 14px; flex-wrap: wrap; margin: 12px 0 20px 0; }
.stat { background: #f9f9f9; border: 1px solid #e0e0e0; padding: 10px 14px; border-radius: 6px; min-width: 110px; }
.stat .num { font-size: 20px; font-weight: 600; color: #333; }
.stat .lbl { font-size: 12px; color: #666; }
.canonical { border: 1px solid #ddd; border-radius: 6px; margin-bottom: 14px; padding: 14px 16px; background: #fafafa; }
.frames { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0; }
.frame { border: 1px solid #ccc; padding: 2px; background: #fff; border-radius: 3px; }
.frame img { display: block; max-height: 180px; }
.frame .label { font-family: monospace; font-size: 10px; color: #777; padding: 2px 4px; text-align: center; }
.section { margin: 8px 0; }
.section .title { font-weight: 600; font-size: 12px; color: #555; text-transform: uppercase; letter-spacing: 0.5px; }
.swatches { margin-top: 6px; }
.kv { font-size: 13px; line-height: 1.7; color: #444; }
.kv b { color: #222; }
.post-palette { background: #fff8ec; border: 1px solid #f0d8a0; padding: 14px 18px; border-radius: 6px; margin: 12px 0 18px 0; }
.tag { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 600; }
.tag-ethnic { background: #d4edda; color: #155724; }
.tag-non { background: #f8d7da; color: #721c24; }
.tag-na { background: #e2e3e5; color: #383d41; }
.dedup-badge { display: inline-block; background: #fff3cd; color: #856404; font-size: 11px; padding: 2px 8px; border-radius: 10px; margin-left: 6px; }
footer { margin-top: 40px; padding-top: 16px; border-top: 1px solid #eee; color: #888; font-size: 11px; }
</style>
""")
    parts.append("</head><body>")

    parts.append("<h1>YouTube 영상 자동 분석 — 데모</h1>")
    parts.append(
        "<div class='lead'>1개 YouTube 영상에서 균등 sampling 으로 frame 을 추출하고, "
        "Vision LLM (Gemini 2.5 Flash) 로 인도 ethnic wear outfit 을 식별 → 같은 의상은 1건으로 병합 → "
        "픽셀 기반 색상 팔레트를 산출합니다.</div>"
    )

    parts.append(
        f"<div class='meta'>"
        f"<b>제목</b>: {html_lib.escape(raw.title)}<br>"
        f"<b>채널</b>: {html_lib.escape(raw.channel)} &nbsp;·&nbsp; "
        f"<b>video_id</b>: <code>{html_lib.escape(raw.video_id)}</code> &nbsp;·&nbsp; "
        f"<b>조회</b> {raw.view_count:,} · <b>좋아요</b> {raw.like_count:,} · <b>댓글</b> {raw.comment_count:,} &nbsp;·&nbsp; "
        f"<b>업로드</b> {raw.published_at} &nbsp;·&nbsp; "
        f"<b>mp4</b> {mp4_size_mb:.1f} MB"
        f"</div>"
    )

    parts.append("<div class='stats'>")
    parts.append(
        f"<div class='stat'><div class='num'>{len(res.canonicals)}</div>"
        f"<div class='lbl'>식별된 의상 (canonical)</div></div>"
    )
    parts.append(
        f"<div class='stat'><div class='num'>{total_frames}</div>"
        f"<div class='lbl'>분석 frame (BBOX)</div></div>"
    )
    parts.append(
        f"<div class='stat'><div class='num'>{multi_frame_canonicals}</div>"
        f"<div class='lbl'>다중 frame 병합 의상</div></div>"
    )
    parts.append(
        f"<div class='stat'><div class='num'>{len(res.post_palette)}</div>"
        f"<div class='lbl'>영상 대표 색상</div></div>"
    )
    parts.append("</div>")

    parts.append(
        "<div class='guide'>"
        "<b>읽는 법</b><br>"
        "• <b>영상 대표 색상 (Post palette)</b>: 이 영상 전체의 시각적 인상을 압축한 최대 3색.<br>"
        "• <b>의상 카드 (Canonical)</b>: 같은 의상이 여러 frame 에 잡히면 1개 카드로 병합 (haul 영상이라 옷 수만큼 카드).<br>"
        "&nbsp;&nbsp;– <b>Member frames</b>: 그 의상이 등장한 모든 frame thumbnail.<br>"
        "&nbsp;&nbsp;– <b>의상 색상 (Cluster palette)</b>: 해당 의상의 픽셀 기반 KMeans 분석 결과.<br>"
        "&nbsp;&nbsp;– <b>preset_picks</b>: Vision LLM 이 50-color 사전에서 직접 고른 의상 색.<br>"
        "• <b>ethnic / non-ethnic</b>: Vision LLM 이 인도 전통 의상인지 직접 판정한 결과."
        "</div>"
    )

    parts.append("<div class='post-palette'>")
    parts.append("<div class='section'><div class='title'>영상 대표 색상 (Post palette)</div></div>")
    parts.append("<div class='swatches'>")
    for c in res.post_palette:
        family_str = c.family.value if c.family else "—"
        parts.append(_swatch(c.hex, f"{family_str} · 비중 {c.share:.0%}"))
    parts.append("</div></div>")

    parts.append(f"<h2>의상별 분석 결과 ({len(res.canonicals)}개)</h2>")

    def _ethnic_tag(flag: bool | None) -> str:
        if flag is True:
            return "<span class='tag tag-ethnic'>ethnic</span>"
        if flag is False:
            return "<span class='tag tag-non'>non-ethnic</span>"
        return "<span class='tag tag-na'>?</span>"

    for i, can in enumerate(res.canonicals):
        rep = can.representative
        parts.append("<div class='canonical'>")
        merge_badge = (
            f"<span class='dedup-badge'>{len(can.members)} frame 병합</span>"
            if len(can.members) > 1
            else ""
        )
        parts.append(
            f"<h3>의상 #{i+1} &nbsp;"
            f"상의: <b>{html_lib.escape(rep.upper_garment_type or '—')}</b> {_ethnic_tag(rep.upper_is_ethnic)} &nbsp; "
            f"하의: <b>{html_lib.escape(rep.lower_garment_type or '—')}</b> {_ethnic_tag(rep.lower_is_ethnic)}"
            f"{merge_badge}</h3>"
        )

        sil = rep.silhouette.value if rep.silhouette else "—"
        kv = (
            f"<div class='kv'>"
            f"<b>원단(fabric)</b>: {html_lib.escape(rep.fabric or '—')} &nbsp;·&nbsp; "
            f"<b>기법(technique)</b>: {html_lib.escape(rep.technique or '—')} &nbsp;·&nbsp; "
            f"<b>실루엣(silhouette)</b>: {html_lib.escape(sil)} &nbsp;·&nbsp; "
            f"<b>frame 면적 비중</b>: {rep.person_bbox_area_ratio:.0%}"
            f"</div>"
        )
        parts.append(kv)

        if rep.color_preset_picks_top3:
            parts.append("<div class='section'><div class='title'>LLM 색상 픽 (preset_picks · LLM 이 50-color 사전에서 직접 선택)</div>")
            parts.append("<div class='swatches'>")
            for pick in rep.color_preset_picks_top3:
                hex_val = preset_map.get(pick, "#cccccc")
                parts.append(_swatch(hex_val, pick))
            parts.append("</div></div>")

        parts.append("<div class='section'><div class='title'>Member frames (그 의상이 잡힌 모든 컷)</div>")
        parts.append("<div class='frames'>")
        for m in can.members:
            mat = _FRAME_RE.match(m.image_id)
            frame_idx = int(mat.group(2)) if mat else None
            data_uri = frame_data.get(frame_idx) if frame_idx is not None else None
            if data_uri:
                parts.append(
                    f"<div class='frame'>"
                    f"<img src='{data_uri}' alt='{html_lib.escape(m.image_id)}'>"
                    f"<div class='label'>frame {frame_idx} · BBOX#{m.outfit_index}</div>"
                    f"</div>"
                )
            else:
                parts.append(
                    f"<div class='frame'><div class='label' style='padding:30px;'>"
                    f"{html_lib.escape(m.image_id)}<br>(no frame)</div></div>"
                )
        parts.append("</div></div>")

        parts.append("<div class='section'><div class='title'>의상 색상 (Cluster palette · 픽셀 기반)</div>")
        parts.append("<div class='swatches'>")
        for c in can.palette:
            family_str = c.family.value if c.family else "—"
            parts.append(_swatch(c.hex, f"{family_str} · 비중 {c.share:.0%}"))
        parts.append("</div></div>")

        parts.append("</div>")

    parts.append(
        "<footer>"
        "ethnic-wear-trend-analysis · M3.H YouTube video frame phase · "
        "Vision LLM: Gemini 2.5 Flash · 픽셀 색상: segformer + KMeans + ΔE76 merge · "
        "self-contained HTML (외부 리소스 없음 — 단독 전송 가능)"
        "</footer>"
    )
    parts.append("</body></html>")
    return "".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="M3.H 1-post YT video smoke")
    parser.add_argument("--post-id", required=True,
                        help="YT row id (e.g. 01KPZ41GH8THB0QZH2N2YMTMMW)")
    parser.add_argument("--blob-cache", default="sample_data/image_cache",
                        help="Azure Blob 다운로드 캐시 디렉토리")
    parser.add_argument("--html", type=Path, default=None,
                        help="HTML 출력 경로 (지정 시 frame 재디코딩 + 시각화)")
    args = parser.parse_args()

    from loaders.starrocks_raw_loader import _build_yt_video  # noqa: E402
    from normalization.normalize_content import normalize_youtube_video  # noqa: E402
    from settings import load_settings  # noqa: E402
    from pipelines.run_daily_pipeline import _select_color_extractor  # noqa: E402

    print(f"== Fetch YT post_id={args.post_id} ==")
    row = _fetch_one_yt(args.post_id)
    raw = _build_yt_video(row)
    if raw is None:
        raise SystemExit("raw YT video build failed")

    print(f"  channel={raw.channel}  title={raw.title[:80]!r}  published={raw.published_at}")
    print(f"  views={raw.view_count} likes={raw.like_count} comments={raw.comment_count}")
    print(f"  video_urls={len(raw.video_urls)}")
    for vurl in raw.video_urls:
        print(f"    video: {vurl[:100]}")

    item = normalize_youtube_video(raw)
    print(f"== Normalized item: image={len(item.image_urls)} video={len(item.video_urls)} ==")

    settings = load_settings()
    cache_dir = Path(args.blob_cache)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("== Loading PipelineBColorExtractor (vision extras) ==")
    extractor = _select_color_extractor(
        choice="pipeline_b",
        settings=settings,
        image_root=None,
        blob_cache=cache_dir,
        vision_llm_choice="gemini",
    )

    print(f"== extract_visual([item]) — Gemini live, video_urls={len(item.video_urls)} ==")
    results = extractor.extract_visual([item])
    if not results:
        raise SystemExit("extract_visual returned empty")

    res = results[0]
    print("== Result ==")
    print(f"  canonicals: {len(res.canonicals)}")
    for i, can in enumerate(res.canonicals):
        rep = can.representative
        member_ids = [m.image_id for m in can.members]
        print(f"    [{i}] members={len(can.members)} image_ids={member_ids[:5]}{'…' if len(member_ids) > 5 else ''}")
        print(f"        upper={rep.upper_garment_type}/{rep.upper_is_ethnic} "
              f"lower={rep.lower_garment_type}/{rep.lower_is_ethnic} "
              f"picks={rep.color_preset_picks_top3}")
        for j, c in enumerate(can.palette[:3]):
            print(f"        cluster[{j}] hex={c.hex} family={c.family} share={c.share:.3f}")
    print(f"  post_palette ({len(res.post_palette)}):")
    for j, c in enumerate(res.post_palette):
        print(f"    [{j}] hex={c.hex} family={c.family} share={c.share:.3f}")

    if args.html is None:
        return

    print(f"== Building HTML → {args.html} ==")
    stem = raw.video_id
    mp4_path = cache_dir / f"{stem}.mp4"
    if not mp4_path.exists():
        raise SystemExit(f"mp4 not in blob_cache: {mp4_path}")
    mp4_size_mb = mp4_path.stat().st_size / 1024 / 1024

    needed_indices: set[int] = set()
    for can in res.canonicals:
        for m in can.members:
            mat = _FRAME_RE.match(m.image_id)
            if mat:
                needed_indices.add(int(mat.group(2)))
    print(f"  decoding {len(needed_indices)} unique frames from {mp4_path.name} ({mp4_size_mb:.1f} MB)…")
    frame_data = _decode_frames(mp4_path, needed_indices)
    print(f"  decoded {len(frame_data)}/{len(needed_indices)} frames")

    preset_map = _load_preset_hex_map()
    if not preset_map:
        print(f"  WARN: preset map empty ({_PRESET_PATH}) — preset_picks 색상 칩 fallback")

    args.html.parent.mkdir(parents=True, exist_ok=True)
    html_str = _render_html(raw, item, res, frame_data, stem, mp4_size_mb, preset_map)
    args.html.write_text(html_str, encoding="utf-8")
    print(f"  HTML written: {args.html} ({len(html_str)/1024:.1f} KB)")


if __name__ == "__main__":
    main()
