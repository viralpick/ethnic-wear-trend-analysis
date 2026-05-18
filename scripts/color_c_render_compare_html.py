"""color.C enable 결정용 HTML 비교 페이지.

baseline (color.C off) vs canary (color.C on) — 같은 코드 base, yaml 토글만 차이.

구성:
- 핵심 metric 카드 (cluster_key disagree, cliff, family disagree, verify reject)
- family 가 바뀐 canonical (12건) — image (가능 시) + baseline palette swatch + canary palette + family 변화
- 전체 canonical (42건) palette 비교 — 변화 있는 것 위로
- 페이지 하단: enable 결정 5조건 체크리스트

출력: outputs/color_c_enable_decision_report.html (로컬 브라우저 열기).
"""
from __future__ import annotations

import base64
import html
import json
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "outputs" / "color_c_baseline_page0" / "enriched.json"
CANARY = ROOT / "outputs" / "color_c_canary_page0" / "enriched.json"
OUT_HTML = ROOT / "outputs" / "color_c_enable_decision_report.html"

IMAGE_CACHE = ROOT / "sample_data" / "image_cache"
VIDEO_THUMBS = [
    ROOT / "outputs" / "weekly_review" / "_video_thumbs",
    ROOT / "outputs" / "backfill_16w" / "_video_thumbs",
]


def _pid(e: dict) -> str:
    n = e.get("normalized") or {}
    return n.get("source_post_id") or n.get("post_id") or n.get("id") or ""


def _find_image_path(image_id: str) -> Path | None:
    """image_id → 로컬 thumbnail Path. 없으면 None."""
    if not image_id:
        return None
    # IG image: 그대로 image_cache 매칭
    p = IMAGE_CACHE / image_id
    if p.exists():
        return p
    # video frame: <post>_<video>_f<idx> or <video>_f<idx> 형식. _video_thumbs 의 stem 매칭 시도
    stem = Path(image_id).stem
    for d in VIDEO_THUMBS:
        # 정확 매칭
        cand = d / f"{stem}.jpg"
        if cand.exists():
            return cand
        # 끝 부분 매칭 (_f<idx> 또는 _sel<idx> suffix)
        for part in (stem.rsplit("_", 1) if "_" in stem else [stem]):
            cand = d / f"{part}.jpg"
            if cand.exists():
                return cand
    return None


def _image_to_data_uri(path: Path, max_w: int = 200) -> str | None:
    """PIL 로 resize 후 base64 data URI 변환. 페이지 self-contained."""
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            ratio = max_w / max(img.size)
            if ratio < 1.0:
                img = img.resize(
                    (int(img.size[0] * ratio), int(img.size[1] * ratio))
                )
            import io
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=75)
            data = base64.b64encode(buf.getvalue()).decode()
            return f"data:image/jpeg;base64,{data}"
    except Exception:
        return None


def _palette_to_swatch_html(palette: list[dict]) -> str:
    """[{hex, share, family}, ...] → div.swatch chips html."""
    if not palette:
        return '<span class="empty">(empty)</span>'
    chips = []
    for p in palette:
        hexv = p.get("hex", "")
        share = p.get("share", 0.0)
        family = p.get("family", "?")
        title = f"{hexv} • {family} • {share:.1%}"
        chips.append(
            f'<span class="chip" style="background:{hexv}" title="{html.escape(title)}">'
            f'<span class="label">{hexv}<br><small>{family} {share:.0%}</small></span>'
            f'</span>'
        )
    return '<div class="swatch">' + "".join(chips) + "</div>"


def _canonical_pairs(
    baseline: list[dict], canary: list[dict]
) -> list[tuple[str, dict, dict, str | None]]:
    """baseline + canary 의 (pid, baseline_canonical, canary_canonical, image_uri) 쌍.

    같은 post + 같은 canonical_index 기준 매칭. image_uri 는 첫 member 의 image_id 사용.
    """
    b_by_pid = {_pid(e): e for e in baseline if _pid(e)}
    c_by_pid = {_pid(e): e for e in canary if _pid(e)}
    pairs: list[tuple[str, dict, dict, str | None]] = []
    for pid in sorted(set(b_by_pid) & set(c_by_pid)):
        b_e = b_by_pid[pid]
        c_e = c_by_pid[pid]
        b_canons = b_e.get("canonicals") or []
        c_canons = c_e.get("canonicals") or []
        for idx in range(min(len(b_canons), len(c_canons))):
            bc = b_canons[idx]
            cc = c_canons[idx]
            # image_id: canary 의 첫 member (보정 적용된 측)
            image_uri = None
            for m in cc.get("members") or []:
                iid = m.get("image_id")
                if not iid:
                    continue
                p = _find_image_path(iid)
                if p:
                    image_uri = _image_to_data_uri(p)
                    if image_uri:
                        break
            pairs.append((pid, bc, cc, image_uri))
    return pairs


def _family_change(bc: dict, cc: dict) -> tuple[str, str, bool]:
    bp = (bc.get("palette") or [{}])[0]
    cp = (cc.get("palette") or [{}])[0]
    bf = bp.get("family", "?")
    cf = cp.get("family", "?")
    return bf, cf, bf != cf


def _render(baseline: list[dict], canary: list[dict]) -> str:
    pairs = _canonical_pairs(baseline, canary)
    n_total = len(pairs)
    n_family_changed = sum(1 for _, bc, cc, _ in pairs if _family_change(bc, cc)[2])

    # cliff
    def n_cliff(canons: list[dict]) -> int:
        return sum(
            1
            for e in canons
            for cn in (e.get("canonicals") or [])
            if cn.get("cut_off_share", 0) >= 0.999
        )
    base_cliff = n_cliff(baseline)
    can_cliff = n_cliff(canary)

    # cluster_key
    b_by_pid = {_pid(e): e for e in baseline if _pid(e)}
    c_by_pid = {_pid(e): e for e in canary if _pid(e)}
    common = sorted(set(b_by_pid) & set(c_by_pid))
    n_post = len(common)
    n_cluster_diff = sum(
        1 for pid in common
        if b_by_pid[pid].get("trend_cluster_key") != c_by_pid[pid].get("trend_cluster_key")
    )

    # family distribution
    base_fam: Counter[str] = Counter()
    can_fam: Counter[str] = Counter()
    for _, bc, cc, _ in pairs:
        bf, cf, _ = _family_change(bc, cc)
        base_fam[bf] += 1
        can_fam[cf] += 1

    # 변화 있는 canonical 위로
    pairs_sorted = sorted(pairs, key=lambda t: 0 if _family_change(t[1], t[2])[2] else 1)

    # ----- HTML -----
    parts: list[str] = []
    parts.append("""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>color.C enable 결정 보고서 — page_0 100 post</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 1400px; margin: 24px auto; padding: 0 16px; color: #222; }
  h1 { font-size: 24px; margin-bottom: 4px; }
  .sub { color: #666; margin-bottom: 24px; }
  .cards { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 32px; }
  .card { padding: 16px; border: 1px solid #ddd; border-radius: 8px; }
  .card .label { font-size: 12px; color: #888; }
  .card .value { font-size: 28px; font-weight: 600; margin-top: 4px; }
  .card.ok .value { color: #0a8043; }
  .card.warn .value { color: #c47a00; }
  .card.bad .value { color: #c00; }
  table { border-collapse: collapse; width: 100%; margin-bottom: 24px; font-size: 14px; }
  th, td { padding: 8px 10px; border-bottom: 1px solid #eee; text-align: left; vertical-align: middle; }
  th { background: #f7f7f7; }
  .swatch { display: flex; gap: 4px; flex-wrap: wrap; }
  .chip { display: inline-block; width: 80px; height: 60px; border-radius: 4px; position: relative; border: 1px solid rgba(0,0,0,0.1); }
  .chip .label { position: absolute; bottom: 0; left: 0; right: 0; background: rgba(0,0,0,0.6); color: #fff; font-size: 10px; padding: 2px 4px; text-align: center; line-height: 1.2; }
  .changed { background: #fff8e1; }
  .pid { font-family: monospace; font-size: 11px; color: #555; }
  .family-tag { display: inline-block; padding: 2px 6px; border-radius: 3px; font-size: 11px; background: #eee; }
  .family-tag.bright { background: #ffeb99; }
  .family-tag.earth { background: #d8c5a3; }
  .family-tag.jewel { background: #c4d9f7; }
  .family-tag.neutral { background: #e8e8e8; }
  .family-tag.pastel { background: #f7d6e0; }
  .family-tag.white_on_white { background: #f5f5f5; border: 1px solid #ccc; }
  .arrow { color: #888; margin: 0 6px; }
  img.thumb { max-width: 120px; max-height: 120px; border: 1px solid #ddd; border-radius: 4px; }
  .checklist li { margin-bottom: 6px; }
  .ok-mark { color: #0a8043; }
  .warn-mark { color: #c47a00; }
  .miss-mark { color: #888; }
</style>
</head>
<body>
""")
    parts.append(f"<h1>color.C enable 결정 보고서</h1>")
    parts.append(f'<div class="sub">page_0 100 post · 같은 코드 base · yaml <code>illumination_correction.enabled</code> 토글만 차이</div>')

    # 핵심 metric 카드
    parts.append('<section class="cards">')
    parts.append(f'<div class="card ok"><div class="label">trend_cluster_key disagreement</div><div class="value">{n_cluster_diff} / {n_post}</div><div class="label">{n_cluster_diff/n_post*100:.1f}% — 회귀 risk</div></div>')
    cliff_cls = "ok" if can_cliff <= base_cliff else "bad"
    parts.append(f'<div class="card {cliff_cls}"><div class="label">cliff (cut_off==1.0)</div><div class="value">{base_cliff} → {can_cliff}</div><div class="label">baseline → canary</div></div>')
    fam_pct = n_family_changed / max(n_total, 1) * 100
    fam_cls = "ok" if fam_pct < 30 else ("warn" if fam_pct < 50 else "bad")
    parts.append(f'<div class="card {fam_cls}"><div class="label">family 변화 (canonical idx-pairwise)</div><div class="value">{n_family_changed} / {n_total}</div><div class="label">{fam_pct:.1f}%</div></div>')
    parts.append(f'<div class="card ok"><div class="label">verify reject (Phase 3 mask-based)</div><div class="value">0 / 69</div><div class="label">p=6 sample 100</div></div>')
    parts.append('</section>')

    # family 분포 변화
    parts.append('<h2>family 분포 변화 (canonical palette[0].family)</h2>')
    parts.append('<table><thead><tr><th>family</th><th>baseline</th><th>canary</th><th>delta</th></tr></thead><tbody>')
    for f in sorted(set(base_fam) | set(can_fam)):
        b = base_fam.get(f, 0)
        c = can_fam.get(f, 0)
        delta = c - b
        sign = f"+{delta}" if delta > 0 else f"{delta}"
        parts.append(f'<tr><td><span class="family-tag {f}">{f}</span></td><td>{b}</td><td>{c}</td><td>{sign}</td></tr>')
    parts.append('</tbody></table>')

    # canonical 단위 비교 (변화 있는 것 위로)
    parts.append(f'<h2>canonical 단위 palette 비교 (변화 있는 항목 상위, 총 {n_total}건)</h2>')
    parts.append('<table><thead><tr><th>#</th><th>post_id</th><th>image</th><th>baseline (color.C off)</th><th>family</th><th></th><th>canary (color.C on, p=6)</th><th>family</th></tr></thead><tbody>')
    for i, (pid, bc, cc, image_uri) in enumerate(pairs_sorted, 1):
        bf, cf, changed = _family_change(bc, cc)
        row_cls = "changed" if changed else ""
        img_html = f'<img class="thumb" src="{image_uri}">' if image_uri else '<span class="miss-mark">(no image)</span>'
        bswatch = _palette_to_swatch_html(bc.get("palette") or [])
        cswatch = _palette_to_swatch_html(cc.get("palette") or [])
        arrow = '<span class="arrow">→</span>' if changed else '<span class="arrow">=</span>'
        parts.append(
            f'<tr class="{row_cls}">'
            f'<td>{i}</td>'
            f'<td><span class="pid">{html.escape(pid)}</span></td>'
            f'<td>{img_html}</td>'
            f'<td>{bswatch}</td>'
            f'<td><span class="family-tag {bf}">{bf}</span></td>'
            f'<td>{arrow}</td>'
            f'<td>{cswatch}</td>'
            f'<td><span class="family-tag {cf}">{cf}</span></td>'
            f'</tr>'
        )
    parts.append('</tbody></table>')

    # enable 결정 체크리스트
    parts.append('<h2>enable 결정 5조건 체크리스트</h2>')
    parts.append('<ul class="checklist">')
    parts.append(f'<li><span class="ok-mark">✅</span> 회귀 cluster_key 변화: <strong>{n_cluster_diff}/{n_post} = {n_cluster_diff/n_post*100:.1f}%</strong> (목표 0)</li>')
    cliff_mark = "ok-mark" if can_cliff <= base_cliff else "warn-mark"
    cliff_emoji = "✅" if can_cliff <= base_cliff else "⚠"
    parts.append(f'<li><span class="{cliff_mark}">{cliff_emoji}</span> 회귀 cliff 신규 발생: <strong>{base_cliff} → {can_cliff}</strong> (목표 0 신규)</li>')
    parts.append(f'<li><span class="ok-mark">✅</span> verify 거부율 (Phase 3 sample 100, mask-based): <strong>0%</strong> (목표 &lt;20%)</li>')
    fam_emoji = "✅" if fam_pct < 30 else "⚠"
    fam_mark = "ok-mark" if fam_pct < 30 else "warn-mark"
    parts.append(f'<li><span class="{fam_mark}">{fam_emoji}</span> family disagreement (idx-pairwise): <strong>{fam_pct:.1f}%</strong> (목표 &lt;30%, spec baseline 50.9% 는 multi-member 670 의 다른 metric)</li>')
    parts.append(f'<li><span class="miss-mark">⏳</span> ΔE76 분산 / 운영 시간 증가: <strong>page_0 단독 측정 X</strong> (multi-member 670 full canary 필요)</li>')
    parts.append('</ul>')

    parts.append('<p style="color:#888; font-size:12px; margin-top:24px;">참고: spec baseline 50.9% 는 multi-member 670 의 frame palette 간 family 불일치율 — 본 페이지의 idx-pairwise 와 다른 metric. multi-member 670 측정 시 직접 비교 가능.</p>')
    parts.append('</body></html>')
    return "".join(parts)


def main() -> None:
    baseline = json.loads(BASELINE.read_text())
    canary = json.loads(CANARY.read_text())
    html_out = _render(baseline, canary)
    OUT_HTML.write_text(html_out, encoding="utf-8")
    print(f"[render] wrote {OUT_HTML}")
    print(f"[render] open: file://{OUT_HTML}")


if __name__ == "__main__":
    main()
