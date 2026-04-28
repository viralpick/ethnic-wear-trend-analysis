"""F-5b: 단계별 β-hybrid 색상 파이프라인 시각화.

각 post 의 **canonical 별**로 4 단계 진화를 한 페이지에 펼친다:

- Step 0: 초기 KMeans (kmeans_clusters_pre_hybrid) + Gemini picks_input
- Step 1: picks_after_R3 (KMeans 검증 통과한 pick) — 변화 diff
- Step 2: weighted_clusters (R1 anchor merge / R2 solo keep, 가중치 적용)
- Step 3: 최종 canonical palette (family + share)

dump 소스: `outputs/2026-04-26_ethnic10/{post_id}.json` 의 `v3_hybrid.canonicals[]`.
post 당 image 여러 장이 union 되어 canonical pool 로 들어가고, KMeans 도 union pool
기준 — 즉 "이미지 1개씩 별도 분석" 이 아니라 post 단위 통합. canonical 마다 별도 파이프.

용도: 단계별 변화를 user 가 직접 확인 (Gemini pick 정합성, KMeans merge 적정성, 최종
palette 의 픽셀 증거 기반 여부).
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ETHNIC_DIR = REPO / "outputs/2026-04-26_ethnic10"
BASELINE_DIR = REPO / "outputs/phase5_stepF_baseline"
PRESET_PATH = REPO / "outputs/color_preset/color_preset.json"
OUT_HTML = ETHNIC_DIR / "stepwise.html"

CSS = """
body { font-family: -apple-system, system-ui, sans-serif; background: #f5f5f5; padding: 16px; margin: 0; }
h1 { margin: 0 0 4px 0; font-size: 18px; }
.totals { color: #666; margin: 0 0 16px 0; font-size: 13px; }
.legend { background: white; padding: 10px 14px; border-radius: 6px; margin-bottom: 14px; font-size: 12px; color: #444; line-height: 1.6; }
.post { background: white; border-radius: 8px; padding: 14px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
.post-head { display: flex; align-items: baseline; gap: 12px; margin-bottom: 8px; flex-wrap: wrap; }
.pid { font-family: monospace; font-size: 12px; color: #444; }
.tag { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 600; }
.tag.ok { background: #d1fae5; color: #065f46; }
.tag.fail { background: #fecaca; color: #7f1d1d; }
.tag.a { background: #fed7aa; color: #7c2d12; }
.tag.b { background: #fde68a; color: #78350f; }
.tag.src { background: #e5e7eb; color: #374151; }
.canonical { border-top: 1px solid #e5e7eb; padding-top: 10px; margin-top: 10px; }
.canonical:first-of-type { border-top: none; padding-top: 0; }
.can-head { font-size: 13px; color: #1f2937; margin-bottom: 8px; }
.can-head b { color: #111; }
.layout { display: grid; grid-template-columns: 280px 1fr; gap: 14px; }
.visual { width: 100%; background: #fafafa; border: 1px solid #e5e7eb; border-radius: 4px; }
.steps { display: flex; flex-direction: column; gap: 8px; }
.step { background: #f9fafb; border-radius: 6px; padding: 8px 10px; }
.step .label { font-size: 11px; color: #6b7280; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px; font-weight: 600; }
.step .label .stage { color: #111; }
.chips { display: flex; flex-wrap: wrap; gap: 6px; }
.chip { display: flex; flex-direction: column; align-items: center; min-width: 64px; }
.chip .swatch { width: 56px; height: 28px; border: 1px solid #d1d5db; border-radius: 3px; }
.chip .meta { font-size: 9px; color: #555; margin-top: 2px; text-align: center; line-height: 1.25; max-width: 70px; word-break: break-word; }
.chip .meta b { color: #111; }
.empty { color: #9ca3af; font-size: 11px; font-style: italic; }
.diff { color: #b91c1c; font-size: 11px; margin-left: 6px; }
.kept { color: #047857; font-size: 11px; margin-left: 6px; }
.cut { color: #6b7280; font-size: 10px; margin-left: 6px; }
.objs { display: flex; flex-direction: column; gap: 6px; }
.obj { background: white; border: 1px solid #e5e7eb; border-radius: 4px; padding: 6px 8px; }
.obj-head { font-size: 10px; color: #6b7280; margin-bottom: 4px; font-family: monospace; }
.row-pair { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
"""


def load_preset_hex() -> dict[str, str]:
    raw = json.loads(PRESET_PATH.read_text(encoding="utf-8"))
    return {entry["name"]: entry["hex"] for entry in raw}


def chip_html(hex_v: str, label: str, sublabel: str = "") -> str:
    sub = f"<br>{sublabel}" if sublabel else ""
    return (
        f'<div class="chip">'
        f'<div class="swatch" style="background: {hex_v}"></div>'
        f'<div class="meta">{label}{sub}</div>'
        f'</div>'
    )


def picks_chips(picks: list[str], preset_hex: dict[str, str]) -> str:
    if not picks:
        return '<span class="empty">— no picks —</span>'
    html = ""
    for name in picks:
        hx = preset_hex.get(name, "#888888")
        html += chip_html(hx, f"<b>{name}</b>", hx)
    return html


def kmeans_chips(clusters: list[dict]) -> str:
    if not clusters:
        return '<span class="empty">— pool too small —</span>'
    html = ""
    for c in clusters:
        share = c.get("share", 0.0)
        html += chip_html(c["hex"], f"<b>{c['hex']}</b>", f"share {share:.2f}")
    return html


def weighted_chips(clusters: list[dict]) -> str:
    if not clusters:
        return '<span class="empty">— no clusters —</span>'
    html = ""
    for c in clusters:
        w = c.get("weight", 0.0)
        html += chip_html(c["hex"], f"<b>{c['hex']}</b>", f"w {w:.1f}")
    return html


def palette_chips(palette: list[dict]) -> str:
    if not palette:
        return '<span class="empty">— empty palette —</span>'
    html = ""
    for c in palette:
        html += chip_html(
            c["hex"],
            f"<b>{c['hex']}</b>",
            f"{c.get('family','?')} {c.get('share',0.0):.2f}",
        )
    return html


def picks_diff(picks_in: list[str], picks_r3: list[str]) -> str:
    if picks_in == picks_r3:
        return '<span class="kept">(R3: 모두 통과)</span>'
    dropped = [p for p in picks_in if p not in picks_r3]
    if dropped:
        return f'<span class="diff">(R3 drop: {", ".join(dropped)})</span>'
    return ""


def object_html(obj: dict, preset_hex: dict[str, str]) -> str:
    img = obj.get("image_id", "?")
    pix = obj.get("obj_pixel_count", 0)
    cov = obj.get("frame_coverage_ratio", 0.0)
    lab_a = obj.get("lab_a_gt20_count", 0)
    lab_a_ratio = (lab_a / pix) if pix else 0.0

    picks_in = obj.get("picks_input", []) or []
    picks_r3 = obj.get("picks_after_R3", []) or []
    kmeans = obj.get("kmeans_clusters_pre_hybrid", []) or []
    weighted = obj.get("weighted_clusters", []) or []

    return (
        f'<div class="obj">'
        f'<div class="obj-head">obj img={img} | pix={pix:,} cov={cov:.2f} '
        f'lab_a&gt;20 {lab_a:,} ({lab_a_ratio*100:.2f}%)</div>'
        f'<div class="row-pair">'
        f'<div class="step">'
        f'<div class="label"><span class="stage">Step 0a</span> · KMeans 초기 cluster (pre-hybrid)</div>'
        f'<div class="chips">{kmeans_chips(kmeans)}</div>'
        f'</div>'
        f'<div class="step">'
        f'<div class="label"><span class="stage">Step 0b</span> · Gemini picks_input (preset name)</div>'
        f'<div class="chips">{picks_chips(picks_in, preset_hex)}</div>'
        f'</div>'
        f'</div>'
        f'<div class="step">'
        f'<div class="label"><span class="stage">Step 1</span> · picks_after_R3 (KMeans 픽셀 증거 검증){picks_diff(picks_in, picks_r3)}</div>'
        f'<div class="chips">{picks_chips(picks_r3, preset_hex)}</div>'
        f'</div>'
        f'<div class="step">'
        f'<div class="label"><span class="stage">Step 2</span> · weighted_clusters (R1 anchor merge / R2 solo keep, frame_area × within-share)</div>'
        f'<div class="chips">{weighted_chips(weighted)}</div>'
        f'</div>'
        f'</div>'
    )


def canonical_html(canonical: dict, preset_hex: dict[str, str]) -> str:
    idx = canonical.get("canonical_index", 0)
    members = canonical.get("members_count", 0)
    objs = canonical.get("objects", []) or []
    palette = canonical.get("palette", []) or []
    cut_off = canonical.get("cut_off_share", 0.0)
    upper = canonical.get("upper_is_ethnic")
    lower = canonical.get("lower_is_ethnic")

    obj_html = "".join(object_html(o, preset_hex) for o in objs) if objs else \
        '<span class="empty">— no objects —</span>'

    return (
        f'<div class="canonical">'
        f'<div class="can-head">'
        f'<b>canonical[{idx}]</b> · members={members} objects={len(objs)} '
        f'· upper_is_ethnic={upper} lower_is_ethnic={lower}'
        f'</div>'
        f'<div class="objs">{obj_html}</div>'
        f'<div class="step" style="margin-top: 6px; background: #ecfdf5; border: 1px solid #a7f3d0;">'
        f'<div class="label"><span class="stage" style="color:#065f46;">Step 3</span> '
        f'· 최종 canonical palette (family + share, top-3) '
        f'<span class="cut">cut_off={cut_off:.2f}</span></div>'
        f'<div class="chips">{palette_chips(palette)}</div>'
        f'</div>'
        f'</div>'
    )


def post_card_html(
    post_id: str,
    info: dict,
    source: str,
    visual_rel: str | None,
    dump: dict,
    preset_hex: dict[str, str],
) -> str:
    verdict = info.get("verdict", "?")
    issue_a = info.get("issue_a", False)
    issue_b = info.get("issue_b", False)
    issue_c = info.get("issue_c", False)
    n_images = len(dump.get("images", []))
    canonicals = dump.get("v3_hybrid", {}).get("canonicals", []) or []

    if verdict == "ok" and not (issue_a or issue_b or issue_c):
        verdict_tag = '<span class="tag ok">OK</span>'
    elif verdict == "ok":
        verdict_tag = '<span class="tag ok">OK</span>'
    else:
        verdict_tag = f'<span class="tag fail">{verdict}</span>'

    issue_tags = ""
    if issue_a:
        issue_tags += '<span class="tag a">A: red shrinkage</span>'
    if issue_b:
        issue_tags += '<span class="tag b">B: NEUTRAL cascade</span>'
    if issue_c:
        issue_tags += '<span class="tag fail">C: empty pool</span>'

    if visual_rel:
        visual_html = (
            f'<img class="visual" src="{visual_rel}" alt="{post_id}">'
        )
    else:
        visual_html = '<div class="visual" style="height: 200px; display:flex; align-items:center; justify-content:center; color:#9ca3af; font-size:12px;">no visual</div>'

    can_html = "".join(canonical_html(c, preset_hex) for c in canonicals) if canonicals else \
        '<span class="empty">— no canonicals (ethnic_not_confirmed?) —</span>'

    return (
        f'<div class="post">'
        f'<div class="post-head">'
        f'<span class="pid">{post_id}</span>'
        f'{verdict_tag}<span class="tag src">{source}</span>{issue_tags}'
        f'<span style="font-size:11px;color:#6b7280;">images={n_images} canonicals={len(canonicals)}</span>'
        f'</div>'
        f'<div class="layout">'
        f'<div>{visual_html}</div>'
        f'<div class="steps">{can_html}</div>'
        f'</div>'
        f'</div>'
    )


def main() -> None:
    preset_hex = load_preset_hex()
    summary = json.loads((ETHNIC_DIR / "summary.json").read_text())
    posts = summary["posts"]
    baseline = summary.get("baseline", {})
    totals = summary["totals"]

    cards: list[str] = []

    for pid, info in posts.items():
        dump_path = ETHNIC_DIR / f"{pid}.json"
        if not dump_path.exists():
            continue
        dump = json.loads(dump_path.read_text())
        v_path = ETHNIC_DIR / f"{pid}_visual.png"
        v_rel = f"{pid}_visual.png" if v_path.exists() else None
        cards.append(post_card_html(pid, info, "ethnic10", v_rel, dump, preset_hex))

    for pid, info in baseline.items():
        dump_path = BASELINE_DIR / f"{pid}.json"
        if not dump_path.exists():
            continue
        dump = json.loads(dump_path.read_text())
        v_path = BASELINE_DIR / f"{pid}_visual.png"
        v_rel = f"../phase5_stepF_baseline/{pid}_visual.png" if v_path.exists() else None
        cards.append(post_card_html(pid, info, "baseline", v_rel, dump, preset_hex))

    totals_html = (
        f'totals: ok_with_palette={totals["ok_with_palette"]}/{totals["candidates"]} '
        f'A={totals["issue_a_count"]} B={totals["issue_b_count"]} C={totals["issue_c_count"]} '
        f'ethnic_not_confirmed={totals["ethnic_not_confirmed"]} + baseline 2'
    )

    legend = (
        '<div class="legend">'
        '<b>단계 가이드</b><br>'
        '<b>Step 0a</b>: canonical pool (post 의 모든 image union, segformer + skin drop 후 픽셀) '
        '에 KMeans(k=5) 적용해 얻은 raw cluster. share 는 pool 내 비율.<br>'
        '<b>Step 0b</b>: Gemini 가 50-color preset 중 1~3 picks. preset name → hex 색상 칩.<br>'
        '<b>Step 1 (R3)</b>: 각 pick 의 LAB 좌표를 KMeans cluster 중심들과 ΔE76 비교. '
        '가장 가까운 cluster 와 ΔE76 ≤ 임계값(R3_DROP_DELTAE76) 이면 keep, 초과면 drop. '
        '"픽셀 증거 없는 pick 차단".<br>'
        '<b>Step 2 (R1/R2)</b>: 통과한 pick 을 anchor 로 KMeans cluster 와 merge. '
        'R1 (≤cluster ΔE76 임계) 은 cluster 에 흡수, R2 는 단독 유지. '
        'weight = frame_area × within-share (image 단위 가중치 비교 가능하게 normalize).<br>'
        '<b>Step 3</b>: 같은 canonical 의 모든 object 의 weighted cluster 를 모아 '
        'aggregate_canonical_palette 통합 weighted KMeans. ΔE76 merge 후 cut_off_share 미만 drop, top-3.'
        '</div>'
    )

    html = (
        f'<!doctype html><html><head><meta charset="utf-8">'
        f'<title>2026-04-26 ethnic stepwise montage</title>'
        f'<style>{CSS}</style></head><body>'
        f'<h1>2026-04-26 ethnic — β-hybrid 단계별 색상 파이프라인</h1>'
        f'<p class="totals">{totals_html}</p>'
        f'{legend}'
        f'{"".join(cards)}'
        f'</body></html>'
    )

    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"wrote {OUT_HTML} ({len(cards)} post cards)")


if __name__ == "__main__":
    main()
