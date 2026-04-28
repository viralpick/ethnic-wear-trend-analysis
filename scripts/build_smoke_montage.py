"""F-5: 12-post smoke 결과 한 페이지 montage HTML 생성.

`outputs/2026-04-26_ethnic10/summary.json` + 각 post 의 `_visual.png` + baseline 2 post
(`outputs/phase5_stepF_baseline/`) 를 single HTML grid 로 묶는다.

각 카드:
- post_id + 출처 (ethnic10 / baseline)
- verdict 배지 (ok / ethnic_not_confirmed / issue_a / issue_b)
- visual.png 임베드 (없으면 placeholder)
- palette swatches (hex + family + share)
- gemini picks

용도: (A)/(B) fix 전후 비교 자료.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ETHNIC_DIR = REPO / "outputs/2026-04-26_ethnic10"
BASELINE_DIR = REPO / "outputs/phase5_stepF_baseline"
OUT_HTML = ETHNIC_DIR / "montage.html"

CSS = """
body { font-family: -apple-system, system-ui, sans-serif; background: #f5f5f5; padding: 16px; margin: 0; }
h1 { margin: 0 0 8px 0; font-size: 18px; }
.totals { color: #666; margin: 0 0 20px 0; font-size: 13px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(380px, 1fr)); gap: 16px; }
.card { background: white; border-radius: 8px; padding: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
.card.issue { border-left: 4px solid #f59e0b; }
.card.fail { border-left: 4px solid #dc2626; }
.card.ok { border-left: 4px solid #10b981; }
.pid { font-family: monospace; font-size: 11px; color: #555; word-break: break-all; }
.tags { margin: 6px 0; }
.tag { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 11px; margin-right: 4px; font-weight: 600; }
.tag.ok { background: #d1fae5; color: #065f46; }
.tag.fail { background: #fecaca; color: #7f1d1d; }
.tag.a { background: #fed7aa; color: #7c2d12; }
.tag.b { background: #fde68a; color: #78350f; }
.tag.src { background: #e5e7eb; color: #374151; }
.visual { width: 100%; max-height: 240px; object-fit: contain; background: #fafafa; border: 1px solid #e5e7eb; }
.placeholder { width: 100%; height: 120px; background: #f3f4f6; display: flex; align-items: center; justify-content: center; color: #9ca3af; font-size: 12px; }
.palette { display: flex; gap: 4px; margin: 8px 0; flex-wrap: wrap; }
.swatch { display: flex; flex-direction: column; align-items: center; min-width: 60px; }
.swatch .chip { width: 50px; height: 30px; border: 1px solid #d1d5db; }
.swatch .meta { font-size: 9px; color: #666; margin-top: 2px; text-align: center; line-height: 1.2; }
.picks { font-size: 11px; color: #555; margin-top: 6px; word-break: break-word; }
.section-title { font-size: 12px; color: #6b7280; text-transform: uppercase; letter-spacing: 0.05em; margin: 8px 0 4px 0; }
"""


def card_html(post_id: str, info: dict, source: str, visual_rel: str | None) -> str:
    verdict = info.get("verdict", "?")
    issue_a = info.get("issue_a", False)
    issue_b = info.get("issue_b", False)
    issue_c = info.get("issue_c", False)
    palette = info.get("palette") or []
    picks = info.get("all_picks") or []

    if verdict == "ok" and not (issue_a or issue_b or issue_c):
        cls = "ok"
        verdict_tag = '<span class="tag ok">OK</span>'
    elif verdict == "ok":
        cls = "issue"
        verdict_tag = '<span class="tag ok">OK</span>'
    else:
        cls = "fail"
        verdict_tag = f'<span class="tag fail">{verdict}</span>'

    issue_tags = ""
    if issue_a:
        issue_tags += '<span class="tag a">A: red shrinkage</span>'
    if issue_b:
        issue_tags += '<span class="tag b">B: NEUTRAL cascade</span>'
    if issue_c:
        issue_tags += '<span class="tag fail">C: empty pool</span>'

    if visual_rel:
        visual_html = f'<img class="visual" src="{visual_rel}" alt="{post_id}">'
    else:
        visual_html = '<div class="placeholder">no visual (ethnic_not_confirmed)</div>'

    swatch_html = ""
    for c in palette:
        hex_v = c.get("hex", "#000")
        fam = c.get("family", "?")
        share = c.get("share", 0.0)
        swatch_html += (
            f'<div class="swatch">'
            f'<div class="chip" style="background: {hex_v}"></div>'
            f'<div class="meta">{hex_v}<br>{fam} {share:.2f}</div>'
            f'</div>'
        )
    if not swatch_html:
        swatch_html = '<div style="color:#9ca3af; font-size:11px;">— no palette —</div>'

    picks_html = ", ".join(picks) if picks else "—"

    return (
        f'<div class="card {cls}">'
        f'<div class="pid">{post_id}</div>'
        f'<div class="tags">{verdict_tag}<span class="tag src">{source}</span>{issue_tags}</div>'
        f'{visual_html}'
        f'<div class="section-title">Canonical Palette</div>'
        f'<div class="palette">{swatch_html}</div>'
        f'<div class="section-title">Gemini Picks (R3 전)</div>'
        f'<div class="picks">{picks_html}</div>'
        f'</div>'
    )


def main() -> None:
    summary = json.loads((ETHNIC_DIR / "summary.json").read_text())
    posts = summary["posts"]
    baseline = summary.get("baseline", {})
    totals = summary["totals"]

    cards: list[str] = []

    # ethnic10 11 posts
    for pid, info in posts.items():
        v_path = ETHNIC_DIR / f"{pid}_visual.png"
        v_rel = f"{pid}_visual.png" if v_path.exists() else None
        cards.append(card_html(pid, info, "ethnic10", v_rel))

    # baseline 2 posts (visual 은 baseline 폴더에 있어 상대경로)
    for pid, info in baseline.items():
        v_path = BASELINE_DIR / f"{pid}_visual.png"
        v_rel = f"../phase5_stepF_baseline/{pid}_visual.png" if v_path.exists() else None
        cards.append(card_html(pid, info, "baseline", v_rel))

    totals_html = (
        f'totals: ok_with_palette={totals["ok_with_palette"]}/{totals["candidates"]} '
        f'A={totals["issue_a_count"]} B={totals["issue_b_count"]} C={totals["issue_c_count"]} '
        f'ethnic_not_confirmed={totals["ethnic_not_confirmed"]} '
        f'+ baseline 2'
    )

    html = (
        f'<!doctype html><html><head><meta charset="utf-8">'
        f'<title>2026-04-26 ethnic smoke montage</title>'
        f'<style>{CSS}</style></head><body>'
        f'<h1>2026-04-26 ethnic smoke montage (F-5)</h1>'
        f'<p class="totals">{totals_html}</p>'
        f'<div class="grid">{"".join(cards)}</div>'
        f'</body></html>'
    )

    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"wrote {OUT_HTML} ({len(cards)} cards)")


if __name__ == "__main__":
    main()
