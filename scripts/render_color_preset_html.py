"""outputs/color_preset/color_preset.json → preview.html.

50색 swatch 를 origin 별 (data_pool / self_generated) 로 묶어 보여준다. hex + LAB 표시.

실행:
  uv run python scripts/render_color_preset_html.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))  # for `from scripts._html_utils import ...`
_JSON = _REPO / "outputs" / "color_preset" / "color_preset.json"
_HTML = _REPO / "outputs" / "color_preset" / "preview.html"

from scripts._html_utils import text_color_for_bg  # noqa: E402

# LAB 의 a/b 평면 chroma 가 이 값 미만이면 "neutral" 로 간주. 중/저채도 전통색 감안해 12 선택.
_NEUTRAL_CHROMA_CUTOFF = 12.0


def _card(entry: dict) -> str:
    hex_code = entry["hex"]
    text_color = text_color_for_bg(hex_code, dark="#111", light="#f5f5f5")
    lab = entry["lab"]
    return (
        f'<div class="card" style="background:{hex_code}; color:{text_color};">'
        f'  <div class="name">{entry["name"]}</div>'
        f'  <div class="hex">{hex_code}</div>'
        f'  <div class="lab">L {lab[0]:.1f}&nbsp;·&nbsp;a {lab[1]:.1f}&nbsp;·&nbsp;b {lab[2]:.1f}</div>'
        f'</div>'
    )


def _section(title: str, entries: list[dict]) -> str:
    cards = "\n".join(_card(e) for e in entries)
    return (
        f'<section>'
        f'  <h2>{title} <span class="count">({len(entries)})</span></h2>'
        f'  <div class="grid">{cards}</div>'
        f'</section>'
    )


def _lab_to_lch(lab: list[float]) -> tuple[float, float, float]:
    """LAB → (L, chroma, hue_deg). hue 는 0~360."""
    L, a, b = lab
    chroma = math.sqrt(a * a + b * b)
    hue = math.degrees(math.atan2(b, a))
    if hue < 0:
        hue += 360.0
    return L, chroma, hue


def _spectrum_sort(entries: list[dict]) -> list[dict]:
    """neutral 은 L 오름차순 (왼쪽), 유채색은 hue 오름차순 (무지개).

    같은 hue 구간에선 L 보조 정렬 (어두움→밝음). 전체 한 줄 띠로 늘어놓을 목적.
    """
    neutrals: list[tuple[float, dict]] = []
    chromatics: list[tuple[float, float, dict]] = []
    for e in entries:
        L, chroma, hue = _lab_to_lch(e["lab"])
        if chroma < _NEUTRAL_CHROMA_CUTOFF:
            neutrals.append((L, e))
        else:
            chromatics.append((hue, L, e))
    neutrals.sort(key=lambda t: t[0])
    chromatics.sort(key=lambda t: (t[0], t[1]))
    return [e for _, e in neutrals] + [e for _, _, e in chromatics]


def _spectrum_section(entries: list[dict]) -> str:
    """스펙트럼 섹션 — 한 행으로 이어붙인 thin swatch band + 카드 grid."""
    sorted_entries = _spectrum_sort(entries)
    band = "\n".join(
        f'<div class="band-cell" style="background:{e["hex"]};" title="{e["name"]} {e["hex"]}"></div>'
        for e in sorted_entries
    )
    cards = "\n".join(_card(e) for e in sorted_entries)
    return (
        f'<section>'
        f'  <h2>스펙트럼 정렬 <span class="count">(neutral→무지개, chroma<{_NEUTRAL_CHROMA_CUTOFF:.0f} neutral)</span></h2>'
        f'  <div class="band">{band}</div>'
        f'  <div class="grid">{cards}</div>'
        f'</section>'
    )


def main() -> int:
    if not _JSON.exists():
        print(f"[preview] {_JSON} 없음. build_color_preset 먼저 실행.")
        return 1
    entries = json.loads(_JSON.read_text(encoding="utf-8"))
    pool = [e for e in entries if e["origin"].startswith("data_pool")]
    self_gen = [e for e in entries if e["origin"] == "self_generated"]

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>50-color ethnic preset</title>
<style>
  body {{ margin: 24px; font-family: -apple-system, Segoe UI, sans-serif; background: #fafafa; color: #222; }}
  h1 {{ margin: 0 0 4px 0; font-size: 20px; }}
  .summary {{ color: #666; font-size: 13px; margin-bottom: 24px; }}
  section {{ margin-bottom: 32px; }}
  h2 {{ font-size: 15px; margin: 0 0 12px 0; }}
  .count {{ color: #999; font-weight: 400; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 10px; }}
  .band {{ display: flex; width: 100%; height: 56px; border-radius: 8px; overflow: hidden;
          box-shadow: 0 1px 3px rgba(0,0,0,0.08); margin-bottom: 14px; }}
  .band-cell {{ flex: 1 1 0; min-width: 0; }}
  .band-cell:hover {{ transform: scaleY(1.12); transition: transform 120ms; }}
  .card {{ border-radius: 8px; padding: 14px 12px; min-height: 92px;
          display: flex; flex-direction: column; justify-content: space-between;
          box-shadow: 0 1px 3px rgba(0,0,0,0.08); font-family: ui-monospace, SF Mono, monospace; }}
  .name {{ font-size: 12px; font-weight: 600; }}
  .hex {{ font-size: 13px; letter-spacing: 0.5px; }}
  .lab {{ font-size: 11px; opacity: 0.85; }}
</style>
</head>
<body>
  <h1>50-color ethnic preset</h1>
  <div class="summary">
    총 {len(entries)}색 · pool {len(pool)} ({pool[0]["origin"] if pool else "-"}) + self-generated {len(self_gen)}
    · 산출: <code>{_JSON.name}</code>
  </div>
  {_section("풀 유래 (data pool)", pool)}
  {_section("자체 보강 (self generated — 전통 ethnic)", self_gen)}
  {_spectrum_section(entries)}
</body>
</html>
"""
    _HTML.write_text(html, encoding="utf-8")
    print(f"[preview] wrote {_HTML}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
