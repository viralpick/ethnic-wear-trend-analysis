"""build_color_preset.py 의 첫 실행 로그를 파싱해 YAML/JSON 을 복구.

`_dump_outputs` 버그(KeyError: 'lab') 때문에 산출물이 저장 안 된 케이스 일회용 복구 스크립트.
로그의 "pool_NN #RRGGBB LAB=[L, a, b]" 라인을 읽어 35 entry 구성 → 자체 15 LAB 재계산 → dump.

실행:
  uv run python scripts/rescue_color_preset_from_log.py \
      --log /tmp/preset_run.log \
      --pool-size 4210
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from vision.color_space import hex_to_rgb, rgb_to_lab  # noqa: E402

# build_color_preset.py 의 보강 15 색과 정확히 동일해야 함.
_SELF_GENERATED_COLORS: list[dict[str, object]] = [
    {"name": "saffron",         "hex": "#FF9933"},
    {"name": "vermillion",      "hex": "#E34234"},
    {"name": "turmeric_yellow", "hex": "#D9B500"},
    {"name": "henna_green",     "hex": "#8B6E3F"},
    {"name": "peacock_blue",    "hex": "#1F6D9E"},
    {"name": "rani_pink",       "hex": "#D04081"},
    {"name": "deep_indigo",     "hex": "#1C2958"},
    {"name": "bottle_green",    "hex": "#0A5F38"},
    {"name": "maroon_red",      "hex": "#80030B"},
    {"name": "mustard_olive",   "hex": "#A27D28"},
    {"name": "blush_peach",     "hex": "#F2C9B4"},
    {"name": "lavender_mauve",  "hex": "#AE94C2"},
    {"name": "mint_green",      "hex": "#98D4BB"},
    {"name": "charcoal_grey",   "hex": "#404347"},
    {"name": "cream_ivory",     "hex": "#F3E5C3"},
]

_LINE_RE = re.compile(
    r"(pool_\d{2})\s+(#[0-9A-Fa-f]{6})\s+LAB=\[([-\d.,\s]+)\]"
)


def _parse_pool_entries(log_text: str, pool_size: int) -> list[dict]:
    entries: list[dict] = []
    seen: set[str] = set()
    for match in _LINE_RE.finditer(log_text):
        name, hex_code, lab_str = match.group(1), match.group(2).upper(), match.group(3)
        if name in seen:
            continue
        seen.add(name)
        lab = [round(float(v.strip()), 2) for v in lab_str.split(",")]
        entries.append({
            "name": name,
            "hex": hex_code,
            "lab": lab,
            "origin": f"data_pool_n={pool_size}",
        })
    entries.sort(key=lambda e: e["name"])
    return entries


def _make_supplemental() -> list[dict]:
    out: list[dict] = []
    for entry in _SELF_GENERATED_COLORS:
        lab = rgb_to_lab(hex_to_rgb(entry["hex"]))
        out.append({
            "name": entry["name"],
            "hex": entry["hex"],
            "lab": [round(float(v), 2) for v in lab.tolist()],
            "origin": "self_generated",
        })
    return out


def _dump(out_dir: Path, all_entries: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "color_preset.json"
    json_path.write_text(
        json.dumps(all_entries, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    yaml_path = out_dir / "color_preset.yaml"
    with yaml_path.open("w", encoding="utf-8") as fh:
        fh.write("# scripts/build_color_preset.py 산출물 (rescue from log)\n")
        fh.write("color_preset:\n")
        for entry in all_entries:
            fh.write(f"  - name: {entry['name']}\n")
            fh.write(f"    hex: '{entry['hex']}'\n")
            fh.write(f"    lab: {entry['lab']}\n")
            fh.write(f"    origin: {entry['origin']}\n")
    print(f"[rescue] wrote {json_path} + {yaml_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True, help="build_color_preset 로그 경로")
    parser.add_argument("--pool-size", type=int, required=True,
                        help="원 로그의 '총 centroid' 값 (origin 표기용)")
    parser.add_argument("--out-dir", type=Path,
                        default=_REPO / "outputs" / "color_preset")
    args = parser.parse_args()

    log_text = args.log.read_text(encoding="utf-8")
    pool_entries = _parse_pool_entries(log_text, args.pool_size)
    if len(pool_entries) != 35:
        print(f"[rescue] WARN pool_entries={len(pool_entries)} (expected 35)")
    supplemental = _make_supplemental()
    all_entries = pool_entries + supplemental
    print(f"[rescue] pool={len(pool_entries)} + supplemental={len(supplemental)} = {len(all_entries)}")
    _dump(args.out_dir, all_entries)
    return 0


if __name__ == "__main__":
    sys.exit(main())
