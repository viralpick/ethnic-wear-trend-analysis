"""Pipeline B end-to-end smoke — sample_data/image 20 JPG 처리 + comparison HTML (Step E).

흐름:
  image_root 에서 JPG 목록 로드 → filename prefix (post ULID) 로 그룹화 → 각 post 를
  ImageFrameSource 로 묶어 Pipeline B extract_palette 호출 → 결과를 palette.json +
  comparison.html 로 저장.

전제: `uv sync --extra vision` (torch + transformers + ultralytics) 완료. 첫 실행 시
yolov8n.pt (~6MB) + segformer_b2_clothes (~200MB) 다운로드 (~2분). 이후 캐시됨.

실행:
  uv run python scripts/pipeline_b_smoke.py
  uv run python scripts/pipeline_b_smoke.py \
      --image-root sample_data/image \
      --output-dir outputs/pipeline_b_smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from settings import load_settings  # noqa: E402
from vision.frame_source import ImageFrameSource  # noqa: E402
from vision.pipeline_b_extractor import extract_palette, load_models  # noqa: E402


def group_by_post_ulid(image_dir: Path) -> dict[str, list[Path]]:
    """JPG 파일명 `{post_ULID}_{image_ULID}.jpg` → post_ULID 로 group."""
    grouped: dict[str, list[Path]] = defaultdict(list)
    for jpg in sorted(image_dir.glob("*.jpg")):
        parts = jpg.stem.split("_", 1)
        if len(parts) != 2:
            continue
        grouped[parts[0]].append(jpg)
    return dict(grouped)


def _relpath(target: Path, start: Path) -> str:
    """HTML 에서 쓸 상대경로 (from start to target). symlink 해석 안 함."""
    try:
        return str(target.relative_to(start))
    except ValueError:
        import os
        return os.path.relpath(target, start)


def _render_html(results: list[dict], output_path: Path) -> None:
    """post 당 썸네일 + palette chip 을 단순 table 로 렌더. Jinja2 의존 X."""
    rows: list[str] = []
    for r in results:
        # 첫 번째 이미지를 썸네일로, 나머지는 작은 miniatures.
        imgs_html = "".join(
            f'<img src="{_relpath(Path(p), output_path.parent)}" '
            f'style="width:120px;margin:2px;border:1px solid #ddd;">'
            for p in r["image_paths"]
        )
        chips_html = "".join(
            f'<div style="display:inline-block;margin:4px;text-align:center;font-size:11px">'
            f'<div style="background:{c["hex_display"]};width:48px;height:48px;'
            f'border:1px solid #888"></div>'
            f'<span>{c["hex_display"]}<br>{c["pct"]:.1%}</span></div>'
            for c in r["palette"]
        ) or '<span style="color:#888">(no palette — no garment pixels detected)</span>'
        rows.append(
            f"<tr><td style='vertical-align:top;padding:8px'><code>{r['post_ulid']}</code>"
            f"<br><small>{len(r['image_paths'])} images</small></td>"
            f"<td style='padding:8px'>{imgs_html}</td>"
            f"<td style='padding:8px'>{chips_html}</td></tr>"
        )
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Pipeline B Smoke</title>
<style>body{{font-family:system-ui;margin:20px}}
table{{border-collapse:collapse}} td,th{{border:1px solid #ccc}}</style></head>
<body><h1>Pipeline B Smoke — sample_data/image</h1>
<p>Rows: {len(results)} posts. HEX = cluster centroid (LAB KMeans), pct = pixel weight.</p>
<table><thead><tr><th>post ULID</th><th>carousel images</th><th>palette (top-k)</th></tr>
</thead><tbody>{''.join(rows)}</tbody></table></body></html>
"""
    output_path.write_text(html, encoding="utf-8")


def run_smoke(image_root: Path, output_dir: Path) -> dict:
    """실 진입점. 결과 dict (summary) 반환 + palette.json / comparison.html 쓰기."""
    settings = load_settings()
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped = group_by_post_ulid(image_root)
    total_imgs = sum(len(v) for v in grouped.values())
    print(f"[smoke] image_root={image_root} posts={len(grouped)} images={total_imgs}")

    print("[smoke] loading YOLO + segformer (first run ~2min for model download)")
    bundle = load_models()
    print(f"[smoke] device={bundle.device}")

    results: list[dict] = []
    for post_ulid, paths in grouped.items():
        source = ImageFrameSource(paths)
        palette = extract_palette(source, bundle, settings.vision)
        results.append({
            "post_ulid": post_ulid,
            "image_paths": [str(p) for p in paths],
            "palette": [item.model_dump(mode="json") for item in palette],
        })
        print(f"[smoke] {post_ulid}: {len(paths)} imgs → {len(palette)} chips")

    (output_dir / "palette.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )
    _render_html(results, output_dir / "comparison.html")
    return {
        "posts": len(results),
        "images": total_imgs,
        "output_dir": str(output_dir),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline B smoke on local JPG directory.")
    parser.add_argument(
        "--image-root", type=Path, default=_REPO_ROOT / "sample_data" / "image",
        help="JPG 디렉토리 (기본: sample_data/image)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=_REPO_ROOT / "outputs" / "pipeline_b_smoke",
        help="palette.json + comparison.html 쓸 디렉토리",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run_smoke(args.image_root, args.output_dir)
    print(f"[smoke] done {summary}")


if __name__ == "__main__":
    main()
