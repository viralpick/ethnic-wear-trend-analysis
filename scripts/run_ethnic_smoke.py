"""F-2: 10-post ethnic smoke wrapper — β-hybrid (A)/(B) 일반화 검증.

흐름:
1. StarRocks 에서 ethnic regex 매칭 (saree|sari|kurta|...) post 18 row LIMIT 추출
   (Sridevi+Fabindia 제외, 이미지 있고 posting_at not null).
2. enriched-like JSON 작성 (`outputs/2026-04-26_ethnic10/enriched.json`) — diag 가 읽음.
3. 각 post image_urls 를 Azure Blob 에서 캐시 (이미 있으면 skip).
4. post 별로 `diag_canonical_pool.py` subprocess 호출 — model 14× load 페널티 ~2.5min 이지만
   diag 코드 0 변경 보장 (단계 E baseline 과 byte 동일 결과).
5. 각 post 의 `{post_id}.json` 결과 파싱해 (A) red shrinkage / (B) NEUTRAL cascade 빈도 집계.
6. summary.json 작성: post-level pass/fail, frequency tables, P1 vs defer 결정 자료.

(A) 빈도 집계: per-canonical pool 의 `lab_a_gt20_count < 5% of obj_pixel_count` 비율 — Gemini
preset_color_names_top3 에 warm 톤 (maroon_red/scarlet/saffron/turmeric_yellow/coral 등) 이
포함된 outfit 중. 즉 "warm pick 했는데 픽셀 누락" 케이스.

(B) 빈도 집계: per-canonical pool 의 `palette family` 가 NEUTRAL 인데 Gemini preset pick top1
이 jewel/bright/earth/saturated_warm 인 비율. 즉 "non-neutral pick 했는데 family rule 이
NEUTRAL 로 cascade" 케이스.

cost: ~14 post × 1 image × $0.00135 ≈ $0.019 (Gemini cache miss 시).
runtime: ~14 × 10s = 2.5min model load + Gemini API 호출.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent

ETHNIC_REGEX = (
    "saree|sari|kurta|lehenga|anarkali|bandhani|banarasi|"
    "chikankari|salwar|dupatta|ethnicwear|indianwear|sherwani|dhoti|palazzo"
)

EXCLUDE_IDS: tuple[str, ...] = (
    "01KPT74FM28H0GT6MQTNHFBY1Q",   # Sridevi (단계 E baseline)
    "01KPWFPXHWJVKPHKK0WKBZ1MGF",   # Fabindia (단계 E baseline)
)

# (A) warm-pick 정의 — preset_color_names_top3 에 포함되면 warm tone 픽
WARM_PICKS: frozenset[str] = frozenset({
    "maroon_red", "scarlet", "saffron", "turmeric_yellow", "coral",
    "rust", "burnt_orange", "brick_red", "tomato_red", "cherry_red",
    "vermillion", "sindoor_red", "rani_pink", "fuchsia",
})

# (B) non-neutral preset family — pick 이 NEUTRAL 이 아닌데 final palette 가 NEUTRAL 로
# cascade 된 케이스 검출용. WARM_PICKS + jewel/earth/bright 톤 이름들 모두 포함.
NON_NEUTRAL_PICK_FAMILIES: frozenset[str] = frozenset({
    "jewel", "bright", "earth", "saturated_warm", "saturated_cool",
})


def _load_dotenv() -> None:
    from dotenv import load_dotenv
    load_dotenv(REPO / ".env")


def fetch_candidates(limit: int) -> list[dict[str, Any]]:
    """StarRocks 에서 ethnic regex 매칭 post 추출. download_urls CSV → image_urls list."""
    import pymysql

    placeholders = ",".join(["%s"] * len(EXCLUDE_IDS))
    sql = (
        "SELECT id, user, posting_at, content, download_urls "
        "FROM india_ai_fashion_inatagram_posting "
        "WHERE LOWER(content) REGEXP %s "
        "AND download_urls IS NOT NULL AND download_urls != '' "
        "AND posting_at IS NOT NULL AND posting_at != '' "
        f"AND id NOT IN ({placeholders}) "
        "ORDER BY posting_at DESC "
        "LIMIT %s"
    )
    conn = pymysql.connect(
        host=os.environ["STARROCKS_HOST"],
        port=int(os.environ.get("STARROCKS_PORT", "9030")),
        user=os.environ["STARROCKS_USER"],
        password=os.environ["STARROCKS_PASSWORD"],
        database=os.environ.get("STARROCKS_RAW_DATABASE", "png"),
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=15,
    )
    rows: list[dict[str, Any]] = []
    with conn:
        with conn.cursor() as cur:
            cur.execute(sql, (ETHNIC_REGEX, *EXCLUDE_IDS, limit))
            for row in cur.fetchall():
                urls = [u.strip() for u in (row["download_urls"] or "").split(",") if u.strip()]
                # 이미지 확장자만 (mp4 reel 제외)
                image_urls = [
                    u for u in urls
                    if Path(u).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".heic"}
                ]
                if not image_urls:
                    continue
                rows.append({
                    "id": row["id"],
                    "user": row["user"],
                    "posting_at": str(row["posting_at"]),
                    "image_urls": image_urls[:1],  # 첫 이미지만 (단계 E baseline 과 동일)
                    "content_preview": (row["content"] or "")[:120],
                })
    return rows


def write_enriched(rows: list[dict[str, Any]], out_path: Path) -> None:
    """diag 가 읽을 enriched-like JSON 작성. normalized.source_post_id + image_urls 만 필요."""
    payload = [
        {
            "normalized": {
                "source_post_id": r["id"],
                "image_urls": r["image_urls"],
            },
            "_meta": {
                "user": r["user"],
                "posting_at": r["posting_at"],
                "content_preview": r["content_preview"],
            },
        }
        for r in rows
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def download_images(rows: list[dict[str, Any]], blob_cache: Path) -> dict[str, str]:
    """image_urls 를 blob_cache 로 다운로드. {post_id: status} 반환."""
    from loaders.blob_downloader import BlobDownloader

    blob_cache.mkdir(parents=True, exist_ok=True)
    dl = BlobDownloader.from_env()
    status: dict[str, str] = {}
    for r in rows:
        ok = True
        for url in r["image_urls"]:
            path = dl.download(url, blob_cache)
            if path is None:
                ok = False
                break
        status[r["id"]] = "ok" if ok else "blob_failed"
    return status


def run_diag(post_id: str, enriched_path: Path, out_dir: Path,
             blob_cache: Path, llm_cache: Path) -> str:
    """diag_canonical_pool.py subprocess 호출. returncode 와 별개로 result JSON 존재 여부로 판단."""
    cmd = [
        ".venv/bin/python", "scripts/diag_canonical_pool.py",
        "--post-id", post_id,
        "--enriched", str(enriched_path),
        "--blob-cache", str(blob_cache),
        "--llm-cache", str(llm_cache),
        "--out", str(out_dir),
    ]
    env = {**os.environ, "PYTHONPATH": "src"}
    proc = subprocess.run(
        cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=600,
    )
    result_json = out_dir / f"{post_id}.json"
    if result_json.exists():
        return "ok"
    # SystemExit (image cache miss / all video) 또는 다른 실패
    last_line = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else ""
    last_stdout = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    return f"diag_skip rc={proc.returncode} err={last_line[:80]} out={last_stdout[:80]}"


def _is_post_ethnic(diag_json: dict[str, Any]) -> bool:
    """Gemini ethnic flag 확인 — image[0].outfits 중 upper_is_ethnic=True 가 하나라도 있으면 True.

    lower_is_ethnic 은 dress (single piece) 일 때 None 이므로 OR 조건 사용.
    """
    for img in diag_json.get("images", []) or []:
        for outfit in img.get("outfits", []) or []:
            if outfit.get("upper_is_ethnic") or outfit.get("lower_is_ethnic"):
                return True
    return False


def _classify_post(diag_json: dict[str, Any]) -> dict[str, Any]:
    """1 post diag 결과에서 (A)/(B)/(C) 분류 + 메트릭 추출.

    verdict 카테고리:
    - ethnic_not_confirmed: Gemini upper/lower_is_ethnic 모두 False/None → (A)/(B)/(C) denom 제외
    - no_canonical: canonical 자체 없음 (LLM bbox 0 또는 SceneFilter stage 모두 reject)
    - empty_pool: canonical 있으나 objects=0 / palette=0 (issue (C))
    - ok: canonical 있고 palette 있음 → (A)/(B) 검사 대상

    (A) red shrinkage: warm pick 있는데 obj 별 lab_a_gt20_count / obj_pixel_count < 5%
    (B) NEUTRAL cascade: non-neutral pick top1 인데 canonical palette family 가 모두 NEUTRAL
    (C) empty pool: ethnic post 인데 canonical 의 objects/palette 가 비어있음
    """
    if not _is_post_ethnic(diag_json):
        return {
            "verdict": "ethnic_not_confirmed",
            "issue_a": False,
            "issue_b": False,
            "issue_c": False,
        }

    canonicals = (diag_json.get("v3_hybrid") or {}).get("canonicals", []) or []
    if not canonicals:
        return {
            "verdict": "no_canonical",
            "issue_a": False,
            "issue_b": False,
            "issue_c": True,  # ethnic 인데 canonical 자체 없음 = (C) 상위집합
        }

    # canonicals 모두 비어있으면 empty_pool
    total_objects = sum(len(c.get("objects", []) or []) for c in canonicals)
    total_palette = sum(len(c.get("palette", []) or []) for c in canonicals)
    if total_objects == 0 or total_palette == 0:
        return {
            "verdict": "empty_pool",
            "issue_a": False,
            "issue_b": False,
            "issue_c": True,
        }

    issue_a = False
    issue_b = False
    a_details: list[dict] = []
    b_details: list[dict] = []
    all_picks: list[str] = []
    all_palette: list[dict] = []

    for canonical in canonicals:
        objects = canonical.get("objects", []) or []
        palette = canonical.get("palette", []) or []
        all_palette.extend(palette)

        picks: list[str] = []
        for obj in objects:
            picks.extend(obj.get("picks_input", []) or [])
        all_picks.extend(picks)

        # (A) warm-pick 검출 — object 별 ratio
        has_warm_pick = any(p in WARM_PICKS for p in picks)
        if has_warm_pick:
            for obj in objects:
                obj_pix = obj.get("obj_pixel_count", 0)
                warm_pix = obj.get("lab_a_gt20_count", 0)
                ratio = (warm_pix / obj_pix) if obj_pix > 0 else 0.0
                if obj_pix > 0 and ratio < 0.05:
                    issue_a = True
                    a_details.append({
                        "obj_pixel_count": obj_pix,
                        "lab_a_gt20_count": warm_pix,
                        "ratio": round(ratio, 4),
                        "picks": picks,
                    })

        # (B) NEUTRAL cascade — canonical 단위
        palette_families = [c.get("family", "").lower() for c in palette]
        all_neutral = bool(palette_families) and all(f == "neutral" for f in palette_families)
        pick_top1 = picks[0] if picks else None
        pick_top1_family = _resolve_pick_family(pick_top1) if pick_top1 else None
        if pick_top1_family in NON_NEUTRAL_PICK_FAMILIES and all_neutral:
            issue_b = True
            b_details.append({
                "pick_top1": pick_top1,
                "pick_family": pick_top1_family,
                "palette_families": palette_families,
            })

    return {
        "verdict": "ok",
        "issue_a": issue_a,
        "issue_b": issue_b,
        "issue_c": False,
        "warm_picks": [p for p in all_picks if p in WARM_PICKS],
        "all_picks": all_picks,
        "a_details": a_details,
        "b_details": b_details,
        "palette": [{"hex": c.get("hex"), "share": c.get("share"),
                     "family": c.get("family")} for c in all_palette],
    }


def _resolve_pick_family(pick: str) -> str:
    """Gemini preset pick 이름 → family 라벨. WARM_PICKS 는 saturated_warm.

    참고: 전체 preset → family 매핑은 src/vision/color_family_preset.py 의 family_map 에 있지만,
    F 측정용으로는 (B) cascade 검출에 필요한 비-neutral 만 분류하면 됨.
    """
    p = pick.lower()
    if p in WARM_PICKS:
        return "saturated_warm"
    if p in {"deep_indigo", "navy_blue", "royal_blue", "midnight_blue", "cobalt"}:
        return "jewel"  # navy 톤은 family rule 상 jewel/dark
    if p in {"emerald", "forest_green", "bottle_green", "teal", "turquoise"}:
        return "jewel"
    if p in {"plum", "wine", "burgundy", "amethyst", "violet", "purple_velvet"}:
        return "jewel"
    if p in {"olive", "khaki", "mustard", "ochre", "henna_brown",
             "coffee_brown", "chocolate_brown", "umber", "sienna"}:
        return "earth"
    if p in {"lemon_yellow", "neon_pink", "neon_green", "electric_blue",
             "fluorescent_orange"}:
        return "bright"
    if p.startswith("pool_") or p.startswith("self_"):
        # 자기 색은 family 미상 — non-neutral 분류 안 함
        return "unknown"
    if p in {"ivory", "cream", "off_white", "stone", "beige", "taupe", "sand",
             "linen", "platinum", "silver", "charcoal", "black", "white",
             "midnight_black"}:
        return "neutral"
    return "unknown"


def aggregate_results(out_dir: Path, dl_status: dict[str, str],
                      diag_status: dict[str, str]) -> dict[str, Any]:
    """모든 {post_id}.json 결과 집계 → summary 반환."""
    summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "posts": {},
        "totals": {
            "candidates": 0,
            "blob_failed": 0,
            "diag_skip": 0,
            # 분류 (verdict 별 카운트)
            "ethnic_not_confirmed": 0,
            "no_canonical": 0,
            "empty_pool": 0,
            "ok_with_palette": 0,
            # issue 카운트 (denom 분리)
            "issue_a_count": 0,  # ok_with_palette 중
            "issue_b_count": 0,  # ok_with_palette 중
            "issue_c_count": 0,  # ethnic-confirmed 중 (no_canonical + empty_pool)
        },
    }
    for post_id, dl in dl_status.items():
        summary["totals"]["candidates"] += 1
        diag = diag_status.get(post_id, "unknown")
        post_summary: dict[str, Any] = {
            "blob": dl,
            "diag": diag,
        }
        if dl != "ok":
            summary["totals"]["blob_failed"] += 1
        elif not diag.startswith("ok"):
            summary["totals"]["diag_skip"] += 1
        else:
            result_json = out_dir / f"{post_id}.json"
            try:
                data = json.loads(result_json.read_text())
                cls = _classify_post(data)
                post_summary.update(cls)
                v = cls["verdict"]
                if v == "ethnic_not_confirmed":
                    summary["totals"]["ethnic_not_confirmed"] += 1
                elif v == "no_canonical":
                    summary["totals"]["no_canonical"] += 1
                    summary["totals"]["issue_c_count"] += 1
                elif v == "empty_pool":
                    summary["totals"]["empty_pool"] += 1
                    summary["totals"]["issue_c_count"] += 1
                else:  # ok
                    summary["totals"]["ok_with_palette"] += 1
                    if cls["issue_a"]:
                        summary["totals"]["issue_a_count"] += 1
                    if cls["issue_b"]:
                        summary["totals"]["issue_b_count"] += 1
            except Exception as exc:  # noqa: BLE001
                post_summary["parse_error"] = str(exc)
        summary["posts"][post_id] = post_summary
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-count", type=int, default=18,
                    help="SQL LIMIT — Stage 1 reject / image cache miss 안전마진 포함")
    ap.add_argument("--out-dir", default=str(REPO / "outputs/2026-04-26_ethnic10"),
                    help="diag 결과 + summary.json 저장 디렉토리")
    ap.add_argument("--blob-cache", default=str(REPO / "sample_data/image_cache"),
                    help="이미지 로컬 캐시 (단계 E baseline 과 공유)")
    ap.add_argument("--llm-cache", default=str(REPO / "outputs/llm_cache"),
                    help="Gemini 캐시 (단계 E hit 가능)")
    ap.add_argument("--skip-diag", action="store_true",
                    help="후보 추출 + 이미지 DL 만, diag run 생략 (dry-run)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    blob_cache = Path(args.blob_cache)
    llm_cache = Path(args.llm_cache)

    _load_dotenv()

    print(f"[F-2] fetching {args.candidate_count} candidates from StarRocks...")
    rows = fetch_candidates(args.candidate_count)
    print(f"[F-2] got {len(rows)} candidates with images")

    enriched_path = out_dir / "enriched.json"
    write_enriched(rows, enriched_path)
    print(f"[F-2] wrote enriched → {enriched_path}")

    print(f"[F-2] downloading images to {blob_cache}...")
    dl_status = download_images(rows, blob_cache)
    ok_count = sum(1 for s in dl_status.values() if s == "ok")
    print(f"[F-2] blob DL ok={ok_count}/{len(rows)}")

    diag_status: dict[str, str] = {}
    if args.skip_diag:
        print("[F-2] --skip-diag: skipping diag runs")
    else:
        for i, r in enumerate(rows):
            pid = r["id"]
            if dl_status.get(pid) != "ok":
                diag_status[pid] = "skipped_blob_failed"
                print(f"[F-2] [{i+1}/{len(rows)}] {pid}: skip (blob_failed)")
                continue
            print(f"[F-2] [{i+1}/{len(rows)}] {pid}: running diag...")
            status = run_diag(pid, enriched_path, out_dir, blob_cache, llm_cache)
            diag_status[pid] = status
            print(f"[F-2] [{i+1}/{len(rows)}] {pid}: {status}")

    summary = aggregate_results(out_dir, dl_status, diag_status)

    # 단계 E baseline 도 summary 에 포함 (Sridevi/Fabindia 의 (A)/(B) 결과)
    baseline_dir = REPO / "outputs/phase5_stepF_baseline"
    summary["baseline"] = {}
    for pid in EXCLUDE_IDS:
        baseline_json = baseline_dir / f"{pid}.json"
        if baseline_json.exists():
            try:
                data = json.loads(baseline_json.read_text())
                summary["baseline"][pid] = _classify_post(data)
            except Exception as exc:  # noqa: BLE001
                summary["baseline"][pid] = {"parse_error": str(exc)}

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[F-2] wrote summary → {summary_path}")
    print(f"[F-2] totals: {json.dumps(summary['totals'], indent=2)}")


if __name__ == "__main__":
    main()
