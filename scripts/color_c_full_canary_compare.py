"""color.C full canary — baseline vs canary enriched.json 비교.

비교 대상:
- baseline: outputs/backfill_16w/page_0_enriched.json (color.C off, 5/2 백필)
- canary  : outputs/color_c_canary_page0/enriched.json (color.C enabled, p=6, verify on)

같은 post 100개 (canary canonical 100, baseline 153 중 매칭 100) 단위 비교.

측정 metric:
- total_canonicals: 전체 canonical 수 (post 합산)
- cliff_count: canonical-level cut_off_share == 1.0 인 canonical 수 (baseline 의 4건 회복 여부)
- cluster_key_disagree: 같은 post 에서 trend_cluster_key 가 다른 비율
- family_disagree: 같은 post + 같은 canonical_index 의 palette[0].family 다른 비율
- canonical_count_diff: post 별 canonical 수 차이 (보정으로 cluster 통합/분할)

출력: docs/color_c_full_canary_page0_results.md
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# 5/18 같은 코드 base 에서 color.C off baseline / color.C on canary 둘 다 실행 — 누적 차이 제거.
BASELINE = ROOT / "outputs" / "color_c_baseline_page0" / "enriched.json"
CANARY = ROOT / "outputs" / "color_c_canary_page0" / "enriched.json"
OUT_MD = ROOT / "docs" / "color_c_full_canary_page0_results.md"


def _pid(e: dict) -> str:
    n = e.get("normalized") or {}
    return n.get("source_post_id") or n.get("post_id") or n.get("id") or ""


def _canonical_summary(canonicals: list[dict]) -> dict:
    """canonical 단위 family / cliff / 수 요약."""
    n_cliff = 0
    family_counter: Counter[str] = Counter()
    family_top1_per_canonical: list[str] = []
    cut_offs: list[float] = []
    for cn in canonicals:
        cut_off = cn.get("cut_off_share")
        if cut_off is not None:
            cut_offs.append(cut_off)
            if cut_off >= 0.999:
                n_cliff += 1
        palette = cn.get("palette") or []
        if palette:
            top1 = palette[0].get("family", "?")
            family_top1_per_canonical.append(top1)
            for p in palette:
                family_counter[p.get("family", "?")] += 1
    return {
        "n_canonicals": len(canonicals),
        "n_cliff": n_cliff,
        "cut_offs": cut_offs,
        "family_top1_per_canonical": family_top1_per_canonical,
        "family_counter": dict(family_counter),
    }


def main() -> None:
    baseline = json.loads(BASELINE.read_text())
    canary = json.loads(CANARY.read_text())

    base_by_pid = {_pid(e): e for e in baseline if _pid(e)}
    canary_by_pid = {_pid(e): e for e in canary if _pid(e)}
    common = set(base_by_pid) & set(canary_by_pid)
    print(f"[compare] baseline {len(base_by_pid)}, canary {len(canary_by_pid)}, common {len(common)}")

    # post-level metric
    n_cluster_disagree = 0
    n_canonical_count_diff = 0
    canonical_delta: list[int] = []  # canary - baseline
    cluster_disagreements: list[tuple[str, str, str]] = []

    # canonical-level (post + canonical_index 매칭) family disagree
    n_family_match_total = 0
    n_family_disagree = 0

    # post 단위 baseline / canary canonical 합산
    base_canon_total = 0
    canary_canon_total = 0
    base_cliff_total = 0
    canary_cliff_total = 0
    base_family_top1: Counter[str] = Counter()
    canary_family_top1: Counter[str] = Counter()
    base_cut_offs: list[float] = []
    canary_cut_offs: list[float] = []

    for pid in common:
        b_e = base_by_pid[pid]
        c_e = canary_by_pid[pid]
        b_sum = _canonical_summary(b_e.get("canonicals") or [])
        c_sum = _canonical_summary(c_e.get("canonicals") or [])

        # totals
        base_canon_total += b_sum["n_canonicals"]
        canary_canon_total += c_sum["n_canonicals"]
        base_cliff_total += b_sum["n_cliff"]
        canary_cliff_total += c_sum["n_cliff"]
        base_cut_offs.extend(b_sum["cut_offs"])
        canary_cut_offs.extend(c_sum["cut_offs"])
        for f in b_sum["family_top1_per_canonical"]:
            base_family_top1[f] += 1
        for f in c_sum["family_top1_per_canonical"]:
            canary_family_top1[f] += 1

        # cluster_key
        b_key = b_e.get("trend_cluster_key")
        c_key = c_e.get("trend_cluster_key")
        if b_key != c_key:
            n_cluster_disagree += 1
            cluster_disagreements.append((pid, str(b_key), str(c_key)))

        # canonical count
        if b_sum["n_canonicals"] != c_sum["n_canonicals"]:
            n_canonical_count_diff += 1
        canonical_delta.append(c_sum["n_canonicals"] - b_sum["n_canonicals"])

        # canonical-level family (idx 매칭 — 같은 outfit 인지 보장 X 지만 page-level 분포 비교)
        b_fams = b_sum["family_top1_per_canonical"]
        c_fams = c_sum["family_top1_per_canonical"]
        for bf, cf in zip(b_fams, c_fams):
            n_family_match_total += 1
            if bf != cf:
                n_family_disagree += 1

    # 결과 markdown
    md: list[str] = []
    md.append("# color.C full canary — page_0 baseline vs canary 비교")
    md.append("")
    md.append("spec: `docs/color_c_illumination_spec.md` Phase 3 enable 결정 5조건 (family / cliff / runtime).")
    md.append("")
    md.append("## 데이터")
    md.append("")
    md.append(f"- baseline: `outputs/color_c_baseline_page0/enriched.json` ({len(baseline)} entries, color.C **off**, **5/18 같은 코드 base**)")
    md.append(f"- canary  : `outputs/color_c_canary_page0/enriched.json` ({len(canary)} entries, color.C **enabled p=6, verify on**, 5/18)")
    md.append(f"- 공통 post: {len(common)} (baseline {len(base_by_pid)}, canary {len(canary_by_pid)})")
    md.append("- **isolated comparison**: 같은 코드 base 에서 yaml `illumination_correction.enabled` 토글만 차이 — vision/v0.10 누적 변경 효과 제거")
    md.append("- 비교 환경: --llm fake (attribute LLM 무관), --vision-llm gemini, --sink dry_run")
    md.append("")
    md.append("## 결과 — canonical totals")
    md.append("")
    md.append("| metric | baseline | canary | delta |")
    md.append("|---|---|---|---|")
    md.append(f"| 공통 post 수 | {len(common)} | {len(common)} | 0 |")
    md.append(f"| canonical 합계 | {base_canon_total} | {canary_canon_total} | {canary_canon_total - base_canon_total:+d} |")
    md.append(f"| **cliff (cut_off_share == 1.0)** | {base_cliff_total} | {canary_cliff_total} | {canary_cliff_total - base_cliff_total:+d} |")
    if base_cut_offs and canary_cut_offs:
        md.append(f"| cut_off median | {statistics.median(base_cut_offs):.4f} | {statistics.median(canary_cut_offs):.4f} | — |")
    md.append("")
    md.append("## cluster_key disagreement")
    md.append("")
    md.append(f"- post 별 `trend_cluster_key` 다른 비율: **{n_cluster_disagree} / {len(common)} = {n_cluster_disagree / len(common) * 100:.1f}%**")
    md.append(f"- canonical 수 다른 post: {n_canonical_count_diff} / {len(common)} = {n_canonical_count_diff / len(common) * 100:.1f}%")
    if canonical_delta:
        med = statistics.median(canonical_delta)
        md.append(f"- canonical 수 delta (canary − baseline) median: {med:+.1f}, range: {min(canonical_delta):+d} ~ {max(canonical_delta):+d}")
    md.append("")
    md.append("## family disagreement (canonical-level palette[0].family)")
    md.append("")
    md.append(f"- canonical 매칭 idx-pairwise: {n_family_disagree} / {n_family_match_total} = **{n_family_disagree / max(n_family_match_total, 1) * 100:.1f}%**")
    md.append("- spec baseline 50.9% (multi-member 670 의 family 불일치율) 와 다른 metric — 본 measurement 는 post-level idx-pairwise.")
    md.append("")
    md.append("## family 분포 비교 (canonical palette[0].family Counter)")
    md.append("")
    md.append("| family | baseline count | canary count |")
    md.append("|---|---|---|")
    all_fams = sorted(set(base_family_top1) | set(canary_family_top1))
    for f in all_fams:
        md.append(f"| {f} | {base_family_top1.get(f, 0)} | {canary_family_top1.get(f, 0)} |")
    md.append("")
    md.append("## cluster_key 변경 sample (max 10)")
    md.append("")
    if cluster_disagreements:
        md.append("| post_id | baseline | canary |")
        md.append("|---|---|---|")
        for pid, bk, ck in cluster_disagreements[:10]:
            md.append(f"| `{pid}` | {bk} | {ck} |")
    else:
        md.append("(모든 post 의 cluster_key 일치 — color.C 가 cluster 결과 변화 0)")
    md.append("")
    md.append("## enable 결정 5조건 vs 현재")
    md.append("")
    md.append("| metric | baseline | 목표 | 현재 (page_0 100 post) |")
    md.append("|---|---|---|---|")
    md.append(
        f"| family disagreement rate | 50.9% (multi-member 670) | < 30% | "
        f"page-level idx-pairwise {n_family_disagree / max(n_family_match_total, 1) * 100:.1f}% (다른 sample)"
    )
    md.append(
        f"| ΔE76 분산 median | 17.7 (multi-member 670) | < 10 | "
        f"page_0 단독 측정 X (full canary 670 필요) |"
    )
    md.append(f"| verify 거부율 | — | < 20% | 0% (Phase 3 sample 100 mask-based) |")
    md.append(
        f"| 회귀 cliff 신규 발생 | 0 | 0 (절대) | "
        f"{'✅ 0' if canary_cliff_total == 0 else f'⚠ {canary_cliff_total}'} "
        f"(baseline {base_cliff_total} → canary {canary_cliff_total}) |"
    )
    md.append(f"| 운영 시간 증가 | 1× | < 30% | (별도 측정 필요 — page_0 단독으로는 산출 어려움) |")
    md.append("")
    md.append("> page_0 은 16w 의 newest 100 post (multi-member 분포가 spec baseline 670 과 다름). full enable 결정에는 multi-member 670 full canary 또는 다중 page 측정이 더 robust.")
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[compare] wrote {OUT_MD}")
    print()
    print("==> summary:")
    print(f"  common post: {len(common)}")
    print(f"  canonical totals: baseline {base_canon_total} -> canary {canary_canon_total} ({canary_canon_total - base_canon_total:+d})")
    print(f"  cliff: baseline {base_cliff_total} -> canary {canary_cliff_total} ({canary_cliff_total - base_cliff_total:+d})")
    print(f"  cluster_key disagree: {n_cluster_disagree}/{len(common)} ({n_cluster_disagree / len(common) * 100:.1f}%)")
    print(f"  family disagree (idx-pairwise): {n_family_disagree}/{n_family_match_total} ({n_family_disagree / max(n_family_match_total, 1) * 100:.1f}%)")


if __name__ == "__main__":
    main()
