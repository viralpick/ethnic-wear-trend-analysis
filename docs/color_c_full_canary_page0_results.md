# color.C full canary — page_0 baseline vs canary 비교

spec: `docs/color_c_illumination_spec.md` Phase 3 enable 결정 5조건 (family / cliff / runtime).

## 데이터

- baseline: `outputs/color_c_baseline_page0/enriched.json` (100 entries, color.C **off**, **5/18 같은 코드 base**)
- canary  : `outputs/color_c_canary_page0/enriched.json` (100 entries, color.C **enabled p=6, verify on**, 5/18)
- 공통 post: 100 (baseline 100, canary 100)
- **isolated comparison**: 같은 코드 base 에서 yaml `illumination_correction.enabled` 토글만 차이 — vision/v0.10 누적 변경 효과 제거
- 비교 환경: --llm fake (attribute LLM 무관), --vision-llm gemini, --sink dry_run

## 결과 — canonical totals

| metric | baseline | canary | delta |
|---|---|---|---|
| 공통 post 수 | 100 | 100 | 0 |
| canonical 합계 | 42 | 42 | +0 |
| **cliff (cut_off_share == 1.0)** | 0 | 0 | +0 |
| cut_off median | 0.0947 | 0.1075 | — |

## cluster_key disagreement

- post 별 `trend_cluster_key` 다른 비율: **0 / 100 = 0.0%**
- canonical 수 다른 post: 0 / 100 = 0.0%
- canonical 수 delta (canary − baseline) median: +0.0, range: +0 ~ +0

## family disagreement (canonical-level palette[0].family)

- canonical 매칭 idx-pairwise: 12 / 42 = **28.6%**
- spec baseline 50.9% (multi-member 670 의 family 불일치율) 와 다른 metric — 본 measurement 는 post-level idx-pairwise.

## family 분포 비교 (canonical palette[0].family Counter)

| family | baseline count | canary count |
|---|---|---|
| bright | 4 | 0 |
| earth | 4 | 4 |
| jewel | 8 | 8 |
| neutral | 17 | 21 |
| pastel | 9 | 8 |
| white_on_white | 0 | 1 |

## cluster_key 변경 sample (max 10)

(모든 post 의 cluster_key 일치 — color.C 가 cluster 결과 변화 0)

## enable 결정 5조건 vs 현재

| metric | baseline | 목표 | 현재 (page_0 100 post) |
|---|---|---|---|
| family disagreement rate | 50.9% (multi-member 670) | < 30% | page-level idx-pairwise 28.6% (다른 sample)
| ΔE76 분산 median | 17.7 (multi-member 670) | < 10 | page_0 단독 측정 X (full canary 670 필요) |
| verify 거부율 | — | < 20% | 0% (Phase 3 sample 100 mask-based) |
| 회귀 cliff 신규 발생 | 0 | 0 (절대) | ✅ 0 (baseline 0 → canary 0) |
| 운영 시간 증가 | 1× | < 30% | (별도 측정 필요 — page_0 단독으로는 산출 어려움) |

> page_0 은 16w 의 newest 100 post (multi-member 분포가 spec baseline 670 과 다름). full enable 결정에는 multi-member 670 full canary 또는 다중 page 측정이 더 robust.
