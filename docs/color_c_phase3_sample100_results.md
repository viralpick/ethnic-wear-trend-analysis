# color.C Phase 3 — sample 100 frame mask-based canary

spec: `docs/color_c_illumination_spec.md` Phase 3 단계 (sample 100 일부).

## 데이터 / 셋업

- frame pool: `outputs/weekly_review/_video_thumbs/` 8057 JPG
- sample: 100 (random, seed=42) — Phase 2 와 동일
- p: 6 (Phase 2 권장값 / spec 권장)
- verify: enabled=True, deltae76_threshold=30
- segformer: segformer_b2_clothes (MPS device), UPPER∪LOWER∪DRESS wear mask
- runtime: 18.5s

## 결과 요약

- mask 보유 frame: 100 / 100 (100.0%)
- detection trigger: 69 / 100 (69.0%) — 이 중 mask 보유 69
- **verify reject (mask-based)**: 0 / 69 = **0.0%**

## garment 단위 보정 효과 (triggered + has_mask)

| metric | median |
|---|---|
| skew_a reduction (garment mask 내) | +3.00 |
| skew_b reduction (garment mask 내) | +3.00 |
| ΔE76 (garment LAB median, 보정 magnitude) | 7.45 |
| ΔE76 p90 | 15.18 |

## Phase 2 vs Phase 3 비교 가이드

- Phase 2 (frame-level surrogate, p=6): ΔE76 median 6.79 / reject 0%
- Phase 3 (mask-based, p=6): ΔE76 median 7.45 / reject 0.0%
- mask-based 가 frame-level 보다 크면: garment 가 보정 영향을 더 받음 (의류 색이 frame 평균에서 멀리)
- mask-based 가 frame-level 보다 작으면: 보정이 배경 위주, garment 는 안정

## p=6 vs p=1 ablation — mask 가드 effectiveness 검증

같은 sample 100 frame, mask-based verify:

| p | verify reject | garment ΔE76 median | garment ΔE76 p90 | garment skew_a↓ | garment skew_b↓ |
|---|---|---|---|---|---|
| **6** (권장) | **0 / 69 = 0%** ✅ | 7.45 | 15.18 | +3.00 | +3.00 |
| 1 (gray-world) | **4 / 69 = 5.8%** ⚠ | 12.83 | 24.11 | +4.00 | +5.00 |

해석:
- p=1 (강한 gray-world 보정) 에서 mask-based verify guard 가 4건 발동 → 원본 회수.
  의류 색을 너무 많이 shift 시킨 frame 4건의 보정 차단 = 가드의 실 effectiveness 검증.
- p=6 은 verify reject 0건 = sample 100 범위에서 의류 색 왜곡 risk 없음.
- Phase 1 sanity (page_0 1 page) 의 reject 0 baseline 보강: 보정 강도 ↑ 면 reject 가 실제로 발동.
- 출원 명세서: 가드 동작 evidence 로 활용 가능 (p=1 vs p=6 reject rate 차이 5.8%p).

> 자세한 p=1 결과: `docs/color_c_phase3_sample100_results_p1.md`

## Phase 3 enable 결정 5조건 vs 현재 측정

| metric | baseline | 목표 | 현재 (sample 100) |
|---|---|---|---|
| family disagreement rate | 50.9% | < 30% | (full canary 필요 — Gemini 호출) |
| ΔE76 분산 median | 17.7 | < 10 | 보정 magnitude 7.45 (다른 metric — 분산은 multi-member 비교)
| verify 거부율 | — | < 20% | **0.0%** ✅
| 회귀 cliff 신규 발생 | 0 | 0 (절대) | (full canary 필요 — cluster matching)
| 운영 시간 증가 | 1× | < 30% | (full canary 필요 — Gemini round-trip 측정)

> 본 Phase 3 sample 100 은 verify 가드 + garment-level 보정 강도 정량까지. family/cliff/runtime 은 별도 full canary 세션 (Gemini Pass 1/2 + Stream Load) 필요.
