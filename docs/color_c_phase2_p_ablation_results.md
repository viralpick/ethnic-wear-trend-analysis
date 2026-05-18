# color.C Phase 2 — Minkowski p ablation

spec: `docs/color_c_illumination_spec.md` 결정 2 Phase 2 단계.

## 데이터 / 셋업

- frame pool: `outputs/weekly_review/_video_thumbs/` 8057 JPG
- sample: 100 (random, seed=42)
- p 후보: [1, 4, 6, 8, 16]
- detection rule: |a|>8 또는 |b|>8 또는 L<30 또는 L>80 (Phase 1 동일)
- verify threshold (frame-level surrogate): ΔE76 > 30
- mask-based verify (segformer): Phase 3 에서 수행

## detection trigger rate

- 69.0% (p 와 무관 — detection rule 동일). spec 예상 50~60% 와 비교.

## p ablation metric

| p | trigger N | skew_a↓ median | skew_b↓ median | L_shift median | ΔE76 median | ΔE76 p90 | verify reject % | runtime |
|---|---|---|---|---|---|---|---|---|
| 1 | 69/100 | +6.81 | +10.19 | 1.32 | 13.45 | 23.03 | 1.4% | 1.02s |
| 4 | 69/100 | +4.24 | +6.02 | 0.86 | 8.17 | 13.99 | 0.0% | 1.74s |
| 6 | 69/100 | +3.17 | +5.12 | 0.72 | 6.79 | 13.39 | 0.0% | 1.74s |
| 8 | 69/100 | +2.58 | +4.12 | 0.69 | 6.13 | 12.97 | 0.0% | 1.76s |
| 16 | 69/100 | +1.88 | +1.96 | 0.55 | 3.96 | 8.75 | 0.0% | 1.75s |

## 해석 가이드

- **skew_a / skew_b reduction (+)**: 보정 후 a/b skew 절대값이 감소한 양.
  양수 = 보정 효과 (gray balance 회복). 음수 = 보정이 오히려 skew 증가.
- **L_shift**: 보정 전후 L mean 의 absolute 차. 클수록 명도 손상 risk.
- **ΔE76**: frame LAB mean 의 보정 magnitude. p 작을수록 (gray-world 에 가까울수록) ↑.
- **verify reject (surrogate)**: ΔE76 > 30 = 과도한 보정 후보. mask-based 와 다를 수 있음.

## 권장값 판단 기준

1. skew_a/skew_b reduction 이 충분히 큰 p (effect size)
2. verify reject 가 낮은 p (안전 margin)
3. L_shift 가 합리적 (e.g. < 5 ~ 10) 한 p
4. tie 시 spec 권장값 p=6 유지

> Phase 3 full canary 에서 mask-based verify + family disagreement / 운영 metric 재측정 후 enable 결정.
