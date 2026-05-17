# color.C — illumination correction spec (석양/조명 보정)

특허 트랙 A 의 enhancement embodiment 후보 + M4.color sub-task P2 (5/6 결정 → 5/16 user 우선순위 변경으로 다음 진입). lighting bias 로 같은 의류의 frame 별 KMeans cluster 색 분산을 보정해 cluster aggregation 의 동일 색 분산 차단.

## 진단 결과 (16w 백필, 5/16 codex 분석)

전체 3,421 canonical, **multi-member 670 (19.6%)** — palette 분산 측정 대상.

### 분산 분포

| ΔE76 (member 간 max-pairwise) | count | % |
|---|---|---|
| < 5 (안정) | 127 | 19.0% |
| 5~10 (경미) | 94 | 14.0% |
| 10~20 (중간) | 136 | 20.3% |
| **20~40 (큰)** | **156** | **23.3%** |
| **≥ 40 (극단)** | **157** | **23.4%** |

- median ΔE76 = 17.7 / p95 = 76.2 / max = 131.3
- **ΔE76 ≥ 20 = 46.7%** — 절반 가까이 lighting bias 의심 수준

### 도메인 편중 — **video 압도적**

- multi-member 670 중 **video-only 651 (97.2%)** / image-only 19 (2.8%)
- video median ΔE76 17.7 (대표 분포)
- carousel/frame 병합이 분산의 거의 유일한 경로 → 보정 대상을 video frame 한정 시 670 중 651 커버

### cut_off_share x 분산 (sweet spot)

| cut_off bucket | n | median ΔE76 |
|---|---|---|
| 0~0.2 | 295 | 14.0 |
| **0.2~0.5** | **337** | **24.0** ★ |
| 0.5~0.8 | 34 | 11.2 |
| 0.8~1.0 | 4 | 0.0 |

**bucket 0.2~0.5 가 sweet spot** — palette 가 어느 정도 잡혔는데 분산 max. 보정 효과 측정에 가장 설득력 있는 구간 (palette 자체가 없는 cliff 와 별개).

### family disagreement

- 같은 canonical 의 member family 일치 = 49.1% (불일치 50.9%)
- top: earth ↔ neutral (412), jewel ↔ neutral (376), neutral ↔ pastel (211), earth ↔ jewel (149), bright ↔ jewel (135)
- **earth ↔ neutral / jewel ↔ neutral 조합이 압도적** — 조명 온도 변화 (warm/cool shift) 의 전형적 lighting bias 패턴. 같은 옷이 frame 마다 earth 또는 neutral 로 분류

### brightness 분포

| LAB L | count |
|---|---|
| < 30 (어두운) | 791 (24.2%) |
| 30~60 | 1054 (32.2%) |
| 60~85 | 1123 (34.4%) |
| ≥ 85 (매우 밝은) | 301 (9.2%) |

밝은 의류 (L≥60) 43.6% — warm shift 영향 더 큼 (석양/실내 텅스텐 조명)

### 후보 top 사례 (보정 대상)

multi-member + ΔE76 ≥ 20 + cut_off < 0.5 = 16건 (top ΔE76):

- `-x3laMckVkY` c=1 — ΔE76 max=131.3 (yt, 3 frame)
- `01KPWFF9771PKVW6ZE92DB3F4G` c=0 — 128.2 (ig)
- `01KPYYF8B7AJ5365CZAMA200CP` c=0 — 103.1 (ig, 5 member)
- `sZxJ9_bcG5E` c=0/2 — 94.3/98.9 (yt)
- 외 12건

## 적용 시점 / 단위

### 옵션 A — pixel 단계 (KMeans 입력 직전)

`extract_dynamic_palette(rgb_pixels)` 호출 전에 rgb_pixels 를 보정. garment-mask 픽셀만 보정 (배경 영향 없음).

- 장점: KMeans cluster 자체가 안정화 → R1/R2 매칭 신뢰도 ↑
- 단점: garment mask 만 가지고 light source 추정이 어려움. mask 가 의류 dominant 색에 편향됐을 수도

### 옵션 B — frame 단계 (segformer 입력 전)

frame 전체 (BGR 또는 RGB) 를 보정 후 segformer + KMeans 흐름 그대로. light source 추정에 frame 전체 정보 활용.

- 장점: 정확한 light estimation (배경 / 피부 / 의류 모두 활용)
- 단점: segformer mask 도 보정된 frame 에 의존 — mask 정확도 변동 risk
- 출원: 더 자연스러운 청구항 (frame illumination → segmentation → palette extraction)

**권장: 옵션 B** — frame 단계 보정. 단 segformer mask 정확도 회귀 가드 (canary 측정 필수)

## 알고리즘 옵션

### 1. Gray-world assumption

`R_mean = G_mean = B_mean` 가정. 각 채널 mean 으로 normalize 후 글로벌 평균 회복.

- 장점: 단순 / 빠름 / 결정론
- 단점: 의류 자체가 한 색 dominant 면 (e.g. saffron 전신) 그 색을 light source 로 오인 → 회색 화

### 2. White-patch (Max-RGB)

각 채널의 max 를 white reference 로 가정. max 로 normalize.

- 장점: 의류 dominant 색 영향 적음
- 단점: 노이즈 / 하이라이트 pixel 1개에 민감 → percentile (top 1% or 5%) 사용 권장

### 3. Shades of gray (general Minkowski norm)

Gray-world 의 일반화. `||I||_p = const` 가정 (p=1 gray-world, p=∞ white-patch). p=6 일반 권장.

- 장점: 두 옵션의 절충 / robust
- 단점: p hyperparameter

### 4. Single-scale Retinex (SSR) / Multi-scale Retinex (MSR)

`L = I / (I * G_σ)` — Gaussian blur 로 illumination 추정 + 분리.

- 장점: 강력 / 명도 균등화
- 단점: 비용 ↑ (Gaussian convolution) / contrast 과장 risk / 의류 본연 색 왜곡 ↑

### 5. CIE LAB 채널 보정 (a/b skew correction)

LAB 의 a (red-green) / b (yellow-blue) 채널의 mean 을 0 으로 shift. L 분포 normalize.

- 장점: 색 채널만 보정 (밝기 무관) / LAB 라 perceptually uniform
- 단점: warm shift 만 가정 → cool shift 도 같이 보정

**권장 시작점: shades of gray (p=6)** — gray-world 안전성 + white-patch robustness 의 절충. SSR/MSR 은 의류 본연 색 왜곡 risk 로 enhancement 단계.

## 조건부 적용 (false-positive 가드)

**모든 frame 을 보정하면 의류 본연 색 왜곡** — 가장 큰 리스크. 단계:

### Detection (보정 대상 식별)

frame 별 light bias 의심 지표:

1. **a/b skew** — LAB a 또는 b 의 mean abs > threshold (e.g. 8) → warm/cool shift 의심
2. **L 분포** — frame L mean 이 < 30 또는 > 80 → 극단 명도. cluster 색 신뢰도 ↓
3. **garment mask 내 channel imbalance** — segformer mask 내 RGB mean 의 max/min 비율 > 1.5

**적용 trigger**: 위 3 지표 중 ≥ 1 trigger → 보정 진행. 0 trigger → skip (원본 유지)

### Apply

trigger 된 frame 에만 shades-of-gray (p=6) 적용. 보정 후 :
- LAB a/b mean 0 근처로 shift
- L 분포 [20, 80] clamp

### Verify (회귀 가드)

보정 후 KMeans cluster 의 share top-1 의 LAB 좌표가 원본 garment_mask 의 LAB median 과 ΔE76 > 30 이면 **보정 거부 (원본 사용)** — 의류 본연 색 왜곡 차단.

## 출원 명세서 청구 매핑

색.C 가 추가되면 트랙 A 출원 명세서에 enhancement embodiment 로 포함 가능:

- **8단계 시스템**: 0. (신규) **frame illumination detection + conditional correction** → 1. SceneFilter → 2. YOLO person → 3. segformer mask → 4. drop_skin → 5. extract_dynamic_palette → 6. R1/R2/etc → 7. canonical aggregate → 8. cluster aggregate
- **결정 룰**: a/b skew threshold = 8 / L extreme = [30, 80] / Minkowski p = 6 / KMeans ΔE76 verify threshold = 30
- **canary metric**: family disagreement rate before/after (50.9% → ?%) + ΔE76 분산 distribution shift (median 17.7 → ?)

## 단계 분할

### Phase 1 — detection + simple gray-world (1~2일)
- frame LAB a/b skew + L extreme detection
- shades-of-gray (p=1, gray-world) 적용
- conditional (detection trigger)
- canary 후보 top 20 + 무작위 100 frame 측정

### Phase 2 — Minkowski p tuning + verify (1~2일)
- Phase 1 결과로 p ∈ {1, 4, 6, ∞} 비교
- KMeans ΔE76 verify 가드 추가
- 보정 거부율 측정

### Phase 3 — full canary (1일)
- multi-member 670 전체 측정 (또는 sample 100)
- family disagreement rate before/after
- 명세서 자료

총 3.5~5일 (color.B 와 유사).

## 결정 1 — 보정 적용 시점 (user 5/16 확정)

**결정: frame 단계 (segformer 입력 전)** ✅ user 확정.

근거:
- light source 추정에 frame 전체 정보 (배경 / 피부 / 의류) 활용 — garment mask 만으로는 의류 dominant 색을 light source 로 오인 risk
- 출원 청구 흐름 자연스러움: frame illumination → segmentation → palette extraction (8단계 system 의 0단계로 신규 추가)
- pixel 단계 (KMeans 직전, garment mask 내) 보정은 정보량 부족
- segformer mask 정확도 회귀 risk — verify 가드 (결정 4) 가 가드

## 결정 2 — 알고리즘 상세 (shades-of-gray, Minkowski p=6)

### 수식 (Finlayson & Trezzi 2004 "Shades of Gray and Colour Constancy")

각 채널 c ∈ {R, G, B} 에 대해 Minkowski p-norm:

```
k_c = ( (1/N) × Σ_x |I_c(x)|^p )^(1/p)
```

보정:

```
gain_c = k_gray / k_c     where k_gray = (k_R + k_G + k_B) / 3
I_c_corrected(x) = I_c(x) × gain_c
```

p hyperparameter:
- **p=1**: gray-world (각 채널 mean) — 단순. 의류 dominant 색 → 회색화 risk
- **p=∞**: white-patch (각 채널 max) — 의류 dominant 영향 적음. single hot pixel (반사광 / sensor noise) 에 민감
- **p=6**: Finlayson 실험 perceptual quality 최고. 둘의 절충 ★ **권장**

### 구현 (pure numpy)

```python
def shades_of_gray(image_rgb: np.ndarray, p: int = 6) -> np.ndarray:
    """frame illumination correction — Minkowski p-norm gray balance.

    image_rgb: (H, W, 3) uint8 RGB.
    p: Minkowski exponent. 1=gray-world, 6=Finlayson optimal, ∞=white-patch.
    """
    img = image_rgb.astype(np.float64) / 255.0
    # Minkowski norm per channel
    k = np.power(img, p).mean(axis=(0, 1)) ** (1.0 / p)  # (3,)
    k_gray = k.mean()
    gain = k_gray / np.clip(k, 1e-6, None)
    corrected = img * gain
    corrected = np.clip(corrected, 0.0, 1.0)
    return (corrected * 255.0).astype(np.uint8)
```

### 알고리즘 비교

| 알고리즘 | 장점 | 단점 | 채택 단계 |
|---|---|---|---|
| Gray-world (p=1) | 단순 / 빠름 | dominant 색 회색화 | Phase 2 ablation |
| White-patch (p=∞) | dominant 무관 | hot pixel 민감 | Phase 2 ablation |
| **Shades-of-gray (p=6)** | 절충 / perceptual 최고 | hyperparam | **Phase 1 권장** |
| Retinex SSR/MSR | 강력 / 명도 균등화 | 비용 ↑, contrast 과장 | enhancement (출원 후) |
| LAB a/b skew correction | 색채널만 보정 | warm/cool 동시 보정 | Phase 2 보완 후보 |

**Phase 1 = shades-of-gray p=6 단일.** Phase 2 에서 p ∈ {1, 4, 6, 8, ∞} ablation 으로 16w 백필 multi-member 670 데이터 기준 최적 p 재선정.

### 차별점 (출원 명세서)

shades-of-gray 자체는 일반적 (prior art 풍부). 본 발명 차별점:
- **garment-targeted detection trigger** (결정 3) — 일반 image 보정이 아니라 의류 frame 의 light bias 만 conditional 적용
- **closed-set anchoring (color.B) 과의 조합** — 보정된 frame 에서 KMeans cluster top-N 이 더 안정 → Pass 2 cluster_index 매칭 신뢰도 향상. 명세서 sub-claim 로 강조

## 결정 3 — detection trigger 상세

frame 별 3 metric 계산. trigger rule: **OR (1+ trigger 시 보정)**.

### (a) LAB a/b skew

- frame 전체 sRGB → LAB 변환 (D65 illuminant, standard CIE)
- 측정: `a_mean = mean(LAB[:, :, 1])`, `b_mean = mean(LAB[:, :, 2])`
- trigger: `|a_mean| > 8 OR |b_mean| > 8`
- 의미:
  - 정상 frame: a_mean / b_mean 모두 0 근처 (gray balanced)
  - |a_mean| > 8: red-green shift (red 우세 또는 green 우세)
  - |b_mean| > 8: yellow-blue shift (석양/텅스텐 = +b, 그늘/형광등 = -b)
- threshold 8 의 근거: LAB JND ≈ 1, 8 = perceptually visible shift. 실측 baseline 으로 Phase 1 에서 재조정

### (b) L extreme

- 측정: `L_mean = mean(LAB[:, :, 0])` (L ∈ [0, 100])
- trigger: `L_mean < 30 OR L_mean > 80`
- 의미:
  - L_mean < 30: 어두운 frame (실내 / 석양 / 야간) — cluster 색 신뢰도 ↓
  - L_mean > 80: overexposed (자연광 strong / flash) — 채도 손실 → washing out
  - 정상: L_mean ∈ [40, 70]
- 16w 백필 실측 분포 (5/16 codex):
  - L<30: 24.2%, L≥85: 9.2%, L∈[30,60]: 32.2%, L∈[60,85]: 34.4%

### (c) garment mask 내 channel imbalance — **Phase 1 에서 제외**

순환 의존 문제: segformer 가 보정된 frame 에 의존하는데 detection 에 segformer mask 사용 시 chicken-egg.

해결 옵션:
- (i) 원본 segformer → mask → imbalance 측정 → trigger → 보정 → segformer 재호출. **비용 2배**
- (ii) frame 전체 channel imbalance (mask 없이). 의류 외 영향
- (iii) **detection 에 (a) + (b) 만**. 의류 본연 색 보정 가드는 결정 4 verify 가 담당. **권장**

### 최종 detection rule

```python
def needs_correction(frame_lab: np.ndarray) -> bool:
    """frame illumination bias trigger — (a) + (b) OR."""
    L_mean = frame_lab[..., 0].mean()
    a_mean = frame_lab[..., 1].mean()
    b_mean = frame_lab[..., 2].mean()
    return (
        abs(a_mean) > 8.0
        or abs(b_mean) > 8.0
        or L_mean < 30.0
        or L_mean > 80.0
    )
```

### 예상 trigger 비율

16w 백필 multi-member 670 기준 추정:
- L extreme (L<30 OR L>80) — L 분포에서 24.2% + (L>80 추정 ~15%) ≈ **40%**
- a/b skew (|a|>8 OR |b|>8) — 실측 미실시, 추정 **~30%**
- 합집합 ≈ **50~60%** trigger 예상

**높은 trigger 비율의 우려**: verify 가드 (결정 4) 가 false-positive 차단. 의류 본연 색 왜곡 발생 시 원본 회수.

### Phase 1 첫 작업

threshold 8 / 30 / 80 의 실측 baseline 측정 — 16w 백필 multi-member 670 frame 의 |a_mean| / |b_mean| / L_mean 분포 보고 quantile 기반 재조정 (e.g. p90 기준).

## 결정 4 — verify 가드 상세

보정 후 의류 색이 왜곡됐는지 객관적 검증. **garment LAB median ΔE76**.

### 측정 절차

```
1. 보정된 frame → segformer → garment mask_c
2. mask_c 내 pixel 의 LAB 변환 → median LAB (M_corrected)
3. 원본 frame → segformer → garment mask_o
4. mask_o 내 pixel 의 LAB 변환 → median LAB (M_original)
5. ΔE76 = √((L_c - L_o)² + (a_c - a_o)² + (b_c - b_o)²)
```

### Threshold

- **ΔE76 ≤ 30**: 보정 채택 — 의도된 색 shift (warm → neutral 등)
- **ΔE76 > 30**: 보정 거부, **원본 frame 으로 segformer + KMeans 재진행**

### Threshold 30 의 근거

ΔE76 perceptual 의미:
- 1 = JND (just noticeable difference)
- 2~3 = small difference
- 10~20 = different color (e.g. light blue vs blue)
- **30+ = clearly different colors (perceptual category 변경)**
- 50+ = vastly different

30 = "보정 의도 (lighting shift 제거) vs 색 본질 왜곡" 경계. Phase 1 에서 50 frame 측정 후 distribution 보고 재조정 (실측 baseline).

### 비용 분석

- segformer 호출 2배 (원본 + 보정) — vision-workers 부담 ↑
- LAB 변환 = numpy O(N pixel) — negligible
- 전체 frame 처리 시간 ≈ 2배 (segformer 비용 dominant)

### 최적화 (Phase 2)

mask 자체는 보정으로 거의 안 변하는 가정 — verify 시 1 segformer mask 만 (e.g. 원본) 사용 + 보정된 pixel 도 같은 mask 로 indexing.

```python
mask_o = segformer(frame_original)  # 1 호출
median_o = lab(frame_original[mask_o]).median(axis=0)
median_c = lab(frame_corrected[mask_o]).median(axis=0)  # 같은 mask
delta_e = sqrt(((median_c - median_o) ** 2).sum())
```

전제 검증 (Phase 2): mask_o vs mask_c 의 IoU > 0.95 이면 안전. 미달 시 2회 segformer 유지.

### 대안 비교

| 대안 | 장점 | 단점 | 채택 |
|---|---|---|---|
| **(a) ΔE76 > 30 verify** | perceptual 정량 / 객관 | segformer 2배 | **권장** |
| (b) family disagreement only | 단순 | 가짜 같은 family (e.g. earth ↔ neutral 50.9%) 차단 불가 | reject |
| (c) verify 없이 강제 보정 | 빠름 / segformer 1회 | false-positive risk ↑ | Phase 2 ablation |
| (d) ΔE76 + family 조합 | 강력 | 복잡도 ↑ | enhancement |

## 결정 5 — default flag 상세

### configs/local.yaml 새 섹션

```yaml
vision:
  ...
  hybrid_palette:
    ...
    color_pick_v010_enabled: false      # color.B
  illumination_correction:               # color.C 신규
    enabled: false                        # default off, canary 후 enable 결정
    minkowski_p: 6                        # shades-of-gray exponent
    detection:
      a_skew_threshold: 8.0               # |LAB a_mean| > 8 trigger
      b_skew_threshold: 8.0               # |LAB b_mean| > 8 trigger
      l_low_threshold: 30.0               # L_mean < 30 trigger (어두운)
      l_high_threshold: 80.0              # L_mean > 80 trigger (밝은)
    verify:
      enabled: true                       # verify 가드 on/off
      deltae76_threshold: 30.0            # 보정 후 ΔE76 > 30 시 원본
```

### 새 Settings 클래스

```python
# src/settings.py
class IlluminationCorrectionDetectionConfig(BaseModel):
    a_skew_threshold: float = 8.0
    b_skew_threshold: float = 8.0
    l_low_threshold: float = 30.0
    l_high_threshold: float = 80.0


class IlluminationCorrectionVerifyConfig(BaseModel):
    enabled: bool = True
    deltae76_threshold: float = 30.0


class IlluminationCorrectionConfig(BaseModel):
    enabled: bool = False                       # default off
    minkowski_p: int = 6
    detection: IlluminationCorrectionDetectionConfig = ...
    verify: IlluminationCorrectionVerifyConfig = ...


class VisionConfig(BaseModel):
    ...
    illumination_correction: IlluminationCorrectionConfig = ...
```

### default off 의 이유 (color.B 패턴)

1. **회귀 risk 0** — main 머지 후 운영 (16w 백필 / cron) 동작 변경 0
2. **canary 측정 후 enable 결정** — Phase 3 결과 보고 enable. 효과 미흡 시 enhancement embodiment 로 출원 후 추가
3. **A/B 비교 가능** — 같은 운영 환경에서 enabled=true/false 토글로 ablation

### enable 기준 (Phase 3 후 의사결정)

다음 조건 모두 충족 시 enable 권장:

- **family disagreement rate**: 50.9% (baseline) → **< 30%** (목표)
- **ΔE76 분산 median**: 17.7 (baseline) → **< 10**
- **verify 거부율**: < 20% (의류 색 왜곡 우려 frame 비율)
- **회귀 cliff (cut_off=1.0 신규 발생)**: 0 (보정으로 새 cliff 만들면 안 됨)
- **운영 시간 증가**: < 30% (segformer verify 2배 영향 제한)

미달 시:
- p tuning (Phase 2 ablation 으로 p ∈ {1, 4, 6, 8, ∞} 비교)
- threshold 재조정 (실측 baseline 기반)
- 출원 후 enhancement embodiment 로 연기

### env override

PoC 환경별 yaml override 또는 env var:
```
ILLUMINATION_CORRECTION_ENABLED=true uv run python src/pipelines/run_daily_pipeline.py ...
```

env var 지원은 settings.py 의 Pydantic Settings env binding 활용 (현재 구조 그대로).

## 미해결 결정 5건 요약 (user 합의 필요)

| # | 결정 | 권장값 | 상세 절 |
|---|---|---|---|
| 1 | 적용 시점 | frame 단계 (segformer 전) ✅ confirmed | 결정 1 |
| 2 | 알고리즘 | shades-of-gray Minkowski p=6 | 결정 2 |
| 3 | detection trigger | (a) a/b skew \|·\|>8 OR (b) L<30 OR L>80. 채널 imbalance 제외 | 결정 3 |
| 4 | verify 가드 | 보정 후 garment LAB median ΔE76 > 30 시 원본 사용 | 결정 4 |
| 5 | default flag | configs/local.yaml `illumination_correction.enabled=false` | 결정 5 |

## 비용 / 리스크

### 비용
- Phase 1 단계 추가 함수 호출만 (gray-world 는 O(N) 단순 stat). pixel 단위 N=mask pixel count. negligible.
- Gemini 호출 0 (frame 단계 보정은 LLM 무관)
- Phase 1 canary: 16w 백필의 multi-member 670 또는 sample 100 만 재실행 — Pass 1 cache hit 다수 + Pass 2 cache hit 도 가능 (image_bytes 그대로면 같은 cache key) → 측정 비용 거의 0
  - 다만 frame illumination correction 이 image_bytes 자체를 변경한다면 cache key 변경 → cache miss. canary 비용 ~$0.2

### 리스크
- **의류 본연 색 왜곡** ★ — 가장 큰 리스크. verify 가드 + canary 측정 필수
- **YT video frame 추출 비결정성** — 5/15 canary 의 6 unmatched canonical 처럼 frame 선정이 변동. 보정 효과 측정 시 같은 frame 사용 보장 필요 (frame_source 의 seed 고정 확인)
- **회귀** — color.B v0.10 의 KMeans cluster top-N 이 보정으로 변경되어 Pass 2 의 cluster_index 매핑 변경. 두 path (v0.9 / v0.10) 모두 보정 영향 — canary 시 동시 측정 필요
- **출원 청구 범위** — illumination correction 자체는 일반적 (prior art 풍부). 본 발명 차별점은 **garment mask 단위 detection trigger** + **closed-set anchoring 과의 조합** (Pass 2 cluster_index 의 robustness 향상). 명세서에서 차별점 강조

## Reference

- 진단: 5/16 codex-data-cruncher 분석 결과 (위 metric)
- spec template: `docs/color_b_prompt_v010_spec.md` (5/12, color.B 패턴)
- 5/8 트랙 A 결정: `memory/project_patent_track_color_2026_05_08`
- 5/16 세션: `memory/project_color_b_v010_session_2026_05_16`
