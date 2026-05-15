# color.B — prompt v0.10 spec (KMeans-anchored VLM color pick, 2-pass)

특허 트랙 A 의 1순위 sub-claim. VLM 의 free-form color pick 을 KMeans cluster 결과 안으로 closed-set anchoring 해 VLM 환각 차단 + R1 ΔE76 매칭률 향상.

## 배경 (5/12 기준)

- 16w 백필 진단: etc cliff 152 canonical (4.4%) 중 VLM picks 매칭 실패 = 152 (동일 현상의 양면)
- 사례 2 `01KQ43Y...` 에서 Gemini 가 `cream_ivory + rani_pink` 픽 → KMeans cluster 와 ΔE76 매칭 실패 → 모든 cluster etc bucket 흡수 → `palette=[]`
- color.A (fallback) 이 4.4% 중 29% (44건) 만 회복. 71% 는 fallback 무효 (cut_off=0.0)
- color.B 는 R1 매칭 자체를 성공시켜 etc cliff 의 다수를 진입 단계에서 차단 목표

## 구조 결정 (옵션 X, 5/12 user 결정)

**2-pass Gemini**:
- **Pass 1** (기존 그대로): image + 50색 preset → 의류 분류 + initial `color_preset_picks_top3`
- **segformer + KMeans** (기존 그대로): Pass 1 분류 결과 → 의류 픽셀 → KMeans cluster (hex + share + lab)
- **Pass 2** (신규): image + 의류 분류 + KMeans cluster hex top-N → KMeans cluster index 중 dominant 색 1~3 pick

## Pass 2 설계

### 입력 (LLM prompt 에 포함)

- 원본 image (또는 person bbox crop)
- 의류 분류 (Pass 1 결과 — upper/lower garment_type, ethnic flag)
- KMeans cluster top-N (N=5 권장):
  - 각 cluster: `{index, hex, share}` (lab 은 LLM 가 hex 로 추론 가능)
  - share desc 정렬, 의류 영역 mask 기반 (현재 `extract_dynamic_palette` 결과)

### 출력 (JSON schema)

```json
{
  "picks": [
    {
      "cluster_index": 0,
      "preset_label": "bottle_green"
    },
    ...
  ]
}
```

- `cluster_index`: 0-based, 입력 KMeans cluster top-N 안에서. **반드시 입력 범위 안의 값** (closed-set)
- `preset_label`: 그 cluster hex 에 가장 가까운 50색 preset 라벨 (dedup / family resolve 용도 — `feedback_gemini_color_dedup_only` 정책 유지)
- `picks` 길이 1~3, garment 가 실제로 단색이면 1, 복합색이면 2~3

### 규칙

- VLM 은 KMeans cluster index 안에서만 pick. 다른 색 introduce 금지
- 의류 dominant 색만 — 배경 / 피부 / 마네킹 영역 제외 (Pass 1 의 의류 분류 + segformer mask 가 이미 거른 결과지만, LLM 이 image 보고 한 번 더 검증)
- 1픽 정책: 진짜 단색이면 1픽 (현재 v0.7~v0.9 정책 유지)

## 기존 R1 매칭 path 변경

현재 `hybrid_palette.build_object_palette`:
```
KMeans cluster + VLM picks → R1 1:1 매칭 (ΔE76 ≤ drop_threshold)
```

v0.10 적용 후:
```
KMeans cluster + Pass 2 pick (cluster_index) → R1 매칭 (cluster_index 직접 anchor, ΔE76 = 0)
```

cluster_index 가 KMeans 결과 안의 직접 index 라 ΔE76 매칭 없이 anchor 확정. 즉 R1 매칭 실패 시나리오 자체가 거의 사라짐 (input 검증 실패 시만).

## 기존 코드 변경

### 1. `src/vision/prompts.py`
- 새 함수 `build_color_pick_v010_prompt(garment_classification, kmeans_clusters)` — Pass 2 prompt 생성
- Pass 1 prompt 는 그대로 유지 (호환성)

### 2. `src/vision/llm_client.py`
- 새 메서드 `VisionLLMClient.pick_colors_from_kmeans(image, classification, kmeans_clusters) -> list[KMeansAnchoredPick]`
- `FakeLLMClient` 도 결정론적 fake 구현 (cluster index 0 → preset_label "test_color" 같은 stub)

### 3. `src/vision/pipeline_b_adapter.py`
- `build_object_palette` 호출 직전에 Pass 2 호출
- Pass 2 결과로 `picks` 갱신 (KMeans cluster index → 그 cluster 의 hex 또는 직접 anchor 마킹)

### 4. `src/contracts/vision.py`
- 새 dataclass `KMeansAnchoredPick(cluster_index: int, preset_label: str)`
- 기존 `EthnicOutfit.color_preset_picks_top3` 호환성 유지 — Pass 2 결과의 preset_label list 로 채움

### 5. `configs/local.yaml`
- 새 옵션 `color_pick_v010_enabled: bool = false` (default off, canary 측정 후 on)
- `color_pick_v010_kmeans_top_n: int = 5` (Pass 2 에 전달할 cluster 수)

## canary 측정 계획

### 대상
앞서 codex-data-cruncher 가 선정한 18 post (cut_off=1.0 인 44건에서 다양성 선정)

### measurement metric

| metric | before (v0.9) | after (v0.10) | 의미 |
|---|---|---|---|
| etc cliff (palette=[]) | 152 | X | 4.4% → ?% |
| VLM picks R1 매칭률 | 95.6% | ~100% expected | 직접 매칭 |
| 시각 정상성 (인간 검수) | baseline | X% 일치 | KMeans hex 의 시각 정확성 |
| cluster palette 오염 | 0건 baseline | X | Pass 2 환각 risk |
| Gemini 비용 | $X | $2X | Pass 2 호출 분 |

### 회귀 가드

- 단위 test: Pass 2 stub 동작 / cluster_index 범위 검증 / fallback (Pass 2 실패 시 Pass 1 picks 그대로 사용)
- snapshot test: 기존 canary 결과와 비교, 의도된 변경만

## 미해결 결정 (사용자 승인 필요)

1. **Pass 2 fallback**: Pass 2 가 invalid cluster_index 반환 시 → (a) Pass 1 picks 그대로 / (b) color.A fallback path 진입 / (c) error raise. 권장 (a)
2. **kmeans_top_n=5 적정성**: 너무 적으면 진짜 색 빠짐, 너무 많으면 환각 risk. 5 가 default 적정?
3. **dedup 정책 유지**: `feedback_gemini_color_dedup_only` 와 모순 없도록 preset_label 만 dedup 사용 (안전)
4. **Pass 1 picks 처리**: Pass 1 의 picks 는 그대로 두고 Pass 2 가 추가 — 또는 Pass 1 의 picks 제거하고 Pass 2 만. 권장: Pass 1 picks 는 enriched 에 보존 (debug 용), 실 사용은 Pass 2 결과만
5. **비용 효과**: Gemini 호출 2배 = 월 ₩90k. 출원 단계 수용 가능. 출원 후 단일 call 통합 (옵션 Y) 또는 후처리 강화 (옵션 Z) 로 비용 절감 후속 결정

## 구현 단계 (합의 후)

1. **spec ack** — 위 5개 미해결 결정 user 합의 (오늘)
2. **prompts.py + contracts/vision.py 추가** (1일)
3. **llm_client.py 새 메서드 + FakeLLMClient stub** (0.5일)
4. **pipeline_b_adapter.py 통합 + 단위 test** (1~2일)
5. **vision extras 설치 + canary 18 post 재실행** (0.5~1일)
6. **before/after 비교 + 명세서 자료** (0.5일)

총 3.5~5일 예상.
