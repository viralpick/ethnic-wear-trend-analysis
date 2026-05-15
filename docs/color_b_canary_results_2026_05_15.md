# color.B v0.10 canary results — 2026-05-15

특허 트랙 A 의 1순위 sub-claim — **KMeans-anchored VLM color pick (2-pass Gemini)** —
의 효과를 16w 백필 cut_off=1.0 cliff 의 40 명시 post 에 대해 측정한 결과.

## Methodology

- **Baseline (v0.9)**: `outputs/backfill_16w/page_*_enriched.json` (5/2 16w 백필).
  filter: `canonical.cut_off_share >= 1.0` → 44 canonical (40 unique source_post_id).
  IG 36 + YT 4.
- **Treatment (v0.10)**: `configs/local.yaml` 의 `color_pick_v010_enabled=true` 임시 override.
  `run_daily_pipeline.py --source-post-ids tests/fixtures/canary_v010_post_ids.txt
  --source starrocks --color-extractor pipeline_b --vision-llm gemini --llm openai
  --sink dry_run --date 2026-05-15`
- **Join**: `(source_post_id, canonical_index)` 단위 before ↔ after.

## Results

### 회복률 (cut_off=1.0 cliff)

| Bucket | v0.9 (before) | v0.10 (after) |
|---|---|---|
| Total (matched) | 34 / 40 | 34 / 40 |
| cut_off = 0.0 (완전 회복) | 0 (0%) | **8 (23.5%)** |
| 0 < cut_off < 1.0 (partial) | 0 (0%) | **26 (76.5%)** |
| cut_off = 1.0 (회복 실패) | 34 (100%) | **0 (0%)** |
| palette length | 0 (모두) | 1 ~ 3 |
| **회복률 = (cut_off < 1.0) / matched** | **0%** | **100%** |

비매칭 6 canonical = YT 4 + IG 2 — v0.10 재실행 시 canonical_index 가 변동 (dedup 순서 / YT video frame 선정 비결정성). recovery 측정의 부정 사례 아님 (단순 매칭 키 mismatch). **Backlog 검토**.

### v0.10 stability

| Indicator | Count |
|---|---|
| Pass 2 fallback (`color_pick_v010_fallback` warning) | **0건** |
| Pass 2 schema violation (`gemini_v010_schema_violation`) | **0건** |
| Pass 2 ValidationError (cluster_index range / duplicate) | **0건** |
| Pass 2 Gemini calls (`gemini_v010_usage`) | 129 |
| Pass 1 Gemini calls (`gemini_usage`, cache miss) | 4 |

Pass 1 의 cache hit 가 대부분이라 추가 Gemini 호출 ≈ Pass 2 의 129 만.

### color.A (5/12) vs color.B (v0.10)

| 정책 | 회복률 (cut_off=1.0 → < 1.0) | 비고 |
|---|---|---|
| **color.A** (etc fallback) | 29% lower-bound | KMeans top-N 의 share/chroma 통과 시 보존. fallback 무효 (cut_off=0.0 → 0.0) 71%. |
| **color.B v0.10** | **100%** | cluster_index 직접 anchor. ΔE76=0 매칭 실패 차단. |

## Sub-claim 핵심

v0.10 의 closed-set anchoring 작동 기제:

1. **KMeans cluster top-N (N=5) 추출** — segformer mask 안의 pixel evidence 가 정의된 색만 후보.
2. **Pass 2 Gemini 호출** — image + 의류 분류 (Pass 1 결과) + cluster top-N hex/share/index 직렬화 입력. dominant 색의 `cluster_index` 1~3 pick (closed-set).
3. **R1 매칭 우회** — cluster_index 가 KMeans 결과 안의 직접 index 라 ΔE76=0 anchor.
4. **R3 (환각 drop) 자동 차단** — VLM 이 KMeans cluster 외 색 도입 불가능.

기존 (v0.9) 의 free-form preset pick 패턴은 50색 preset 의 hex 와 KMeans cluster hex 사이 ΔE76 가 필연적으로 > 0 → R1 매칭 임계 (28) 초과 시 cluster 가 etc bucket 으로 흡수 → palette=[].

5/12 사례 2 (`01KQ43YDK0NZKEH1Q10C34VNZM`, Gemini 가 cream_ivory + rani_pink 픽했으나 KMeans cluster 와 ΔE76 모두 > 28) 도 v0.10 후 회복: Pass 2 가 cluster top-N 안에서 pick.

## 비용

- Pass 1 (`extract_garment`): cache hit 다수 — 운영 비용 거의 0.
- Pass 2 (`pick_colors_from_kmeans`): 40 post × ~3.2 image/post ≈ 129 call.
  Gemini 2.5 Flash baseline ~$0.0009/call → 약 $0.12 (40 post 한정).
  full 16w 백필 (3,421 canonical) 환산 시 약 ~$3 / 16주 = ₩4,500. cache 통합 시 60~80% 절감 예상.
- spec 의 "월 ₩90k" 수용 결정 보다 훨씬 낮음 (canary 40 post 만 측정한 한계로 보수적 추정).

## 백로그 (canary 후 추가)

1. **Pass 2 cache 통합** — `compute_cache_key` 확장 (image + prompt_version + model_id + cluster_top_n_hex). full 16w 백필 재실행 비용 절감.
2. **fallback rate telemetry counter** — 현 `logger.warning` 만. `color_pick_v010_fallback_total` counter 추가.
3. **6 unmatched canonical 분석** — YT video frame 선정 비결정성 / dedup 순서 차이.
4. **timeout / rate-limit 처리** — Pass 2 Gemini SDK transient 분기 추가 (canary 빈도 측정 후).
5. **시각 정상성 인간 검수** — recovery 후 palette 가 garment 색을 정확히 반영하는지 18~40 사례 캐나리 HTML 검수.

## 다음 단계

- PR #80 머지 결정 (user). default off 라 머지 안전.
- 머지 후 PoC 환경에서 v0.10 enabled 운영 + full 16w 백필 재실행 (cache 통합 우선).
- prior art 조사 (출원 전 필수).
- 출원 자료 정리 (이 문서 + 5/12 spec + 코드 reference).

## Reference

- spec: `docs/color_b_prompt_v010_spec.md` (5/12)
- 5/12 진입점: `memory/project_session_state_2026_05_12.md`
- 5/14 구현: `memory/project_color_b_v010_impl_2026_05_14.md`
- canary fixture: `tests/fixtures/canary_v010_post_ids.txt` (40 post id)
- canary 결과: `outputs/canary_v010_after_enriched.json` / `outputs/canary_v010_before_after.tsv` (gitignored)
- PR: #80 `feat/color-b-v010-prompt` (commit `0f518c4` + `dc35a4a`)
