# Roadmap — ethnic-wear-trend-analysis

**최근 업데이트**: 2026-04-21

이 문서는 분석 레포의 **완료 / 진행중 / 후보** 작업을 한곳에 정리한다. spec (`docs/spec.md`)
이 "5/4 데모 까지의 범위" 라면, 이 문서는 "5/4 이후에 무엇을 더 할 수 있는가" 를 포함한다.

---

## 완료 (M1 ~ M2)

### M1 — 2026-04-21 오전: 속성 추출 baseline

- Python 3.14 + uv + hatchling + ruff + pytest 셋업
- contracts/ (Pydantic v2, StrEnum 14종 + TrendClusterSummary/ScoreBreakdown/DrilldownPayload)
- attributes/ (rule 매핑 + FakeLLM + unknown 해시태그 트래커)
- clustering/, normalization/, loaders/, utils/
- `uv run attributes` CLI — 속성 추출 + 클러스터 배정만 (스코어링 전)

### M2 — 2026-04-21 오후: 스코어링 skeleton + color Pipeline B 이식

- scoring/ (minmax_same_run, direction ±5% + early rules, 4 sub-scores)
- aggregation/ (LAB KMeans palette)
- vision/ (color_space — 동료 PoC 의 color_utils 전량 이식)
- exporters/ (atomic JSON writer, backend payload placeholder)
- pipelines/ (scoring, daily CLI)
- 113 unit + snapshot 테스트 / ruff clean
- 데이터 파이프라인 상/하류 합의: Azure Blob → 분석 → StarRocks + BE API

---

## 진행중 (M3) — 4/24 ~ 5/1

4/24 싱크 결과 기반 실 구현. 자세한 agenda 는 `docs/sync_2026-04-24_agenda.md`.

### M3.A — Pipeline B 완성 (vision 실 구현)

- **Step C (진행중)**: `vision/frame_source.py` + `vision/pipeline_b_extractor.py`
  - `FrameSource` Protocol + `ImageFrameSource` + `VideoFrameSource` (stub → 실 ffmpeg)
  - YOLO + segformer + drop_skin + extract_colors 통합
  - `[vision]` optional extras (torch, transformers, ultralytics, pillow, opencv)
- **Step D**: `VLMClient` → `ColorExtractor` 리네임. core 파이프라인이 Protocol 뒤의 실
  구현 (PipelineBExtractor) 을 호출하게 DI
- **Step E**: 20 JPG 로 end-to-end smoke + comparison HTML (PoC 의 `comparison.html` 스타일)

### M3.B — 상/하류 인프라 연결

- `loaders/blob_raw_loader.py` — Azure Blob SDK, 페이지네이션, SAS 인증
- `loaders/tsv_raw_loader.py` — 크롤러가 전달한 TSV (`png_india_ai_fashion_*.tsv`) 를
  raw contract 로 변환 (로컬 개발용)
- `exporters/starrocks_writer.py` — Stream Load 또는 INSERT
- `exporters/backend_poster.py` — BE API 호출 (최소한의 backoff)
- pipeline CLI 에 `--source blob|local|tsv` / `--sink starrocks|file|backend` 토글

### M3.C — 스코어링 값 실제화 (현재 skeleton 은 0 seed)

- `momentum_post_growth` / `momentum_hashtag_velocity` / `momentum_new_account_ratio`
  를 전일/전주 summaries.json 로부터 계산 (history loader 필요)
- `cultural_festival_match` 의 Akshaya Tritiya 윈도우 매칭 실제 작동
- `cultural_bollywood_presence` — IG source_type=bollywood_decode post 주입
- daily/weekly direction baseline 실 비교 (현재 0 이라 전부 FLAT)
- `influencer_tier` 가중치 실제 반영 — raw IG `account_followers` → ClusterScoringContext

### M3.D — 4/24 싱크 결정 반영 (외부 팀 합의 후)

- `contracts/raw.py` 를 크롤러 팀 TSV 스키마에 맞춰 조정
- `BackendScoreboardPayload` 를 백엔드 팀 endpoint 에 맞춰 확정
- StarRocks 테이블 스키마 확정 (단일 vs 분리, schema_version 컬럼, 재처리 멱등성)

---

## M4 — 5/1 이후 후보 (데모 영상 완료 후)

spec §7.4 "VLM 으로 안 하는 것" + 기타 확장 범위. **지금은 하지 말 것** — 데모 우선.

### M4.A — YouTube 색상 추출

**현재 상태**: spec §7.2 / §4.1 ④ 에서 명시적 배제.
**이유 (spec 원문)**: "한 영상에 여러 옷이 나와서 프레임별 추출 비현실적"

**M4 에서 재고할 이유**:
- 데모 후 production 으로 갈 때 YT 가 보조 시그널이 되면 값어치 있음
- Pipeline B 가 pixel-level 결정론 이라 frame 수만 많을 뿐 원리는 같음
- 인도 리테일러가 YT haul 영상을 "스타일 브리프 입력" 으로 쓴다면 색 분포가 중요

**활성화 조건**:
- IG 기반 demo 가 business value 입증 (5/8 이후)
- YT 프레임당 garment count 분포 측정 결과, "평균 ≤ 2 garment" 이면 aggregate 의미 있음
  (실측 필요)
- `VideoFrameSource` 가 ffmpeg 연결 완료

**예상 작업**:
1. `VideoFrameSource` 의 fps / sampling strategy 를 clip-level 메타 기반으로 (scene change
   detect 등) — naive 1fps 는 noise 많음
2. frame 당 **모든 person × 모든 garment** 의 color 를 aggregate — IG 는 보통 1인이지만 YT
   는 N인. `person_id` 트래킹 (ByteTrack 등) 이 필요해짐
3. video 단위 "dominant garment color over time" 메트릭 — 한 영상의 대표 색이 시점에 따라
   변하는 걸 어떻게 단일 `ColorPaletteItem` 으로 요약할지 (구간 평균 vs 최빈 vs 길이 가중)
4. 비용 — ffmpeg frame extract + Pipeline B 를 video 하나당 수백 frame 돌리면 GPU 시간 큼.
   영상당 samples = N_{keyframes} 정도로 줄여야 함
5. YouTube 에서 **영상 속 상품 tagging** (Shoppable moment) 라이선스 고려 — 제3자 콘텐츠

**트레이드오프**:
- 장점: 모션/스타일링 맥락 데이터 확보 (IG 정적 이미지 대비 풍부)
- 단점: frame cost 가 O(영상 길이) 이고 garment 분리가 본질적으로 어려움

### M4.B — IG Reel 색상 추출

**현재 상태**: sample_data 의 `instagram_sample_list.txt` 에 4개 Reel URL 이 있으나 blob
SAS 가 placeholder 라 실 download 불가.

**M4 에서 할 것**:
- `VideoFrameSource` 를 Reel 에 먼저 적용 (YT 보다 짧고 1인 중심이라 더 쉬움)
- IG 이미지 포스트 (주력) + Reel (보조) 을 같은 cluster 에서 합치는 aggregate 설계
- `posting.tsv` row 10 처럼 image 0장인 post 를 "video 포스트" 로 식별하는 스키마 확정 필요

**frame sampling + quality filter** (phase 3 instance palette 와 연계):

*sampling 전략* (기본: 고정 frame count, uniform):
```yaml
vision:
  video_sample:
    fps: null              # null 이면 frame_count 우선
    frame_count: 20        # 영상 길이 무관 고정 샘플
    max_duration_sec: 300  # 5분 초과 영상은 앞부분만
```

장점: 영상 길이 무관 비용 일정, 결정론. 이후 필요하면 scene change detection
(`ffmpeg scdet` 또는 `cv2.absdiff`) 으로 outfit 전환 시점 기반 샘플링으로 upgrade.

*frame quality filter* (저품질 drop):

| 지표 | 계산 | drop 조건 | 라이브러리 |
|---|---|---|---|
| **motion blur** | `cv2.Laplacian(gray, CV_64F).var()` | < `blur_min` (예: 100) | opencv (vision extras) |
| **exposure** | `np.mean(gray)` + 극단값 비율 | mean < 30 or > 225, 또는 극단값 > 70% | numpy |
| **garment presence** | YOLO + segformer pre-check | WEAR_KEEP pixel < `min_pixels` | 기존 Pipeline B 재활용 |

```yaml
vision:
  frame_quality:
    blur_min: 100.0
    exposure_mean_min: 30.0
    exposure_mean_max: 225.0
    saturation_ratio_max: 0.7
```

프레임당 ms 단위라 오버헤드 없음. 20 frame 샘플 중 보통 3~5 장 drop 예상 (인도 IG Reel
셀카 조명 편차 고려).

`FramePalette` 에 quality 메타데이터 추가:
```python
@dataclass(frozen=True)
class FramePalette:
    ...existing...
    blur_score: float
    exposure_ok: bool
    quality_filtered: bool   # drop 처리됐나
```

**활성화 조건**: M4.A 보다 먼저 (더 쉬운 선행 검증). M3.B blob_raw_loader 가 Reel 경로
확보 완료 후.

### M4.C — YouTube 영상 ASR + 텍스트 attribute 보강

**현재 상태**: spec §7.4 "영상 프레임별 분석 / ASR → PoC에서 시도".

**M4 에서 할 것**:
- 영상 음성 → 텍스트 → 기존 rule/LLM 속성 추출 파이프라인 재사용
- "이 쿠르타는 linen 소재입니다" 같은 영상 내 구두 설명이 caption 에 없을 수 있음

**트레이드오프**:
- 장점: title + description 에 없는 attribute 확보
- 단점: ASR 비용 + 인도 악센트 영어/힌디어 혼용 인식 품질 미지수

### M4.D — 세분 속성 (넥라인 / 소매 / 기장)

**현재 상태**: spec §7.4 "넥라인/소매/기장 같은 세부 속성 → PoC에서 시도".

**M4 에서 할 것**:
- Fashionpedia 46-class 같은 세분 segformer fine-tune
- 또는 VLM 을 "label 보조" 로만 활용 (hybrid — color 는 Pipeline B, label 은 VLM)
- `contracts/common.py` 의 `Silhouette` enum 을 확장해서 sleeve_length / neckline 추가 enum

### M4.E — 인도 직물 세부 분류

**현재 상태**: spec §7.4 "치칸카리와 럭나우식 치칸카리의 차이 → R&D 영역. 약속하면 안 됨".

**M4 에서 할 것**: 보수적. 2-tier 분류 (coarse / fine) 구조만 준비. 실 데이터로 학습은
도메인 전문가 합의 후.

### M4.F — 학습 기반 랭커 / re-ranking

**현재 상태**: 우리 mandate 로 v1 금지 ("explainable from raw counts").

**M4 에서 재고할 이유**: business user 가 "급상승 여부" 를 설명력보다 정확도로 원하면.

**조건**: 설명력을 **잃지 않는** 구조로만. 예: 스코어링 공식은 유지하고, "notable change
detection" 만 학습 기반으로 추가하는 hybrid.

### M4.H — Palette cluster → spatial 역매핑 overlay

**현재 상태**: phase 2 에서 kept/dropped 2색 overlay 는 구현 (quality/phase-2 브랜치).
다만 "palette chip 이 이미지의 어느 영역에서 왔는지" 는 아직 안 보임.

**M4 에서 할 것**:
- post-level KMeans 결과를 post 전체 pixel 로 label assignment
- 각 cluster 에 속한 pixel 을 해당 cluster hex 색으로 덧칠 (heat-map 아니라 실 색 overlay)
- 드릴다운에서 "chip #B8D4C3 (sage 32%) 는 이 post 의 셔츠 오른쪽 소매에서 나왔다" 같은
  정보가 시각적으로 매핑

**기술 난이도**:
- 현재 구조는 모든 frame pixel 을 concat 해서 KMeans 돌리므로 "어느 pixel 이 어느 cluster"
  정보를 유지하려면 pixel 좌표 (frame_id, y, x) 를 원래 순서 그대로 tracking 해야 함
- frame 별 cluster label 을 배열로 받아 frame rgb 에 역매핑 (palette_mask per frame)
- `extract_palette_with_diagnostics` 확장 또는 새 함수 `extract_palette_with_spatial_labels`

**활용 시나리오**:
- "왜 이 chip 이 나왔나" 즉각 확인 (소품 오분류 / 배경 침입 진단)
- 캐러셀에서 "이 frame 은 어느 cluster 에 많이 기여했나" 정량 측정
- drill-down 화면에서 chip 클릭 시 대표 post 의 overlay 보여주기

**선행 조건**: Phase 2 검증 완료 후 (131 post smoke 결과 보고 개선 방향 확정).

### M4.I — Demographic filter (여성 × 성인만 타겟)

**현재 상태**: Pipeline B 가 detected person 전원을 처리. 인도 ethnic wear 여성 타겟
데모에서 남성·아동 포스트는 시그널이 되지 않고 오히려 오염 소스.

**Primary 접근 (다음 PR 에서 구현)**: CLIP zero-shot prompt 기반.
- scene + gender + age 대역을 단일 모델 (CLIP ViT-B/32) forward 1회로 판정
- image-level pre-filter — YOLO/segformer 돌기 전 drop → 비용 절감
- `[vision]` extras 에 CLIP 만 추가 (~600MB). 현재 `transformers` 이미 의존
- prompts:
  - scene: `["a person wearing clothing", "a statue", "a product-only shot", "a landscape"]`
  - gender: `["a woman", "a man"]`
  - age: `["a child", "an adult"]`
- 판정: argmax + confidence threshold. 세 판정 모두 통과한 이미지만 Pipeline B 돌림
- drop 사유는 HTML 에 노출 (수동 검증 + bias 감사)

**업그레이드 후보 (CLIP 정확도가 부족하면)**: `insightface` (buffalo_s pack).
- face detection + age (숫자) + gender 전용 학습 모델 — 정확도 CLIP 대비 높음 (~95%)
- 단점: 얼굴 안 보이는 샷은 판정 불가 (CLIP 은 전체 이미지로 판정 가능), 추가 ~300MB
- `[demographics]` optional extra 로 분리
- 4/24 싱크 후 실 데이터에서 CLIP 오분류율 측정 → 10% 초과시 전환 검토

**공통 위험**:
- 인도인 female/male / 전통 복식 (kurta 남성, shalwar kameez) bias — 서양 중심 학습 데이터
- adult/child 경계 (특히 10대 후반) 가 모호
- false negative 로 demo 샘플 축소 리스크 → drop 투명성 설계 (HTML 노출) 필수

**활성화 조건**: 이번 PR (phase 3) 머지 직후. 4/24 싱크 이전 완성 목표.

### M4.G — Semantic similarity / embedding

**현재 상태**: contract v2 후보 (per-attribute confidence / evidence span 와 함께 미룸).

**M4 에서 재고할 이유**: 해시태그 매핑이 커버 못 하는 긴 꼬리 표현 ("summer office look
in breathable fabric") 을 jaccard 가 아닌 embedding 으로 매칭.

---

## 영구 비스코프 (우리 레포가 **절대 하지 않을 것**)

- 백엔드 API 서버 구현 → 백엔드 팀 레포
- 크롤러 구현 → 크롤러 팀 레포
- 프론트엔드 UI → 프론트 팀 레포 (또는 COS)
- DB 스키마 관리 → DW 팀
- 모델 학습 인프라 (train / MLOps) — 이 레포는 inference 전용
- 실시간 스트리밍 / 메시지 큐 consumer — spec §10 이 batch 전제

---

## 우선순위 매트릭스 (M4 candidate)

| 후보 | 효과 | 비용 | 선행 조건 | 제안 순서 |
|---|---|---|---|---|
| M4.B IG Reel 색 추출 | 중 | 중 (VideoFrameSource 필요) | M3.A 완료 | 1 |
| M4.A YouTube 색 추출 | 중-고 | 고 (ByteTrack + scene detect) | M4.B 검증 | 2 |
| M4.C YT ASR | 중 | 중 (ASR API 비용) | M4.A 또는 독립 | 3 |
| M4.D 세분 속성 | 고 | 고 (세분 라벨 데이터 확보) | M3.A 완료 | 4 |
| M4.F 학습 랭커 | 저 (v1) | 중 | business 요구 있을 때 | 5 |
| M4.G Embedding | 저-중 | 저 | 긴 꼬리 표현 실측 확인 | 6 |
| M4.H Palette cluster 역매핑 | 중 | 중 | Phase 2 검증 완료 | 4 |
| M4.E 인도 직물 세분 | 저 (R&D) | 고 | 도메인 전문가 합의 | 7 |

**Rule of thumb**: M4 는 data-driven 으로 우선순위 조정. 실 사용에서 빈번히 실패하는 지점을
먼저 막는다 ("사용자가 wow 라 하는 기능" 보다 "사용자가 trust 를 잃는 실패 지점" 우선).
