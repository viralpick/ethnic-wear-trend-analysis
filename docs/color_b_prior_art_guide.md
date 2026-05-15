# color.B v0.10 prior art 조사 가이드

특허 트랙 A 의 1순위 sub-claim (**KMeans-anchored VLM color pick, 2-pass Gemini**) 의 출원 전 prior art 사전 조사. 출원 거절 / 진보성 결여 / 신규성 결여 리스크 사전 차단.

## 핵심 청구항 (search 대상)

1. **VLM (vision-language model) + KMeans cluster anchoring** 결합 — VLM 이 KMeans cluster index 안에서만 색 pick (closed-set anchoring)
2. **2-pass VLM**: Pass 1 = 의류 분류 + free-form 색 pick / Pass 2 = cluster_index 기반 dominant 색 pick
3. **ΔE76 = 0 매칭** — cluster_index 가 KMeans 결과 직접 index 라 매칭 실패 자체 차단
4. **garment-region pixel evidence 기반 closed-set** — KMeans 가 segformer mask 안의 pixel 만 cluster

## 검색 키워드 (3 layer 권장)

### Layer 1 — 정공 (가장 가까운 prior art)

| 키워드 | 의도 |
|---|---|
| `"VLM" AND "color" AND "KMeans" AND "anchoring"` | 직격 |
| `"vision-language model" AND "palette" AND "cluster"` | 일반화 |
| `"GPT-4V" OR "Gemini" AND "color extraction" AND "garment"` | 모델 특정 |
| `"closed-set" AND "color" AND "VLM"` | 핵심 메커니즘 |
| `"two-pass" AND "VLM" AND "color"` | 구조 |

### Layer 2 — 인접 분야 (VLM 일반 + 색상 추출)

| 키워드 | 의도 |
|---|---|
| `"VLM" AND "structured output" AND "image"` | output schema 측 |
| `"image segmentation" AND "color clustering"` | 픽셀 기반 색 추출 |
| `"fashion" AND "VLM" AND "attribute"` | 도메인 (인도 ethnic 의류 분류) |
| `"KMeans" AND "garment" AND "palette"` | 의류 색상 추출 일반 |
| `"dominant color" AND "VLM"` | dominant 색 추론 |

### Layer 3 — 방어적 (혹시 비슷한 패턴)

| 키워드 | 의도 |
|---|---|
| `"hallucination" AND "color" AND "VLM"` | LLM 환각 차단 일반 |
| `"reference set" AND "VLM" AND "pick"` | reference-based pick 패턴 |
| `"anchor" AND "VLM" AND "perception"` | anchoring 일반 |
| `"ΔE76" OR "CIE" AND "VLM"` | color metric 결합 |

## 검색 DB (우선순위)

1. **Google Patents** — 정공. 미국 특허 / WIPO 국제 모두 검색. AND/OR/NEAR/() 지원.
   - https://patents.google.com/
   - filter: 출원일 ≤ 2026-04-23 (Gemini 2.5 Flash 등장 이후 기간 한정 가능)
   - filter: CPC G06V (image processing) + G06N (ML)
2. **KIPRIS** — 한국 특허. 사내 출원 전 한국어 prior art 필수.
   - http://www.kipris.or.kr/
3. **arxiv.org** — 학술 prior art. VLM 관련 논문 다수.
   - search: cs.CV / cs.CL / cs.AI categories
   - 키워드 + 2024 ~ 2026 기간
4. **IEEE Xplore / ACM Digital Library** — peer-reviewed 학술
5. **Google Scholar** — 일반 학술 + citation 추적

## 검토 우선순위 매트릭스

| 발견 prior art 유형 | 위협도 | 대응 |
|---|---|---|
| (A) 정확히 동일한 청구항 (VLM + KMeans + closed-set) | **HIGH** | 출원 보류. 사내 IP 위원회 의견 청취. 청구항 범위 narrow 화 또는 출원 포기 |
| (B) VLM + 다른 anchoring (e.g. dictionary lookup) | **MID** | 청구항을 KMeans-specific 으로 narrow 화. 차별점 (pixel evidence + ΔE76=0) 강조 |
| (C) KMeans + 의류 palette (VLM 없음) | **LOW** | 청구항의 VLM 결합 부분으로 차별화. 명세서에서 KMeans-only 의 한계 (free-form VLM pick 부재로 dominant 색 부정확) 명시 |
| (D) VLM + garment classification (color pick 부재) | **LOW** | Pass 2 color anchoring 의 신규성 강조 |
| (E) Fashion VLM 일반 | **NONE** | 무관. 본 발명은 색상 extraction 의 환각 차단 메커니즘 |

## 검토 절차

1. **각 Layer 검색어 별 상위 10 결과 review** — title + abstract
2. **HIGH risk 발견 시**: 전문 (claim + spec) full read → 차별점 정리
3. **LOW/MID 발견 시**: 본 출원 명세서에 differentiator 명시 (e.g. "기존 X 와 달리 본 발명은 ...")
4. **0 발견 시도 신중**: 검색어 부족 가능성. 동의어 / 반의어 / 도메인 외 일반화 추가
5. **결과 정리**: `docs/color_b_prior_art_results.md` 에 발견 / 위협도 / 대응 기록

## 참고 — 본 발명의 차별점 (출원 명세서 강조 포인트)

- **closed-set anchoring**: VLM 이 KMeans cluster top-N 안의 cluster_index 만 출력. cluster 외 색 도입 불가능 (LLM 환각 차단).
- **pixel evidence 기반**: KMeans 가 segformer mask 안의 garment pixel 만 cluster — 배경 / 피부 / 액세서리 leakage 차단.
- **ΔE76=0 매칭**: cluster_index 가 KMeans 결과 직접 index 라 좌표계 변환 / 임계 매칭 불필요.
- **2-pass 구조**: Pass 1 의 의류 분류 결과를 Pass 2 의 input context 로 활용 (단일 LLM call 의 multi-task 부담 분리).
- **결정론 보장**: temperature=0, seed=42, cache_key 에 cluster hex 시퀀스 포함 — 재현 가능.

## 외부 공개 정책 (5/8)

- 검색 결과 / 분석 메모는 **사외 비공개** — 내부 IP 위원회까지만 공유.
- arxiv preprint 검색 시 본 발명 키워드 직접 입력 OK (검색 행위 자체는 prior art 안 됨).
- 다만 prior art 가 발견되어 본 출원에 반영 시 출원 명세서 표현 신중 (검색 행위 노출 회피).

## 검토 시점

- **이번 주** — Layer 1 / Layer 2 검색 1차 review (PM 결정)
- **5/22 (금)** 까지 — Layer 3 + KIPRIS / 학술 DB 보충 review
- 발견 prior art 없거나 LOW/MID 만이면 다음 주 출원 명세서 작성 진입
- HIGH 발견 시 사내 IP 위원회 emergency call

## Reference

- spec: `docs/color_b_prompt_v010_spec.md`
- canary 결과: `docs/color_b_canary_results_2026_05_15.md`
- 트랙 A 결정: `memory/project_patent_track_color_2026_05_08.md`
