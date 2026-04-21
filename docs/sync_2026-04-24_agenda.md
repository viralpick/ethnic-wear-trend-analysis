# 2026-04-24 1차 싱크 — 분석 레포 전달 사항

**작성**: 2026-04-21
**대상**: 크롤러 팀 / DW 팀 / 백엔드 팀 / spec 저자 (종휘님)
**이 문서의 목적**: 싱크 회의 전에 각 팀이 미리 확인할 수 있도록 결정 필요 사항 / 질문 / 제안을 한 곳에 모음.

---

## 1. 크롤러 팀에 전달

### 1.1 DB export 테이블명 오타

현재 `sample_data/` 로 전달된 TSV 5개 중 3개가 테이블명 오타:

| 현재 파일명 | 올바른 이름(추정) | 비고 |
|---|---|---|
| `png_india_ai_fashion_inatagram_posting.tsv` | `..._instagram_posting` | `inatagram` → `instagram` |
| `png_india_ai_fashion_inatagram_profile.tsv` | `..._instagram_profile` | 동일 오타 |
| `png_india_ai_fashionagram_profile_posting.tsv` | `..._fashion_instagram_profile_posting`? | 언더스코어 누락 추정 |
| `png_india_ai_fashionash_tag_search_result.tsv` | `..._fashion_hash_tag_search_result`? | `h` 중복 제거 추정 |

→ DB export script 의 자동 테이블 네이밍 로직 버그로 보임. 스키마 컨트랙트 단계에서 교정 요청.

### 1.2 `inatagram_posting.tsv` vs `fashionagram_profile_posting.tsv` 중복

- 10개 포스트 URL 이 **10/10 완전 일치** (masoomminawala 의 같은 포스트)
- `inatagram_posting` 은 13 cols (인게이지먼트 + 이미지 경로 포함)
- `fashionagram_profile_posting` 은 9 cols (텍스트 중심, image_paths 없음)

→ 한 테이블로 통합 가능한지, 두 개가 별도 수집 파이프라인 산출인지 확인. 분석 레이어 입장에선 한 뷰로 충분.

### 1.3 `hashtag_search_result.tsv` 의 `likes=0 / comments=0`

41개 전체 row 가 `likes=0, comments=0`. 실 포스트에 당연히 0 일 수 없으니:

- (a) 크롤러가 해시태그 검색 엔드포인트에서 인게이지먼트를 수집 안 하는 건지
- (b) 수집 실패 default 인지

→ **spec §9.2 Social score 의 분모에 이 데이터가 들어오면 왜곡됨**. Social 계산은 `influencer_fixed` / `hashtag_tracking` 소스 posts 기준이라 OK 지만, `hashtag_search_result` 가 어떤 source_type 으로 들어오는지 명세 필요.

### 1.4 YouTube blob 경로의 `null`

`collectify/youtube/null/{video_id}.mp4` — `null` 자리가 channel_id 였을 가능성. 경로 규약 확정 요청.

### 1.5 Blob 경로 규약 (Azure Blob loader 개발용)

spec §10 이후 우리 `src/loaders/blob_raw_loader.py` 를 구현할 예정. 아래 규약 확정 필요:

- `container/{source}/{date}/{post_id}.json` (파일 단위) vs 날짜별 단일 JSONL?
- `date` 는 post_date 기준인지 collected_at 기준인지?
- re-crawl / 재처리 시 같은 key 덮어쓰나, 버저닝 (suffix) 두나?
- SAS URL 만료 시간 / 재발급 주기?

→ 현재 `instagram_sample_list.txt` / `youtube_sample_list.txt` 의 blob URL 포맷이 placeholder 라 실 경로 확정 어려움.

### 1.6 raw contract shape (우리 `contracts/raw.py`) 정합

현재 `RawInstagramPost` 는 12 필드인데 TSV `inatagram_posting` 은 13 cols:

| TSV cols (추정 순서) | 우리 contract | 상태 |
|---|---|---|
| `date_partition` (2026042023) | — | 누락: 수집 기준 시간 bucket |
| ULID | — | 우리는 `post_id` (insta short code) 사용. ULID 와의 관계? |
| handle | `account_handle` | ✅ |
| insta URL | — | 우리 contract 엔 없음. 필요 여부 확인 |
| CDN image URL | `image_urls[0]` | ✅ but 단일 thumbnail 인지 전체 캐러셀인지 |
| post_date | `post_date` | ✅ |
| caption | `caption_text` | ✅ |
| comments (pipe-joined) | — | 우리 contract 엔 없음. spec §8.1 의 `top_comments` 는 YT 만 정의 |
| likes | `likes` | ✅ |
| comments_count | `comments_count` | ✅ |
| image_paths (csv-joined blob 경로) | `image_urls` | **포맷 불일치** — 우리 contract 는 URL 이고 TSV 는 blob path |
| created_at | `collected_at` | ✅ |
| updated_at | — | 새 필드 |

→ `date_partition`, `updated_at`, insta URL, 펼친 comments 의 취급을 합의. `image_urls` 를 "CDN URL 또는 blob path" 로 확장할지, loader 단에서 변환할지 결정.

---

## 2. spec 저자 (종휘님) 에게 전달 — spec §4.1 ④ + §7 갱신 제안

### 2.1 VLM → Pipeline B 전환 권고

동료 (`~/dev/clothing-color-extraction-poc`) 의 2026-04-17 PoC 결과 (40 프레임 기준):

| 메트릭 | 기준 | Claude Opus 4.5 | GPT-5.2 | Pegasus (video) | **Pipeline B** |
|---|---|---|---|---|---|
| dominant_pass_rate (ΔE76≤10) | ≥80% | **26.4%** ❌ | **28.9%** ❌ | ~0% ❌ | **100% ✅** |
| label_acc | ≥85% | 100% ✅ | 97.8% ✅ | 부분 라벨 | 100% ✅ |
| range_pass_rate (recall@ΔE15≥70%) | ≥70% | 41.5% ❌ | 66.7% ≈❌ | n/a | 100% ✅ |
| skin_leak_rate | ≤10% | 0% ✅ | 0% ✅ | — | 0% ✅ |

**결론**: VLM 은 색상 "계열 인식" 만 가능, pixel-level hex 추출은 실패. Pegasus 는 schema 강제 시 `#000000`/`#FFFFFF` mode collapse. 현실적으로 **OpenCV KMeans in LAB space 가 단일 신뢰 경로**.

### 2.2 spec 변경 범위 제안

| spec 섹션 | 현재 서술 | 제안 변경 |
|---|---|---|
| §4.1 ④ color | "Instagram 이미지에서만 VLM 으로 추출. Plan B: OpenCV 픽셀 샘플링" | "YOLOv8 + segformer_b2_clothes + LAB KMeans 파이프라인 (주). VLM 은 label 보조 옵션 (hybrid)" |
| §7.1 | "텍스트가 먼저, VLM 은 보조" | "텍스트 + Pipeline B 주. VLM 은 garment taxonomy 세분화가 필요한 경우만" |
| §7.2 Case 1/2 | "VLM 호출" | "Pipeline B 호출" |
| §7.3 VLM 컬러 추출 프롬프트 | 유지 (hybrid 모드 대비) | 단 "우선 사용 X" 명시 |
| §7.5 비용 | VLM Case1/2 합 10~17만원 | **$0 runtime** (로컬 GPU/MPS). 비용 줄어듦 |

### 2.3 ATR taxonomy vs 우리 `GarmentType` 간극

segformer_b2_clothes 의 ATR 18-class 는 서양 복식 (upper-clothes / pants / dress / skirt / hat / shoes) 기준. 우리 `GarmentType` enum (kurta_set / anarkali / saree 등) 과 **직접 매칭 안 됨**.

→ Pipeline B 의 역할을 **"의류 영역 vs 피부/배경 분리 + 픽셀 확보"** 에 한정. `garment_type` 분류는 계속 텍스트/LLM 파이프라인이 담당. 이 경계 spec 에 명시 필요.

---

## 3. DW 팀 (StarRocks) 에 전달

### 3.1 테이블 분리 방안

현재 우리 output contract 기준 3개 테이블로 분리 제안:

- `trend_summary` (= `TrendClusterSummary`, 클러스터 × 일 1 row)
- `enriched_content` (= `EnrichedContentItem`, 포스트 1 row)
- `unknown_signals` (= `UnknownAttributeSignal`, 해시태그 1 row)

→ 분리할지, 단일 테이블 + view 로 갈지 DW 컨벤션 확인.

### 3.2 `schema_version` 컬럼

우리 contract 에 `schema_version: "0.1.0"` 이 TrendClusterSummary 에 박혀있음. 테이블에도 컬럼으로 올릴지, 별도 metadata 테이블 보유할지.

### 3.3 재처리 멱등성 규약

같은 날짜 분석을 두 번 돌리면:

- (a) `(cluster_key, date)` UPSERT?
- (b) append + `latest view` (run_id 기준)?

→ 현재 로컬 파이프라인은 atomic rename 으로 덮어쓰기 (b 패턴과 유사). DW 측 관행 확인.

### 3.4 적재 방식

- **INSERT per row**: 간단하지만 StarRocks 에 부하
- **Stream Load (JSON batch)**: 권장 방식이지만 우리 쪽 writer 구현 복잡도 증가

→ 어느 쪽 선호하는지 확인. 우리는 writer Protocol 뒤에 구현체 숨길 예정이라 어느 것이든 가능.

---

## 4. 백엔드 팀에 전달

### 4.1 `BackendScoreboardPayload` endpoint shape

현재 placeholder (`src/exporters/build_backend_payload.py`):

```python
class BackendScoreboardPayload(BaseModel):
    schema_version: str
    target_date: date
    clusters: list[TrendClusterSummary]
```

→ 실 API endpoint 정의:

- `POST /scoreboard/{date}` (일별 replace) vs `PUT /scoreboard` (전체 replace)?
- 멱등성 (같은 payload 두 번 POST 시 중복 생성 X)?
- auth (internal service mesh vs shared token)?

### 4.2 Drill-down 전송 단위

현재 `TrendClusterSummary.drilldown` 에 `DrilldownPayload` 통째로 포함. 응답 크기가 커질 수 있음:

- color_palette (top 5 HEX chip)
- silhouette/occasion/styling distribution
- top_posts ID list
- top_videos ID list
- top_influencers list

→ 스코어보드 초기 렌더는 drilldown 빼고, drill 클릭 시 `GET /cluster/{key}?date=...` 로 lazy fetch 가 UX 상 나을지? BE 팀 선호도 + 프론트 측 의견 확인.

### 4.3 실시간성

분석 파이프라인은 일 1회 배치. BE 는 캐시 전략 어떻게 갈 건지 (TTL 24h / pub-sub / manual invalidate).

---

## 5. 내부 결정 필요 (3일치 데이터로 캘리브레이션)

### 5.1 Scoring 가중치

현재 초안 (`configs/local.yaml`): Social 40 / YouTube 25 / Cultural 15 / Momentum 20.

→ 3일치 데이터 돌리고 클러스터 간 점수 차이 히스토그램 확인. 전부 비슷하게 나오면 재배분.

### 5.2 Direction threshold

`scoring.direction_threshold_pct: 5.0` — 일별 변동 실제 폭 보고 3~7 사이에서 조정.

### 5.3 Lifecycle 점수 대역

`early_below=30, growth_until=65` — 실 점수 분포 보고 조정.

### 5.4 Palette bucket_size

현재 `palette.bucket_size: 32` (거친 격자, 8 chip/channel). Pipeline B 이식 후엔 **bucket 방식 자체를 LAB KMeans 로 교체 가능**. 교체 여부 + 교체 후 테스트 방법.

### 5.5 VLM Case 1/2 cap

Pipeline B 로 전환되면 VLM cap 설정 (`vlm.case1_daily_cap=150`, `case2_per_cluster_cap=10`) 은 **의미 없어짐**. 대신 Pipeline B 의 `min_pixels=150`, `k=5` 등이 tuning target.

### 5.6 data_maturity 밴드

`bootstrap<3, partial 3-6, full≥7` — 수집 시작일 2026-04-21 기준. 데모 5/4 까지 13일 → FULL 진입은 4/27. 4/24 싱크 시점엔 아직 PARTIAL.

---

## 6. M3 (5/1 이후) 전제 작업

4/24 에서 결정된 사항 기반으로 M3 에 진행할 것:

1. `src/loaders/blob_raw_loader.py` — Azure Blob SDK, 페이지네이션, SAS 인증
2. `src/exporters/starrocks_writer.py` — Stream Load or INSERT
3. `src/exporters/backend_poster.py` — BE API 호출 + 최소한의 backoff
4. `src/vision/pipeline_b_extractor.py` — 동료 레포의 YOLO+segformer+KMeans 이식
5. `src/scoring/` 의 momentum/cultural/direction baseline 을 전일/전주 summaries.json 로부터 실제로 계산 (현재 skeleton 은 전부 0)
6. influencer_tier 가중치 실제 적용 — raw IG `account_followers` → ClusterScoringContext 경유

각각 Protocol 뒤에 구현체 숨기는 방식 유지.

### 6.1 이미지/영상 이중 소스 대응 — frame source 추상화

동료 PoC (`~/dev/clothing-color-extraction-poc`) 는 **영상 전용** 가정 (`FRAMES_DIR/{video_id}/NNNN.jpg`).
우리 소스는 IG 정적 이미지가 주, IG Reel (영상) 이 보조. YT 영상은 color 추출 대상 아님 (spec §7.2).

frame 단위 추상화 (Step C 에서 신설):

```python
class FrameSource(Protocol):
    def iter_frames(self) -> Iterator[Frame]: ...

@dataclass(frozen=True)
class Frame:
    id: str          # 예: "{post_ulid}_{image_ulid}" or "{video_id}_frame_000012"
    rgb: np.ndarray  # (H, W, 3) uint8
    source_type: str # "image" | "video"

class ImageFrameSource:
    """IG 캐러셀 N장 등. JPG 경로 리스트 → frame 당 이미지 1장."""

class VideoFrameSource:
    """IG Reel 등. ffmpeg 서브프로세스로 fps 고정 추출. YT 는 호출 대상 아님 (type guard)."""
```

`pipeline_b_extractor.analyze(source: FrameSource)` 는 FrameSource 를 받아 frame 단위 반복.
IG 포스트 → `ImageFrameSource(image_paths)`, IG Reel → `VideoFrameSource(reel_path, fps=1)`.

**스코프 경계**:
- YT 영상: FrameSource 호출 안 함. `source == ContentSource.YOUTUBE` 입력 시 raise (spec §7.2).
- sample_data 기준: 20 JPG 는 Step E smoke 로 ImageFrameSource 커버. IG Reel 4건 + YT 5건은
  blob SAS URL 이 placeholder 라 M3 에서 실 download 연결 후 VideoFrameSource 테스트.

### 6.2 frame 단위 집계 flow (Step B / C 에서 구현)

```
[cluster A]
  ├── post 1 (IG 이미지, 4장)   → ImageFrameSource → 4 frame → 각 frame top-5 color
  ├── post 2 (IG Reel, 30초)    → VideoFrameSource(fps=1) → 30 frame → 각 frame top-5 color
  ├── post 3 (IG 이미지, 1장)   → ImageFrameSource → 1 frame → top-5 color
  └── ...
        │
        ▼ concat frame pixel samples (또는 frame top-k centroid 들을 post-level weight 로 재KMeans)
        │
        ▼ cluster-level LAB KMeans (top_k=5)
        │
        ▼ ColorPaletteItem[5] ← DrilldownPayload.color_palette
```

**미결 질문** (4/24 싱크 또는 4/28 싱크):
- frame-level top-5 vs cluster-level top-5 의 aggregate 방식 (pixel concat vs centroid reclustering)
- IG 캐러셀에서 각 이미지를 동등 가중 vs 대표 이미지 1장만 사용
- Reel frame 수가 이미지 수 대비 훨씬 크므로 가중치 정규화 필요한가

---

## 7. 참고 / 현재 상태 스냅샷

- 레포: https://github.com/viralpick/ethnic-wear-trend-analysis (init/step-3b-skeleton 브랜치)
- 테스트: 85 unit + 2 snapshot 전부 pass, ruff 클린
- `uv run daily --date 2026-04-21` 로 로컬 sample_data 기준 11 클러스터 summaries.json 생성 검증
- VLM 은 현재 `FakeVLMClient` (해시 결정론 stub) — 실 pixel 추출은 Pipeline B 이식 이후
- 데이터 파이프라인 상/하류: Azure Blob (raw) → 분석 레이어 → StarRocks (DW) + BE API
- 로컬 POC 실행 시 **Pritunl VPN 기본 OFF** (외부 리소스만 사용, shadow consumer 사고 재발 방지)
