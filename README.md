# ethnic-wear-trend-analysis

인도 여성 ethnic wear (kurta set, co-ord, anarkali 등)의 단기 트렌드를 Instagram 포스트와
YouTube 영상에서 분석하는 harness. POC 단계, 1인 오너, 3주 일정.

데모일은 2026-05-04 (벵갈루루). 1차 산출물은 2026-05-01 데모 영상.

## 이 레포가 하는 일

이 레포는 **분석 레이어만** 다룬다. 크롤링/백엔드 API/프론트는 별도 레포에서 다른 담당자가
관리한다. 우리는 크롤러가 이미 수집해 둔 Instagram 포스트 + YouTube 영상(JSON/DB 레코드)을
입력으로 받아 세 가지를 산출한다:

1. **콘텐츠 단위 enriched 레코드** — 8개 속성(garment_type, fabric, technique, color,
   silhouette, occasion, styling_combo, brand_mentioned) + 배정된 trend_cluster_key
2. **트렌드 클러스터 배정** — `garment_type × technique × fabric` 조합 키 (15~20개 목표)
3. **클러스터별 일간 summary** — 스코어(0~100) + 방향(▲▼→) + 라이프사이클 + 속성 분포 +
   color palette(RGB → HEX) + drill-down용 top posts/videos

세부 정의는 두 문서로 분리되어 있다:
- [`docs/pipeline_spec.md`](docs/pipeline_spec.md) — 데이터 계층 (4-tier) / 매칭 multiplier / weekly 스코어링 / DB 적재 스키마. **canonical**
- [`docs/spec.md`](docs/spec.md) — 데모 화면 요건 / 속성 체계 (8개) / 텍스트 추출 / VLM 사용 / 비용 / 일정 (§5/§8/§9.2~§9.4 는 pipeline_spec 으로 redirect)

## 이 레포가 하지 않는 일

v1 스코프에서 다음은 **하지 않는다.** 필요해지면 PR 전에 질문으로 먼저 올릴 것.

- Instagram/YouTube 크롤링 — 크롤러 레포 책임
- 백엔드 API/프론트 — 각각 별도 레포
- 프로덕션 DB 영속화 — 이번 단계에서는 JSON in, JSON out
- Docker / nginx / 배포 스캐폴딩
- FastAPI app, routers, DI 컨테이너, DDD bounded context 분리
- 속성별 confidence score, evidence span 추적
- Embeddings, semantic similarity, vector search
- Caption 다국어 번역, 자유 텍스트 unknown 시그널 추출 (v1은 해시태그만)
- 학습 기반 랭커, ML 스코어 플러그인

v1은 결정론적이고 검수 가능한 분석 결과를 먼저 확보하는 것이 목표다. confidence/evidence
같은 v2 후보는 base pipeline이 안정화된 뒤 검토한다.

## 의존성과 원칙

- Python 3.14
- Pydantic v2 (도메인 모델은 `ConfigDict(frozen=True)`)
- pydantic-settings (YAML 기반 로컬 설정)
- 표준 `logging` 라이브러리만 사용
- uv + ruff + pytest (+ pytest-asyncio)
- src 레이아웃, flat 멀티 패키지(`contracts/`, `loaders/`, `pipelines/`, `utils/`) + 탑레벨 `settings` 모듈
- 한국어 주석/docstring OK — 식별자는 영어 유지

## 프로젝트 구조

```
src/
  settings.py              # pydantic-settings → configs/local.yaml
  contracts/               # Pydantic 입출력 계약
    raw.py                 # RawInstagramPost, RawYouTubeVideo (크롤러 출력 shape)
    enriched.py            # EnrichedContentItem + post_palette (PaletteCluster)
    output.py              # TrendClusterSummary + ColorPaletteItem
  loaders/
    sample_loader.py       # JSON 파일 로드 + Pydantic 검증
                           # (future: db_loader.py stub이 들어갈 자리)
  pipelines/
    run_local_smoke_test.py  # 샘플 로드 → 부분 클러스터링 → mock summary 기록
                             # (future: run_daily_pipeline.py)
  utils/
    logging.py             # stdlib 얇은 래퍼
configs/
  local.yaml               # 로컬 실행용 설정
sample_data/               # 크롤러 계약 예시 (우리가 요구하는 shape)
  sample_instagram_posts.json
  sample_youtube_videos.json
tests/unit/
notebooks/
scripts/
outputs/                   # gitignored. 파이프라인이 여기에 기록
  {YYYY-MM-DD}/summaries.json
  unknown_signals.json     # 전체 런에 걸쳐 누적
docs/pipeline_spec.md      # canonical: 데이터 계층 + DB 적재 + weekly 스코어링
docs/spec.md               # 참조: 데모 화면 요건 + 속성 체계 + 비용/일정
```

## 셋업

```bash
uv sync --extra dev
```

## 스모크 테스트 실행

```bash
uv run python -m pipelines.run_local_smoke_test
```

실행 후 확인:

- 콘솔 로그에 Instagram 8건 + YouTube 3건 로드, source_type별 카운트,
  클러스터별 포스트 수가 출력된다
- `outputs/<오늘날짜>/summaries.json` 이 생성된다 (mock `TrendClusterSummary` 레코드들)
- `outputs/unknown_signals.json` 이 생성/갱신된다 (매핑에 없는 해시태그 누적)

테스트:

```bash
uv run pytest
```

## Pipeline B color smoke (vision extras 필요)

동료 PoC 에서 인수인계받은 YOLO + segformer + LAB KMeans 파이프라인을 로컬 JPG 에 돌려
결과를 시각 확인하는 스크립트. `sample_data/image/` 의 20장 실 IG 이미지 → post ULID 로
그룹화 → palette 추출 → HTML 비교 페이지 생성.

첫 실행 전제:

```bash
uv sync --extra vision        # torch + transformers + ultralytics + pillow + opencv
```

실행:

```bash
uv run python scripts/pipeline_b_smoke.py
# 선택 옵션:
#   --image-root PATH    (기본: sample_data/image)
#   --output-dir PATH    (기본: outputs/pipeline_b_smoke)
```

첫 실행 시 yolov8n.pt (~6MB) + segformer_b2_clothes (~200MB) 다운로드 후 캐시. Apple
Silicon 은 MPS 자동 사용. CPU only 환경에서도 동작하지만 프레임당 ~1s 소요.

산출:

- `outputs/pipeline_b_smoke/palette.json` — post 별 팔레트 원본 데이터
- `outputs/pipeline_b_smoke/comparison.html` — 썸네일 + palette chip 시각 비교 (브라우저로 열기)

스코프: 동료 PoC 결과를 우리 레포 데이터/구조로 재현하는 smoke. 대시보드 연결은 M3,
Blob SAS URL download 도 M3 (현재는 로컬 JPG 만 처리).

daily CLI 에서 Pipeline B 를 사용하려면:

```bash
uv run daily --date 2026-04-21 --color-extractor pipeline_b --image-root sample_data/image
```

기본값은 `--color-extractor fake` (vision extras 없이 동작).

## Blob 다운로드 + Pipeline B 전수 처리 (vision + blob extras)

`posting.tsv` 가 참조하는 62장 이미지를 Azure Blob 에서 로컬 캐시로 받아 Pipeline B 를 전 post
에 적용. 전체 퀄리티를 `comparison.html` 로 확인.

준비:

```bash
uv sync --extra vision --extra blob    # 또는 --all-extras
```

그리고 프로젝트 루트에 `.env` 파일 생성 (gitignored). 아래 예시 내용 중 connection
string 의 `REPLACE_ME` 자리에 Azure Portal → Storage account → "Access keys" 의
Connection string 값 붙여넣기:

```dotenv
# 옵션 A (권장): connection string 한 줄
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=enhansprodpriceagentsa;AccountKey=REPLACE_ME;EndpointSuffix=core.windows.net

# 옵션 B (연결 문자열 비워둘 때 fallback):
# AZURE_STORAGE_ACCOUNT_NAME=enhansprodpriceagentsa
# AZURE_STORAGE_ACCOUNT_KEY=REPLACE_ME

AZURE_STORAGE_CONTAINER=collectify
```

다운로드:

```bash
uv run python scripts/download_blobs.py --dry-run   # 경로 미리보기
uv run python scripts/download_blobs.py             # 실 다운로드 → sample_data/image_cache/
```

전수 실행:

```bash
# TSV 소스 + Pipeline B 로 전 50 post 처리
uv run daily --date 2026-04-21 --source tsv --tsv-dir sample_data \
    --color-extractor pipeline_b --image-root sample_data/image_cache

# 또는 smoke 스크립트로 comparison.html 갱신
uv run python scripts/pipeline_b_smoke.py --image-root sample_data/image_cache
open outputs/pipeline_b_smoke/comparison.html
```

Instagram CDN 이미지 (hashtag_search.tsv) 는 현재 blob downloader 대상이 아니고, IG CDN 의
referer/토큰 expiry 제약 때문에 별도 검토 필요 (roadmap M4).

## `src/settings.py` 수정 시 주의

`pyproject.toml`의 `[tool.hatch.build.targets.wheel.force-include]` 는 `src/settings.py` 를
editable 설치 시에도 **`.venv/lib/.../site-packages/settings.py` 로 복사**한다 (symlink 가
아니다). 따라서 `src/settings.py` 를 편집한 후에는 복사본을 갱신해야 한다:

```bash
uv sync --extra dev --reinstall-package ethnic-wear-trend-analysis
```

다른 모듈 (`contracts/`, `loaders/`, `pipelines/`, `utils/`) 은 `packages = [...]` 로 등록되어
있어 editable 설치의 .pth 경유로 즉시 반영된다. 복사본 문제는 `settings.py` 만 해당.

yaml 설정(`configs/local.yaml`) 은 런타임에 CWD 부터 상향 탐색하여 찾으므로, 수정 후 재설치
불필요.

## 다음 마일스톤 (docs/spec.md §12 기준)

- 2026-04-21: daily 수집 시작 (크롤러 팀). 이 레포는 샘플 로더 + 룰 기반 매핑 스켈레톤으로 대기
- 2026-04-23까지: docs/spec.md §6.2 해시태그 매핑 테이블을 코드화하여 `src/analysis/`에 배치 (Step 2)
- 2026-04-24 1차 싱크: 3일치 실제 데이터로 매핑 검증 + 클러스터 granularity 조정
- 2026-04-28 2차 싱크: 스코어보드 데이터 + 화면 리뷰
- 2026-05-01: 데모 영상 촬영·편집 완료

## Early-data caveat

수집 시작(2026-04-21) 후 **첫 3일간**은:

- **weekly direction / weekly_change_pct**: 이전 주 베이스라인이 없으므로 신뢰할 수 없다
- **momentum score**: 최근 7일 일평균이 부족하여 수렴하지 않는다
- **lifecycle_stage**: 대부분 Early로 분류된다

스모크 테스트와 일일 파이프라인은 이 구간에서 크래시 없이 기본값(`"flat"`, `0.0`,
`"early"`)으로 폴백한다. 2026-04-24 벌크 수집(과거 1개월, §12) 완료 이후 주간/모멘텀이
의미 있는 값으로 수렴하기 시작한다.

## 참고

- **데이터 계층 / DB 적재 / 매칭 multiplier / weekly 스코어링 (canonical)**: [`docs/pipeline_spec.md`](docs/pipeline_spec.md) §1, §2, §3.4~§3.5, §5
- **속성 정의 (8개)**: spec §4.1
- **데모 화면 요건**: spec §2
- **텍스트 추출 방법론**: spec §6
- **VLM 사용 정책**: spec §7
- **비용 / 일정**: spec §11, §12

> spec.md §5 (클러스터링) / §8 (스키마) / §9.2~§9.4 (스코어링·방향성·lifecycle) 은 pipeline_spec.md 가 canonical 이며 spec.md 에는 redirect 노트만 남아 있다.
