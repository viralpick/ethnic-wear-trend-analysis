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

세부 정의와 스코어링 공식은 [docs/spec.md](docs/spec.md) 참조.

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
    enriched.py            # EnrichedContentItem + ColorInfo
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
docs/spec.md               # canonical project spec
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

- canonical spec: [`docs/spec.md`](docs/spec.md)
- 속성 정의: spec §4.1
- 클러스터 키와 배정 규칙: spec §5
- 스코어링 공식: spec §9
- 데이터 스키마: spec §8
