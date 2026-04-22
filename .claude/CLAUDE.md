# ethnic-wear-trend-analysis — 프로젝트 규칙

이 레포 작업 시 adhere 해야 할 핵심 규칙. 글로벌 `~/.claude/CLAUDE.md` + `rules/*.md` 을
**보완** 하고 충돌 시 이 파일이 우선.

## 이 프로젝트의 절대 규칙

1. **vision extras 격리** — core 코드 (`contracts/`, `attributes/`, `clustering/`,
   `normalization/`, `scoring/`, `aggregation/`, `exporters/`, `pipelines/`) 에 `torch` /
   `transformers` / `ultralytics` / `insightface` / `clip` / `onnxruntime` 의 **top-level
   import 금지**. 반드시 Protocol 뒤 DI (`VLMClient`, `ColorExtractor`, `SceneFilter` 등).
   이유: `uv sync --no-extras` 환경 (CI, 최소 개발자) 에서도 core tests 가 돌아야 함.
   `src/vision/` 내부에서만 top-level import 허용.

2. **scoring.normalization_method 동결** — `minmax_same_run` 외 방식은 `configs/local.yaml`
   의 override 가 아니면 `apply_normalization` dispatcher 에서 `ValueError`. z-score, log,
   percentile, softmax 등 **코드에 추가 금지**. raw counts → same-run minmax → weight 가
   유일한 재현 경로 (spec mandate).

3. **persistence / framework 금지** — DB persistence / ORM / service layer / DI 컨테이너 /
   스케줄러 / 메시지 큐 consumer 금지. 입력은 raw 파일/DB reader, 출력은 enriched JSON +
   summaries JSON + 선택적 BE poster. spec §3/§10 batch 전제.

4. **실패 숨김 금지** — retry 루프 / 묵시적 fallback 으로 에러 삼킴 금지. **timeout 가드**
   만 허용. LLM/VLM 은 `seed=42` / `temperature=0` 결정론. `FakeLLMClient` / `FakeVLMClient`
   는 image_url / source_post_id 해시 기반 결정론. 외부 호출에서 빈 응답 / 파싱 실패 시
   raise 또는 `unknown` sentinel 로 명시 기록.

5. **스코어 재현성** — raw counts + 공식만으로 재현 가능한 스코어만 생성. ML ranker /
   heuristic re-rank / 학습 기반 가중치 금지. 각 `score_*.compute(ctx, cfg) -> float` 는
   pure raw 반환, 정규화는 orchestrator 1회. spec §9 가 canonical.

## 작업 패턴

- 큰 작업은 `/work-kickoff` 로 시작 (범위/리스크/성공기준 명시 후 승인)
- 다축 변경 후 `/review-triple` 로 재사용성/품질/효율 3축 리뷰
- 구현 완료 시 자동 테스트 → 수동 체크 → memory 갱신 순서
- vision extras 동작이 바뀌면 `scripts/pipeline_b_smoke.py` 로 mini (6 posts) + full
  (131 posts) smoke 해서 HTML 비교

## 주요 파일 지도

- `docs/spec.md` — canonical spec. 이 문서와 코드 충돌 시 spec 우선. 변경은 user 동의
- `docs/roadmap.md` — M1~M4 milestone + 영구 비스코프 + 우선순위 매트릭스
- `docs/sync_2026-04-24_agenda.md` — 4/24 싱크 agenda (크롤러/BE/DW 팀 결정 기록)
- `configs/local.yaml` — 모든 magic number 가 여기. 코드에 하드코드 금지
- `src/vision/pipeline_b_extractor.py` — YOLO + segformer + KMeans Pipeline B
- `src/vision/garment_instance.py` — phase 3 instance 단위 palette
- `tests/unit/` + `tests/snapshot/` — 182+ tests, snapshot golden drift 가드

## 한국어 커뮤니케이션

- 기술 용어 (embedding, KMeans, fallback 등) 는 영어 유지
- 문서 / 주석 / commit 메시지 본문은 한국어
- 함수/변수명은 영어 (code style)
