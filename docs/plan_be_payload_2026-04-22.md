# BE Payload 구성 계획 (2026-04-22)

## 목표

spec에 정의된 스코어링 + 드릴다운 데이터를 기반으로, BE가 프론트에 내려줄 payload를 구성한다.
지금 당장 쓸 수 있는 데이터 범위에서 최대한 채우고, 역사 데이터가 필요한 부분만 명시적 stub으로 남긴다.

---

## 현재 상태 → 목표

| 항목 | 현재 | 목표 |
|------|------|------|
| Social score | engagement 단순 합산 | 인플루언서 티어 가중치 (mega×3, macro×2, mid×1.5, micro×1) |
| Cultural score | 0.0 고정 | Akshaya Tritiya 윈도우 (4/13~5/11) + Bollywood source_type |
| Momentum score | 0.0 고정 | 7일치 없어 지금은 유지 (명시) |
| 일별/주차 방향 | 항상 → (flat) | score_history.json 저장 → 다음날부터 실제 direction |
| BE payload 파일 | 없음 | `outputs/{date}/payload.json` 직렬화 |

## 포함

1. Social: `account_followers` → tier 분류 → 가중 engagement 합산 (spec §9.2)
2. Cultural: Akshaya Tritiya 해시태그 매칭 + Bollywood source_type 기반 presence (spec §9.2, §9.5)
3. 스코어 히스토리: `outputs/score_history.json` 저장/로드 → `daily_direction` / `weekly_direction` 실제 계산
4. `payload.json` write (HTTP POST는 별도 — 4/24 field 합의 후)

## 제외

- Momentum: `momentum_post_growth`, `hashtag_velocity`, `new_account_ratio` — 7일치 필요. 0.0 stub 유지.
- YouTube view growth: 이전 날 view count 비교 필요. 히스토리 구조 잡으면 내일부터.
- BE HTTP POST: 4/24 싱크 전 field 합의 안 됨.

## 리스크

- `account_followers` 정보가 enriched 레코드까지 흘러오는지 확인 필요 (normalized에만 있고 scoring context 주입 안 됨)
- Cultural festival 해시태그: normalized.hashtags에서 읽어야 하는데 경로 확인 필요
- 주차 direction은 7일 전 값 필요 → 4/29 이후에야 실제 작동

## 성공 기준

1. `uv run python -m pipelines.run_daily_pipeline --source tsv` 오류 없이 완료
2. `outputs/{date}/payload.json` 생성, `BackendScoreboardPayload` schema 유효
3. Social 스코어가 influencer tier 간 실제 차이를 가짐 (테스트 검증)
4. Cultural 스코어가 Akshaya Tritiya 기간(현재)에 0 초과 값 가짐
5. 기존 182+ 테스트 전통과

## 구현 순서

1. `run_scoring_pipeline._build_contexts`에 followers 데이터 주입 + tier 가중치 적용
2. Cultural context 계산 (festival window 체크 + bollywood presence)
3. 스코어 히스토리 저장/로드 + direction 연결
4. `payload.json` write 추가
5. 테스트 보강 + 전체 파이프라인 smoke

---

*선행 조건: 속성 추출 커버리지 확인 (docs/plan_attribute_coverage_2026-04-22.md)*
