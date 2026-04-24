"""Vision (VLM) 패키지 (spec §7).

Case 1: 텍스트로 분류 안 된 포스트 → canonical outfit + palette 추출.
Case 2: 클러스터 × top-engagement IG 포스트 → canonical palette 보강.

silhouette 는 canonicals[*].representative.silhouette 로 접근 — post-level 단일값 없음
(B3d, feedback_post_level_single_value).

안전 baseline (3-A LLM 과 동일 discipline):
- temperature=0, seed configurable
- enum / 범위 검증 실패 시 해당 필드 None 으로 drop (coerce 금지)
- no retries in skeleton
- YouTube 는 이 패키지 호출 대상이 아니다 (§7.2)
"""
