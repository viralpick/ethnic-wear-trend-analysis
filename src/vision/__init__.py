"""Vision (VLM) 패키지 (spec §7).

Case 1: 텍스트로 분류 안 된 포스트 → garment_type + silhouette + color 추출.
Case 2: 클러스터 × top-engagement IG 포스트 → palette/silhouette 보강.

안전 baseline (3-A LLM 과 동일 discipline):
- temperature=0, seed configurable
- enum / 범위 검증 실패 시 해당 필드 None 으로 drop (coerce 금지)
- no retries in skeleton
- YouTube 는 이 패키지 호출 대상이 아니다 (§7.2)
"""
