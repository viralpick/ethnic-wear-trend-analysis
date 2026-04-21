"""Scoring 패키지 (spec §9).

공용 진입점: `compute_scores.score_clusters(contexts, cfg)` — ScoreBreakdown 배치 반환.
sub-score 별 raw 계산은 `score_social / score_youtube / score_cultural / score_momentum`.
정규화는 `normalize.minmax_same_run` 로 동결 (override 는 configs/local.yaml).
"""
