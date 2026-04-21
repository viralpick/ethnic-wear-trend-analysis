"""Aggregation 패키지 (spec §5.4, §8.2).

책임:
- 클러스터별 포스트 묶음 → DrilldownPayload (색상 팔레트, 분포, top_posts)
- Palette: RGB → HEX 결정론적 bucketing (k-means 아님, v1 explainable chip)
- Summary 빌더: 스코어링과 드릴다운을 결합해 TrendClusterSummary 를 구성
"""
