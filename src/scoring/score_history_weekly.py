"""Weekly score history — pipeline_spec_v1.0 §3.2 / §3.5 (Q-D2 weekly bucket).

기존 `ScoreHistory` (daily) 와 disjoint. 데이터 양 적은 환경에서 daily noise 흡수 위해
weekly bucket (월~일 IST) 단위로 score 누적.

파일: outputs/score_history_weekly.json (날짜 서브폴더 밖, 누적 보존).
포맷:
  {cluster_key: {"YYYY-Www": {
    "score": float,               # 그 주 weekly score (= run 시점 계산 결과)
    "post_count": int,            # 그 주 post 수
    "youtube_views_total": float,
    "hashtag_counts": {tag: int},
    "accounts": [str],
    "week_start_date": "YYYY-MM-DD",  # 그 주 월요일 (조회 편의)
  }}}

ISO week key = `date.isocalendar()` 기반. IST timezone 변환은 caller 책임 (이 클래스는
naive date 만 받는다 — caller 가 IST 변환 후 전달).

trajectory: §3.4 의 12주 시계열 — `get_trajectory_12w` 가 12개 list 반환, 부족분 0 패딩.
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path


def iso_week_key(d: date) -> str:
    """date → ISO week key 'YYYY-Www' (e.g. 2026-04-27 월 → '2026-W18')."""
    year, week, _ = d.isocalendar()
    return f"{year}-W{week:02d}"


def week_start_monday(d: date) -> date:
    """주어진 날짜가 속한 ISO week 의 월요일."""
    return d - timedelta(days=d.weekday())


class WeeklyScoreHistory:
    """Weekly bucket score history.

    backward compat: 기존 daily `score_history.json` 와 분리 (별도 파일). daily 와
    weekly 둘 다 유지하는 phase 동안 caller 가 둘 다 갱신.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._data: dict[str, dict[str, dict]] = {}
        if path.exists():
            self._data = json.loads(path.read_text(encoding="utf-8"))

    def get_weekly_score(self, cluster_key: str, target_date: date) -> float | None:
        bucket = self._data.get(cluster_key, {}).get(iso_week_key(target_date))
        if bucket is None:
            return None
        return bucket.get("score")

    def get_baseline(self, cluster_key: str, target_date: date) -> float | None:
        """spec §3.4 weekly_change_pct 분모 — 지난 주 score."""
        last_week = target_date - timedelta(days=7)
        return self.get_weekly_score(cluster_key, last_week)

    def get_trajectory_12w(
        self, cluster_key: str, target_date: date
    ) -> list[float]:
        """target_date 가 속한 주 포함 최근 12주 score (oldest → newest, 부족분 0).

        spec §3.4: trajectory 배열 길이 12, 부족분 = 0.
        """
        scores: list[float] = []
        for offset in range(11, -1, -1):  # 11..0 (오래된 → 최근)
            wk_date = target_date - timedelta(days=offset * 7)
            score = self.get_weekly_score(cluster_key, wk_date)
            scores.append(score if score is not None else 0.0)
        return scores

    def update_weekly(
        self,
        cluster_key: str,
        target_date: date,
        score: float,
        post_count: float = 0.0,
        youtube_views_total: float = 0.0,
        hashtag_counts: dict[str, int] | None = None,
        accounts: list[str] | None = None,
    ) -> None:
        """주의: 같은 (cluster_key, week) 호출 시 덮어쓰기. 같은 주 재처리 시 의도된 동작."""
        bucket = self._data.setdefault(cluster_key, {})
        bucket[iso_week_key(target_date)] = {
            "score": score,
            "post_count": post_count,
            "youtube_views_total": youtube_views_total,
            "hashtag_counts": hashtag_counts or {},
            "accounts": accounts or [],
            "week_start_date": week_start_monday(target_date).isoformat(),
        }

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self._data, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
