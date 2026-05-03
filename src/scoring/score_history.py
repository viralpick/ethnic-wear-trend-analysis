"""Score history 저장/로드 — daily / weekly direction + momentum 계산 기반 (spec §9.3).

파일 위치: outputs/score_history.json (날짜 서브폴더 밖, 누적 보존).
포맷:
  {cluster_key: {"YYYY-MM-DD": {
    "score": float,
    "post_count": float,                # Phase γ: int → float (β2/β4 share-weighted 정합)
    "youtube_views_total": float,
    "hashtag_counts": {tag: int},
    "accounts": [str],                  # IG handle (B-2 sub-signal: new_ig_account_ratio)
    "channels": [str],                  # YT channel (B-2 sub-signal: new_yt_channel_ratio)
  }, ...}}

backward compat:
- 구버전 float 값("YYYY-MM-DD": 40.0) 및 score/post_count 만 있는 구버전 dict 도 읽기.
- post_count 가 int 로 저장된 기존 파일 → `_read_count` 의 `float()` cast 로 자연 호환
  (Phase γ 마이그 read-cast 정책).
- B-2 (M3.G/H 후): channels 키 없는 기존 entry → default `[]` (자연 read-cast).
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path


def _read_score(entry: object) -> float | None:
    if entry is None:
        return None
    if isinstance(entry, dict):
        return entry.get("score")
    return float(entry)  # backward compat (구버전 float 포맷)


def _read_count(entry: object) -> float:
    """Phase γ: post_count 를 float 로 반환. 기존 int json 도 float() cast 로 자연 호환."""
    if isinstance(entry, dict):
        return float(entry.get("post_count", 0))
    return 0.0


def _read_youtube_views(entry: object) -> float:
    if isinstance(entry, dict):
        return float(entry.get("youtube_views_total", 0.0))
    return 0.0


def _read_hashtag_counts(entry: object) -> dict[str, int]:
    if isinstance(entry, dict):
        return entry.get("hashtag_counts", {})
    return {}


def _read_accounts(entry: object) -> list[str]:
    if isinstance(entry, dict):
        return entry.get("accounts", [])
    return []


def _read_channels(entry: object) -> list[str]:
    """B-2 (M3.G/H 후): YT channel 신규 비율 추적용. 기존 entry 에 channels 없으면 []."""
    if isinstance(entry, dict):
        return entry.get("channels", [])
    return []


class ScoreHistory:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._data: dict[str, dict[str, object]] = {}
        if path.exists():
            self._data = json.loads(path.read_text(encoding="utf-8"))

    def get_score(self, cluster_key: str, target_date: date) -> float | None:
        entry = self._data.get(cluster_key, {}).get(target_date.isoformat())
        return _read_score(entry)

    def get_daily_baseline(self, cluster_key: str, target_date: date) -> float | None:
        return self.get_score(cluster_key, target_date - timedelta(days=1))

    def get_weekly_baseline(self, cluster_key: str, target_date: date) -> float | None:
        return self.get_score(cluster_key, target_date - timedelta(days=7))

    def get_total_post_count(self, cluster_key: str) -> float:
        """Phase γ: 히스토리 전체 날짜의 post_count 합 (float). 오늘은 아직 미포함."""
        return sum(
            (_read_count(v) for v in self._data.get(cluster_key, {}).values()),
            0.0,
        )

    def get_post_count_history(
        self, cluster_key: str, target_date: date, days: int
    ) -> list[float]:
        """Phase γ: target_date 이전 days 일간 post_count 목록 (float). 없는 날 = 0.0,
        index 0 = 어제."""
        bucket = self._data.get(cluster_key, {})
        return [
            _read_count(bucket.get((target_date - timedelta(days=i)).isoformat()))
            for i in range(1, days + 1)
        ]

    def get_youtube_view_growth(self, cluster_key: str, target_date: date) -> float:
        """spec §9.2 — (이번 주 합산 - 지난 주 합산) / 지난 주. 지난 주 0 이면 0."""
        bucket = self._data.get(cluster_key, {})

        def _week_views(offset: int) -> float:
            return sum(
                _read_youtube_views(bucket.get((target_date - timedelta(days=i)).isoformat()))
                for i in range(offset, offset + 7)
            )

        this_week = _week_views(1)
        last_week = _week_views(8)
        if last_week == 0:
            return 0.0
        return (this_week - last_week) / last_week

    def get_hashtag_velocity(self, cluster_key: str, target_date: date) -> float:
        """주간 해시태그 총 사용량 증가율 (이번 주 vs 지난 주)."""
        bucket = self._data.get(cluster_key, {})

        def _week_total(offset: int) -> int:
            return sum(
                sum(_read_hashtag_counts(
                    bucket.get((target_date - timedelta(days=i)).isoformat())
                ).values())
                for i in range(offset, offset + 7)
            )

        this_week = _week_total(1)
        last_week = _week_total(8)
        if last_week == 0:
            return 0.0
        return (this_week - last_week) / last_week

    def get_new_ig_account_ratio(
        self,
        cluster_key: str,
        target_date: date,
        window_days: int,
        today_accounts: list[str],
    ) -> float:
        """B-2 (M3.G/H 후): IG handle 신규 비율 — momentum sub-signal."""
        return self._compute_new_entity_ratio(
            cluster_key, target_date, window_days, today_accounts, _read_accounts
        )

    def get_new_yt_channel_ratio(
        self,
        cluster_key: str,
        target_date: date,
        window_days: int,
        today_channels: list[str],
    ) -> float:
        """B-2 (M3.G/H 후): YT channel 신규 비율 — momentum sub-signal."""
        return self._compute_new_entity_ratio(
            cluster_key, target_date, window_days, today_channels, _read_channels
        )

    def _compute_new_entity_ratio(
        self,
        cluster_key: str,
        target_date: date,
        window_days: int,
        today_entities: list[str],
        reader,
    ) -> float:
        """window_days 내 미등장 entity 비율. today_entities 없으면 0."""
        if not today_entities:
            return 0.0
        bucket = self._data.get(cluster_key, {})
        seen: set[str] = set()
        for i in range(1, window_days + 1):
            seen.update(reader(
                bucket.get((target_date - timedelta(days=i)).isoformat())
            ))
        new_count = sum(1 for e in today_entities if e not in seen)
        return new_count / len(today_entities)

    def update(
        self,
        cluster_key: str,
        target_date: date,
        score: float,
        post_count: float,
        youtube_views_total: float = 0.0,
        hashtag_counts: dict[str, int] | None = None,
        accounts: list[str] | None = None,
        channels: list[str] | None = None,
    ) -> None:
        bucket = self._data.setdefault(cluster_key, {})
        bucket[target_date.isoformat()] = {
            "score": score,
            "post_count": post_count,
            "youtube_views_total": youtube_views_total,
            "hashtag_counts": hashtag_counts or {},
            "accounts": accounts or [],
            "channels": channels or [],
        }

    def save(self, *, compact: bool = False) -> None:
        """`compact=True` 면 indent 없이 한 줄 dump — production 16w 누적 cumulative cost
        절감용. default 는 indent=2 (audit / diff 친화).
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        indent = None if compact else 2
        self._path.write_text(
            json.dumps(self._data, indent=indent, ensure_ascii=False),
            encoding="utf-8",
        )
