"""로컬 스모크 테스트 파이프라인.

흐름:
    1. configs/local.yaml 로드
    2. sample_data/*.json 로드 + Pydantic 검증
    3. source / source_type 별 카운트 로깅
    4. 해시태그로 부분 클러스터 키 힌트 추출 (최소 매핑만) — 3축 전부 잡히면 완전 키,
       일부만 잡히면 "unknown" 플레이스홀더, 전부 null 이면 "unclassified"
    5. 클러스터별 mock TrendClusterSummary 를 outputs/{YYYY-MM-DD}/summaries.json 에 기록
       (early-data caveat 상 score=0, direction=flat, lifecycle=early, maturity=bootstrap 폴백)

Step 1 스코프: 최소 룰 매핑만 inline 으로 둠. unknown signal 추적은 Step 3-A 의
attributes.unknown_signal_tracker 가 담당하므로 이 스모크 테스트에서는 다루지 않는다
(outputs/unknown_signals.json 포맷 충돌 방지).
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import date

from contracts.output import (
    DrilldownPayload,
    ScoreBreakdown,
    TrendClusterSummary,
)
from contracts.raw import RawInstagramPost, RawYouTubeVideo
from loaders.sample_loader import load_instagram_samples, load_youtube_samples
from settings import Settings, load_settings
from utils.logging import get_logger

logger = get_logger(__name__)

# spec §6.2 의 축소판. 스모크 테스트에서 "클러스터 키가 실제로 생성된다"를 보여주기 위한 최소셋.
_GARMENT_HINTS: dict[str, str] = {
    "kurtaset": "kurta_set",
    "kurtasets": "kurta_set",
    "kurtapalazzoset": "kurta_set",
    "coordset": "co_ord",
    "coordsets": "co_ord",
    "kurtadress": "kurta_dress",
    "anarkali": "anarkali",
    "kurti": "tunic",
    "tunic": "tunic",
}
_TECHNIQUE_HINTS: dict[str, str] = {
    "chikankari": "chikankari",
    "blockprint": "block_print",
    "handblockprint": "block_print",
    "floralprint": "floral_print",
}
_FABRIC_HINTS: dict[str, str] = {
    "cotton": "cotton",
    "cottonkurta": "cotton",
    "linen": "linen",
    "rayon": "rayon",
}


def _normalize_tag(raw_tag: str) -> str:
    return raw_tag.lstrip("#").lower()


def _cluster_key(hashtags: list[str]) -> str:
    normalized = [_normalize_tag(t) for t in hashtags]
    garment = next((_GARMENT_HINTS[t] for t in normalized if t in _GARMENT_HINTS), None)
    technique = next((_TECHNIQUE_HINTS[t] for t in normalized if t in _TECHNIQUE_HINTS), None)
    fabric = next((_FABRIC_HINTS[t] for t in normalized if t in _FABRIC_HINTS), None)

    if garment and technique and fabric:
        return f"{garment}__{technique}__{fabric}"
    if garment and technique:
        return f"{garment}__{technique}__unknown"
    if garment:
        return f"{garment}__unknown__unknown"
    return "unclassified"


def _display_name(cluster_key: str) -> str:
    if cluster_key == "unclassified":
        return "Unclassified"
    parts = [p for p in cluster_key.split("__") if p != "unknown"]
    return " ".join(p.replace("_", " ").title() for p in parts)


def _mock_summary(
    cluster_key: str,
    posts: list[RawInstagramPost],
    videos: list[RawYouTubeVideo],
    today: date,
) -> TrendClusterSummary:
    # Early-data caveat (README): weekly_* / momentum 베이스라인 부재로 data_maturity=bootstrap,
    # direction=flat, lifecycle=early, score=0 폴백. 실제 스코어링은 후속 단계의 §9 공식으로 대체.
    return TrendClusterSummary(
        cluster_key=cluster_key,
        display_name=_display_name(cluster_key),
        date=today,
        score=0.0,
        score_breakdown=ScoreBreakdown(
            social=0.0,
            youtube=0.0,
            cultural=0.0,
            momentum=0.0,
        ),
        daily_direction="flat",
        weekly_direction="flat",
        daily_change_pct=0.0,
        weekly_change_pct=0.0,
        lifecycle_stage="early",
        data_maturity="bootstrap",
        drilldown=DrilldownPayload(
            color_palette=[],
            silhouette_distribution={},
            occasion_distribution={},
            styling_distribution={},
            top_posts=[p.post_id for p in posts[:3]],
            top_videos=[v.video_id for v in videos[:3]],
            top_influencers=sorted({p.account_handle for p in posts})[:3],
        ),
        post_count_total=len(posts),
        post_count_today=len(posts),
        avg_engagement_rate=0.0,
        total_video_views=sum(v.view_count for v in videos),
    )


def _log_source_distribution(
    posts: list[RawInstagramPost],
    videos: list[RawYouTubeVideo],
) -> None:
    # IG 는 source_type 이 A/B/C 세 가지로 의미 있음. YT 는 단일 "youtube_channel" 이라 고정.
    counts: Counter[str] = Counter(f"instagram/{p.source_type.value}" for p in posts)
    counts["youtube/youtube_channel"] = len(videos)
    for source_type, count in sorted(counts.items()):
        logger.info("source=%s count=%d", source_type, count)


def _run(settings: Settings, today: date) -> None:
    sample_dir = settings.paths.sample_data
    outputs_dir = settings.paths.outputs

    posts = load_instagram_samples(sample_dir / "sample_instagram_posts.json")
    videos = load_youtube_samples(sample_dir / "sample_youtube_videos.json")
    logger.info("loaded instagram=%d youtube=%d", len(posts), len(videos))

    _log_source_distribution(posts, videos)

    buckets: dict[str, list[RawInstagramPost]] = defaultdict(list)
    for post in posts:
        buckets[_cluster_key(post.hashtags)].append(post)
    for key, bucket in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        logger.info("cluster=%s posts=%d", key, len(bucket))

    run_outputs = outputs_dir / today.isoformat()
    run_outputs.mkdir(parents=True, exist_ok=True)
    summaries_path = run_outputs / "summaries.json"
    summaries = [_mock_summary(key, bucket, videos, today) for key, bucket in buckets.items()]
    summaries_path.write_text(
        json.dumps(
            [s.model_dump(mode="json") for s in summaries],
            indent=2,
            ensure_ascii=False,
        )
    )
    logger.info("wrote summaries=%d path=%s", len(summaries), summaries_path)


def main() -> None:
    _run(load_settings(), date.today())


if __name__ == "__main__":
    main()
