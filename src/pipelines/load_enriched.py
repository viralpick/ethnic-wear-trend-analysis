"""enriched.json 파일 로더 + 날짜 필터.

phase=representative 모드에서 여러 batch 의 enriched JSON 파일을 통째로 읽어
posted_at IST 기준으로 [start_date, end_date] 윈도우만 추출.

설계 원칙:
- 같은 source_post_id 가 여러 파일에 있으면 마지막 파일 (glob iteration 순서) 우선.
  enriched.json 은 대부분 unique 라 dedup 은 안전망.
- post_date 는 normalized.post_date — IG: posting_at 파싱, YT: upload_date 파싱 (raw_loader.py).
  둘 다 datetime 으로 정규화되어 있으니 IST 변환 후 date 비교.
"""
from __future__ import annotations

import glob
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from contracts.enriched import EnrichedContentItem
from utils.logging import get_logger

logger = get_logger(__name__)

_IST = timezone(timedelta(hours=5, minutes=30))


def load_enriched_files(glob_pattern: str) -> list[EnrichedContentItem]:
    """glob 패턴으로 enriched.json 파일들 로드 → flat list.

    같은 source_post_id 중복 시 마지막 파일 우선 (glob 정렬 순서). 단 같은
    url_short_tag 의 multi-snapshot 은 모두 유지 (Phase 3, 2026-04-30) — growth rate
    계산용 시계열.
    """
    paths = sorted(glob.glob(glob_pattern))
    if not paths:
        logger.warning("load_enriched_files no_match pattern=%s", glob_pattern)
        return []

    by_post_id: dict[str, EnrichedContentItem] = {}
    for path in paths:
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("load_enriched_skip path=%s reason=%r", path, exc)
            continue
        for raw in data:
            try:
                item = EnrichedContentItem.model_validate(raw)
            except Exception as exc:
                logger.warning(
                    "load_enriched_validate_skip path=%s reason=%r", path, exc,
                )
                continue
            by_post_id[item.normalized.source_post_id] = item

    logger.info(
        "load_enriched_files loaded=%d files=%d pattern=%s",
        len(by_post_id), len(paths), glob_pattern,
    )
    return list(by_post_id.values())


def _to_ist_date(dt: datetime) -> date:
    """datetime → IST date. tzinfo 없으면 UTC 가정 후 IST 변환."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(_IST).date()


def load_raw_post_urls() -> dict[str, str]:
    """raw DB 의 IG/YT 테이블에서 post_id → 외부 URL 매핑 로드. 실패 시 빈 dict.

    dedup_by_raw_url 의 사전 입력. raw DB 의 동일 url 가진 다른 post_id 들 (재크롤링
    등) 을 중복으로 인식하기 위해 필요. STARROCKS_* 환경변수 미설정 시 빈 dict 반환
    (dedup 미적용).
    """
    import os
    out: dict[str, str] = {}
    try:
        import pymysql  # noqa: I001
        from dotenv import load_dotenv
        from pathlib import Path
        load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")
        if not os.environ.get("STARROCKS_HOST"):
            return out
        conn = pymysql.connect(
            host=os.environ["STARROCKS_HOST"],
            port=int(os.environ.get("STARROCKS_PORT", "9030")),
            user=os.environ["STARROCKS_USER"],
            password=os.environ["STARROCKS_PASSWORD"],
            database=os.environ.get("STARROCKS_RAW_DATABASE", "png"),
            connect_timeout=15,
            cursorclass=pymysql.cursors.DictCursor,
        )
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, url FROM india_ai_fashion_inatagram_posting "
                "WHERE url IS NOT NULL AND url != ''"
            )
            for r in cur.fetchall():
                out[r["id"]] = r["url"]
            cur.execute(
                "SELECT id, url FROM india_ai_fashion_youtube_posting "
                "WHERE url IS NOT NULL AND url != ''"
            )
            for r in cur.fetchall():
                out[r["id"]] = r["url"]
        conn.close()
    except Exception as exc:
        logger.warning("load_raw_post_urls failed: %r — dedup will skip", exc)
    return out


def dedup_by_url_short_tag(
    items: list[EnrichedContentItem],
) -> list[EnrichedContentItem]:
    """url_short_tag 기준 dedup — 동일 short_tag 중 가장 최근 (post_date 최대) 1건 keep.

    Phase 3 (2026-04-30): score 계산 시 동일 게시물의 multi-snapshot 이 cluster 점수
    inflate 시키는 걸 방지. growth rate 계산은 별도로 시계열 유지 (compute_growth_rate
    가 모든 snapshot 받아서 처리).

    short_tag 미존재 (url parse 실패) item 은 source_post_id (ULID) 기준 유지 — fallback.
    """
    by_tag: dict[str, list[EnrichedContentItem]] = {}
    no_tag: list[EnrichedContentItem] = []
    for it in items:
        tag = it.normalized.url_short_tag
        if tag:
            by_tag.setdefault(tag, []).append(it)
        else:
            no_tag.append(it)
    out: list[EnrichedContentItem] = list(no_tag)
    dropped = 0
    for tag, group in by_tag.items():
        if len(group) > 1:
            dropped += len(group) - 1
        # 가장 최근 snapshot keep (post_date max). 같은 ULID 의 re-crawl 은 같은 post_date
        # 라 secondary tiebreak 으로 engagement_raw_count desc.
        group.sort(
            key=lambda it: (
                -(it.normalized.post_date.timestamp()),
                -(it.normalized.engagement_raw_count or 0),
            )
        )
        out.append(group[0])
    logger.info(
        "dedup_by_url_short_tag in=%d out=%d dropped=%d no_tag=%d",
        len(items), len(out), dropped, len(no_tag),
    )
    return out


# 옛 dedup_by_raw_url 호환 alias (backwards-compat. Phase 3 후속 cleanup)
def dedup_by_raw_url(
    items: list[EnrichedContentItem],
    post_urls: dict[str, str] | None = None,
) -> list[EnrichedContentItem]:
    """deprecated — Phase 3 이후 url_short_tag 사용. 호환 위해 wrapper 유지.

    post_urls 인자는 backwards-compat 더미 (사용 안 함).
    """
    _ = post_urls
    return dedup_by_url_short_tag(items)


def compute_growth_rate(
    items: list[EnrichedContentItem],
) -> dict[str, tuple[str, float]]:
    """url_short_tag 기준 시계열 → (source, growth_rate) (Δ growth_metric / Δ days).

    growth_metric (2026-04-30): IG=likes, YT=view_count. source 별 다른 단위 (likes
    수십~수천 vs views 수만~수백만) 라 정규화 시 source 별 분리 필요. tuple 의 source
    는 growth_rate_factor_map 에서 source 별 max 로 정규화 위해 함께 반환.

    Δ days 분모는 `collected_at` (크롤 수집 시점). post_date 는 게시일 불변이라
    multi-snapshot 에 동일 → Δ days = 0 → 전부 skip 되는 버그 회피.

    같은 short_tag 의 multiple snapshot 이 있으면 (collected_at asc 정렬):
    - 첫 ↔ 마지막 snapshot 의 growth_metric 차이 / Δ days = growth_rate
    - snapshot 1개, collected_at 누락, Δ days = 0 이면 미수록 → factor=1.0 (default)
    """
    by_tag: dict[str, list[EnrichedContentItem]] = {}
    for it in items:
        tag = it.normalized.url_short_tag
        if not tag:
            continue
        if it.normalized.collected_at is None:
            continue
        by_tag.setdefault(tag, []).append(it)
    out: dict[str, tuple[str, float]] = {}
    for tag, group in by_tag.items():
        if len(group) < 2:
            continue
        group.sort(key=lambda it: it.normalized.collected_at)
        first = group[0]
        last = group[-1]
        delta_days = (
            last.normalized.collected_at - first.normalized.collected_at
        ).total_seconds() / 86400.0
        if delta_days <= 0:
            continue
        delta_metric = (
            (last.normalized.growth_metric or 0)
            - (first.normalized.growth_metric or 0)
        )
        out[tag] = (last.normalized.source.value, delta_metric / delta_days)
    return out


def growth_rate_factor_map(
    growth_by_tag: dict[str, tuple[str, float]],
) -> dict[str, float]:
    """growth_rate dict → item_base_unit 가중 factor dict — source 별 분리 정규화.

    factor = 1 + max(growth_rate, 0) / max_growth_in_source
    - IG (likes/day) 와 YT (views/day) 는 단위가 다르므로 각 source 의 max 로 정규화
    - 음수 growth_rate (감소) → factor = 1.0 (감소 무시, 1.0 floor)
    - growth_rate 미존재 (snapshot 1건만) → factor = 1.0 (caller default)
    - factor 범위: [1.0, 2.0] per source. 같은 source 안 viral post 가 base 의
      최대 2배 가중

    Returns: {url_short_tag: factor}.
    """
    by_source: dict[str, list[float]] = {}
    for source, rate in growth_by_tag.values():
        if rate > 0:
            by_source.setdefault(source, []).append(rate)
    max_by_source = {s: max(rs) for s, rs in by_source.items()}

    out: dict[str, float] = {}
    for tag, (source, rate) in growth_by_tag.items():
        max_src = max_by_source.get(source, 0.0)
        if max_src <= 0 or rate <= 0:
            out[tag] = 1.0
        else:
            out[tag] = 1.0 + rate / max_src
    return out


def filter_by_date_range(
    items: list[EnrichedContentItem],
    *,
    start_date: date,
    end_date: date,
) -> list[EnrichedContentItem]:
    """posted_at IST 기준 [start_date, end_date] 포함 범위 필터.

    start_date / end_date 둘 다 inclusive. start_date <= IST(post_date) <= end_date.
    """
    if start_date > end_date:
        raise ValueError(f"start_date {start_date} > end_date {end_date}")
    out = [
        item for item in items
        if start_date <= _to_ist_date(item.normalized.post_date) <= end_date
    ]
    logger.info(
        "filter_by_date_range start=%s end=%s in=%d out=%d",
        start_date, end_date, len(items), len(out),
    )
    return out


def extract_vision_raw_tags(
    enriched: list[EnrichedContentItem],
) -> dict[str, dict[str, list[str]]]:
    """post_id → category → vision LLM raw 단어 list (Phase 2 Tier 4).

    각 EnrichedContentItem 의 canonicals[*].representative 에서 free-form lowercase
    단어를 category 별로 모음. spec §4.2 v2.3 Tier 4 — unknown_signal_tracker 의
    extra_tags input. signal_type 분류용 source category 동시 보존.

    category 매핑:
    - vision_garment: upper_garment_type / lower_garment_type / outer_layer
    - vision_fabric:  fabric
    - vision_technique: technique

    vision LLM 이 이미 ethnic 판정한 post 라 fashion-context 자동 인정 (build_counters
    안 has_vision_signal 분기). dedup 은 단어 set, 빈 단어/None 제외.
    """
    category_attrs: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("vision_garment", ("upper_garment_type", "lower_garment_type", "outer_layer")),
        ("vision_fabric", ("fabric",)),
        ("vision_technique", ("technique",)),
    )
    out: dict[str, dict[str, list[str]]] = {}
    for item in enriched:
        per_category: dict[str, set[str]] = {}
        for canonical in item.canonicals:
            ref = canonical.representative
            for category, attrs in category_attrs:
                bucket = per_category.setdefault(category, set())
                for attr in attrs:
                    v = getattr(ref, attr, None)
                    if v:
                        bucket.add(str(v).lower())
        if any(per_category.values()):
            out[item.normalized.source_post_id] = {
                category: sorted(words)
                for category, words in per_category.items()
                if words
            }
    return out
