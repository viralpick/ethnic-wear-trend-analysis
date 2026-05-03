"""enriched JSON 들의 normalized 필드를 raw 에서 통째 재생성.

Phase 3 (2026-04-30): NormalizedContentItem 에 url_short_tag / engagement_score /
growth_metric / collected_at 등 신 필드 추가됨. 옛 enriched (12주 backfill) 는 옛
schema 라 신 필드 없음 → growth-rate 가중 / dedup-by-url 사실상 비활성.

raw 테이블에서 IG/YT row 를 다시 가져와 normalize 함수로 NormalizedContentItem 재생성.
canonicals / post_palette / brands / occasion 등 Vision/Gemini 결과는 그대로 보존
(Gemini 호출 0). 즉 enriched.normalized 만 in-place 교체.

실행:
  uv run python scripts/backfill_collected_at.py
  uv run python scripts/backfill_collected_at.py --glob 'outputs/backfill/page_*_enriched.json'

전제: `.env` 의 STARROCKS_* 크리덴셜 + `uv sync --extra db`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

import pymysql  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from contracts.common import InstagramSourceType  # noqa: E402
from contracts.raw import RawInstagramPost, RawYouTubeVideo  # noqa: E402
from loaders.starrocks_raw_loader import _BOLLYWOOD_HANDLES, _split_csv  # noqa: E402
from loaders.url_parsing import extract_url_short_tag  # noqa: E402
from normalization.normalize_content import (  # noqa: E402
    normalize_instagram_post,
    normalize_youtube_video,
)

_DEFAULT_GLOB = "outputs/backfill/page_*_enriched.json"
_IG_TABLE = "india_ai_fashion_inatagram_posting"
_YT_TABLE = "india_ai_fashion_youtube_posting"
_PROFILE_TABLE = "india_ai_fashion_inatagram_profile"
_HASHTAG_TABLE = "india_ai_fashion_inatagram_hash_tag_search_result"
_HASHTAG_PLACEHOLDER_DATE = datetime(2025, 1, 1, tzinfo=timezone.utc)


def _connect() -> pymysql.connections.Connection:
    """raw DB 연결 — `loaders.starrocks_connect.connect_raw` 위임 (drift 방지)."""
    from loaders.starrocks_connect import connect_raw
    return connect_raw()


def _chunked(seq: list[str], n: int) -> list[list[str]]:
    return [seq[i:i + n] for i in range(0, len(seq), n)]


def _split_image_video(urls: list[str]) -> tuple[list[str], list[str]]:
    images: list[str] = []
    videos: list[str] = []
    for u in urls:
        if not u:
            continue
        low = u.lower().split("?", 1)[0]
        if low.endswith((".mp4", ".mov", ".webm", ".m4v")):
            videos.append(u)
        else:
            images.append(u)
    return images, videos


def _ig_source_type(entry: str | None, user: str) -> InstagramSourceType:
    if (entry or "").lower() == "hashtag":
        return InstagramSourceType.HASHTAG_TRACKING
    if (user or "").lstrip("@").lower() in _BOLLYWOOD_HANDLES:
        return InstagramSourceType.BOLLYWOOD_DECODE
    return InstagramSourceType.INFLUENCER_FIXED


def _hashtags_from_caption(caption: str) -> list[str]:
    import re
    return re.findall(r"#[A-Za-z0-9_가-힣]+", caption or "")


def _load_followers_index(cur) -> dict[str, int]:
    cur.execute(
        f"SELECT user, MAX(follower_count) AS f FROM {_PROFILE_TABLE} GROUP BY user"
    )
    return {row["user"]: int(row["f"] or 0) for row in cur.fetchall() if row["user"]}


def _lookup_ig_posting(cur, ids: list[str], followers: dict[str, int]) -> dict[str, RawInstagramPost]:
    out: dict[str, RawInstagramPost] = {}
    cols = (
        "id, user, url, posting_at, content, like_count, comment_count, "
        "download_urls, created_at, entry"
    )
    for chunk in _chunked(ids, 500):
        placeholders = ",".join(["%s"] * len(chunk))
        cur.execute(
            f"SELECT {cols} FROM {_IG_TABLE} WHERE id IN ({placeholders})",
            chunk,
        )
        for row in cur.fetchall():
            try:
                images, videos = _split_image_video(_split_csv(row["download_urls"] or ""))
                user = row["user"] or ""
                followers_count = followers.get(user, 0)
                created = row["created_at"]
                if created and created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                posting_at = row["posting_at"]
                if isinstance(posting_at, str):
                    # raw posting_at 이 ISO 'Z' 문자열로 저장된 경우
                    posting_at = datetime.fromisoformat(
                        posting_at.replace("Z", "+00:00")
                    )
                if posting_at and posting_at.tzinfo is None:
                    posting_at = posting_at.replace(tzinfo=timezone.utc)
                out[row["id"]] = RawInstagramPost(
                    post_id=row["id"],
                    source_type=_ig_source_type(row.get("entry"), user),
                    post_url=row.get("url") or None,
                    account_handle=user or None,
                    account_followers=followers_count,
                    image_urls=images,
                    video_urls=videos,
                    caption_text=row["content"] or "",
                    hashtags=_hashtags_from_caption(row["content"] or ""),
                    likes=int(row["like_count"] or 0),
                    comments_count=int(row["comment_count"] or 0),
                    saves=None,
                    post_date=posting_at,
                    collected_at=created or datetime.now(timezone.utc),
                )
            except Exception as exc:
                print(f"  [skip ig id={row.get('id')}] {exc}", file=sys.stderr)
    return out


def _lookup_ig_hashtag_search(cur, ids: list[str]) -> dict[str, RawInstagramPost]:
    """hashtag_search_result 테이블 — 일부 enriched 의 source_post_id 가 여기에서 옴."""
    out: dict[str, RawInstagramPost] = {}
    # row layout from tsv_raw_loader: id, original_post_id?, hashtag, ?, image_url,
    # caption, likes, comments, created_at, ... — 정확한 column 명은 알 수 없으므로
    # SELECT * 후 dict key 추측.
    for chunk in _chunked(ids, 500):
        placeholders = ",".join(["%s"] * len(chunk))
        try:
            cur.execute(
                f"SELECT * FROM {_HASHTAG_TABLE} WHERE id IN ({placeholders})",
                chunk,
            )
            rows = cur.fetchall()
        except Exception as exc:
            print(f"  [hashtag table skip] {exc}", file=sys.stderr)
            return out
        for row in rows:
            try:
                created = row.get("created_at")
                if created and getattr(created, "tzinfo", None) is None:
                    created = created.replace(tzinfo=timezone.utc)
                tag = row.get("hashtag") or ""
                content = row.get("content") or row.get("caption") or ""
                image_url = row.get("image_url") or ""
                out[row["id"]] = RawInstagramPost(
                    post_id=row["id"],
                    source_type=InstagramSourceType.HASHTAG_TRACKING,
                    post_url=row.get("url") or None,
                    account_handle=None,
                    account_followers=0,
                    image_urls=[image_url] if image_url else [],
                    video_urls=[],
                    caption_text=content,
                    hashtags=[f"#{tag}"] if tag else _hashtags_from_caption(content),
                    likes=int(row.get("like_count") or row.get("likes") or 0),
                    comments_count=int(row.get("comment_count") or row.get("comments") or 0),
                    saves=None,
                    post_date=_HASHTAG_PLACEHOLDER_DATE,
                    collected_at=created or datetime.now(timezone.utc),
                )
            except Exception as exc:
                print(f"  [skip hashtag id={row.get('id')}] {exc}", file=sys.stderr)
    return out


def _lookup_yt(cur) -> dict[str, RawYouTubeVideo]:
    """YT 는 비교적 작아 전체 SELECT — video_id 키로 매핑."""
    cur.execute(
        f"SELECT id, url, channel, channel_follower_count, title, description, tags, "
        f"thumbnail_url, upload_date, view_count, like_count, comment_count, "
        f"comments, download_urls, created_at FROM {_YT_TABLE}"
    )
    out: dict[str, RawYouTubeVideo] = {}
    for row in cur.fetchall():
        try:
            url = row.get("url") or ""
            short = extract_url_short_tag(url)
            if not short:
                continue
            _, videos = _split_image_video(_split_csv(row.get("download_urls") or ""))
            published = (
                datetime.strptime(row["upload_date"], "%Y%m%d").replace(tzinfo=timezone.utc)
                if row.get("upload_date") else datetime.now(timezone.utc)
            )
            created = row.get("created_at")
            if created and getattr(created, "tzinfo", None) is None:
                created = created.replace(tzinfo=timezone.utc)
            out[short] = RawYouTubeVideo(
                video_id=short,
                video_url=url,
                channel=row.get("channel") or "",
                channel_follower_count=int(row.get("channel_follower_count") or 0),
                title=row.get("title") or "",
                description=row.get("description") or "",
                tags=_split_csv(row.get("tags") or ""),
                thumbnail_url=row.get("thumbnail_url") or "",
                view_count=int(row.get("view_count") or 0),
                like_count=int(row.get("like_count") or 0),
                comment_count=int(row.get("comment_count") or 0),
                top_comments=[
                    x for x in (row.get("comments") or "").split("|") if x
                ],
                published_at=published,
                collected_at=created or datetime.now(timezone.utc),
                video_urls=videos,
            )
        except Exception as exc:
            print(f"  [skip yt id={row.get('id')}] {exc}", file=sys.stderr)
    return out


def _backfill_file(
    path: Path,
    ig_map: dict[str, RawInstagramPost],
    yt_map: dict[str, RawYouTubeVideo],
) -> tuple[int, int, int]:
    """returns (rebuilt, kept_old, total)."""
    items = json.loads(path.read_text())
    if not isinstance(items, list):
        return (0, 0, 0)
    rebuilt = 0
    kept_old = 0
    total = len(items)
    for item in items:
        normalized: dict[str, Any] = item.get("normalized") or {}
        source = normalized.get("source")
        spid = normalized.get("source_post_id") or ""
        new_normalized: dict[str, Any] | None = None
        if source == "instagram" and spid in ig_map:
            new_normalized = normalize_instagram_post(ig_map[spid]).model_dump(mode="json")
        elif source == "youtube":
            short = normalized.get("url_short_tag") or spid
            if short in yt_map:
                new_normalized = normalize_youtube_video(yt_map[short]).model_dump(mode="json")
        if new_normalized is not None:
            item["normalized"] = new_normalized
            rebuilt += 1
        else:
            kept_old += 1
    if rebuilt:
        path.write_text(json.dumps(items, ensure_ascii=False, indent=2))
    return (rebuilt, kept_old, total)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glob", default=_DEFAULT_GLOB)
    args = parser.parse_args()

    paths = sorted(_REPO.glob(args.glob))
    if not paths:
        print(f"no files matched: {args.glob}", file=sys.stderr)
        return 1
    print(f"[backfill_normalized] enriched files: {len(paths)}")

    # 1) source_post_id 모으기
    ig_ids: set[str] = set()
    for p in paths:
        try:
            items = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        for it in items:
            n = it.get("normalized") or {}
            src = n.get("source")
            spid = n.get("source_post_id") or ""
            if not spid:
                continue
            if src == "instagram":
                ig_ids.add(spid)
    print(f"[backfill_normalized] unique IG ids: {len(ig_ids)}")

    # 2) raw lookup
    with _connect() as conn, conn.cursor() as cur:
        followers = _load_followers_index(cur)
        ig_map = _lookup_ig_posting(cur, sorted(ig_ids), followers)
        # posting 테이블에 없는 hashtag_search id 들 채움
        missing = sorted(ig_ids - ig_map.keys())
        if missing:
            ig_map.update(_lookup_ig_hashtag_search(cur, missing))
        yt_map = _lookup_yt(cur)
    print(
        f"[backfill_normalized] resolved IG: {len(ig_map)} / "
        f"YT: {len(yt_map)}"
    )

    # 3) 각 enriched 파일 rebuild
    grand_rebuilt = 0
    grand_kept = 0
    grand_total = 0
    for p in paths:
        rebuilt, kept_old, total = _backfill_file(p, ig_map, yt_map)
        grand_rebuilt += rebuilt
        grand_kept += kept_old
        grand_total += total
        print(f"  {p.name}: rebuilt={rebuilt} kept_old={kept_old}/{total}")
    print(
        f"[backfill_normalized] DONE rebuilt={grand_rebuilt} "
        f"kept_old={grand_kept}/{grand_total}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
