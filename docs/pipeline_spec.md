# Pipeline Spec — 데이터 계층 / 계산 로직 / DB 적재 (2026-04-30)

> **Schema 버전**: `pipeline_spec_v2.1` (2026-04-30). 모든 4 base table 의 `schema_version` 컬럼에 `"pipeline_v2.0"` 적재 (DDL breaking 없음 — minor bump v2.0 → v2.1 은 cluster fan-out 알고리즘 변경, BE/FE 영향 없음).
>
> v2 major bump 사유: cluster axis 3축 → 2축 breaking, NormalizedContentItem schema 확장 (engagement rate / growth metric / url_short_tag / collected_at), text rule G/F/T 폐지.
> v2.1 minor: canonical 단위 cluster fan-out (cross-product 폐기). 가짜 매칭 (multi-canonical post 의 axis mix) 차단.
>
> **본 문서의 위치**: 데이터 계층 모델 (4-tier) / 기여도 합성 / 매칭 multiplier / weekly representative 적재 / DB 스키마 의 canonical. score / lifecycle / direction 공식의 weekly 정의 (§3.4, §3.5) 도 본 문서 self-contained — `docs/spec.md` 별도 참조 없이 적재 구현 가능.
>
> `docs/spec.md` 는 (1) 데모 화면 요건, (2) 수집 대상, (3) 8개 속성 체계 (§4), (4) VLM 사용 정책 (§7), (5) Daily 파이프라인 순서 (§10), (6) 비용 / 일정 의 참조 source. **데이터 계층 / 적재 / 스코어링 공식 (§5, §8, §9.2~§9.4) 은 본 문서가 우선**하며 spec.md 의 해당 섹션은 redirect 처리.

---

## 0. v1 → v2 변경 요약 (2026-04-30 sync 누적)

### 트렌드 클러스터 / 기여도

| 항목 | v1 (~2026-04-29) | v2 (2026-04-30~) |
|---|---|---|
| Cluster axis | `g__t__f` (3축) | **`g__f` (2축)**. Technique 은 cluster 안 distribution |
| 매칭 multiplier | 5.0 / 2.5 / 1.0 (N=3/2/1) | **2.5 / 1.0** (N=2 full / N=1 partial). N=3 폐지 |
| Cluster 최소 단위 | item (post) 1 = 1 contribution | **canonical_group** (post 안 outfit). multi-outfit post 는 outfit 별 fan-out |
| **Cluster fan-out 산출 (v2.1)** | **G × F distribution cross-product** (multi-canonical 의 garment ↔ fabric mix) | **canonical 단위 직접 매핑** (각 canonical 의 (g_i, f_i) 가 자기 cluster 에 group_to_item_contrib 비례 mass). cross-product 폐기 |
| G/F/T distribution source | text rule + vision blend (text weight 6.0 / 3.0) | **vision only** (canonical 1개 이상 시 text 무시) |
| canonicals=[] post | text rule fallback 으로 cluster fan-out | **차단** — vision empty 면 cluster contribute 0 |

### Item / 측정값

| 항목 | v1 | v2 |
|---|---|---|
| URL dedup | 없음 (raw ULID 별 별개 row) | **url_short_tag** 기준 dedup (IG shortcode / YT video_id) |
| Engagement Score | `likes + comments×2 + saves×3` 절대량 | **`(likes/max(followers,100))×1 + (comments/max(followers,100))×2`** rate-based. saves 제외 |
| Engagement raw 보존 | `engagement_raw` (절대량) | `engagement_raw_count` rename + 별도 보존 |
| Growth rate 가중 | 없음 (item_base_unit=1.0) | **growth_metric × Δ days 시계열** → source 별 max 정규화 → factor [1.0, 2.0] |
| Growth Δ days 분모 | — | **`collected_at`** (post_date 는 게시일 불변이라 부적합) |
| `brands_mentioned` | 단일 brand string | **list[BrandInfo]** JSON 배열 (1:N. account_handle ∪ caption mention) |
| `representative_weekly.brand_distribution` | 없음 | **top 5 brand share map** (로직 C, log-scale 균등) |
| `representative_weekly.technique_distribution` | 항상 NULL | **share-weighted distribution map** 채움 |

### 새 시그널 / 검수

| 항목 | v1 | v2 |
|---|---|---|
| 새 해시태그 추천 | 없음 | **`unknown_signal` 테이블 + `_latest` view** (v2.2 emergence rule: baseline 부재 + spike + co-occurrence). **`hashtag_weekly`** raw count source. 검수 HTML weekly panel |
| `_latest` view 정책 | freeze 명목으로 backup 가리킴 | **main 가리킴 + 신 데이터 filter** (G__F 2축만 / url_short_tag IS NOT NULL) |
| 옛 데이터 보존 | — | `*_v1_gtf_backup` 4 테이블 보존 (롤백 가능) |

### Contract / DDL 변경

| 변경 | 위치 |
|---|---|
| `RawYouTubeVideo.collected_at: datetime` 신규 | `contracts/raw.py` |
| `NormalizedContentItem.url_short_tag / collected_at / engagement_score / engagement_raw_count / growth_metric` 신규 | `contracts/normalized.py` |
| `representative_weekly.effective_item_count DOUBLE` ALTER ADD | DDL 정의됐으나 prod 미적용 → 4단계에서 ALTER |
| `representative_weekly.brand_distribution JSON` (migration 003) | 머지 완료 |
| `item / canonical_group / canonical_object.url_short_tag VARCHAR(64)` (migration 004) | 머지 완료 |
| `unknown_signal` 테이블 + view (DDL 07/08) | 신규 |
| `hashtag_weekly` 테이블 + view (DDL 09/10, v2.2) | 신규 |

### 검수 페이지 (review HTML)

- contributor 카드: post 안 모든 canonical 한 줄씩 펼침 + 매칭 cluster 의 canonical 파란 highlight + ★ this cluster badge
- thumbnail grid: cluster 매칭 canonical 의 image_id / video frame 만 (다른 컷 자동 제외)
- YT thumbnail CDN fallback (`img.youtube.com/vi/{vid}/hqdefault.jpg`)
- counts 라인 flex-wrap (옆 셀과 시각적 겹침 fix)

---

## 1. 데이터 계층 모델 (4-tier)

```
Representative (G × F 조합, weekly)            ← 트렌드 단위, 화면 노출
   ▲ (다대다, 매칭 1.0×/2.5×)
Item (1 IG post 또는 1 YT video, URL-dedup)    ← DB 영구 row
   ▲ (1 : N)
CanonicalGroup (한 item 내 동일 옷 군)         ← Cluster fan-out 단위
   ▲ (1 : N)
CanonicalObject (group 의 1 멤버 = 1 BBOX)     ← 추적 / 디버깅
```

### 1.1 Representative (`representative_weekly`)

| 필드 | 정의 | 출처 |
|---|---|---|
| `representative_id` | `blake2b(representative_key, digest_size=8)` signed BIGINT — deterministic surrogate PK | `row_builder.representative_id` |
| `representative_key` | **`{garment_type}__{fabric}`** (2축, 신 v2). 사람이 읽는 명시적 키, unique | `clustering/assign_trend_cluster.py` |
| `display_name` | 사람 읽기용 한글/영문 이름 | 기존 코드 |
| `week_start_date` | 월요일 (IST) | 윈도우 정의 §3 |
| `granularity` | `"weekly"` 고정 (monthly = 4주 합성, §3.3) | — |
| `score_total` | §3.5 weekly score 0~100 | §3.5 |
| `score_breakdown` | `{social, youtube, cultural, momentum, momentum_components: {post_growth, hashtag_velocity, new_ig_account_ratio, new_yt_channel_ratio}}` | §3.5 |
| `lifecycle_stage` | early / growth / maturity / decline | §3.4 |
| `weekly_change_pct` | 직전 주 score 대비 변화율 | §3.4 |
| `weekly_direction` | up / flat / down (±5% 임계) | §3.4 |
| `factor_contribution` | `{instagram, youtube}` (합=1.0) | §2.4 |
| `evidence_ig_post_ids` | top-K (k=4), contribution desc. 부족 시 padding 없음 | §4 |
| `evidence_yt_video_ids` | top-K (k=4), 동상 | §4 |
| `color_palette` | top 5 + etc bucket, ΔE76=10 머지 | §2.3 |
| `silhouette_distribution` | `{value: pct}` map (로직 B) | §2.2 |
| `occasion_distribution` | `{value: pct}` map (text-only) | §2.5 |
| `styling_combo_distribution` | `{value: pct}` map (로직 B, text+vision blend) | §2.2 |
| `garment_type_distribution` | **항상 NULL** (representative 단위 단일값이라 redundant) | sink_runner |
| `fabric_distribution` | **항상 NULL** (동상) | sink_runner |
| `technique_distribution` | **share-weighted distribution map 채움** (v2 신규 — 2축 cluster 의 3rd axis) | §2.2 share-weighted |
| `brand_distribution` | top 5 brand `{name: share}` (로직 C, log-scale 균등). share<0.05 drop. 빈 dict 시 NULL | §2.8 |
| `trajectory` | 최근 12주 score 시계열 (부족분 = 0) | §3.4 |
| `total_item_contribution` | representative 에 누적된 item contribution 합 (sparse 적재 분모) | §2.4 |
| `effective_item_count` | batch 분모 (multiplier-scaled, view normalize 용). β1 phase. **v2 prod ALTER 적용 (2026-04-30)** | §2.4 |
| `schema_version` | `"pipeline_v2.0"` | 머리말 |
| `computed_at` | 적재 timestamp (UTC) | 머리말 |

### 1.2 Item (post / video) — `item`

> **대시보드 노출**: representative drill-down 시 evidence top 4 의 item 상세 클릭 → 본 테이블 row 노출 (post 본문 / palette / distribution). raw 데이터 join 시 `(source, source_post_id)` 가 raw png DB (`posting`, `youtube`) PK 와 1:1 매칭.

| 필드 | 정의 | 출처 |
|---|---|---|
| `source` | `"instagram" | "youtube"` — DB PK 일부 | normalized |
| `source_post_id` | raw DB PK 매칭 — DB PK 일부 | normalized |
| `url_short_tag` | **신 v2** — IG shortcode (`/p/{shortcode}`) / YT video_id (`?v={video_id}`). URL 기반 unique 식별. _latest view dedup 키 | `loaders/url_parsing.py` |
| `posted_at` | post created_at (IST 변환) | normalized |
| `garment_type_dist` | 다중값 + % (vision-only, 로직 A) | §2.1 |
| `fabric_dist` | 다중값 + % (vision-only, 로직 A) | §2.1 |
| `technique_dist` | 다중값 + % (vision-only, 로직 A) | §2.1 |
| `silhouette_dist` | 다중값 + % (로직 B, vision-only) | §2.2 |
| `styling_combo_dist` | 다중값 + % (로직 B, text + vision blend, M3.I) | §2.2 |
| `occasion` | 단일값 (rule 또는 gpt) — text-only | §2.5 |
| `brands_mentioned` | **multi-brand JSON 배열** `[{name, tier}, ...]` (1:N, M3.F brand registry). 매칭 source: `account_handle` 1차 + caption `@mention` dedup | `extract_rule_based` + `data/brand_registry.json` |
| `color_palette` | post_palette top 3 + etc bucket | §2.3 |
| `engagement_raw` | 절대량 보존 (IG: likes+comments×2, YT: views+likes+comments×2) | normalized.engagement_raw_count |
| `account_handle` | IG 핸들 또는 YT 채널 | normalized |
| `account_follower_count` | tier 분류용 + engagement rate 분모 | normalized |
| `schema_version`, `computed_at` | — | 머리말 |

> **적재 cadence**: post 처음 분석된 시점에 적재. **spec/알고리즘 변경 시 upsert 가능** — 같은 `(source, source_post_id)` 에 대해 새 `computed_at` 으로 누적, `_latest` view 가 최신 row 노출. **v2 추가**: `url_short_tag` 별 dedup (같은 URL multi-snapshot 중 가장 최근 1건만 cluster 점수 사용, 나머지는 growth rate 측정용).

### 1.3 CanonicalGroup — `canonical_group`

| 필드 | 정의 | 출처 |
|---|---|---|
| `group_id` | `(item_source, item_source_post_id, canonical_index)` 합성 | vision pipeline |
| `(item_source, item_source_post_id)` | 부모 item FK | — |
| `url_short_tag` | **신 v2** — _latest view dedup 키 | row_builder |
| `garment_type` | 단일값 (다수결 + tie-break: 평균 면적 큰 group 우선) | §2.6 |
| `fabric` | 동상 | §2.6 |
| `technique` | 동상 | §2.6 |
| `silhouette` | 동상 | §2.6 |
| `styling_combo` | canonical 단일값 (M3.I — `derive_styling_from_outfit(canonical.representative)`). P0: standalone / with_palazzo / with_churidar / with_pants / with_jeans. P1 (prompt v0.8): co_ord_set / with_dupatta / with_jacket | `derive_styling_from_outfit` |
| `color_palette` | canonical palette top 3 + etc bucket | §2.3 |
| `item_contribution_score` | item 내 기여도 (등장횟수 × 면적 곱셈 log scale) | §2.7 |
| `n_objects` | 멤버 객체 수 | — |
| `mean_area_ratio` | 객체 면적 비율 평균 | — |
| `schema_version`, `computed_at` | — | 머리말 |

### 1.4 CanonicalObject — `canonical_object`

| 필드 | 정의 | 출처 |
|---|---|---|
| `object_id` | `(group_id, member_index)` 합성 | vision |
| `group_id`, `(item_source, item_source_post_id)` | FK | — |
| `url_short_tag` | **신 v2** — _latest view dedup 키 | row_builder |
| `media_ref` | IG image: Azure Blob full path raw URL (예: `collectify/poc/.../...jpg`, SAS query 제외). IG video / YT: `video_id` (raw ULID) — frame 단위는 `OutfitMember.frame_index` sub-record. M3.G (IG VideoFrameSource) + M3.H (YT video_urls) e2e 검증 완료 | `row_builder._resolve_media_ref` + `vision/frame_source.py` |
| `garment_type` | gemini 원시값 (free-form word) | vision |
| `fabric`, `technique` | 동상 | vision |
| `silhouette` | gemini 원시값 (Silhouette enum) | vision |
| `styling_combo` | canonical 단일값 (group 측과 동일) | `derive_styling_from_outfit` |
| `color_palette` | object 단위 픽셀 KMeans top 3 + etc bucket. `OutfitMember.palette` + `cut_off_share` | `pipeline_b_adapter.finalize_object_palette` |
| `area_ratio` | person_bbox area | vision |
| `group_contribution_score` | group 내 기여도 (면적 log scale) | §2.7 |
| `bbox` | `[x, y, w, h]` normalized | vision |
| `schema_version`, `computed_at` | — | 머리말 |

### 1.5 Unknown Signal — `unknown_signal` (v2.2 emergence rule)

> spec §4.2 / §8.3. 매핑 외 hashtag 가 emergence 룰 (baseline 부재 + spike + co-occurrence) 통과하면 surface. weekly cadence (representative_weekly 와 정합).

#### Surface 룰 (모두 통과)

1. **baseline 부재** — 직전 N일 (default 56) 동안 등장 ≤ floor (default 0)
2. **spike 발생** — 최근 M일 (default 14) 동안 등장 ≥ K (default 3)
3. **ethnic_co_share** — tag 가진 post 들 중 known fashion hashtag 도 가진 비율 ≥ R (default 0.5)
4. **min_posts** — 최소 N posts (default 5) 에서 등장 — measurement stability

CLI override: `--unknown-baseline-days / --unknown-spike-days / --unknown-spike-threshold / --unknown-baseline-floor / --unknown-co-share / --unknown-min-posts`.

#### 컬럼

| 필드 | 정의 |
|---|---|
| `tag` | 해시태그 (# prefix 포함, lowercase) |
| `week_start_date` | weekly anchor 의 주 시작일 (월요일, IST). v2.2 신규 |
| `computed_at` | 적재 시각 (UTC), append-only sort key |
| `count_recent_window` | spike window 안 등장 instance 수. v2.2 신규 |
| `count_3day` | v1 호환 컬럼 — `count_recent_window` 와 같은 값 dump (deprecated) |
| `first_seen` | 최초 발견일 (IST). counters 안 가장 오래된 bucket date |
| `likely_category` | technique? / fabric? 등 추정 (NULL 허용) |
| `reviewed` | 0=pending, 1=reviewed |
| `schema_version` | `"pipeline_v2.2"` |

`unknown_signal_latest` view = (tag, week_start_date) 별 MAX(computed_at) 1 row (week_start_date IS NOT NULL filter).

### 1.6 Hashtag Weekly — `hashtag_weekly` (v2.2 신규)

> spec §4.2/§8.3. emergence rule 평가 source. 모든 hashtag (known + unknown) 의 주별 raw count + co-occurrence. LLM 분류 도입 시 input cache 로 reuse.

| 필드 | 정의 |
|---|---|
| `tag` | 해시태그 (# 미포함, lowercase) |
| `week_start_date` | IST 월요일 (anchor 의 주) |
| `computed_at` | 적재 시각 (UTC) |
| `n_posts` | post-level dedup 카운트 (post 안 같은 tag 여러 번 = +1) |
| `n_instances` | raw instance 카운트 (post 안 중복 포함) |
| `n_posts_with_known_fashion` | 같은 post 에 known fashion hashtag 도 있는 post 수 (co-occurrence numerator) |
| `is_known_mapping` | 0=매핑 외, 1=mapping_tables 의 known hashtag |
| `schema_version` | `"pipeline_v2.2"` |

`hashtag_weekly_latest` view = (tag, week_start_date) 별 MAX(computed_at) 1 row.

ethnic_co_share = `n_posts_with_known_fashion / n_posts`.

---

## 2. 계산 로직

### 2.1 로직 A — vision-only G/F/T distribution (v2)

대상: `garment_type`, `fabric`, `technique`.

> **v2 정책 변경 (2026-04-30)**: text rule 폐지. canonicals 1개 이상이면 vision 만. canonicals=[] 면 빈 distribution (cluster fan-out 차단).
>
> 근거: caption / hashtag 는 post-level 단일 신호라 multi-canonical post 에서 어느 outfit 매칭인지 알 수 없음 → 균등 분배 → canonical 별 정확도 오염. vision (segment-aware, canonical 별 single 매핑) 만 사용.

```
입력:
  - canonical_groups: [(group_value, n_objects_i, mean_area_ratio_i), ...]
출력:
  - distribution: {value: pct} (합=1.0, value=None 제외)

canonicals=[] (vision 결과 빈) → 빈 dict 반환 (cluster fan-out 자동 차단)

가중치:
  group 기여도 합 G = log2(Σ_i n_objects_i + 1)
  group_i 기여도 = G × (n_objects_i / Σ_j n_objects_j)

로직:
  totals = defaultdict(float)
  for (group_value, group_share_of_G) in canonical_groups:
      if group_value:
          totals[group_value] += group_share_of_G
  total_sum = sum(totals.values())
  if total_sum == 0: return {}
  return {k: v / total_sum for k, v in totals.items()}
```

### 2.2 로직 B — vision-canonical fan-out 속성

대상: `silhouette`, `styling_combo`.

- `silhouette`: vision-only (LLMExtractionResult 에 silhouette 필드 없음).
- `styling_combo`: text + vision blend — text 채널 (post-level enum, rule 6.0 / LLM 3.0) + vision 채널 (canonical 단위 `derive_styling_from_outfit`). 합산.

공통: vision 채널은 canonical 단위 단일값을 `group_to_item_contrib (= log2(n+1) × log2(area×100+1))` 비율로 그룹별 분배.

```
입력:
  - text_value: 단일값 또는 None (silhouette 은 항상 None)
  - text_method: rule | gpt | None
  - canonical_groups: [(group_value, n_objects_i, mean_area_ratio_i), ...]
출력:
  - distribution: {value: pct}

가중치:
  text 기여도 = 6.0 (rule) | 3.0 (gpt) | 0.0 (None)
  G = log2(Σ_i n_objects_i + 1)
  group_share_of_G_i = G × group_to_item_contrib_i / Σ_j group_to_item_contrib_j

로직:
  totals = defaultdict(float)
  if text_value:
      totals[text_value] += text_weight  # silhouette 은 항상 0
  for (group_value, group_share_of_G) in canonical_groups:
      if group_value:
          totals[group_value] += group_share_of_G
  total_sum = sum(totals.values())
  if total_sum == 0: return {}
  return {k: v / total_sum for k, v in totals.items()}
```

**Cluster (representative) level aggregation (β4 share-weighted)**:
- 각 (item, cluster_share) 페어에서 per-item dist `dist_i` 계산
- cluster value 가중치 = `Σ_i dist_i[value] × cluster_share_i` → 정규화
- `_share_weighted_dict_aggregate` (build_cluster_summary) 처리

### 2.3 Color palette merge (4-tier)

| 단계 | 입력 | 알고리즘 | 출력 max | 코드 위치 |
|---|---|---|---|---|
| Object | object 픽셀 (BBOX 단위 segformer mask) | KMeans (object 단위) | **3** (+ etc) | `OutfitMember.palette` + B1 멤버별 KMeans |
| Group | object palette 들 | hybrid β-merge → weighted KMeans (frame_area normalize) → top_n 절단 + cut_off_share | **3** (+ etc) | `vision/hybrid_palette.py` + `canonical_palette.py` |
| Item | group palette 들 | one-group-one-vote → ΔE76 ≤ 10.0 greedy merge | **3** (+ etc) | `vision/post_palette.py` |
| Representative | item palette 들 | one-item-one-vote → ΔE76 ≤ 10.0 greedy merge | **5** (+ etc) | `aggregation/cluster_palette.py` → `TrendClusterSummary.drilldown.color_palette`. sink_runner 가 summary palette 그대로 representative palette 로 재사용 |

**ΔE76 ≤ 10.0 머지 동작**:
- 두 cluster 의 LAB 거리 ≤ 10 이면 1개로 병합
- 병합 시 share 합산: `merged.share = c1.share + c2.share`
- 병합 hex = share-weighted LAB centroid 평균
- greedy: 가장 가까운 쌍부터 병합, ≤ 10 인 쌍 없을 때까지
- 병합 후 share desc 정렬하여 max N cap, 잘린 잔여 = etc bucket

**etc bucket**: 머지/cap 후 share 합 < 1.0 이면 잔여 = `etc` 키로 표시 (시각화 회색).

### 2.4 Item → Representative 매칭 / 기여도 합성 (v2.1 canonical 단위 fan-out)

> v2 v2.1 변경 (2026-04-30): cross-product 폐기. **canonical 단위 fan-out** — 각 canonical 의 (g_i, f_i) 가 자기 cluster_key 에 `group_to_item_contrib_i / Σ group_to_item_contrib_j` 비례 mass 등록. multi-canonical post 의 garment ↔ fabric mix 가 만들던 가짜 cluster 매칭 차단.

```
canonicals = [
  c0: (kurta, cotton, 4 members, area=0.35),
  c1: (kurta, cotton, 2 members, area=0.20),    # 같은 cluster 합산
  c2: (saree, georgette, 1 member, area=0.40),
  c3: (kurta, satin, 3 members, area=0.15),
  c4: (kurta, silk, 1 member, area=0.30),
]

각 canonical 의 contrib weight:
  contrib_i = log2(n_objects_i + 1) × log2(mean_area_ratio_i × 100 + 1)

cluster_shares (per item):
  cluster_key 별 합산 share = Σ_i (contrib_i / Σ_j contrib_j)  [ i: cluster 에 매칭되는 canonical만 ]

사례 (위 5 canonical):
  straight_kurta__cotton:    (c0 + c1 의 contrib_share 합)
  casual_saree__georgette:   (c2 의 contrib_share)
  straight_kurta__satin:     (c3 의 contrib_share)
  straight_kurta__silk:      (c4 의 contrib_share)
  ※ casual_saree__satin 같은 가짜 cluster — 어느 canonical 도 (saree, satin) 매칭 X → 등장 X

매칭 multiplier (cluster_key 의 unknown axis 갯수로 결정):
  N=2 (G/F 둘 다 resolved cluster) → 2.5x (full match)
  N=1 (한 axis unknown placeholder, partial) → 1.0x
  N=0 → 후보 아님

per-item mass:
  multiplier_for_n(N) / multiplier_for_n(2) = N=2: 1.0 / N=1: 0.4 / N=0: 0

item 의 representative_k contribution
  = cluster_shares[representative_k]                 [canonical 단위 mass]
    × multiplier(N_k)                                 [1.0 / 2.5]
    × item_base_unit                                  [1.0 default + growth_rate factor 가중]
```

**partial canonical (g O / f X 또는 그 반대)**:
- (g, None) → cluster_key = `{g}__unknown`
- (None, f) → cluster_key = `unknown__{f}`
- (None, None) → drop (의미 없는 mass)

**Growth rate factor (v2 신규)** — `item_base_unit` 이 default 1.0 에서 growth 가중 [1.0, 2.0]:
```
growth_rate_factor[url_short_tag] = 1 + max(growth_rate, 0) / max_growth_in_source
  (source 별 분리 정규화 — IG max likes/day, YT max views/day)
  growth_rate = (last.growth_metric - first.growth_metric) / Δ days
  Δ days = (last.collected_at - first.collected_at).days
  같은 url_short_tag 의 multi-snapshot 시계열 비교
```

**Representative score 입력**:
- social/youtube/cultural/momentum (§3.5) 계산 시 contribution-weighted (item 의 cluster_share × multiplier × growth_factor 분만 합산)

**factor_contribution (instagram vs youtube)**:
```
factor_contribution[s] = Σ_{item ∈ source s} contribution_to_representative
                       ─────────────────────────────────────────────────────
                       Σ_{all items} contribution_to_representative
```

**Partial-share representative 적재 정책**:
- `aggregation.representative_builder.build_contributions` 가 N=1, 2 모두 emit (multiplier 1.0 / 2.5 비례). 비어있는 axis 는 `unknown` placeholder (예: `kurta_set__unknown` = N=1 partial cluster)
- `assign_shares` / `item_cluster_shares` 도 N<2 시 multiplier_ratio 가중 share 반환
- `effective_item_count` (β1) 와 단위 정합 — `Σ assign_shares(item) per item = effective_item_count`
- partial cluster 도 representative_weekly 적재 (multiplier 가 가중치 자동 조절)

**Share-weighted summary path (β2 ~ ζ + 옵션 C 누적, v1 시점에 main 종료, v2 에서 2축으로 자동 적용)**:
- Cluster space 가 G×F cross-product 로 확장됐고 partial 활성화로 N=1/2 모두 적재 → `minmax_same_run` 분모 row 수 증가 (γ 단위 정합)
- Fan-out-only cluster 도 β3/β4 로 summary wire
- `EnrichedContentItem.trend_cluster_shares` (ζ) 가 score / summary / picking 의 단일 fan-out source. legacy `trend_cluster_key` 단일 read 는 fallback
- `TrendClusterSummary` / `ClusterDecision.post_count_*` 는 fractional mass 보존 (옵션 C, int → float)

### 2.5 occasion 단일값 결정

text-only. rule 또는 gpt 결과 그대로 (둘 다 None 시 None). distribution 만들 때는 1표 (vision 으로 추론 어려운 메타라 text 유지).

### 2.6 Group 단일값 결정 (다수결 + tie-break)

대상: `garment_type`, `fabric`, `technique`, `silhouette`, `styling_combo`.

```
1. 멤버 객체들의 (attribute, value) 카운트
2. 최다 카운트 value 가 1개면 그것
3. 동률이면 tie-break: 동률 value 들의 각 객체 area_ratio 평균 비교 → 큰 쪽
4. 그래도 동률이면 enum sort 의 첫 값 (deterministic)
```

### 2.7 기여도 곱셈 수식

**Group → Item**:
```
group_to_item_contrib = log2(n_objects + 1) × log2(mean_area_ratio × 100 + 1)
                        ──────────────────    ─────────────────────────────
                          등장 횟수 축              면적 축 (0~1 → 0~100)
```
- offset `+1`: log(0) 발산 방지
- log2 base 통일

§2.1 / §2.2 의 group_share_of_G:
```
G = log2(Σ_i n_objects_i + 1)
group_share_of_G_i = G × (group_to_item_contrib_i / Σ_j group_to_item_contrib_j)
```

**Object → Group**:
```
object_to_group_contrib = log2(area_ratio × 100 + 1)
```
단축 (객체 단위 등장 1회 고정).

### 2.8 로직 C — brand_distribution (categorical multi-label)

대상: `brand_distribution` (representative_weekly only).

> **의도**: post 가 여러 brand 동시 언급 가능. linear 1/N 분배는 multi-brand post 영향력을 너무 가파르게 깎고, 1.0 분배는 haul/collab 한 건이 노이즈로 dominate. 절충: log-scale 영향력 + 균등 분배.

```
입력:
  - cluster 안 (item, cluster_share) 페어. item.brands = list[BrandInfo] (post-level)
  - top_n (default 5), min_share (default 0.05)
출력:
  - distribution: {brand_name: share}, share desc insertion order, sum=1.0 또는 빈 dict

per-post log-scale 영향력:
  N = post.brands list 길이 (dedup 후 — account_handle ∪ caption mention)
  post_log_weight = 1 / log2(N + 1)
    N=1 → 1.0     (단독 brand)
    N=2 → 0.631
    N=3 → 0.500
    N=5 → 0.387

균등 분배:
  per_brand_contribution = cluster_share × post_log_weight × (1/N)
  for brand in dedup(post.brands):
      raw[brand] += per_brand_contribution

후처리 (top N + threshold drop):
  1차 정규화 → share desc 정렬 (name asc tiebreak)
  filter: share ≥ min_share
  cut: top_n
  2차 정규화 (살아남은 entry 합 = 1.0)
```

**구현**: `aggregation/brand_distribution.compute_brand_distribution`. `make_drilldown` 호출. DDL: `representative_weekly.brand_distribution JSON NULL` (migration 003).

### 2.9 Engagement Score (v2 신규 정의)

> **의도**: 절대량 합 (likes + comments × 2 + saves × 3) 이 follower 1M 셀럽을 follower 100K 인플루언서 보다 항상 높게 만드는 문제 해소. follower 대비 rate 측정.

```
engagement_score = (likes / max(followers, 100)) × 1
                 + (comments / max(followers, 100)) × 2
saves 제외 (raw DB 미수집)
100 follower fallback (신규/private 계정 가드)

absolute 보존:
  engagement_raw_count = likes + comments × 2  (IG)
                       = view_count + likes + comments × 2  (YT)
  cluster summary 의 top_posts 정렬에 절대량 사용
```

### 2.10 Growth Metric (v2 신규)

> **의도**: 같은 콘텐츠의 multi-snapshot 시계열로 성장률 측정. Cluster fan-out 시 빠르게 자라는 콘텐츠를 가중.

```
growth_metric (per item, source 별):
  IG → likes (post 인기 변화 직접 시그널)
  YT → view_count (영상 노출 확대)

시계열 분모:
  Δ days = (last.collected_at - first.collected_at) / 86400
  collected_at = 크롤 수집 시점 (post_date 는 게시일 불변이라 부적합 — 이전 v1 misuse 시
  Δ days = 0 으로 모든 entry 미수록되던 버그)

source 별 정규화:
  growth_rate = (last.growth_metric - first.growth_metric) / Δ days
  factor = 1 + max(growth_rate, 0) / max_growth_in_source
  range: [1.0, 2.0]  (음수 growth → 1.0 floor)
  IG / YT 단위 다른 metric 이므로 source 별 max 분리 정규화

미수록 조건:
  - snapshot 1개만 (시계열 비교 불가)
  - Δ days = 0 (동시 수집 snapshot)
  - collected_at 누락 (옛 enriched JSON)
  → factor = 1.0 (default, 가중 없음)
```

---

## 3. 시간 윈도우

### 3.1 기준 시간대
- IST (Asia/Kolkata, UTC+5:30)
- post `created_at` / video `published_at` 모두 IST 변환 후 bucket

### 3.2 Weekly bucket
- 월요일 00:00:00 IST ~ 일요일 23:59:59 IST
- `week_start_date` = 그 주 월요일 (date 단위)

### 3.3 Monthly = 4-week rolling
- monthly 화면 요청 시 최근 4주 weekly representative 합성 (별도 monthly table 없음)
- 합성 = 4주 분 distribution / palette 의 contribution-weighted 평균. trajectory 는 12주 그대로
- spec §3.3 sparse 0 padding: 등장 안한 주는 score=0 으로 4주 평균 (분모 = 4 고정)
- calendar month 가 필요해지면 별도 `monthly_representative` 테이블 (현 phase 외)

### 3.4 Direction / Trajectory / Lifecycle (weekly canonical)

**Direction**:
- `weekly_change_pct` = `(이번 주 score - 지난 주 score) / 지난 주 score × 100`
- `weekly_direction` = +5% 이상 → up / -5% 이하 → down / 그 외 → flat
- `trajectory` = 최근 12주 score 시계열 (배열 길이 12, 부족분 = 0)

**Lifecycle 단계**:
- **early**: score < 30 + unique_account < 10 + hashtag volume 낮음
- **growth**: score 30~65, 3주 연속 상승, mega/macro influencer 1+ 참여, 주간 post 증가율 +20%+
- **maturity**: score ≥ 65, 주간 변동 ±5% 이내, mega 다수 참여
- **decline**: 3주 연속 하락, hashtag 감소, engagement 하락

> 임계값은 데모 데이터 캘리브레이션 기준. 별도 참조 문서 없이 본 §3.4 가 canonical.

### 3.5 Score 공식 (weekly canonical)

가중치 합산 100점, 4 factor:

| factor | 가중치 | weekly raw | normalize | 비고 |
|---|---|---|---|---|
| social | **40** | 클러스터 내 모든 post 의 `weighted_engagement` 합. `weighted_engagement = engagement_score × influencer_weight × source_type_weight × growth_factor` | 전체 클러스터 max 로 0~1 | engagement_score = rate-based (§2.9). influencer_weight: mega(1M+)=3.0 / macro(100K~1M)=2.0 / mid(10K~100K)=1.5 / micro(<10K)=1.0 |
| youtube | **25** | `V × 0.3 + normalize(views) × 0.7` | youtube_raw 자체 normalize | V = 최근 7일 영상 수. view_growth 제외 |
| cultural | **15** | `festival_match × 0.6 + bollywood_presence × 0.4` | 0~1 | festival 매칭 시 ×1.5 부스트. bollywood = source_type=bollywood_decode 1건 이상 +0.3. **source_type_weight**: hashtag_haul tier max 2.0 (M3.E). `haul_tags` = 21개 (15 baseline + M3.F 자동 확장 6개) |
| momentum | **20** | `post_growth × 0.4 + hashtag_velocity × 0.3 + new_ig_account_ratio × 0.15 + new_yt_channel_ratio × 0.15` | 0~1 | post_growth = (이번 주 - 지난 주) / 지난 주. new_account_ratio B-2 분리 (M3.G/H 후) |

```
total_score = social_score + youtube_score + cultural_score + momentum_score   (0~100)
```

> 가중치 비율 근거: Social 40 (IG 데이터 밀도 최고) / YouTube 25 (수집 빈도 낮음) / Cultural 15 (인도 차별화 포인트) / Momentum 20 (급상승 트렌드).

---

## 4. Representative 화면 데이터 (13 항목)

| # | 화면 항목 | 데이터 source | 비고 |
|---|---|---|---|
| 1 | trend lifecycle | §3.4 lifecycle | early/growth/maturity/decline |
| 2 | trend score | §3.5 score | 0~100 |
| 3 | trajectory | 최근 12주 score 배열 | 부족분 0 |
| 4 | momentum | (이번 주 score - 지난 주 score) | 절대값 |
| 5 | growth rate | weekly_change_pct (%) | §3.4 |
| 6 | weekly/monthly direction | weekly_direction. monthly = 최근 4주 평균 변화율 | up/flat/down |
| 7 | factor contribution | §2.4 (instagram, youtube 비율) | sum=1.0 |
| 8 | color palette top 5 | §2.3 (ΔE76=10, max 5, etc bucket) | hex + share |
| 9 | silhouette distribution | §2.2 로직 B | %map |
| 10 | occasion distribution | §2.5 text-only | %map |
| 11 | styling combo distribution | §2.2 로직 B (text + vision blend) | %map |
| 12 | brand distribution top 5 | §2.8 로직 C | %map (≤5 entries) |
| 13 | technique distribution | §2.2 share-weighted (v2 신규 — 2축 cluster 의 3rd axis) | %map |
| 14 | IG / YT evidence top 4~5 | item.contribution_to_representative desc | source 각각 4~5개 |

Default = weekly. monthly toggle 시 §3.3 4-week rolling 합성.

---

## 5. DB 적재 단위

### 5.1 4 base table + 4 _latest view + 3 ethnic view + unknown_signal

실 DDL 은 StarRocks **DUPLICATE KEY** 모델 (append-only) + `_latest` view 로 최신 row 조회.

| Table | 1 row 단위 | DUPLICATE KEY | 적재 정책 |
|---|---|---|---|
| `representative_weekly` | `(representative_id, week_start_date)` 매주 snapshot | `(representative_id, week_start_date, computed_at)` | append-only. `representative_id` = `blake2b(representative_key, digest_size=8)` signed BIGINT |
| `item` | `(source, source_post_id)` post 단위 | `(source, source_post_id, computed_at)` | append-only. `_latest` view dedup 키 = `url_short_tag` |
| `canonical_group` | `(item, canonical_index)` outfit 단위 | `(item_source, item_source_post_id, canonical_index, computed_at)` | append-only. `group_id VARCHAR(96)` redundant |
| `canonical_object` | `(group, member_index)` BBOX 단위 | `(item_source, item_source_post_id, canonical_index, member_index, computed_at)` | append-only. `object_id VARCHAR(112)` + `group_id` redundant |
| `unknown_signal` (v2.2 emergence) | `(tag, computed_at)` | `(tag, computed_at)` | append-only. weekly cadence (week_start_date 컬럼). emergence rule 통과만 surface |
| `hashtag_weekly` (v2.2 신규) | `(tag, week_start_date, computed_at)` | `(tag, week_start_date, computed_at)` | append-only. 모든 hashtag (known+unknown) 의 주별 raw count + co-occurrence |

**FK 연결 (논리적, StarRocks 실 FK 없음)**:
- `canonical_object.group_id` ↔ `canonical_group.group_id`
- `canonical_object.(item_source, item_source_post_id)`, `canonical_group.(item_source, item_source_post_id)` ↔ `item.(source, source_post_id)`
- `representative_weekly` ↔ `item` 직접 join 없음. FE drilldown 시 `evidence_ig/yt_post_ids` 로 join

### 5.2 _latest view 정책 (v2)

**v1 → v2 변경**: backup 테이블 가리키던 검수 freeze view 를 main + 신 데이터 filter 로 swap (2026-04-30).

| view | source | filter | dedup 키 |
|---|---|---|---|
| `representative_weekly_latest` | `representative_weekly` (main) | `LENGTH(rk) - LENGTH(REPLACE(rk, '__', '')) = 2` (G__F 2축만) | `(representative_id, week_start_date)` MAX(computed_at) |
| `item_latest` | `item` (main) | `url_short_tag IS NOT NULL` | `(source, url_short_tag)` MAX(computed_at) |
| `canonical_group_latest` | `canonical_group` (main) | 동상 | `(item_source, url_short_tag, canonical_index)` MAX |
| `canonical_object_latest` | `canonical_object` (main) | 동상 | `(item_source, url_short_tag, canonical_index, member_index)` MAX |
| `item_ethnic_latest` | `item_latest` wrapper | `color_palette IS NOT NULL` | — |
| `canonical_group_ethnic_latest` | `canonical_group_latest` wrapper | `color_palette IS NOT NULL` | — |
| `canonical_object_ethnic_latest` | `canonical_object_latest` wrapper | `color_palette IS NOT NULL` | — |
| `unknown_signal_latest` | `unknown_signal` | week_start_date IS NOT NULL | `(tag, week_start_date)` MAX(computed_at) |
| `hashtag_weekly_latest` | `hashtag_weekly` | — | `(tag, week_start_date)` MAX(computed_at) |

**옛 데이터 보존**: `*_v1_gtf_backup` 4 테이블 — view 가 가리키지 않으나 DB 에 보존 (롤백 / 비교용).

### 5.3 Sparse 적재 정책
- representative 는 `total_item_contribution > 0` 인 조합만 적재
- 모든 가능한 G × F 조합을 매주 row 만들 필요 없음
- (representative_key, week_start_date) 가 한 번도 안 나오면 그 주는 row 없음

### 5.4 멱등성 / 재처리
- `schema_version` (string, `"pipeline_v2.0"`) + `computed_at` (timestamp UTC) 모든 테이블
- 재처리 시 append-only history — 같은 (PK, computed_at) 조합으로 row 누적
- 조회는 `(PK)` 기준 최신 `computed_at` row (`_latest` view 자동)

### 5.5 Granularity
- `representative_weekly.granularity = "weekly"` 고정
- monthly 별도 row 없음 — read-time 4-week rolling 합성

---

## 6. 잔존 미구현

| 항목 | 상태 | 비고 |
|---|---|---|
| `occasion` (vision part) | 미구현 | text-only |
| `text` 본문 분석 (caption 마이닝) | 비스코프 | unknown_signal 의 다음 단계로 가능 |
| Calendar month rollup | 이연 | 현재 monthly = 4-week rolling. 정확한 월 합산은 별도 |
| ε live smoke metric (picking 다양성) | BLOCKED | Azure Blob VPN ON + Gemini API 비용 |

---

## 7. 정책 결정 사항

### 7.1 v2 정책 요약

- **B.1** rule 잡히면 gpt 안잡힘 (OR, 가중치 6 vs 3) — styling_combo / occasion 만 적용. **G/F/T 는 v2 부터 vision-only**
- **B.2** group 기여도 = `log2(Σ n_objects + 1)`
- **B.3** group 단일값 동률 시 tie-break = 평균 면적 큰 group
- **C.1** representative 매칭 multiplier = (1.0, 2.5) — v2 부터 2축 (옛 5/2.5/1 폐지)
- **C.2** representative 는 N≥1 (G/F 중 1개 이상 결정) 모두 적재. unknown axis 는 partial placeholder. multiplier 가 partial 매칭 가중 자동 조절
- **C.7** factor_contribution = source 별 contribution score 합산 비율
- **D.1** (v2 신규) canonicals=[] post 는 cluster fan-out 차단 — vision 결과 빈 게시물의 text rule fallback 폐지
- **D.2** (v2 신규) G/F/T distribution 은 vision-only — caption / hashtag 키워드 매칭 폐지 (multi-canonical false positive 차단)
- **D.3** (v2 신규) URL dedup — 같은 콘텐츠 multi-snapshot 의 가장 최근 1건만 cluster 점수에 사용
- **D.4** (v2 신규) Engagement Score = rate-based ((likes + comments×2) / max(followers, 100))
- **D.5** (v2 신규) Growth rate factor [1.0, 2.0] — collected_at 시계열, source 별 max 정규화
- **D.6** (v2.1 신규) Cluster fan-out 산출 — **canonical 단위 직접 매핑** (cross-product 폐기). 각 canonical 의 (g, f) 가 자기 cluster 에 group_to_item_contrib 비례 mass. multi-canonical post 의 가짜 cluster 매칭 (예: canonical 0=kurta+cotton, canonical 3=saree+satin → cross-product `casual_saree__satin` 가짜) 차단

### 7.2 Backend 통신 path — `backend_poster` 폐기

- 분석 결과 → StarRocks 적재 → BE 가 StarRocks 직접 read (4 base + 4 `_latest` + 3 ethnic + 1 unknown_signal_latest + 1 hashtag_weekly_latest view)
- 별도 push 채널 (HTTP POST sender) 불필요. M3.B 폐기
- BE 측 read pattern 은 `_latest` / ethnic view 기준 (raw DUPLICATE KEY 직접 조회 금지)

---

## 8. 구현 갭 체크 (v2 시점)

1. ✅ `frame_source.py` frame_index/total emit — M3.G (IG) + M3.H (YT video_urls). e2e smoke 검증 완료
2. ✅ `LLMExtractionResult` 에 silhouette 필드 부재 → 의도 (vision-only)
3. ✅ `EnrichedContentItem` post-level 단일값 → distribution 변환은 sink_runner / row_builder 가 in-flight 합성
4. ✅ `assign_trend_cluster.py` partial 매칭 — N=1, 2 모두 emit. N<2 multiplier_ratio 가중. β2~ζ + 옵션 C 모두 main 종료 (ε live smoke 만 BLOCKED)
5. ✅ cluster_palette / canonical_palette ↔ representative max=5 정합 (`aggregation/cluster_palette.py` ΔE76=10)
6. ⏸ monthly 4주 합성 헬퍼 — read-time. M3 P1 lifecycle 단계로 이연
7. ✅ trajectory history (`score_history_weekly.json`) — `WeeklyScoreHistory` 12주 window + 부족분 0 패딩
8. ✅ `_apply_extraction_result` 무관 — sink_runner / row_builder 안에서만 distribution 빌드
9. ✅ `representative_id` = `blake2b(representative_key, digest_size=8)` signed BIGINT
10. ✅ `factor_contribution` (instagram/youtube 합=1.0) — `RepresentativeAggregate.factor_contribution` 적재 단계 계산
11. ✅ Object palette — `OutfitMember.palette` + `cut_off_share` + B1 BBOX 단위 KMeans
12. ✅ (v2 신규) `effective_item_count` 컬럼 prod ALTER 적용 + v6 적재로 채움
13. ✅ (v2 신규 + v2.2 갱신) `unknown_signal` 테이블 + view + sink_runner emit + 검수 HTML weekly panel — emergence rule (baseline + spike + co-occurrence)
14. ✅ (v2.2 신규) `hashtag_weekly` 테이블 + view — emergence rule source / LLM 분류 cache
14. ✅ (v2 신규) `url_short_tag` 컬럼 ALTER + `_latest` view dedup 키 swap
15. ✅ (v2 신규) `NormalizedContentItem.collected_at` + `compute_growth_rate` 시계열 정상 동작 (옛 enriched 는 `scripts/backfill_collected_at.py` 로 마이그)

---

## 9. 적재 부록

### 9.1 적재 path 이중화

| sink | path | writer | 용도 |
|---|---|---|---|
| `--sink starrocks` | AKS HTTPS ingress (Stream Load) | `StarRocksStreamLoadWriter` | **운영 path** (`https://starrocks.enhans.ai/api/{db}/{table}/_stream_load`) |
| `--sink starrocks_insert` | 9030 query INSERT | `StarRocksInsertWriter` | 비상 fallback. PARSE_JSON('escaped'), multi-row VALUES, autocommit. 대량 시 느림 |
| `--sink fake` | in-memory | `FakeWriter` | 단위 테스트 |

두 writer 모두 `StarRocksWriter` Protocol — `write_batch(table, rows) -> int`.

### 9.2 cluster_match 메트릭

`sink_runner._build_representative_rows` 가 cluster_key (summary 측) ↔ representative_key (aggregate 측) 매칭 진단:

- `summary_coverage = |matched| / |summary_keys|`. 1.0 목표 (vision-aware reassign)
- `rep_with_summary = |matched| / |aggregate_keys|`. 분포 확산으로 1.0 미만 정상

### 9.3 Phase 분리 CLI

`run_daily_pipeline.py --phase {all, item, item-resync, representative}`:

| phase | Gemini 호출 | 적재 |
|---|---|---|
| `all` | ✅ 신규 raw 처리 | item + group + object + representative |
| `item` | ✅ 신규 raw 처리 | item + group + object (rep X) |
| `item-resync` | ❌ enriched 재사용 | item + group + object 재적재 (algorithm 변경 반영) |
| `representative` | ❌ enriched 재사용 | representative_weekly only (12주 backfill / score_history 누적) |
