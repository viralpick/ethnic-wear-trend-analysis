# Pipeline Spec — 데이터 계층 / 계산 로직 / DB 적재 (2026-04-27)

> **본 문서의 위치**: 데이터 계층 모델 (4-tier) / 기여도 합성 / 매칭 multiplier / weekly
> representative 적재 / DB 스키마 의 canonical. score / lifecycle / direction 공식의 weekly
> 정의 (§3.4, §3.5) 도 본 문서 self-contained — `docs/spec.md` 별도 참조 없이 적재 구현 가능.
>
> `docs/spec.md` 는 (1) 데모 화면 요건, (2) 수집 대상, (3) 8개 속성 체계 (§4), (4) 텍스트 추출
> 방법론 (§6), (5) VLM 사용 정책 (§7), (6) Daily 파이프라인 순서 (§10), (7) 비용 / 일정 의
> 참조 source. **데이터 계층 / 적재 / 스코어링 공식 (§5, §8, §9.2~§9.4) 은 본 문서가 우선**
> 하며 spec.md 의 해당 섹션은 redirect 처리되어 있다.

---

## 1. 데이터 계층 모델 (4-tier) — **Step 7 적재 완료 (2026-04-27)**

```
Representative (G × T × F 조합, weekly)        ← 트렌드 단위, 화면 노출
   ▲ (다대다, 매칭 1/2.5/5x)
Item (1 IG post 또는 1 YT video)               ← DB 영구 row
   ▲ (1 : N)
CanonicalGroup (한 item 내 동일 옷 군)         ← 추적/디버깅
   ▲ (1 : N)
CanonicalObject (group 의 1 멤버 = 1 BBOX)     ← 추적/디버깅
```

### 1.1 Representative

| 필드 | 정의 | 출처 |
|---|---|---|
| `representative_id` | **숫자 surrogate PK** (auto-increment 또는 hash). FE/BE 가 짧은 키로 참조하기 위함 | DDL 생성 시 부여 |
| `representative_key` | `{garment_type}__{technique}__{fabric}` (3개 모두 결정). 사람이 읽는 명시적 키, unique | `clustering/assign_trend_cluster.py` |
| `display_name` | 사람 읽기용 한글/영문 이름 | 기존 코드 |
| `week_start_date` | 월요일 (IST) | 윈도우 정의 §3 |
| `granularity` | `"weekly"` 고정 (monthly 는 weekly 4 row 합성, §3.3) | — |
| `score_total` | §3.5 weekly score 공식 0~100 | §3.5 |
| `score_breakdown` | social/youtube/cultural/momentum | §3.5 |
| `lifecycle_stage` | early/growth/maturity/decline | §3.4 |
| `weekly_change_pct` | 직전 주 score 대비 변화율 | §3.4 |
| `weekly_direction` | up/flat/down (±5% 임계) | §3.4 |
| `factor_contribution` | `{instagram: float, youtube: float}` (합=1.0) | §2.4 |
| `evidence_ig_post_ids` | **top-K (k=4)**, contribution score desc. 부족 시 그대로 적게 적재 (padding 없음) | §4 |
| `evidence_yt_video_ids` | **top-K (k=4)**, 동상 | §4 |
| `color_palette` | top 5 + etc bucket, ΔE76=10 머지 | §2.3 |
| `silhouette_distribution` | `{value: pct}` map | §2.2 (로직 B) |
| `occasion_distribution` | `{value: pct}` map | §2.1 (로직 A) |
| `styling_combo_distribution` | `{value: pct}` map | §2.1 (로직 A) |
| `garment_type_distribution` | **항상 NULL 적재** (representative 단위 단일값이라 redundant). DDL column 은 보존 (M3 P1 monthly 합성 시 활용 여지) | sink_runner |
| `fabric_distribution` | **항상 NULL 적재** (동상) | sink_runner |
| `technique_distribution` | **항상 NULL 적재** (동상) | sink_runner |
| `trajectory` | 최근 12주 score 시계열 (부족분 = 0) | §3.4 |
| `total_item_contribution` | 이 representative 에 누적된 item contribution 합 (sparse 적재 기준) | §2.4 |
| `schema_version` | 본 문서 버전 | §6.4 |
| `computed_at` | 적재 timestamp (UTC) | §6.4 |

### 1.2 Item (post / video)

> **대시보드 노출**: representative drill-down 시 evidence top 4 의 item 상세를 클릭하면
> 본 테이블의 row 가 노출 가능하다 (post 본문, palette, distribution 모두). 따라서
> raw 데이터 (image url, caption 등) 와의 join 이 핵심 — `(source, source_post_id)` 가
> raw png DB (`posting`, `youtube` 테이블) 의 PK 와 1:1 매칭.

| 필드 | 정의 | 출처 |
|---|---|---|
| `item_id` | **DDL column 아님** — sink_runner 내부 mapping (`item_id ↔ source_post_id`) 합성 식별자. DB PK 는 `(source, source_post_id, computed_at)` composite | sink_runner |
| `source` | `"instagram" | "youtube"` (이번 phase 양쪽 모두 적재. 향후 다른 source 추가 시 확장). **DB PK 일부** | normalized |
| `source_post_id` | raw DB PK 와 매칭. join 시 직접 사용. **DB PK 일부** | normalized |
| `posted_at` | post created_at (IST 변환) | normalized |
| `garment_type_dist` | 다중값 + % (로직 A) | §2.1 |
| `fabric_dist` | 다중값 + % (로직 A) | §2.1 |
| `technique_dist` | 다중값 + % (로직 A) | §2.1 |
| `silhouette_dist` | 다중값 + % (로직 B, vision-only) | §2.2 |
| `styling_combo_dist` | 다중값 + % (로직 A). text + **vision (M3.I P0 5매핑 적용)** 합성 — `derive_styling_from_outfit` 가 Gemini `dress_as_single`/`lower_garment_type` → StylingCombo 파생 (추가 LLM call 0) | §2.1 |
| `occasion` | 단일값 (rule 또는 gpt) | §2.5 |
| `brands_mentioned` | **multi-brand JSON 배열** `[{name, tier}, ...]` (M3.F brand registry rule). 한 post = 여러 brand 가능 (haul / collab / styled-by). 매칭 source: `account_handle` 1차 + caption `@mention` 모두 dedup. 매칭 0건 시 `[]` | §6.1 |
| `color_palette` | post_palette top 3 + etc bucket | §2.3 |
| `engagement_raw` | IG: like+comment, YT: views | normalized |
| `account_handle` | IG 핸들 또는 YT 채널 | normalized |
| `account_follower_count` | tier 분류용 | normalized |
| `schema_version`, `computed_at` | — | §6.4 |

> **적재 cadence (정정)**: post 처음 분석된 시점에 적재.
> **spec / 알고리즘 변경에 따른 upsert 가능** — 같은 `(source, source_post_id)` 에 대해
> `computed_at` 갱신과 함께 attribute 재추출 결과를 덮어쓸 수 있다. history 보존이 필요한
> 경우 `(source, source_post_id, computed_at)` 단위 append-only 도 가능 (§5.3).

### 1.3 CanonicalGroup

| 필드 | 정의 | 출처 |
|---|---|---|
| `group_id` | `(item_id, canonical_index)` 합성 | vision pipeline |
| `item_id` | 부모 item | FK |
| `garment_type` | 단일값 (다수결 + tie-break: 평균 면적 큰 group 우선) | §2.6 |
| `fabric` | 동상 | §2.6 |
| `technique` | 동상 | §2.6 |
| `silhouette` | 동상 | §2.6 |
| `styling_combo` | canonical 단일값 (M3.I 매핑 — `derive_styling_from_outfit(canonical.representative)`). P0 (2026-04-24): standalone / with_palazzo / with_churidar / with_pants / with_jeans. P1 (2026-04-28, prompt v0.8): co_ord_set (`is_co_ord_set` 슬롯) / with_dupatta / with_jacket (`outer_layer` 슬롯). 추가 LLM call 0 | §6.1 |
| `color_palette` | canonical palette top 3 + etc bucket | §2.3 |
| `item_contribution_score` | item 내 기여도 (등장횟수 × 면적 곱셈 log scale) | §2.7 |
| `n_objects` | 멤버 객체 수 | — |
| `mean_area_ratio` | 객체 면적 비율 평균 | — |
| `schema_version`, `computed_at` | — | §6.4 |

> **적재 cadence**: item 적재 시점에 같이 적재. **spec / 알고리즘 변경 시 upsert 가능**
> (같은 `(item_id, canonical_index)` 덮어쓰기). history 누적 시 `computed_at` PK 포함 (§5.3).

### 1.4 CanonicalObject

| 필드 | 정의 | 출처 |
|---|---|---|
| `object_id` | `(group_id, member_index)` 합성 | vision |
| `group_id` | 부모 group | FK |
| `item_id` | 부모 item (편의 redundant key) | FK |
| `media_ref` | IG: **Azure Blob full path 형태 raw URL** (예: `collectify/poc/.../...jpg`, SAS query string 제외). `OutfitMember.image_id` (path basename) ↔ `normalized.image_urls` basename 매칭 후 raw URL 적재. 매칭 없으면 NULL. YT: `video_id` (raw `youtube.id` ULID, video 단위 단일 reference) | §6.2 + `row_builder._resolve_media_ref` |
| `garment_type` | gemini 원시값 (free-form word) | vision |
| `fabric` | 동상 | vision |
| `technique` | 동상 | vision |
| `silhouette` | gemini 원시값 (Silhouette enum) | vision |
| `styling_combo` | canonical 단일값 (group 측과 동일 — object 단위 변동 없음). M3.I P0 5매핑 | §6.1 |
| `color_palette` | object 단위 픽셀 KMeans top 3 + etc bucket (**이번 phase 신규 구현 — §6.5**) | vision |
| `area_ratio` | person_bbox area | vision |
| `group_contribution_score` | group 내 기여도 (면적 log scale 단축) | §2.7 |
| `bbox` | `[x, y, w, h]` normalized | vision |
| `schema_version`, `computed_at` | — | §6.4 |

> **적재 cadence**: item 적재 시점에 같이 적재. **spec / 알고리즘 변경 시 upsert 가능**
> (같은 `(group_id, member_index)` 덮어쓰기).

---

## 2. 계산 로직

### 2.1 로직 A — text 기반 또는 다중 값 가능 속성

대상: `garment_type`, `fabric`, `technique`, `styling_combo`, **and item-level distribution 일반**.

```
입력:
  - text_value: rule 또는 gpt 로 뽑힌 단일값 (None 가능)
  - text_method: "rule" | "gpt" | None
  - canonical_groups: [(group_value, group_to_item_contrib), ...]
출력:
  - distribution: {value: pct} (합=1.0, value=None 제외)

가중치:
  text 기여도 = 6.0 (rule 인 경우)
              | 3.0 (gpt 인 경우)
              | 0.0 (둘 다 None)
  group 기여도 합 G = log2(Σ_i n_objects_i + 1)        [모든 group 의 객체 수 합 → log2 → +1 offset]
  group_i 가 distribution 에 보태는 가중치
    = G × (n_objects_i / Σ n_objects_j)                 [객체 수 비례 분배]

로직:
  totals = defaultdict(float)
  if text_value:
      totals[text_value] += text_weight  # 6 또는 3
  for (group_value, group_share_of_G) in canonical_groups:
      if group_value:
          totals[group_value] += group_share_of_G

  total_sum = sum(totals.values())
  if total_sum == 0: return {}
  return {k: v / total_sum for k, v in totals.items()}
```

**주의**:
- text 와 vision 이 같은 value 를 내면 가중치 합산 (예: 둘 다 "kurta" → 6 + group_share).
- 다른 value 를 내면 별 항목으로 나뉨 (예: text=kurta(6) + group_a=saree(2) → kurta 75%, saree 25%).

### 2.2 로직 B — vision-only 다중 값 속성

대상: `silhouette`. text 채움 없음 (LLMExtractionResult 에 silhouette 필드 없음, vision 만 추출).

```
입력:
  - canonical_groups: [(group_value, group_share_of_G)]
출력:
  - distribution: {value: pct}

로직:
  totals = defaultdict(float)
  G = log2(Σ_i n_objects_i + 1)
  for (group_value, group_share_of_G) in canonical_groups:
      if group_value:
          totals[group_value] += group_share_of_G

  total_sum = sum(totals.values())
  if total_sum == 0: return {}
  return {k: v / total_sum for k, v in totals.items()}
```

### 2.3 Color palette merge (4-tier)

Object → Group → Item → Representative 4 tier 모두 merge 형태.

| 단계 | 입력 | 알고리즘 | 출력 max | 현 코드 위치 |
|---|---|---|---|---|
| Object | object 픽셀 (BBOX 단위 segformer mask) | KMeans (object 단위) | **3** (+ etc) | **이번 phase 신규** — `OutfitMember.palette` + B1 멤버별 KMeans (§6.5) |
| Group | object palette 들 | hybrid β-merge → weighted KMeans (frame_area normalize) → top_n 절단 + cut_off_share | **3** (+ etc) | `vision/hybrid_palette.py` + `canonical_palette.py` (`CanonicalOutfit.palette` max 3) |
| Item | group palette 들 | one-group-one-vote → ΔE76 ≤ 10.0 greedy merge | **3** (+ etc) | `vision/post_palette.py` (`post_palette` max 3) |
| Representative | item palette 들 | one-item-one-vote → ΔE76 ≤ 10.0 greedy merge | **5** (+ etc) | `aggregation/cluster_palette.py` → `TrendClusterSummary.drilldown.color_palette`. sink_runner 는 별도 KMeans 안 돌리고 **summary 의 cluster_palette 를 그대로 representative palette 로 재사용** (cluster_key ↔ representative_key 매칭 시) |

**ΔE76 ≤ 10.0 머지 동작 정의 (사용자 확인 사항)**:
- 두 cluster 의 LAB 거리 ≤ 10 이면 **모두 1개로 병합**.
- 병합 시 share (점유율) 도 **합산**: `merged.share = c1.share + c2.share`.
- 병합 hex 는 share-weighted LAB centroid 평균.
- greedy: 가장 가까운 쌍부터 차례로 병합, 더 이상 ≤ 10 인 쌍이 없을 때까지.
- 병합 후 share desc 정렬하여 max N 으로 cap, 잘린 잔여 share = etc bucket.

**etc bucket**: 머지/cap 후 share 합 < 1.0 이면 잔여 = `etc` 키로 표시 (시각화 회색).
- representative palette 은 정확히 `[{hex, share}, ..., {key: "etc", share: cut_off_share}]` 형태.
- cut_off_share = 0 이면 etc 항목 생략 (FE 가 합 1.0 으로 그릴 수 있음).

### 2.4 Item → Representative 매칭 / 기여도 합성

한 item 은 그 item 의 `garment_type` / `technique` / `fabric` 분포에서 등장한 모든 값의
조합으로 다수의 representative 에 부분 기여한다.

```
item.distributions = {
    "garment_type": {kurta: 0.7, saree: 0.3},
    "technique":    {block_print: 1.0},
    "fabric":       {cotton: 0.6, silk: 0.4},
}

representative_keys 후보 = cross-product 후
  (kurta, block_print, cotton) → 매칭 share = 0.7 × 1.0 × 0.6 = 0.42
  (kurta, block_print, silk)   → 0.7 × 1.0 × 0.4 = 0.28
  (saree, block_print, cotton) → 0.3 × 1.0 × 0.6 = 0.18
  (saree, block_print, silk)   → 0.3 × 1.0 × 0.4 = 0.12

매칭 multiplier:
  N = item 의 garment/tech/fabric 중 representative key 와 일치하는 결정값 개수
  - N=3 (모두 일치) → 5.0x
  - N=2             → 2.5x
  - N=1             → 1.0x
  - N=0 → representative 후보 아님

⚠ N 의 정의 보강:
  item 의 garment/tech/fabric 중 distribution 에 1개 value 만 있고 그게 representative 와
  일치하면 N 에 포함. distribution 에 여러 value 가 있어도 각 value 별로 cross-product
  키가 별도 representative 로 매칭되며, 그 representative 입장에서는 그 value 가 일치하므로
  N 에 포함.
  즉 N 은 "각 representative 입장에서 해당 item 의 G/T/F 가 representative 와 일치한
  필드 개수" — 위 예시는 G/T/F 모두 결정되어 있으므로 항상 N=3, 5.0x multiplier.

item 의 representative_k 에 대한 contribution
  = match_share(item, representative_k)              [위 cross-product 의 share]
    × multiplier(N_k)                                [1.0 / 2.5 / 5.0]
    × item_base_unit                                 [1.0 (현 phase). 향후 engagement 가중 가능]
```

**Representative 의 score 입력**:
- social/youtube/cultural/momentum (§3.5) 계산 시 "이 representative 에 속하는 post" 합산은
  contribution-weighted (이 item 이 이 representative 에 1.0 contribution 이라면 fully count,
  0.42 contribution 이라면 0.42 분만 합산).

**factor_contribution (instagram vs youtube)**:
```
factor_contribution[s] = Σ_{item ∈ source s} contribution_to_representative
                       ─────────────────────────────────────────────────────
                       Σ_{all items} contribution_to_representative
```

**Partial-share representative 적재 정책 (Phase partial(g) 활성화 후, 2026-04-28)**:
- `aggregation.representative_builder.build_contributions` 가 **N=1, 2, 3 모두 emit** —
  multiplier 1.0 / 2.5 / 5.0 비례. 비어있는 axis 는 `unknown` placeholder
  (예: `kurta_set__unknown__cotton` = N=2 partial cluster).
- per-item mass: N=3=1.0 / N=2=0.5 / N=1=0.2 / N=0=0 (multiplier_for_n(N) / multiplier_for_n(3)).
- `assign_shares` / `item_cluster_shares` 도 N<3 시 multiplier_ratio 가중 share 반환 →
  β2 의 N<3 zero contribution 정책이 partial 활성화로 자연스럽게 뒤집힘 (`_build_contexts`
  코드 변경 없이 `assign_shares` 동작 변화로 자동 따라옴).
- effective_item_count (β1) 와 단위 정합 — `Σ assign_shares(item) per item = effective_item_count`.
- partial cluster 도 representative_weekly 에 적재 (multiplier 가 가중치 자동 조절). user 의도:
  "3개 다 매칭되는 post 가 없어도 partial 매칭 representative 가 화면에 떠야". evidence top-K
  정렬은 `contribution = share × multiplier` desc 라 N=3 매칭이 자연 우선, 없으면 N=2/1 노출.

**Share-weighted summary path 진척 (2026-04-28)**:

§2.4 의 share-weighted contribution 정책은 `representative_builder` 측에 적용돼 있었으나
summary path (cluster_summary / scoring) 는 winner-takes-all 이었던 drift 가 6-phase plan
(α → β1 → β2 → β3 → β4 → γ → δ → ε → ζ) 으로 정리 중. 첫 3 phase 완료:

- **Phase α (PR #9 squash, `0bb71d5`)** — `aggregation.representative_builder.item_cluster_shares(item)` —
  G×T×F cross-product share dict (multiplier 미적용 raw share, sum=1.0 per item, N<3 빈 dict).
  scaffolding 만, 호출자 X.
- **Phase β1 (PR #10 squash, `a6ea6a8`)** — `aggregation.representative_builder.effective_item_count(items)` —
  multiplier-scaled denominator (1.0/2.5/5.0 → 1.0/0.5/0.2 정규화). DDL `*_summary` 테이블에
  `effective_item_count DOUBLE` 컬럼 추가. `row_builder` / `sink_runner` wiring + 8 pinning.
  단위 정합 (β1 raw scale ↔ β2 minmax 0~100) 은 γ 에서 align 예정 (advisor blocking).
- **Phase β2 (PR #11 squash, `3889852`, 2026-04-28)** —
  `pipelines.run_scoring_pipeline._build_contexts` 가 winner-takes-all 에서 G×T×F cross-product
  share-weighted fan-out 으로 재작성:
  - 1 item 당 contribution 이 cross-product cluster 들에 share 비례로 분산
  - N<3 (G/T/F 한 축이라도 비어있는) item 은 어떤 cluster 에도 기여 0 (assign_shares 빈 dict 정책 정합)
  - mass preservation invariant: G×T×F share 합 = 1.0 per N=3 item
  - `ClusterScoringContext` schema 부분 완화: `youtube_video_count`, `post_count_today` int→float
    (`post_count_total` 는 history schema 호환 위해 int 유지, γ 에서 마이그)
  - `accounts` 는 winner-keyed 유지 (fan-out 으로 부풀려지면 account 식별 의미 손상)
  - `avg_engagement_rate` 분모: `max(len(items), 1)` → `max(post_count_today, 1.0)` (fractional share 합)
  - KeyError 가드: `grouped` 의 모든 winner key 는 N<3 라도 zero-aggregate context 를 받음
    (`score_and_export` 의 `decisions[key]` 룩업 안전)
  - `_decide_clusters` / `score_and_export` 의 history.update 경로는 `round(ctx.post_count_today)` 로
    mixed precision 처리 (γ 에서 history schema float 마이그 후 round 제거)

**Behavior change 결과 (β2 도입 후)**:
- Cluster space 가 G×T×F cross-product 로 확장 → `minmax_same_run` 분모 row 수 증가, 같은 raw count
  라도 normalized score 가 달라질 수 있음 (γ 에서 numerator/denominator align 예정).
- Fan-out-only cluster (β2 score acc 에 있지만 grouped winner 없는 cluster) 은 score 만 계산되고
  summary 는 β3 에서 wire 예정.
- N<3 item 은 이전엔 winner cluster 에 0.x 기여, 지금은 어떤 cluster 에도 0 (representative 측과 정합).

**β3/β4/γ 후속 (미진입)**:
- **β3** — `aggregation.build_cluster_summary.group_by_cluster` share-weighted. fan-out-only cluster
  의 summary wire.
- **partial(g)** — `representative_builder.build_contributions` + `assign_shares` 가 N=1, 2 도 emit.
  multiplier 가중. β2 의 N<3 zero 정책 자동 revisit. user 의도 "partial 매칭도 화면에 떠야" 충족.
- **β4 (origin δ)** — drilldown share-vote (silhouette / occasion / styling 분포 share-weighted).
- **γ** — `score_history` / `score_history_weekly` schema int→float 마이그 + minmax numerator/denominator
  align (β1 effective_item_count 와 단위 정합).
- **ε** — metric watch: `rep_with_summary` 0.230 → 0.9+ 검증.
- **ζ** — `EnrichedContentItem.trend_cluster_key` (winner) → `trend_cluster_shares: dict[str, float]`
  contract 교체 + Case2 picking share-based + `_vision_reassign` 재구현. 5/4 데모 후 후속 (contract
  breaking 큼).

### 2.5 occasion 단일값 결정

Item 단일값. rule 또는 gpt 결과 그대로 (둘 다 None 시 None).
distribution 만들 때는 1표 (다른 attribute 와 동일하게 contribution-weighted vote).

### 2.6 Group 단일값 결정 (다수결 + tie-break)

대상: `garment_type`, `fabric`, `technique`, `silhouette`, `styling_combo` (미구현 NULL).

```
1. 멤버 객체들의 (attribute, value) 카운트
2. 최다 카운트 value 가 1개면 그것
3. 동률이면 tie-break: 동률 value 들의 각 객체 area_ratio 평균 비교 → 큰 쪽
4. 그래도 동률이면 enum sort 의 첫 값 (deterministic)
```

### 2.7 기여도 곱셈 수식

#### Group → Item (group 의 item 내 기여도)
```
group_to_item_contrib = log2(n_objects + 1) × log2(mean_area_ratio × 100 + 1)
                        ──────────────────    ─────────────────────────────
                          등장 횟수 축              면적 축 (0~1 → 0~100)
```
- offset `+1`: log(0) 발산 방지, 두 축 모두 0 일 때 contribution = 0
- log2 base 통일

이 값은 §2.1 / §2.2 의 `group_share_of_G` 계산에 그대로 사용:
```
G = log2(Σ_i n_objects_i + 1)
group_share_of_G_i = G × (group_to_item_contrib_i / Σ_j group_to_item_contrib_j)
```

#### Object → Group (object 의 group 내 기여도)
```
object_to_group_contrib = log2(area_ratio × 100 + 1)
```
단축 (객체 단위 등장 1회 고정).

---

## 3. 시간 윈도우

### 3.1 기준 시간대
- **IST (Asia/Kolkata, UTC+5:30)**.
- post `created_at` / video `published_at` 모두 IST 변환 후 bucket.

### 3.2 Weekly bucket
- **월요일 00:00:00 IST ~ 일요일 23:59:59 IST**.
- `week_start_date` = 그 주 월요일 (date 단위).

### 3.3 Monthly = 4-week rolling
- monthly 화면 요청 시 **최근 4주 weekly representative 를 합성** (별도 monthly table 없음).
- 합성 = 4주 분 distribution / palette 의 contribution-weighted 평균. trajectory 는 12주 그대로.
- **calendar month 가 필요해지면 별도 `monthly_representative` 테이블 추가 (현 phase 외)**.

### 3.4 Direction / Trajectory / Lifecycle (weekly canonical)

**Direction**:
- `weekly_change_pct` = `(이번 주 score - 지난 주 score) / 지난 주 score × 100`
- `weekly_direction` = +5% 이상 → up / -5% 이하 → down / 그 외 → flat
- `trajectory` = 최근 12주 score 시계열 (배열 길이 12, **부족분 = 0**)

**Lifecycle 단계 (weekly 기준)** — early/growth/maturity/decline 4단계:
- **early**: score < 30 + unique_account < 10 + hashtag volume 낮음
- **growth**: score 30~65, **3주 연속 상승**, mega/macro influencer 1+ 참여, 주간 post 증가율 +20%+
- **maturity**: score ≥ 65, **주간 변동 ±5% 이내**, mega 다수 참여
- **decline**: **3주 연속 하락**, hashtag 감소, engagement 하락

> 위 임계값은 데모 데이터 캘리브레이션 기준 초안. 별도 참조 문서 없이 본 §3.4 가 canonical.

### 3.5 Score 공식 (weekly canonical)

가중치 합산 100점, 4 factor:

| factor | 가중치 | weekly raw | normalize | 비고 |
|---|---|---|---|---|
| social | **40** | 이번 주 (월~일 IST) 클러스터 내 모든 post 의 `weighted_engagement` 합. `weighted_engagement = (likes + comments×2 + saves×3) × influencer_weight` (saves null → 0) | 전체 클러스터 중 max 로 나눈 0~1 | influencer_weight: mega(1M+)=3.0 / macro(100K~1M)=2.0 / mid(10K~100K)=1.5 / micro(<10K)=1.0 |
| youtube | **25** | `V × 0.3 + normalize(views) × 0.7` | youtube_raw 자체 normalize | V = 최근 7일 영상 수. **view_growth 제외** (크롤링 미대응) |
| cultural | **15** | `festival_match × 0.6 + bollywood_presence × 0.4` | normalize 0~1 | festival window 는 주간, 매칭 시 ×1.5 부스트. bollywood = source_type=bollywood_decode 1건 이상이면 +0.3. **source_type_weight 추가**: `hashtag_haul` tier max 2.0 (M3.E) — social/momentum factor 의 weighted_engagement 에 곱적용 |
| momentum | **20** | `post_growth × 0.4 + hashtag_velocity × 0.3 + new_account_ratio × 0.3` | normalize 0~1 | post_growth = (이번 주 - 지난 주) / 지난 주. new_account_ratio = 이번 주 first-seen 계정 / 전체 |

```
total_score = social_score + youtube_score + cultural_score + momentum_score   (0~100)
```

> 가중치 비율 근거: Social 40 (IG 데이터 밀도 최고, 가장 강한 선행 시그널) / YouTube 25 (수집 빈도 낮음, 보조) / Cultural 15 (인도 축제·볼리우드 차별화 포인트) / Momentum 20 (급상승 트렌드 부각).

---

## 4. Representative 화면 데이터 (12 항목)

| # | 화면 항목 | 데이터 source | 비고 |
|---|---|---|---|
| 1 | trend lifecycle | §3.4 lifecycle | early/growth/maturity/decline |
| 2 | trend score | §3.5 score 공식 | 0~100 |
| 3 | trajectory | 최근 12주 score 배열 | 부족분 0 |
| 4 | momentum | (이번 주 score - 지난 주 score) | 절대값. weekly_change_pct 와 별도 |
| 5 | growth rate | weekly_change_pct (%) | §3.4 direction |
| 6 | weekly/monthly direction | weekly_direction. monthly = 최근 4주 평균 변화율 | up/flat/down |
| 7 | factor contribution | §2.4 (instagram, youtube 비율) | sum=1.0 |
| 8 | color palette top 5 | §2.3 (ΔE76=10, max 5, etc bucket) | hex + share |
| 9 | silhouette distribution | §2.2 로직 B | %map |
| 10 | occasion distribution | §2.1 로직 A | %map |
| 11 | styling combo distribution | §2.1 로직 A (vision part NULL) | %map |
| 12 | IG / YT evidence top 4~5 | item.contribution_to_representative desc | source 각각 4~5개 |

Default = weekly. monthly toggle 시 §3.3 4-week rolling 합성.

---

## 5. DB 적재 단위

### 5.1 4 테이블

실 DDL 은 StarRocks **DUPLICATE KEY** 모델 (append-only) + `_latest` view 로 최신 row 조회.
"UPSERT" 표현은 의미상 "조회 시 최신 computed_at row 의 의미" 로 해석 (§5.3 참조).

| Table | 1 row 단위 | DUPLICATE KEY (실 DDL) | 적재 정책 |
|---|---|---|---|
| `representative_weekly` | (representative_id, week_start_date) 의 매주 snapshot | `(representative_id, week_start_date, computed_at)` | append-only. `representative_id` = `blake2b(representative_key, digest_size=8)` signed BIGINT |
| `item` | (source, source_post_id) post 단위 | `(source, source_post_id, computed_at)` | append-only. 재계산 결과는 새 `computed_at` 으로 누적, `_latest` view 가 최신 row 노출 |
| `canonical_group` | (item, canonical_index) outfit 단위 | `(item_source, item_source_post_id, canonical_index, computed_at)` | append-only. `group_id VARCHAR(96)` redundant key 보존 (FE join 편의) |
| `canonical_object` | (group, member_index) BBOX 단위 | `(item_source, item_source_post_id, canonical_index, member_index, computed_at)` | append-only. `object_id VARCHAR(112)` + `group_id` redundant key 보존 |

**FK 연결 (논리적, StarRocks 실 FK 없음)**:
- `canonical_object.group_id` ↔ `canonical_group.group_id`
- `canonical_object.(item_source, item_source_post_id)`, `canonical_group.(item_source, item_source_post_id)` ↔ `item.(source, source_post_id)`
- `representative_weekly` 와 `item` 은 직접 join 없음 (다대다, 매칭 share + multiplier 가
  적재 전 representative 계산에 흡수). FE drilldown 시 `evidence_ig/yt_post_ids` 로 join.

### 5.2 Sparse 적재 정책 (Q2-b)

- representative 는 `total_item_contribution > 0` 인 조합만 적재.
- 모든 가능한 G × T × F 조합 (수천) 을 매주 row 만들 필요 없음.
- (representative_key, week_start_date) 가 한 번도 안 나오면 그 주는 row 없음.

### 5.3 멱등성 / 재처리 (Q8-b)

- `schema_version` (string) + `computed_at` (timestamp UTC) 컬럼 모든 테이블 추가.
- 재처리 시 **append-only history** — 같은 (PK, computed_at) 조합으로 row 누적.
- 조회는 `(PK)` 기준 최신 `computed_at` row.
- StarRocks Primary Key 모델 활용 시 `computed_at` 도 PK 일부 (history 보존).

### 5.4 Granularity

- `representative_weekly.granularity = "weekly"` 고정.
- monthly 별도 row 만들지 않음 (Q9-a).
- 향후 monthly 별도 적재 필요해지면 같은 schema 의 `representative_monthly` 또는
  granularity 확장 — 본 phase 외.

---

## 6. 미해결 / 미구현 항목

### 6.1 vision LLM 미구현
- ~~`styling_combo` (vision part)~~ — **M3.I P0+P1 매핑 완료**. P0 (2026-04-24, 5매핑): standalone / with_palazzo / with_churidar / with_pants / with_jeans (`dress_as_single` / `lower_garment_type` 재해석). P1 (2026-04-28, prompt v0.8 슬롯 추가): co_ord_set (`is_co_ord_set`) / with_dupatta / with_jacket (`outer_layer` ∈ {dupatta/shawl/stole/jacket/cardigan/nehru/shrug}). 추가 LLM call 0 — 모두 기존 vision call 결과 재해석. 우선순위: `dress_as_single` > `is_co_ord_set` > `lower_garment_type` > `outer_layer` (specific bottom 이 high-prevalence outer 보다 우선).
- `occasion` (vision part) — vision 으로 안 뽑음. text-only.
- ~~`brand_mentioned`~~ — **M3.F brand registry 적용 완료 (2026-04-28)**. `data/brand_registry.json` 60 brand (handle/alias → display_name + 3-tier). `extract_rule_based` 가 `account_handle` 1차 + caption `@mention` 모두 dedup 수집 → multi-brand JSON 배열 적재. logo OCR 은 추후 별도 phase (brand registry 로 충분히 cover).

### 6.2 YouTube / IG video media_ref (M3.G/H 완료 2026-04-28)
- **IG VideoFrameSource (M3.G)** + **YT video_urls 매핑 (M3.H)** 완료. video 도 frame 단위 vision 처리.
- frame 단위 BBOX 추적은 `OutfitMember.frame_index` (sub-record) 로 처리. canonical_object row 도 frame 별 emit.
- `media_ref` 정책:
  - IG image post: Azure Blob full URL (path basename ↔ `normalized.image_urls` basename 매칭, SAS query 제외).
  - IG video / YT video: `video_id` (raw ULID) + `frame_index` (OutfitMember level). canonical_object row 의 `media_ref` 는 `video_id`.
- 검증: masoomminawala 사리 영상 (IG) e2e + Khushi Malhotra Meesho Haul (YT) e2e. text-only YT 환각 (nose piercing → saree 강제) 해소 확인.

### 6.3 사용자 정정으로 정리된 정책 요약
- B.1 rule 잡히면 gpt 안잡힘 (OR, 가중치 6 vs 3).
- B.2 group 기여도 = `log2(Σ n_objects + 1)`.
- B.3 group 단일값 동률 시 tie-break = 평균 면적 큰 group.
- C.1 representative 매칭 multiplier = (1, 2.5, 5).
- C.2 ~~representative 자체는 "G/T/F 모두 결정된 (unknown 없는)" 조합만 적재~~ —
  **Phase partial(g) 활성화 (2026-04-28) 로 정책 변경**. representative 는 N≥1 (G/T/F
  중 1개 이상 결정) 모두 적재. unknown axis 는 partial key placeholder. multiplier
  (1.0 / 2.5 / 5.0) 가 partial 매칭 가중 자동 조절. user 의도: "3개 다 매칭되는 post
  없어도 partial representative 가 화면에 떠야". per-item mass = multiplier_ratio.
- C.7 factor contribution = source 별 contribution score 합산 비율.
- ~~Q4 vision 결과 제외, brand 제외~~ — vision/brand 모두 **이 phase 에서 적용 완료** (M3.F brand registry + Step 7 4-tier 적재).

### 6.5 Object palette 신규 구현 (이번 phase 적재) — **완료 (2026-04-27 Step 7.6)**

CanonicalObject 단위 palette 가 §1.4 / §2.3 에 명시되어 있으나, 현재 `OutfitMember`
contract 에는 palette 필드 없음. **이번 phase 에 신규 구현 — 검수 페이지 운영 의도**
(BBOX 단위까지 색상 추적 가능해야 review 시 이상 detection 디버깅 가능).

구현 작업:
- contract: `OutfitMember.palette: list[PaletteCluster]` (max 3) + `cut_off_share` (Step 7.6 완료)
- 추출: B1 `canonical_extractor` + `pipeline_b_adapter.finalize_object_palette` 가 멤버별
  top_n cap palette + cut_off_share 채움
- DB: `canonical_object.color_palette` JSON column 적재 검증 완료 (live INSERT 7 row)

### 6.4 Schema 버전
- 본 문서 = `pipeline_spec_v1.0` (2026-04-27).
- `schema_version` 컬럼 = `"pipeline_v1.0"` 적재.
- 본 문서 수정 시 minor (v1.1) 또는 major (v2.0) bump + DDL 수정 검토.

### 6.6 Backend 통신 path — `backend_poster` 폐기
- 분석 결과 → **StarRocks 적재** → BE 가 StarRocks 직접 read (4 base table + 4 `_latest` view + 3 ethnic view).
- 별도 push 채널 (`backend_poster`, HTTP POST sender) **불필요. M3.B 폐기**.
- BE 측 read pattern 은 `_latest` / ethnic view 기준 (raw DUPLICATE KEY 직접 조회 금지).

### 6.7 Haul 시그널 (M3.E 완료 2026-04-24)
- `normalization.haul_tags` = 21개 (15 baseline + M3.F 자동 확장 6개: `amazonhaul`, `tryonhaul`, `ethnicwearhaul`, `haulvideo`, `festivehaul`, `earringhaul`).
- 자동 확장 기준: `*haul$ / *finds$ / tryon* / unboxing` substring + ethnic_share≥0.6 + n≥3.
- `_classify_ig_source_type` 가 `haul_tags ∩ post.hashtags` 검사 → `HASHTAG_HAUL` enum 승격 → `source_type_weight` (max 2.0) social/momentum 가중.

---

## 7. 구현 갭 체크 (Step 7 완료 시점)

본 spec 으로 구현 들어가기 전에 다음을 §2 단계에서 점검:

1. ✅ `frame_source.py` frame_index/total emit — M3.G (IG) + M3.H (YT video_urls) 완료. `OutfitMember.frame_index` sub-record + canonical_object frame 단위 row 적재. e2e smoke (masoomminawala 사리 영상 + Khushi Meesho Haul) 검증 완료
2. ✅ `LLMExtractionResult` 에 silhouette 필드 부재 → 의도 (vision-only). 확인 완료
3. ✅ `EnrichedContentItem` post-level 단일값 → distribution 변환은 **β 결정**: contract 무변경, sink_runner / row_builder 가 in-flight 합성. 구현 완료
4. ✅ `assign_trend_cluster.py` partial 매칭 — **Phase partial(g) 활성화 (2026-04-28)**.
  `build_contributions` 가 N=1, 2, 3 모두 emit (multiplier 1.0 / 2.5 / 5.0). `assign_shares` 도
  N<3 시 multiplier_ratio 가중 share 반환 (unknown placeholder, per-item mass = multiplier_ratio).
  β2 의 N<3 zero contribution 정책은 partial 활성화로 자연 revisit (`_build_contexts` 코드 변경
  없이 `assign_shares` 동작 변화로 따라옴). β3 group_by_cluster share-weighted (2026-04-28) +
  β4 (drilldown share-vote) / γ (history schema 마이그) 미진입. 자세한 내용 §2.4 share-weighted
  summary path 진척 참조
5. ✅ cluster_palette / canonical_palette max ↔ representative max=5 정합 — `aggregation/cluster_palette.py` max 5 + ΔE76=10. 확인 완료
6. ⏸ monthly 4주 합성 헬퍼 부재 — 신규 (M3 P1 lifecycle 단계로 이연)
7. ✅ trajectory history (score_history.json → score_history_weekly.json) — `WeeklyScoreHistory` 12주 window + 부족분 0 패딩. 구현 완료 (Step 5)
8. ✅ `_apply_extraction_result` 무관 — sink_runner / row_builder 안에서만 distribution 빌드. β 결정 적용 완료
9. ✅ `representative_id` surrogate — `blake2b(representative_key, digest_size=8)` signed BIGINT (§5.1). row_builder.py 구현 완료 (xxhash 의존성 부재로 blake2b 채택)
10. ✅ `factor_contribution` (instagram/youtube 합=1.0) — `RepresentativeAggregate.factor_contribution` 적재 단계 계산. 구현 완료
11. ✅ Object palette 신규 구현 (§6.5) — `OutfitMember.palette` + `cut_off_share` + B1 BBOX 단위 KMeans. 구현 완료 (Step 7.6)

---

## 8. Step 7 적재 부록 (2026-04-27)

### 8.1 적재 path 이중화

prod StarRocks 환경에 **8030 (Stream Load HTTP) 가 server-side allow-list 로 차단** 발견 (2026-04-27).
fallback 으로 9030 (query) INSERT writer 채택, 정공법 8030 은 allow-list 수령 후 1회 smoke 로 전환.

| sink | path | writer | 용도 |
|---|---|---|---|
| `--sink starrocks` | 8030 Stream Load HTTP | `StarRocksStreamLoadWriter` | **운영 path** (대량 적재 빠름). 차단 해소 후 진입 |
| `--sink starrocks_insert` | 9030 query INSERT | `StarRocksInsertWriter` | 검증 / 8030 차단 임시 우회. PARSE_JSON('escaped') wrap, multi-row VALUES, autocommit. 대량 시 느림 |
| `--sink fake` | in-memory | `FakeWriter` | 단위 테스트 |

두 writer 모두 `StarRocksWriter` Protocol — `write_batch(table, rows) -> int` 동일 시그니처.

### 8.2 cluster_match 메트릭

`sink_runner._build_representative_rows` 가 cluster_key (summary 측) ↔ representative_key
(aggregate 측) 매칭 진단 emit:

- `summary_coverage` = `|matched| / |summary_keys|`. **1.0 목표** (vision-aware reassign).
  현재 1.000 (Step 7 첫 적재 기준). 1.0 미만이면 cluster_key top-1 ↔ representative_key top-1 어긋남.
- `rep_with_summary` = `|matched| / |aggregate_keys|`. 분포 확산으로 1.0 미만이 정상.
  실측 0.230 — 다수 partial-share representative 가 score 없이 factor_contribution 만 적재.
