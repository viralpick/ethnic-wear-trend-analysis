# spec.md ↔ 현재 구현 갭 감사 (작성일: 2026-04-29)

**원칙**: `spec.md` 는 참고사항 (2026-04-24 정책). 갭이 있으면 `spec.md` 가 outdated 이다. 코드를 `spec.md` 에 맞추지 않는다.

**범례**:
- `✅` spec ↔ code match
- `⚠️` partial drift — some change, core behavior preserved
- `❌` removed — spec item is obsolete/deleted
- `🆕` added — exists in code but not in spec
- `🔁` redirected — delegated to `pipeline_spec.md` or another doc/repo

## §1~§3 배경/수집

| § ref | spec text summary | code reality | status | code path:line |
|---|---|---|---|---|
| §1 | 프로젝트 배경/데모 목적 설명 | 분석 레포에는 배경 로직이 없고, 수집 이후 분석/적재 코드만 있다. | 🔁 | `src/loaders/raw_loader.py:1`, `src/pipelines/run_daily_pipeline.py:1` |
| §2 | 스코어보드/드릴다운 UI 요구 | FE 구현은 없고, BE/분석용 payload contract 만 있다. 스코어보드/드릴다운에 해당하는 shape 는 `TrendClusterSummary`/`DrilldownPayload` 로 제공된다. | ⚠️ | `src/contracts/output.py:37`, `src/exporters/build_backend_payload.py:18` |
| §3, §3.1, §3.2 | Instagram/YouTube 수집 소스, 계정/해시태그/채널 목록 | 현재 레포는 크롤러가 아니라 로더 인터페이스만 가진다. 실제 입력은 `sample_data` 또는 StarRocks raw DB 이고, 수집 목록 관리는 이 레포 밖이다. | 🔁 | `src/loaders/raw_loader.py:1`, `src/loaders/sample_loader.py:1`, `src/loaders/starrocks_raw_loader.py:1` |
| §3.1 C | 볼리우드 디코딩 5계정 | raw loader 는 spec 의 5개 계정을 고정 handle 집합으로 분류해 `bollywood_decode` 로 태깅한다. | ✅ | `src/loaders/starrocks_raw_loader.py:42`, `src/loaders/starrocks_raw_loader.py:82` |

## §3.5 전처리

| § ref | spec text summary | code reality | status | code path:line |
|---|---|---|---|---|
| §3.5 필터링 대상 | 패션/여성 ethnic wear 무관 포스트 제거, dedup | 현재 메인 파이프라인에는 spec 의 "키워드 포함/제외 1차 필터 + LLM 점수화 2차 필터" 가 없다. 대신 vision path 에서 scene/person filter 와 person-bbox 검증을 한다. | ⚠️ | `src/pipelines/run_daily_pipeline.py:233`, `src/vision/scene_filter.py:1`, `src/vision/pipeline_b_extractor.py:223` |
| §3.5 구현 방식(2단계) | 1차 룰 기반 키워드 필터, 2차 LLM 분류 | 구현된 2-stage 는 텍스트 필터가 아니라 CLIP 기반 `Stage 1 accept` + `Stage 2 classify_persons` 이다. spec 의 전처리 절차와 단계 의미가 달라졌다. | ⚠️ | `src/vision/scene_filter.py:3`, `src/vision/scene_filter_clip.py:3`, `src/vision/pipeline_b_extractor.py:228` |
| §3.5 이미지 품질 기준 | 저해상도/blur/의류 작음/얼굴만 보임 제거 | 영상 프레임은 blur/brightness 기반 품질 점수로 거르고, person bbox 는 최소 짧은 변과 skin/garment ratio 로 거른다. 품질 가드는 존재한다. | ✅ | `src/vision/frame_quality.py:18`, `src/vision/video_frame_selector.py:37`, `src/settings.py:310`, `src/vision/pipeline_b_extractor.py:292` |

## §4 속성

| § ref | spec text summary | code reality | status | code path:line |
|---|---|---|---|---|
| §4.1① garment_type | 11개 enum, 텍스트 중심 추출 | enum 과 rule/LLM 추출이 구현돼 있고 canonical fan-out 시 item distribution 으로 합성된다. | ✅ | `src/contracts/common.py:30`, `src/attributes/extract_text_attributes.py:163`, `src/aggregation/item_distribution_builder.py:69` |
| §4.1② fabric | 13개 enum, 텍스트 중심 추출 | enum, rule/LLM, item distribution 합성이 구현돼 있다. | ✅ | `src/contracts/common.py:45`, `src/attributes/extract_text_attributes.py:164`, `src/aggregation/item_distribution_builder.py:102` |
| §4.1③ technique | 14개 enum + embellishment intensity | enum, rule/LLM, embellishment 파생이 구현돼 있다. | ✅ | `src/contracts/common.py:62`, `src/attributes/extract_text_attributes.py:115`, `src/attributes/extract_text_attributes.py:165` |
| §4.1④ color | 포스트당 대표색 1개, RGB 저장/HEX 표시 | **Known drift #4, #11**. 단일 post color 는 폐기됐고 `canonical/post/cluster` 3-tier palette 로 재설계됐다. VLM 은 RGB 를 내지 않고 preset pick 만 하며, 실제 palette 는 pixel/KMeans + F-13 R1/R2/R3/hybrid 규칙으로 계산한다. | ⚠️ | `src/contracts/enriched.py:51`, `src/contracts/common.py:228`, `src/vision/prompts.py:14`, `src/vision/hybrid_palette.py:6` |
| §4.1⑤ silhouette | post-level 단일 silhouette + 텍스트/VLM 보강 | **Known drift #4**. post-level 단일값은 제거됐고 canonical silhouette 을 기반으로 per-item/per-cluster distribution 만 만든다. | ⚠️ | `src/contracts/enriched.py:53`, `src/aggregation/item_distribution_builder.py:147`, `src/aggregation/build_cluster_summary.py:168` |
| §4.1⑥ occasion | 6개 enum, 텍스트 중심 추출 | enum, rule/LLM, drilldown 분포가 구현돼 있다. | ✅ | `src/contracts/common.py:115`, `src/attributes/extract_text_attributes.py:166`, `src/aggregation/build_cluster_summary.py:204` |
| §4.1⑦ styling_combo | 8개 enum, 텍스트 50% + VLM 보강 | **Known drift #10, #14**. 현재는 text 단일값 + vision-side `derive_styling_from_outfit` 를 합산하는 로직 B 이고, `outer_layer`/`is_co_ord_set` 슬롯이 prompt v0.8 에 추가됐다. | ⚠️ | `src/contracts/common.py:125`, `src/attributes/derive_styling_from_vision.py:1`, `src/contracts/vision.py:88`, `src/vision/prompts.py:42`, `src/aggregation/item_distribution_builder.py:163` |
| §4.1⑧ brand_mentioned | brand 1개 자유 텍스트 + tier 후처리 | **Known drift #1**. post-level contract 는 `BrandInfo` 리스트(`brands`) 로 바뀌었고, handle + caption mention 을 dedup 수집한다. | ⚠️ | `src/contracts/enriched.py:27`, `src/contracts/enriched.py:62`, `src/attributes/extract_text_attributes.py:153`, `src/attributes/brand_registry.py:58` |
| §4.2 unknown 속성 자동 감지 | 3일 10건 이상이면 `unknown_attributes` 테이블 등록 | 감지 자체는 구현됐지만 DB table 이 아니라 `outputs/unknown_signals.json` 을 쓴다. 또한 hashtag-only 이고 caption free-text mining 은 없다. | ⚠️ | `src/attributes/unknown_signal_tracker.py:1`, `src/pipelines/run_daily_pipeline.py:247` |
| §4.3 값이 안 나오면 | 해당 값/클러스터가 그냥 안 뜨면 됨 | 구현은 `None`/빈 dict/DB `NULL` 을 허용한다. 분포 합이 0 이면 `{}` 로, StarRocks 적재 시 `NULL` 로 떨어진다. | ✅ | `src/aggregation/distribution_builder.py:96`, `src/exporters/starrocks/row_builder.py:137`, `src/exporters/starrocks/row_builder.py:340` |

## §5 트렌드 클러스터링

| § ref | spec text summary | code reality | status | code path:line |
|---|---|---|---|---|
| §5 전체 | §5 는 historical, canonical 은 `pipeline_spec.md` 로 redirect | **Known drift #12**. 현재 canonical 은 `pipeline_spec_v1.0` 4-tier/weekly representative 이고, code 도 그 모델을 따른다. | 🔁 | `docs/spec.md:503`, `docs/pipeline_spec.md:1`, `src/aggregation/representative_builder.py:1`, `src/exporters/starrocks/ddl/04_representative_weekly.sql:1` |
| §5.1 | cluster key = `garment_type__technique__fabric` | 키 포맷은 그대로 유지된다. | ✅ | `src/clustering/assign_trend_cluster.py:24`, `src/aggregation/representative_builder.py:70` |
| §5.2 | 각 post 는 1개의 primary cluster 에만 1:1 배정 | **Known drift #8, #9**. 현재 canonical path 는 `trend_cluster_shares: dict[str,float]` 와 G×T×F cross-product fan-out 을 사용하고, summary/scoring/picking 경로가 모두 share-weighted 로 합쳐졌다. | 🆕 | `src/contracts/enriched.py:77`, `src/contracts/enriched.py:85`, `src/pipelines/run_daily_pipeline.py:128`, `src/aggregation/build_cluster_summary.py:67`, `src/pipelines/run_scoring_pipeline.py:156` |
| §5.3 | 클러스터 수가 적으면 granularity 낮춤 / 많으면 technique 상위 그룹화 | 옛 granularity 조정 정책은 코드에 없다. 대신 sparse representative 적재와 partial(g) activation 으로 대체됐다. | ❌ | `src/aggregation/representative_builder.py:107`, `src/exporters/starrocks/sink_runner.py:177` |
| §5.4 | Drill-down 속성 분포 | 분포는 유지되지만 post 단일값 카운트가 아니라 share-weighted aggregation 이다. palette 도 `PaletteCluster` 형식이다. | ⚠️ | `src/aggregation/build_cluster_summary.py:179`, `src/aggregation/cluster_palette.py:1`, `src/contracts/common.py:228` |

## §6 텍스트 속성 추출

| § ref | spec text summary | code reality | status | code path:line |
|---|---|---|---|---|
| §6.1 | rule 전수 적용 + 미분류만 LLM fallback | 그대로 구현돼 있다. rule 이 먼저, `None` 필드만 LLM 이 채운다. | ✅ | `src/attributes/extract_text_attributes.py:147`, `src/attributes/extract_text_attributes_llm.py:91` |
| §6.2 | hashtag/keyword 매핑 테이블 | 매핑 테이블은 구현돼 있으나 spec 초안보다 태그/키워드가 확장돼 있고, rule path 는 5개 속성만 다룬다. color/silhouette/brand 는 별도 경로다. | ⚠️ | `src/attributes/mapping_tables.py:1`, `src/attributes/mapping_tables.py:7`, `src/attributes/mapping_tables.py:26` |
| §6.3 | LLM prompt 에 8개 속성 + `brand_mentioned` scalar | 실제 client 는 batch JSON schema 로 호출하고, LLM brand 출력은 `BrandInfo` 리스트의 1건 fallback 으로 흡수된다. scalar 저장은 더 이상 canonical 이 아니다. | ⚠️ | `src/attributes/azure_openai_llm_client.py:39`, `src/attributes/extract_text_attributes_llm.py:35`, `src/attributes/extract_text_attributes_llm.py:144` |

## §7 VLM

| § ref | spec text summary | code reality | status | code path:line |
|---|---|---|---|---|
| §7.1 | 텍스트 우선, VLM 보조 | 현재 파이프라인도 text extraction 후 color/vision extraction 을 돈다. | ✅ | `src/pipelines/run_daily_pipeline.py:233`, `src/pipelines/run_daily_pipeline.py:241` |
| §7.2 | Case1 미분류 포스트, Case2 대표 IG 포스트 | Case1/Case2 선택 로직은 구현돼 있다. 다만 현재 target selection 은 IG-only 이고, case1 조건도 `garment_type is None` 중심으로 좁다. | ⚠️ | `src/pipelines/run_daily_pipeline.py:85`, `src/pipelines/run_daily_pipeline.py:96` |
| §7.3 | VLM color pick prompt, RGB 직접 반환 안 함 | **Known drift #10, #11**. Gemini 2.5 Flash + prompt v0.8 이고, RGB 직접 반환 없이 preset pick 만 한다. prompt 에 `outer_layer`/`is_co_ord_set` 가 추가됐다. | ⚠️ | `src/vision/gemini_client.py:3`, `src/vision/gemini_client.py:45`, `src/vision/prompts.py:7`, `src/vision/prompts.py:42`, `configs/local.yaml:168` |
| §7.4 | YT thumbnail VLM 분석 안 함, 컬러는 영상 프레임 사용 | 현재 raw loader 는 YT `video_urls` 를 로드하고, frame selector/frame source/pipeline B 가 영상 프레임 경로를 지원한다. 썸네일 대신 영상 프레임 컬러 추출 방향으로 구현돼 있다. | ⚠️ | `src/loaders/starrocks_raw_loader.py:115`, `src/contracts/normalized.py:37`, `src/vision/video_frame_selector.py:1`, `src/vision/frame_source.py:95` |
| §7.5 | VLM 비용 | 비용 수치는 코드로 검증할 수 없는 운영 추정치다. | 🔁 | `docs/spec.md:872` |

## §8 데이터 스키마

| § ref | spec text summary | code reality | status | code path:line |
|---|---|---|---|---|
| §8 전체 | §8.1/§8.2 는 pipeline spec 으로 redirect | **Known drift #12, #13**. 현재 StarRocks schema 는 4 base table + 4 `_latest` view + 3 `*_ethnic_latest` view 이다. | 🔁 | `docs/spec.md:884`, `src/exporters/starrocks/ddl/01_item.sql:1`, `src/exporters/starrocks/ddl/05_views_latest.sql:6`, `src/exporters/starrocks/ddl/06_views_ethnic.sql:15` |
| §8.1 | 단일 post 테이블, `brand_mentioned` VARCHAR, 단일값 속성 | item 테이블은 distribution JSON 중심 4-tier 스키마로 바뀌었고, **Known drift #2** 로 `brand_mentioned` 는 `brands_mentioned JSON` 으로 마이그레이션됐다. | ⚠️ | `src/exporters/starrocks/ddl/01_item.sql:4`, `src/exporters/starrocks/migrations/001_brand_1_to_n_2026_04_28.sql:16`, `src/exporters/starrocks/row_builder.py:132` |
| §8.2 | daily cluster table | representative 는 weekly table 로 교체됐고 `_latest`/ethnic view 패턴을 쓴다. summary/output contract 의 `post_count_total/post_count_today` 는 **Known drift #5, #6** 에 따라 float 이지만, StarRocks representative DDL 은 그 필드 대신 `total_item_contribution`/`effective_item_count` 를 저장한다. | ⚠️ | `src/exporters/starrocks/ddl/04_representative_weekly.sql:6`, `src/contracts/output.py:90`, `src/aggregation/build_cluster_summary.py:59`, `src/exporters/starrocks/sink_runner.py:17` |
| §8.3 | `unknown_attributes` 테이블 | 테이블이 아니라 JSON state 파일이다. | ⚠️ | `src/attributes/unknown_signal_tracker.py:7`, `src/pipelines/run_attribute_pipeline.py:102` |

## §9 스코어링

| § ref | spec text summary | code reality | status | code path:line |
|---|---|---|---|---|
| §9.1 | Social 40 / YouTube 25 / Cultural 15 / Momentum 20 | top-level factor cap 은 그대로다. 다만 influencer tier 가중치가 spec 과 달라졌고, `source_type_weights` 가 추가됐다. | ⚠️ | `configs/local.yaml:67`, `configs/local.yaml:89`, `configs/local.yaml:103`, `src/pipelines/run_scoring_pipeline.py:63` |
| §9.2~§9.4 | score/direction/lifecycle 는 `pipeline_spec.md` weekly canonical 로 redirect | redirect 취지는 맞지만 현재 런타임은 daily history 와 `daily_direction` 도 계속 유지하고, YouTube raw 는 여전히 `view_growth` 를 계산해 쓴다. **Known drift #5, #6, #7** 는 반영됐지만 pipeline spec 과는 완전 일치하지 않는다. | ⚠️ | `src/scoring/score_history.py:1`, `src/scoring/score_history_weekly.py:1`, `src/pipelines/run_scoring_pipeline.py:268`, `src/scoring/score_youtube.py:12`, `src/scoring/direction.py:37` |
| §9.5 | Akshaya Tritiya 4/20~4/27 부스트 | 축제 boost/태그 매칭은 구현돼 있지만 실제 window 는 `2026-04-13~2026-05-11` 로 더 넓다. | ⚠️ | `configs/local.yaml:121`, `configs/local.yaml:124`, `src/pipelines/run_scoring_pipeline.py:122` |

## §10 파이프라인 실행

| § ref | spec text summary | code reality | status | code path:line |
|---|---|---|---|---|
| §10 REDIRECT | canonical 은 weekly cadence + pipeline spec | 현재 실행 진입점 이름은 여전히 `run_daily_pipeline` 이고, daily summary/history 와 weekly sink 가 공존한다. redirect 는 맞지만 구현은 순수 weekly-only 로 단순화되어 있지 않다. | ⚠️ | `src/pipelines/run_daily_pipeline.py:1`, `src/pipelines/run_scoring_pipeline.py:312`, `src/scoring/score_history_weekly.py:1` |
| §10.1 현재 단계 | collect → enrichment → matching → scoring → StarRocks | 순서 자체는 대체로 맞고, 중간에 color extraction 이 scoring 전에 들어간다. StarRocks sink 는 optional 이다. | ⚠️ | `src/pipelines/run_daily_pipeline.py:221`, `src/pipelines/run_daily_pipeline.py:241`, `src/pipelines/run_daily_pipeline.py:249`, `src/pipelines/run_daily_pipeline.py:253` |
| §10 / BE 전달 | backend_poster 로 push | **Known drift #3**. `backend_poster.py` 는 현재 레포에 없고, 결과는 StarRocks 4-base-table sink 로 적재한 뒤 BE 가 read 하는 구조다. | ❌ | `src/exporters/starrocks/sink_runner.py:1`, `src/exporters/starrocks/writer.py:21`, `src/exporters/starrocks/ddl/05_views_latest.sql:1` |

## §11~§13 비용/타임라인/참고

| § ref | spec text summary | code reality | status | code path:line |
|---|---|---|---|---|
| §11 | 비용 추정 | 운영/예산 정보로 코드 검증 대상이 아니다. | 🔁 | `docs/spec.md:1197` |
| §12 | 일정/싱크 체크리스트 | 프로젝트 관리 정보로 코드 검증 대상이 아니다. | 🔁 | `docs/spec.md:1213` |
| §13 | 참고/운영 메모 | 참고사항으로 코드 검증 대상이 아니다. | 🔁 | `docs/spec.md:1244` |

## 종합 요약
- 상태 건수: `✅ 10`, `⚠️ 22`, `❌ 2`, `🆕 1`, `🔁 8`
- 가장 큰 드리프트:
  - §4 color/silhouette/brand/styling 이 spec 단일값 모델에서 distribution/3-tier palette/1:N brand/vision-derived styling 으로 바뀌었다.
  - §5 는 사실상 전면 교체됐다. 현재 코드는 `trend_cluster_shares` 기반 다대다 fan-out, share-weighted summary/scoring/picking, partial(g) activation 을 사용한다.
  - §8 은 단일 post/daily cluster 스키마가 아니라 StarRocks 4-tier + `_latest`/`*_ethnic_latest` view 로 운영된다.
  - §9 redirect 이후에도 런타임에는 daily history, `daily_direction`, YouTube `view_growth` 가 남아 있어 `pipeline_spec.md` 와 완전 합치지는 않는다.
  - §10 에서 `backend_poster` 는 폐기됐고 StarRocks direct-read 구조가 canonical 이다.
- 권고:
  - `spec.md` §4 는 body update 가 필요하다. 특히 color, silhouette, styling_combo, brand_mentioned 는 현재 구현을 반영하지 못한다.
  - `spec.md` §5, §8, §9.2~§9.4, §10 은 redirect annotation 이 이미 있지만, 본문 상단에 "historical only" 경고를 더 강하게 넣는 편이 맞다.
  - `spec.md` §3.5 는 현재의 CLIP 2-stage scene/person filter 와 차이가 크므로 별도 redirect 또는 rewrite 가 필요하다.
  - `pipeline_spec.md` 기준으로도 현재 scoring path 에 남아 있는 daily history / `daily_direction` / YouTube `view_growth` 는 추가 정리가 필요하다.
