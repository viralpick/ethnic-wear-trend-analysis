FYI - [260416_AI Fashion Tour India — 데모 개발 작업명세서_ver1](https://www.notion.so/260416_AI-Fashion-Tour-India-_ver1-34402490cbe080aba25fcd9172837ee6?pvs=21) 

# AI Fashion Tour India — 데모 개발 작업명세서

**작성일**: 2026.04.17
**수신**: 이의현, 황현수, 민지홍
**참조**: 김도균
**발신**: 김종휘
**데모일**: 5/4 벵갈루루 ~ 5/8 뭄바이
**수집 시작**: 4/21 (월)
**데모 영상 완성**: 5/1
**예산**: 인건비 제외 500만원 이내
**Feasibility**: 전 항목 Go (260410_AI_Fashion_Tour_Feasibility_v2.xlsx 참조)

---

## 1. 배경 — 이 프로젝트가 뭔지

Microsoft가 인도 패션 리테일러 5개사(Myntra, AJIO, Tata CLiQ, Pantaloons, Tanishq)를 모아서 1:1 미팅을 함. 여기서 Enhans가 20분짜리 데모를 발표함.

데모의 핵심은 이것:

- 인도 여성들이 매일 입는 ethnic wear (특히 cotton/linen kurta set)의 트렌드를
- Instagram과 YouTube에서 자동으로 수집해서
- "지금 뭐가 뜨고 있고, 왜 뜨는지, 올라가는 중인지 내려가는 중인지"를 보여주는 것

만들어야 하는 화면은 딱 2개:

- **트렌드 스코어보드**: 15~20개 트렌드를 점수·방향·라이프사이클과 함께 리스트로 보여줌
- **Explainability Drill-Down**: 트렌드 하나를 클릭하면 "왜 이 점수인지"를 인스타 포스트, 유튜브 영상, 컬러 팔레트 등으로 상세히 보여줌

데이터 소스는 Instagram + YouTube만. 고객사 사이트, 매거진, Pinterest 등은 수집하지 않음. 1차 산출물은 5/1까지 데모 영상 확보. 라이브 시연은 영상 확보 전제 하에 추가 대응.

---

## 2. 데모 화면 요건 — 개발자가 뭘 만들어야 하는지

### 화면 1: 트렌드 스코어보드

이 화면은 "한눈에 지금 뭐가 뜨는지"를 보여주는 메인 대시보드.

15~20개 트렌드 클러스터를 리스트로 표시한다. 각 행에 들어가는 필드:

- **트렌드명**: 시스템이 자동 생성한 이름. 예를 들어 "Chikankari Cotton Kurta Set"처럼 garment_type + technique + fabric 조합으로 만들어짐 (Section 5에서 상세 설명)
- **스코어**: 0~100점. 소셜 버즈, 유튜브 조회수, 문화적 적합도, 상승 모멘텀을 종합한 점수 (Section 9에서 계산식 설명)
- **일별 방향**: 어제 대비 ▲(올라감) ▼(내려감) →(변동 없음)
- **주차별 방향**: 지난 주 대비 ▲▼→
- **라이프사이클**: 이 트렌드가 지금 어느 단계인지 — Early(막 시작), Growth(급성장), Maturity(정점), Decline(하락) 중 하나

UI 요건:

- 기본 정렬은 스코어 내림차순. 클릭하면 다른 기준(예: Momentum순)으로 정렬 변경 가능해야 함
- 라이프사이클에 색상 코딩 적용: Growth=녹색, Early=파랑, Maturity=회색, Decline=적색
- 각 트렌드 행을 클릭하면 화면 2(Drill-Down)로 진입

### 화면 2: Explainability Drill-Down

이 화면이 데모의 하이라이트. 발표에서 여기에 6분을 씀. "이 트렌드가 왜 이 점수인지"를 구체적 증거와 함께 보여주는 화면.

스코어보드에서 트렌드 하나 클릭하면 이 화면으로 진입.

**상단 — 트렌드 요약 영역**:

- 트렌드명 + 스코어 + 라이프사이클 태그
- 팩터별 기여도: 파이 차트 또는 바 차트로 Social / YouTube / Cultural / Momentum 각각의 기여 비율
- **속성 프로필** — 이 부분이 핵심. 카테고리 매니저가 "이걸로 다음 달 상품 기획서 쓸 수 있겠다"고 느끼게 하는 정보:
    - 컬러 팔레트: RGB 분포 기반 클러스터링 → 대표값을 **HEX 변환하여 색상 칩으로 표시** + 비율. 예: `[#B8D4C3] sage 32% [#FFFFFF] white 28% [#E8D5C4] peach 18%`. 저장은 RGB, 표시는 HEX.
    - 실루엣 분포: Straight 55% / A-line 35% / Other 10%
    - 맥락 분포: Office 70% / Casual 20% / Campus 10%
    - 스타일링 분포: With Palazzo 45% / With Pants 30% / Standalone 15%

**중단 — Instagram 근거**:

- 이 트렌드에 속하는 인스타 포스트들을 카드 형태로 나열. 각 카드에: 썸네일 이미지, 계정명, 날짜, 좋아요/댓글 수
- 카드 클릭하면 실제 인스타 포스트로 이동 (외부 링크)
- 이 트렌드 관련 해시태그의 일별 성장률 라인 차트

**하단 — YouTube 근거**:

- 관련 유튜브 영상 카드 나열. 각 카드에: 썸네일, 제목, 채널명, 조회수, 날짜
- 카드 클릭하면 유튜브로 이동
- 조회수 추이 차트

구현 방식: COS 없이 자체 프론트+백엔드 우선 개발. 시간 여유 시 프론트만 COS로 이관. 5/1까지 데모 영상 촬영·편집 완료가 1차 목표.

---

## 3. 수집 대상 — 뭘 어디서 긁어오는지

Instagram과 YouTube만 수집한다. 다른 소스(고객사 사이트, 매거진, Pinterest 등)는 이 데모에서 사용하지 않으므로 수집하지 않음.

**소스 우선순위 (ver2 확정):**
- **해시태그 UGC = 메인 소스** — ethnic wear 관련 포스트를 가장 직접적으로 수집 가능, 최신성 확보
- 인플루언서/볼리우드 계정 = 보조 소스 (초기 seed + cultural signal)
- YouTube = 해시태그 트렌드 크로스 체크 (haul/try-on 중심, 설명/정리 성격)
- 매거진 → 제외 (noise 대비 효용 낮음)
- 해시태그 수집 3일 후 → ethnic wear 비중 높은 계정을 데이터에서 자동 추출 → 4/24 싱크에서 인플루언서 목록 추가 확정

### ⚠️ 해시태그 리스트가 Feasibility 시트 기준에서 변경됨. 아래 리스트가 확정판.

데모 카테고리가 "everyday ethnic"으로 확정되면서, 축제/웨딩 중심 해시태그 6개를 빼고 오피스/일상 중심 해시태그 6개로 교체했음.

### 3-1. Instagram — 매일 수집

3개 소스에서 매일 수집. 전부 기존 Instagram 수집기를 그대로 사용하되, 타겟만 인도 패션 계정/해시태그로 세팅하면 됨.

### A. Top 10 인플루언서 — 인도 여성 패션 메가/매크로 인플루언서

이 10명은 팔로워가 크고 트렌드 확산력이 강한 인플루언서. 이들이 특정 스타일을 입으면 그 자체가 트렌드 시그널.

| # | 계정 |
| --- | --- |
| 1 | @masoomminawala |
| 2 | @juhigodambe |
| 3 | @komalpandeyofficial |
| 4 | @thatbohogirl |
| 5 | @diipakhosla |
| 6 | @shrads |
| 7 | @santoshishetty |
| 8 | @rxjvee |
| 9 | @avneetkaur_13 |
| 10 | @gima_ashi |

각 계정에서 수집할 항목 (포스트 단위):

- 포스트 이미지 URL (여러 장이면 전부)
- 캡션 텍스트 전문
- 해시태그 목록
- 좋아요 수, 댓글 수, 저장 수 (가능한 범위에서)
- 포스트 날짜
- 계정 팔로워 수 (하루 1번 스냅샷 — 인플루언서 티어 분류에 사용)
- 댓글 1 page (수집 중, 분석 미사용 — 후순위)

수집 방식: 매일 1회 체크. 신규 포스트만 수집 (이미 수집한 건 skip).

### B. 해시태그 15개 × Top 50 포스트 — 미드/마이크로 인플루언서 + 일반 사용자까지 포착

A의 Top 10은 메가 인플루언서만 잡힘. 하지만 트렌드는 종종 중소 인플루언서나 일반 사용자에서 먼저 시작됨. 그래서 해시태그 트래킹으로 더 넓은 범위를 본다.

각 해시태그별로 인스타에서 "최신 인기 포스트 Top 50"을 매일 수집. 15개 × 50 = 최대 750건/일이지만, 중복이 많으므로 dedup 처리 필요.

| # | 해시태그 | 변경 여부 |
| --- | --- | --- |
| 1 | #ethnicwear | 유지 |
| 2 | #indianwear | 유지 |
| 3 | #sareelove | 유지 |
| 4 | #kurti | 유지 |
| 5 | #indianfashion | 유지 |
| 6 | #indowestern | 유지 |
| 7 | #bollywoodfashion | 유지 |
| 8 | #EthnicWearGoals | 유지 |
| 9 | #FusionFashion | 유지 |
| 10 | **#officekurta** | **신규** — 구: #indowesternoutfit |
| 11 | **#everydayethnic** | **신규** — 구: #festivefashion2025 |
| 12 | **#kurtaset** | **신규** — 구: #sareepact |
| 13 | **#chikankari** | **신규** — 구: #DesiLook2025 |
| 14 | **#cottonkurta** | **신규** — 구: #lehenga |
| 15 | **#indowesternoffice** | **신규** — 구: #festivewear |
| 16 | #kurtastyle | 신규 — 상품 특화 |
| 17 | #cottonsuit | 신규 — 상품 특화 |
| 18 | #linenkurta | 신규 — 상품 특화 |
| 19 | #summerkurta | 신규 — 상품 특화 |
| 20 | #printedkurta | 신규 — 상품 특화 |
| 21 | #chickenkurta | 신규 — 상품 특화 |
| 22 | #jaipurikurta | 신규 — 상품 특화 |
| 23 | #mulmulkurta | 신규 — 상품 특화 |
| 24 | #palazzoset | 신규 — 상품 특화 |
| 25 | #anarkalisuit | 신규 — 상품 특화 |
| 26 | #blockprintkurta | 신규 — 상품 특화 |
| 27 | #handblockprint | 신규 — 상품 특화 |
| 28 | #bandhani | 신규 — technique/fabric 특화 |
| 29 | #kalamkari | 신규 — technique/fabric 특화 |
| 30 | #lucknowi | 신규 — technique/fabric 특화 |
| 31 | #mirrorwork | 신규 — technique/fabric 특화 |
| 32 | #gotapatti | 신규 — technique/fabric 특화 |
| 33 | #chanderi | 신규 — technique/fabric 특화 |
| 34 | #khadikurta | 신규 — technique/fabric 특화 |
| 35 | #rayonkurta | 신규 — technique/fabric 특화 |
| 36 | #schiffli | 신규 — technique/fabric 특화 |
| 37 | #ikat | 신규 — technique/fabric 특화 |
| 38 | #pintuck | 신규 — technique/fabric 특화 |
| 39 | #georgettekurta | 신규 — technique/fabric 특화 |
| 40 | #alinekurta | 신규 — garment/silhouette 특화 |
| 41 | #straightkurti | 신규 — garment/silhouette 특화 |
| 42 | #ethnicdress | 신규 — garment/silhouette 특화 |
| 43 | #kurtadress | 신규 — garment/silhouette 특화 |
| 44 | #indowesterndress | 신규 — garment/silhouette 특화 |
| 45 | #fusionwear | 신규 — garment/silhouette 특화 |
| 46 | #tunic | 신규 — garment/silhouette 특화 |
| 47 | #peplumkurta | 신규 — garment/silhouette 특화 |
| 48 | #myntrahaul | 신규 — 구매/하울 |
| 49 | #ajiohaul | 신규 — 구매/하울 |
| 50 | #kurtahaul | 신규 — 구매/하울 |
| 51 | #ethnichaul | 신규 — 구매/하울 |
| 52 | #myntrafinds | 신규 — 구매/하울 |
| 53 | #officewearethnic | 신규 — 맥락/시즌 |
| 54 | #workwearethnic | 신규 — 맥락/시즌 |
| 55 | #akshayatritiya2026 | 신규 — 맥락/시즌 |
| 56 | #desifa | 신규 — 맥락/시즌 |

수집 항목: A와 동일.

### C. 볼리우드 디코딩 계정 5개 — 셀럽 착장 트렌드

인도에서 볼리우드 셀럽이 입은 옷은 며칠 내에 대중 트렌드로 전파됨. 이 5개 계정은 셀럽이 뭘 입었는지를 전문적으로 디코딩(브랜드, 스타일 분석)하는 계정.

@bollywoodwomencloset, @celebrity_fashion_decode, @bollywoodalmari, @celebrities_outfit_decode, @bollydressdecode

수집 항목: A와 동일. A와 같은 파이프라인으로 통합 처리하면 됨.
이 데이터는 스코어링에서 "Cultural Fit" 점수 산출에 사용됨 (Section 9 참조).

### 3-2. YouTube — 주 2~3회 수집

인스타보다 업로드 빈도가 낮으므로 매일 수집할 필요 없음.

### D. Top 10 패션 채널

인도 여성 패션 전문 유튜버 10명. "Office Kurta Haul", "Myntra Try-On" 같은 콘텐츠를 주로 올림.

Jhanvi Bhatia (952K), Madhureddy official (866K), Khushi Malhotra (665K), Rooh Dreamz (657K), Sowbaraniya Ramesh (575K), Manisha Malik (514K), Aruna Krithi (323K), Pranavi Anakali (301K), Saniya Iyappan (244K), Rachel D'cruz (166K)

각 채널에서 수집할 항목 (영상 단위):

- 영상 제목
- 영상 설명(description) 텍스트
- 태그 목록 (YouTube Data API의 snippet.tags 필드로 접근)
- 썸네일 이미지 URL
- 조회수, 좋아요 수, 댓글 수
- 업로드 날짜
- 댓글 Top 50개 (텍스트만)

수집 방식: 화/목/토 주 3회 체크, 신규 영상만.

---

## 3.5 전처리 — 수집 후 분석 대상 선별

수집된 포스트에서 분석 가치 없는 것을 제거한 뒤 속성 추출 단계로 넘긴다.

### 필터링 대상 (제거)

- 패션과 무관한 게시글 (개인 일상 등)
- 패션이지만 **여성 ethnic wear와 무관**한 게시글 (남성 패션 / 럭셔리 주얼리 / 메이크업 등)
- 볼리우드 계정 포스트 중 여성 ethnic wear 키워드 미포함 (남성 착장 등)
- 중복 게시글 (post_id dedup)

### 이미지 품질 기준 (제거)

- 해상도가 분석 불가 수준
- 의류가 너무 작아 VLM 처리 어려움
- 얼굴/배경만 있고 의류가 거의 안 보임
- 콜라주/텍스트 배너 위주로 의류 영역 판정 어려움

### 구현 방식 (2단계)

**1차 — 룰 기반 키워드 필터:**

포함 키워드: saree, ethnic wear, styling, haul, try-on, outfit, look, wardrobe + garment_type/fabric 키워드 (§6.2 매핑 테이블과 동일)

제외 키워드: 현재 ethnic wear 포스트 캡션 특성상 명확한 제외어 정의가 어려움 → 키워드 미매칭 포스트를 2차로 넘김

**2차 — LLM 분류:**

1차 통과 + garment_type null 포스트 → LLM에 여성 ethnic wear 여부 점수화. 특정 점수 미만 시 제외. 애매한 경계 케이스는 수동 검수.

---

## 4. 속성 체계 — 수집한 포스트에서 뭘 뽑아내는지

여기가 이 프로젝트의 기술적 핵심. 인스타 포스트나 유튜브 영상을 수집한 뒤, 각 콘텐츠에서 **8개 속성**을 추출한다. 이 속성들이 트렌드 분류(Section 5), 스코어링(Section 9), 대시보드 표시(Section 2) 전부의 기반이 됨.

쉽게 말하면: 인스타에 누군가 쿠르타 셋 사진을 올리면, 그 포스트의 캡션과 해시태그에서 "이건 cotton 소재의 chikankari 기법으로 만든 kurta set이고, office 맥락이고, palazzo와 함께 입었다"를 자동으로 뽑아내는 것.

### 4.1 8개 속성 — 각각의 정의, 허용값, 추출 방법

### ① garment_type — "이 옷이 뭐냐" (어떤 종류의 의류인지)

추출 방법: 텍스트(캡션+해시태그)에서 80% 이상 추출 가능. 인도 패션 포스트에서 "#kurtaset", "#anarkali" 같은 해시태그가 매우 보편적이기 때문.

허용값과 의미:

```
kurta_set          = 쿠르타 상의 + 바텀(팔라조, 팬츠 등) 세트. 가장 큰 볼륨.
kurta_dress        = 원피스형 쿠르타. 바텀 없이 단독 착용.
co_ord             = 상하의가 같은 원단/디자인으로 된 코디 세트. Gen Z에서 급성장.
anarkali           = 밑단이 크게 퍼지는(flare) 전통 스타일. 데모에서는 라이트 버전만.
straight_kurta     = 일자 실루엣 쿠르타. 바텀 별매. 오피스웨어의 기본.
a_line_kurta       = A라인(위가 좁고 아래로 퍼지는) 쿠르타. 바텀 별매.
tunic              = 힙 기장의 짧은 쿠르티/튜닉. 진이나 레깅스와 착용.
ethnic_dress       = 티어드/셔츠/피트앤플레어 등 서양 드레스 형태에 에스닉 원단.
casual_saree       = 캐주얼/레디투웨어(미리 주름 잡힌) 사리.
fusion_top         = 페플럼/크롭/하이로우 탑. 에스닉 프린트지만 서양 실루엣.
ethnic_shirt       = 밴드칼라(목 부분 짧은 칼라) 셔츠. 핸들룸/프린트 원단.
```

### ② fabric — "무슨 소재로 만들었나"

추출 방법: 텍스트에서 60~70% 추출 가능. "#cottonkurta", "linen set" 같은 표현이 많음. 다만 일부 포스트는 소재를 명시 안 함.

허용값:

```
cotton             = 순면 (캠브릭, 보일, 물물/뮤즐린, 슬럽 코튼 포함)
cotton_blend       = 면혼방 (면+폴리, 면+비스코스 등)
linen              = 순리넨
linen_blend        = 리넨혼방 (리넨+코튼 등)
rayon              = 레이온/비스코스. 인도 everyday ethnic에서 가장 흔한 소재 중 하나.
modal              = 모달/모달혼방. 부드러운 촉감.
chanderi           = 찬데리 코튼. 인도 전통 직조. 약간 광택.
georgette          = 조젯. 가벼운 합성 소재.
crepe              = 크레이프.
chiffon            = 시폰. 주로 두파타(스카프)에 사용.
khadi              = 카디. 인도 수방직 면. 간디 운동과 관련된 상징적 소재.
polyester_blend    = 폴리혼방. 가격 저렴. 밸류 세그먼트.
jacquard           = 자카드 직조. 문양이 직조에 들어간 것.
```

### ③ technique — "어떤 기법/장식으로 만들었나"

추출 방법: 텍스트에서 80% 추출 가능. "#chikankari", "#blockprint" 등 기법 관련 해시태그가 인도 패션에서 매우 활발.

허용값:

```
solid              = 무지. 장식 없음.
self_texture       = 셀프 디자인/셀프 스트라이프. 같은 색으로 미세한 문양.
chikankari         = 치칸카리. 인도 럭나우 지역의 전통 화이트/파스텔 자수.
block_print        = 블록 프린트. 나무 블록으로 찍는 전통 프린트. 자이푸르가 유명.
floral_print       = 플로럴 프린트. 꽃무늬 (디지, 마이크로, 보태니컬 등).
geometric_print    = 지오메트릭. 스트라이프, 체크, 셰브론, 이캇, 추상 무늬.
ethnic_motif       = 에스닉 모티프. 페이즐리, 부타, 칼람카리, 반다니 등 전통 문양.
digital_print      = 디지털 프린트. 기계로 인쇄.
thread_embroidery  = 스레드 자수. 실로 수놓은 것 (토널/컨트라스트).
mirror_work        = 미러워크/시샤. 작은 거울 조각을 넣는 장식.
schiffli           = 시플리 자수. 기계 레이스 자수.
pintuck            = 핀턱. 가는 주름을 잡아 꿰매는 디테일.
lace_cutwork       = 레이스/컷워크. 구멍 뚫린 레이스 장식.
gota_patti         = 고타파티. 금/은색 리본 장식. 라이트 버전만 everyday에 해당.
```

추가로 Embellishment Intensity(장식 강도) 플래그를 함께 부여:

- `everyday` = solid, self_texture, 라이트 프린트, 서틀한 자수, 핀턱 등. 매일 입어도 되는 수준.
- `festive_lite` = 포일 하이라이트, 요크(윗부분) 자수, 미디엄 고타/미러. 작은 행사용.
- `heavy` = 데모 스코프 밖. dense sequin, 헤비 자리(금사), 올오버 미러워크 등.

### ④ color — "무슨 색이냐"

추출 방법: 텍스트로는 잘 안 잡힘 (캡션에 "pastel mint"라고 쓰는 경우 드묾). Instagram 이미지 + IG Reel + YouTube 영상에서 Pipeline B 로 추출. IG 정적 이미지가 1차, Reel / YouTube 는 VideoFrameSource 로 대표 프레임 샘플링 후 동일 로직 적용 (중간 우선순위 — VideoFrameSource 구현 후).

**3층 palette 구조**

color 는 "canonical outfit / post / trend cluster" 세 레벨 모두에서 palette (PaletteCluster list) 로 표현된다. 과거 (~2026-04-23) "post 당 대표색 1" 구조는 멀티톤 의류 (saree base+drape, layered kurta 등) 에서 dark neutral 로 수렴하는 버그 (pool_02 수렴 진단, 2026-04-24) 때문에 폐기.

각 레벨 palette 최대 크기:

| 레벨             | max 색상 | 계산 방법                                           |
|------------------|----------|----------------------------------------------------|
| canonical outfit | 3        | segformer ethnic pool → KMeans → ΔE76 greedy merge |
| post             | 3        | canonical palette 들을 ΔE76 greedy merge           |
| trend cluster    | 5        | post palette 들을 ΔE76 greedy merge                |

**Gemini (VLM) 의 역할과 한계**

VLM 은 이미지 1장에서 50-color preset 중 1~3개 `color_preset_picks_top3` 를 semantic highlight 로 pick 한다. 이 pick 의 **유일한 용도는 post 내 canonical dedup 의 비교 키** (같은 post 안의 outfit 들이 같은 garment 인지 판단).

canonical / post / cluster palette 의 RGB 값은 **오직 픽셀 증거** (segformer ethnic 영역 pool → KMeans → ΔE76 greedy merge) 로 계산한다. Gemini pick 은 palette 의 RGB 값에 섞이지 않으며, 어떤 merge 에도 참여하지 않는다 (LLM 환각이 palette 에 오염되는 것을 방지).

**segmentation pool — ethnic 영역만**

canonical 당 pool 은 그 outfit 에서 ethnic 으로 판정된 class 만 union.
- `upper_is_ethnic=True` → segformer upper class pool 포함
- `lower_is_ethnic=True` → segformer lower class pool 포함
- `dress_as_single=True` + `upper_is_ethnic=True` → segformer dress class pool 포함 (upper 슬롯 재활용)
- non-ethnic part (jeans, western_pants 등) 는 pool 에서 제외하되 라벨 자체는 보존 (upper=kurta / lower=jeans 형태로 "western matched" 흔적 추적)

is_ethnic 판정은 Gemini vision LLM 이 `EthnicOutfit.upper_is_ethnic` / `lower_is_ethnic` 필드로 직접 수행 (prompts v0.4+). 이전 `configs/garment_vocab.yaml` + `src/attributes/ethnic_vocab.py` 기반 로컬 vocab 매핑은 B1 에서 폐기.

**저장 형식 — PaletteCluster**

```json
{ "hex": "#B8D4C3", "share": 0.32, "family": "pastel" }
```

- hex: KMeans centroid 의 "#RRGGBB" 문자열 (대시보드 칩 표시용)
- share: 이 cluster 의 pixel 비중 (0~1, 같은 레벨 palette 내 share 합 = 1.0)
- family: 50-color preset 매핑 경유로 결정, 아래 family vocab 중 하나

PaletteCluster 는 canonical / post / cluster 세 레벨 모두 동일 구조. 세 레벨 모두 share (%) 를 포함한다.

**저장 형식 — Gemini pick**

`color_preset_picks_top3` 는 50-color preset 이름 array (1~3). 비중 / 순위 없음. 이 pick 은 canonical dedup 비교 키로만 사용되고, palette 계산 / merge 에는 참여하지 않음.

```json
"color_preset_picks_top3": ["pool_13", "self_maroon_red"]
```

color_family 분류 (색상 계열):

```
pastel             = 파우더 블루, 라벤더, 민트, 블러시, 셀레스트 등 연한 색
earth              = 테라코타, 러스트, 머스타드, 올리브, 탄 등 흙빛
neutral            = 화이트, 오프화이트, 베이지, 그레이, 블랙
white_on_white     = 화이트온화이트. 특히 치칸카리에서 흰 천에 흰 자수.
jewel              = 에메랄드, 로열블루, 라니핑크, 와인 등 보석 톤
bright             = 터머릭옐로우, 퓨시아 등 강렬한 색
dual_tone          = 투톤/컬러블록 (두 가지 색 조합)
multicolor         = 멀티컬러 프린트
```

post / cluster 레벨 family 는 "palette 의 dominant cluster family" 로 계산 (참고용).

### ⑤ silhouette — "옷의 형태/실루엣이 뭐냐"

추출 방법: 텍스트에서 50% 정도만 잡힘 ("A-line kurta"처럼 명시적으로 쓴 경우만). 나머지는 VLM으로 이미지에서 보강.

허용값:

```
straight       = 일자 실루엣
a_line         = 위가 좁고 아래로 퍼지는 A형
flared         = 넓게 퍼지는 (아나르칼리 아닌 일반 플레어)
anarkali       = 아나르칼리 특유의 패널 플레어
fit_and_flare  = 상체는 타이트, 하체는 퍼지는
tiered         = 여러 단으로 주름잡힌 (티어드 드레스)
high_low       = 앞은 짧고 뒤는 긴
boxy           = 박시/릴랙스드/오버사이즈
kaftan         = 카프탄 스타일 (느슨한 직선형)
shirt_style    = 셔츠 스타일/셔츠 드레스
angrakha       = 앙그라카 (앞이 교차되는 랩 프론트)
empire         = 엠파이어 라인 (가슴 아래에서 퍼지는)
```

### ⑥ occasion — "어떤 상황에서 입는 옷이냐"

추출 방법: 텍스트에서 90% 추출 가능. "office look", "casual day out", "#workwearethnic" 같은 표현이 매우 흔함.

허용값:

```
office         = 오피스/회사/직장
casual         = 캐주얼/일상
campus         = 대학교/캠퍼스
weekend        = 주말 브런치/외출
festive_lite   = 가벼운 축제/소규모 행사/푸자(기도)
travel         = 여행/휴가
```

### ⑦ styling_combo — "뭐랑 같이 입었냐"

추출 방법: 텍스트에서 50% + VLM 보강. "palazzo set"처럼 캡션에 쓰는 경우도 있지만, 이미지를 봐야 하는 경우도 많음.

허용값:

```
with_palazzo   = 팔라조(넓은 통 바지)와 함께
with_pants     = 스트레이트/시가렛 팬츠와 함께
with_churidar  = 추리다르(발목에서 조이는 전통 바지)와 함께
with_dupatta   = 두파타(스카프/숄)와 함께
standalone     = 원피스/드레스로 단독 착용
with_jacket    = 재킷/슈러그와 레이어드
with_jeans     = 진과 착용 (인스타 UGC에서 자주 보임)
co_ord_set     = 코디 세트로 상하의 세트 착용
```

### ⑧ brand_mentioned — "어떤 브랜드가 언급됐나"

추출 방법: 텍스트. 캡션에서 브랜드명이나 마켓플레이스명 추출.

자유 텍스트로 저장하고, 후처리로 tier 매핑:

- value: 마켓플레이스 자체 브랜드 (Anouk 등)
- mid: Aurelia, Libas, Biba 등
- premium_everyday: Fabindia, W, House of Pataudi Rozana 등

### 4.2 우리 리스트에 없는 새로운 속성값이 나오면 — Unknown 속성 자동 감지

현실적으로, 위에 정해둔 허용값 리스트가 인도 everyday ethnic의 모든 것을 커버할 수는 없음. 예를 들어 수집 기간 중 "#bandhani"(반다니라는 인도 전통 염색 기법) 관련 포스트가 갑자기 많이 나올 수 있는데, 이게 매핑 테이블에 안 들어 있을 수 있음.

처리 방법:

```
1. 수집된 해시태그 중 매핑 테이블(Section 6)에 없는 것을 자동으로 빈도 카운트
2. 3일간 빈도가 10건을 넘으면 → unknown_attributes 테이블에 자동 등록
   저장 형식: { tag: "#bandhani", count_3day: 34, first_seen: "2026-04-23",
               likely_category: "technique?", reviewed: false }
3. 4/24 1차 싱크 때 사람이 리뷰 → 매핑에 추가하거나, noise로 무시
```

이렇게 자동 감지된 게 데모에서 하나라도 있으면 오히려 좋음: "리스트에 없던 새로운 트렌드 시그널을 시스템이 자동으로 잡았습니다"라고 보여줄 수 있음.

### 4.3 정해둔 속성값이 소셜에서 안 나오면

예를 들어 "gota_patti"를 technique 리스트에 넣어뒀는데, 수집 기간 동안 관련 포스트가 0건이면? → 아무것도 안 해도 됨. 해당 값을 포함하는 클러스터가 그냥 생성 안 되거나 스코어가 0이라서 스코어보드에 안 뜸.

단, 4/24 1차 싱크에서 활성 클러스터가 5개 미만이면 클러스터 키의 granularity를 낮춰야 함 (Section 5.3에서 설명).

---

## 5. 트렌드 정의 및 클러스터링 — 수집된 포스트를 어떻게 "트렌드"로 묶는지

> ⚠️ **Canonical: [`docs/pipeline_spec.md`](pipeline_spec.md) §1.1 (Representative) + §2.4 (item ↔ representative 매칭) + §4 (화면 데이터 12 항목)**.
> 본 §5 의 1:1 매칭 모델은 **2026-04-27 phase 에서 다대다 + multiplier 모델로 교체**. 아래 원문은 attribute 정의 (G × T × F 조합) 의 historical context 참조용으로만 보존. 실 구현은 pipeline_spec.md 우선.

**핵심 차이 요약**:

| 항목 | 본 §5 (옛 v0) | pipeline_spec.md (canonical) |
|---|---|---|
| 클러스터 단위 | 1 post → 1 primary cluster (1:1) | 1 item (post/video) → **다수 representative** (cross-product) |
| 매칭 가중 | "가장 specific" 1개만 | **multiplier 1/2.5/5x** (N=일치 결정 필드 수) — `pipeline_spec §2.4` |
| 부분 매칭 | "임시 배정" | distribution share 기반 cross-product |
| 키 구조 | `g__t__f` 3-tuple (단일) | 동일 `g__t__f` 이지만 representative 쪽 → `representative_id = blake2b(key, 8)` BIGINT 신규 |
| Drill-down 분포 | post 단일값 카운트 | **distribution map** (text 가중 6/3 + vision contribution log scale, `pipeline_spec §2.1, §2.2`) |
| color_palette 형식 | `{r, g, b, name, family, pct}` | `{hex, share, family}` (PaletteCluster) — `pipeline_spec §2.3` |
| 활성 클러스터 수 조정 정책 (§5.3) | fabric 제거 / technique 그룹화 | **현 phase 미적용** (cross-product 후 sparse 적재 정책으로 대체, `pipeline_spec §5.2`) |

<details>
<summary>옛 §5 본문 (v0, 참조용 보존)</summary>

이 섹션이 전체 시스템의 구조를 결정함. "트렌드 1개"가 뭔지를 정의하지 않으면 스코어링도, 대시보드도 만들 수 없음.

### 5.1 트렌드 클러스터 키 — "트렌드 1개 = garment_type × technique × fabric"

수집된 포스트를 묶는 기준은 **의류 종류 × 기법 × 소재** 조합.

이 3개를 조합하면 카테고리 매니저가 인식하는 "상품 단위"와 일치하는 트렌드 묶음이 됨. 예를 들어 "면 소재로 치칸카리 기법으로 만든 쿠르타 세트"는 실제로 인도 패션 리테일러가 하나의 상품군으로 기획하는 단위.

```
클러스터 키 형식: {garment_type}__{technique}__{fabric}

예시:
kurta_set__chikankari__cotton        → "Chikankari Cotton Kurta Set"
co_ord__block_print__linen           → "Block Print Linen Co-ord"
kurta_dress__floral_print__rayon     → "Floral Print Rayon Kurta Dress"
kurta_set__solid__cotton             → "Solid Cotton Kurta Set"
straight_kurta__thread_embroidery__linen_blend → "Embroidered Linen Straight Kurta"
```

이 조합으로 **예상 15~20개** 활성 클러스터가 생김.

display_name(화면에 보이는 트렌드명)은 클러스터 키에서 자동 생성.

### 5.2 각 포스트를 어떤 클러스터에 넣는지 — 배정 규칙 (옛 1:1 모델)

```
1. 각 포스트는 1개의 primary cluster에만 배정한다 (1:1).

2. 한 포스트가 여러 클러스터에 해당할 수 있으면 → 가장 구체적인(specific) 것을 선택.

3. 3개 속성(garment_type, technique, fabric) 중 일부만 추출됐으면 → 부분 매칭.

4. 3개 전부 null이면 → "unclassified" 버킷으로 분류.
```

### 5.3 활성 클러스터 수가 너무 적거나 많으면 — 옛 정책

- **5개 미만**이면: fabric 키에서 제외, garment_type × technique만으로 키 구성.
- **30개 이상**이면: technique을 상위 그룹으로 묶음.

(현 phase 는 sparse 적재 정책으로 대체, pipeline_spec.md §5.2 참조)

### 5.4 Level 2 속성 분포 — Drill-Down 화면

클러스터 내 포스트들의 속성 분포 시각화. 옛 형식 예시 (color_palette 형식과 distribution 계산법은 pipeline_spec.md §2.1~§2.3 가 canonical):

```
color_palette: 최대 5색
silhouette / occasion / styling: distribution map (% 합 = 1.0)
```

이 분포가 카테고리 매니저에게 **"디자인 브리프의 입력값"** 역할.

</details>

---

## 6. 텍스트 속성 추출 — 캡션/해시태그에서 속성을 뽑는 구체적 방법

### 6.1 2단계 혼합 방식

비용과 속도를 고려해서 2단계로 처리:

**Step 1 — 룰 기반 매칭 (전체 포스트, 비용 0원)**

모든 신규 포스트에 대해 해시태그와 캡션 키워드를 아래 매핑 테이블과 대조. 매칭되면 해당 속성값을 자동 입력.

이 방식의 장점: 빠르고 무료. 디버깅도 쉬움 (왜 이렇게 분류됐는지 매핑 테이블 보면 바로 알 수 있음).
한계: "summer work look in breathable fabric" 같은 간접 표현은 못 잡음.

**Step 2 — LLM 배치 분류 (미분류 포스트만)**

Step 1에서 garment_type 또는 technique이 null로 남은 포스트만 선별하여 LLM에게 분류 요청.
예상 일 100~200건. 비용 미미 (4주 3~5만원).

### 6.2 해시태그 → 속성 매핑 테이블

이 테이블이 Step 1(룰 기반 매칭)의 핵심. 이 테이블에 해시태그/키워드가 어떤 속성값에 매핑되는지가 정의돼 있어야 코드를 짤 수 있음.

**중요: 이 매핑은 리서치 기반 초안.** 실제 수집 데이터를 보면서 검증하고 조정해야 함. "이 해시태그가 정말 이 속성에 맞는지"는 4/21~23에 샘플 포스트 50~100개를 눈으로 확인하면서 튜닝.

### garment_type 매핑

```
kurta_set:
  tags: [#kurtaset, #kurtasets, #kurtiset, #kurtapalazzoset, #kurtasetsonline]
  keywords: ["kurta set", "kurta palazzo set", "kurta pant set", "3 piece set"]

  → 이 해시태그나 키워드가 포스트에 있으면 garment_type = "kurta_set"

anarkali:
  tags: [#anarkali, #anarkalisuit, #anarkalikurta, #anarkalidress]
  keywords: ["anarkali"]

co_ord:
  tags: [#coordset, #coordsets, #twinningset, #matchingset]
  keywords: ["co-ord", "coord set", "matching set"]

kurta_dress:
  tags: [#kurtadress, #ethnicdress, #indiandress]
  keywords: ["kurta dress", "ethnic dress"]

casual_saree:
  tags: [#saree, #sareelove, #readytowearsaree, #casualsaree, #officesaree]
  keywords: ["saree", "sari", "ready to wear saree"]

straight_kurta:
  tags: [#straightkurta, #straightcut]
  keywords: ["straight kurta", "straight cut"]

tunic:
  tags: [#kurti, #tunic, #ethnictunic]
  keywords: ["kurti", "tunic"]

fusion_top:
  tags: [#fusiontop, #croptop, #peplumtop]
  keywords: ["fusion top", "peplum", "crop top"]

ethnic_shirt:
  tags: [#ethnicshirt, #indianshirt, #bandcollarshirt]
  keywords: ["ethnic shirt", "band collar"]
```

### technique 매핑

```
chikankari:
  tags: [#chikankari, #chikan, #chikanwork, #lucknowi, #lucknowichikankari,
         #chikankarisuit, #chikankarikurta, #chikankaricollection]
  keywords: ["chikankari", "chikan", "lucknowi"]

block_print:
  tags: [#blockprint, #handblockprint, #jaipuriprint, #ajrakh, #bagru, #handblock]
  keywords: ["block print", "hand block", "ajrakh", "bagru", "jaipur print"]

solid:
  tags: [#solid, #solidkurta, #plain, #minimal]
  keywords: ["solid", "plain"]

floral_print:
  tags: [#floralprint, #floralkurta, #floralsuit, #ditsy, #botanical]
  keywords: ["floral", "ditsy", "botanical"]

geometric_print:
  tags: [#stripes, #checks, #ikat, #geometric, #chevron, #abstract]
  keywords: ["stripe", "check", "ikat", "geometric"]

ethnic_motif:
  tags: [#paisley, #kalamkari, #bandhani, #bandhej, #buta]
  keywords: ["paisley", "kalamkari", "bandhani", "bandhej"]

thread_embroidery:
  tags: [#embroidery, #threadwork, #threadembroidery, #embroidered]
  keywords: ["embroidery", "thread work", "embroidered"]

mirror_work:
  tags: [#mirrorwork, #shisha, #mirrorembroidery]
  keywords: ["mirror work", "shisha"]

self_texture:
  tags: [#selfdesign, #selftexture, #selfstripe, #dobby, #jacquard]
  keywords: ["self design", "self texture", "jacquard", "dobby"]

digital_print:
  tags: [#digitalprint, #digitalprintkurta]
  keywords: ["digital print"]

pintuck:
  tags: [#pintuck, #pintuckkurta]
  keywords: ["pintuck"]

lace_cutwork:
  tags: [#lace, #cutwork, #crochet, #schiffli]
  keywords: ["lace", "cutwork", "schiffli"]

gota_patti:
  tags: [#gotapatti, #gota, #gotawork]
  keywords: ["gota patti", "gota work"]
```

### fabric 매핑

```
cotton:
  tags: [#cotton, #cottonkurta, #cottonsuit, #purecotton, #mulmul, #cambric,
         #cottonkurtaset, #summercotton]
  keywords: ["cotton", "mulmul", "muslin", "cambric"]

linen:
  tags: [#linen, #linenkurta, #linenkurti, #purelinen, #linenblend, #linencotton]
  keywords: ["linen"]

rayon:
  tags: [#rayon, #viscose, #rayonkurta]
  keywords: ["rayon", "viscose"]

modal:
  tags: [#modal, #modalsatin, #modalcotton]
  keywords: ["modal"]

chanderi:
  tags: [#chanderi, #chanderikurta, #chandericotton]
  keywords: ["chanderi"]

georgette:
  tags: [#georgette, #georgettekurta]
  keywords: ["georgette"]

khadi:
  tags: [#khadi, #khadicotton, #handspun]
  keywords: ["khadi", "handspun"]
```

### occasion 매핑

```
office:
  tags: [#officewear, #officekurta, #workwear, #workwearethnic,
         #indianofficewear, #officelook, #corporateethnic, #indowesternoffice]
  keywords: ["office", "workwear", "work wear", "corporate", "desk to dinner",
             "professional", "9 to 5"]

casual:
  tags: [#casualethnic, #everydayethnic, #dailywear, #casualwear]
  keywords: ["casual", "everyday", "daily wear", "easy wear"]

campus:
  tags: [#collegewear, #campuslook, #collegefashion]
  keywords: ["college", "campus", "university"]

weekend:
  tags: [#weekendlook, #brunch, #weekendoutfit, #sundaylook]
  keywords: ["weekend", "brunch", "outing", "day out"]

festive_lite:
  tags: [#festivevibes, #pujalook, #festivelook, #akshayatritiya]
  keywords: ["festive", "puja", "small function", "get together", "akshaya tritiya"]
```

### styling_combo 매핑

```
with_palazzo:
  tags: [#palazzo, #palazzoset, #kurtapalazzo]
  keywords: ["palazzo"]

with_pants:
  tags: [#kurtapants, #cigarettepants, #straightpants, #trousers]
  keywords: ["pants", "cigarette pants", "straight pants", "trousers"]

with_churidar:
  tags: [#churidar, #churidarset]
  keywords: ["churidar"]

with_dupatta:
  tags: [#dupatta, #3pieceset, #suitset]
  keywords: ["dupatta", "3 piece", "suit set"]

standalone:
  tags: [#kurtadress, #ethnicdress, #onepieceethnic, #maxidress]
  keywords: ["dress", "one piece", "standalone"]

with_jacket:
  tags: [#jacket, #shrug, #overlay, #layered]
  keywords: ["jacket", "shrug", "overlay"]

with_jeans:
  tags: [#kurtawithjeans, #ethnicwithjeans, #fusionlook]
  keywords: ["with jeans", "denim"]
```

### 6.3 LLM 배치 프롬프트 — Step 2에서 미분류 포스트에 사용

Step 1(룰 기반)에서 garment_type 또는 technique이 null로 남은 포스트에만 사용.

```
You are classifying an Indian fashion Instagram post.
Extract attributes from caption and hashtags.
Use ONLY the allowed values. If uncertain, output null.

POST:
Caption: {caption_text}
Hashtags: {hashtags}

EXTRACT:
1. garment_type: [kurta_set, kurta_dress, co_ord, anarkali, straight_kurta,
   a_line_kurta, tunic, ethnic_dress, casual_saree, fusion_top, ethnic_shirt]
2. technique: [solid, self_texture, chikankari, block_print, floral_print,
   geometric_print, ethnic_motif, digital_print, thread_embroidery,
   mirror_work, schiffli, pintuck, lace_cutwork, gota_patti]
3. fabric: [cotton, cotton_blend, linen, linen_blend, rayon, modal,
   chanderi, georgette, crepe, chiffon, khadi, polyester_blend, jacquard]
4. embellishment_intensity: [everyday, festive_lite, heavy]
5. occasion: [office, casual, campus, weekend, festive_lite, travel]
6. styling_combo: [with_palazzo, with_pants, with_churidar, with_dupatta,
   standalone, with_jacket, with_jeans, co_ord_set]
7. brand_mentioned: (brand name or null)

Output JSON only.
```

---

## 7. VLM (Vision Language Model) 사용 — 이미지에서 속성을 뽑는 방법

### 7.1 원칙: 텍스트가 먼저, VLM은 보조

캡션+해시태그만으로 70~80%의 포스트는 분류 가능. VLM은 텍스트로 못 잡는 것만 처리. 이유: VLM은 건당 비용이 있고, 속도도 느림. 전체 포스트에 VLM을 돌리면 비용과 시간이 폭증.

### 7.2 VLM을 쓰는 경우 — 딱 2가지

**케이스 1: 텍스트로 분류가 안 된 포스트**

캡션이 이모지만 있거나 해시태그가 "#fashion #ootd" 같은 generic뿐이어서 Step 1(룰)+Step 2(LLM)를 거쳐도 속성이 안 잡힌 포스트. 전체의 10~20% 예상. 일 약 100~150건.

이 포스트의 이미지를 VLM에게 보내서 추출: garment_type, silhouette, color(RGB+family)

**케이스 2: 클러스터 속성 프로필 보강 — Drill-Down 화면용 (Instagram 이미지만)**

각 트렌드 클러스터에서 인게이지먼트(좋아요+댓글) 상위 Instagram 포스트 중 vlm_processed=false인 것을 골라서 VLM 처리. 클러스터당 최대 10~20개.

이 처리의 목적: drill-down 화면의 **색상 팔레트 HEX 칩**과 **실루엣 분포**를 만들기 위해. 텍스트만으로는 정확한 색상과 실루엣을 알 수 없으니, 대표 이미지들에서 VLM으로 뽑는 것.

YouTube 영상에서도 컬러 추출함. VideoFrameSource로 대표 프레임 샘플링 후 Pipeline B와 동일 로직 적용. IG Reel 지원과 함께 구현 (VideoFrameSource 공용).

### 7.3 VLM 컬러 pick 프롬프트

VLM 은 RGB 값을 직접 반환하지 않는다 (과거 초안의 `{ "r": 184, "g": 212, ... }` 구조 폐기, 2026-04-24). VLM 의 역할은 **50-color preset 중 1~3개 pick** 뿐이고, 이 pick 은 canonical dedup 비교 키로만 쓰인다. RGB 값은 Pipeline B 의 segformer + KMeans 가 담당 (§4.1 ④ 3층 palette 참조).

프롬프트 (요지):

```
For each person bbox in the image, return:
- upper_garment_type, lower_garment_type, dress_as_single
- silhouette, fabric, technique
- color_preset_picks_top3: array of 1~3 preset names from the 50-color preset
  - solid color garment → 1 pick
  - two-tone garment → 2 picks
  - multicolor garment → 3 picks
  - do not pad to 3 unnecessarily
```

family 는 별도 추출하지 않음. Pipeline B palette 의 `family` 필드는 preset 매핑 경유로 계산한다.

### 7.4 VLM으로 안 하는 것 — 이건 PoC 범위

- YouTube 썸네일 VLM 분석 → 제목+설명+태그로 충분. 컬러는 영상 프레임에서 추출.
- YouTube 영상 음성 텍스트 변환(ASR) → PoC에서 시도
- 넥라인/소매/기장 같은 세부 속성 → PoC에서 시도
- 인도 직물 세부 분류 (예: 치칸카리와 럭나우식 치칸카리의 차이) → R&D 영역. 약속하면 안 됨.

### 7.5 VLM 비용

| 항목 | 일간 건수 | 4주 합계 |
| --- | --- | --- |
| 케이스 1 (미분류 포스트) | ~100~150건 | ~7~10만원 |
| 케이스 2 (대표 이미지 속성) | ~50~100건 | ~3~7만원 |
| **합계** | **~150~250건/일** | **~10~17만원** |

---

## 8. 데이터 스키마 — DB에 어떤 구조로 저장하는지

> ⚠️ **§8.1 / §8.2 Canonical: [`docs/pipeline_spec.md`](pipeline_spec.md) §1 (4-tier 데이터 계층) + §5 (DB 적재 단위)**.
> 본 §8 의 v0 스키마 (단일 post 테이블 + daily 클러스터 테이블) 는 **2026-04-27 phase 에서 4-tier (Representative / Item / CanonicalGroup / CanonicalObject) + weekly representative 로 전면 교체**. §8.3 (Unknown 속성 테이블) 은 pipeline_spec 무관이라 그대로 유효.

**핵심 차이 요약**:

| 항목 | 본 §8 (옛 v0) | pipeline_spec.md (canonical) |
|---|---|---|
| 테이블 수 | 2 (post + cluster) | **4** (item + canonical_group + canonical_object + representative_weekly) — `pipeline_spec §5.1` |
| post PK | `post_id` 단일 | `(source, source_post_id, computed_at)` composite + DUPLICATE KEY append-only — `pipeline_spec §5.1, §5.3` |
| post attribute | 단일값 (`garment_type`, `fabric`, ...) | **distribution** (`garment_type_dist` 등 % map) — `pipeline_spec §1.2, §2.1` |
| cluster 적재 cadence | daily (`date` 컬럼, `daily_direction`, `daily_change_pct`) | **weekly only** (`week_start_date`, `weekly_direction`) — `pipeline_spec §3.2, §3.4` |
| 매칭 모델 | post → 1 cluster (`trend_cluster_key` 단일) | item → **다수 representative** (cross-product + multiplier 1/2.5/5) — `pipeline_spec §2.4` |
| representative PK | `cluster_key` string | `representative_id = blake2b(key, 8)` BIGINT surrogate + `representative_key` 사람용 — `pipeline_spec §1.1` |
| color_palette 형식 | `{r, g, b, hex_display, name, family, pct}` | `{hex, share, family}` (PaletteCluster, B3a 에서 r/g/b/name drop) — `pipeline_spec §2.3` |
| representative 신규 컬럼 | — | `factor_contribution` (IG/YT 비율) + `total_item_contribution` + `trajectory` (12주) + `evidence_ig/yt_post_ids` (top-K=4) + `schema_version` + `computed_at` — `pipeline_spec §1.1` |
| representative `garment/fabric/technique_distribution` | distribution % map | **항상 NULL** (representative 단위 단일값이라 redundant, DDL column 만 보존) — `pipeline_spec §1.1` |
| evidence | `top_posts: [post_id]` 단일 배열 | `evidence_ig_post_ids` + `evidence_yt_video_ids` 분리 (각 top-K=4, 부족 시 padding 없이 적게 적재) — `pipeline_spec §1.1` |
| canonical (post 내 outfit) | post sub-document `canonicals[]` | **별도 테이블** `canonical_group` + `canonical_object` (item 1:N group 1:N object) — `pipeline_spec §1.3, §1.4` |
| canonical_object.media_ref | post 내 image_urls 참조 | **Azure Blob full path raw URL** (SAS 제외) 또는 YT video_id ULID — `pipeline_spec §1.4` |

<details>
<summary>옛 §8.1 / §8.2 본문 (v0, 참조용 보존)</summary>

### 8.1 포스트 테이블 — 옛 v0 단일 post 테이블

```json
{
  "post_id": "string",
  "source": "instagram | youtube",
  "source_type": "influencer_fixed | hashtag_tracking | bollywood_decode | youtube_channel",
  "account_handle": "string",
  "account_followers": "number",
  "influencer_tier": "mega | macro | mid | micro",
  "image_urls": ["string"],
  "caption_text": "string",
  "hashtags": ["string"],
  "likes": "number",
  "comments_count": "number",
  "saves": "number | null",
  "post_date": "datetime",
  "collected_at": "datetime",
  "garment_type": "string | null",
  "fabric": "string | null",
  "technique": "string | null",
  "embellishment_intensity": "everyday | festive_lite | heavy | null",
  "occasion": "string | null",
  "styling_combo": "string | null",
  "brand_mentioned": "string | null",
  "vlm_processed": "boolean",
  "canonicals": [
    {
      "canonical_index": "number",
      "upper_garment_type": "string | null",
      "lower_garment_type": "string | null",
      "dress_as_single": "boolean",
      "silhouette": "string | null",
      "fabric": "string | null",
      "technique": "string | null",
      "color_preset_picks_top3": ["string"],
      "palette": [{"hex": "#B8D4C3", "share": 0.32, "family": "pastel"}]
    }
  ],
  "post_palette": [{"hex": "#B8D4C3", "share": 0.32, "family": "pastel"}],
  "trend_cluster_key": "string | null",
  "classification_method": "rule | llm | vlm | null"
}
```

YouTube 영상은 위와 동일한 구조에 추가 필드 (video_title, description, tags, view_count, thumbnail_url).
**현 phase 는 본 단일 테이블 대신 item / canonical_group / canonical_object 3 테이블로 분리** — 자세한 컬럼 정의는 `pipeline_spec.md §1.2, §1.3, §1.4`.

### 8.2 트렌드 클러스터 테이블 — 옛 v0 daily cluster 테이블

```json
{
  "cluster_key": "kurta_set__chikankari__cotton",
  "display_name": "Chikankari Cotton Kurta Set",
  "date": "date",
  "score": "number 0~100",
  "score_social": "number",
  "score_youtube": "number",
  "score_cultural": "number",
  "score_momentum": "number",
  "daily_direction": "up | down | flat",
  "weekly_direction": "up | down | flat",
  "daily_change_pct": "number",
  "weekly_change_pct": "number",
  "lifecycle_stage": "early | growth | maturity | decline",
  "color_palette": [{"r": 184, "g": 212, "b": 195, "hex_display": "#B8D4C3", "name": "sage", "family": "pastel", "pct": 0.32}],
  "silhouette_distribution": {"straight": 0.55, "a_line": 0.35, "other": 0.10},
  "occasion_distribution": {"office": 0.70, "casual": 0.20, "campus": 0.10},
  "styling_distribution": {"with_palazzo": 0.45, "with_pants": 0.30},
  "top_posts": ["post_id"],
  "top_influencers": ["account_handle"],
  "post_count_total": "number",
  "post_count_today": "number",
  "avg_engagement_rate": "number",
  "top_videos": ["video_id"],
  "total_video_views": "number"
}
```

**현 phase 는 본 daily 테이블 대신 `representative_weekly` 단일 테이블 + `_latest` view 로 전면 교체**. daily 적재 / `daily_direction` / `daily_change_pct` 필드는 폐기. 자세한 컬럼 정의는 `pipeline_spec.md §1.1, §5.1`.

</details>

### 8.3 Unknown 속성 테이블 — 매핑에 없는 새 시그널 자동 감지용

```json
{
  "tag": "#bandhani — 감지된 해시태그",
  "count_3day": 34,
  "first_seen": "2026-04-23",
  "likely_category": "technique? — 어떤 속성 카테고리에 해당할 것 같은지 (추정)",
  "reviewed": false,
  "action": "pending — 1차 싱크에서 결정: 매핑 추가 / noise 무시"
}
```

---

## 9. 스코어링 — 트렌드 점수를 어떻게 계산하는지

### 9.1 가중치

4개 팩터의 가중치. 합산 100점.

| 팩터 | 가중치 | 왜 이 비율인지 |
| --- | --- | --- |
| Social (Instagram) | 40 | 데이터 밀도 최고 (매일 수집). 트렌드의 가장 강한 선행 시그널. |
| YouTube | 25 | 주 2~3회 수집이라 밀도가 낮음. 보조 시그널. |
| Cultural Fit | 15 | 인도 축제+볼리우드 맥락. 다른 트렌드 도구가 못 하는 차별화 포인트. |
| Momentum | 20 | "지금 급상승 중인 것"을 스코어보드 상위에 띄우기 위해 높게 잡음. 데모에서 "어제 없다가 오늘 갑자기 뜬 트렌드"가 보이면 wow moment. |

### 9.2 계산식 / 9.3 방향성 / 9.4 라이프사이클

> ⚠️ **Canonical: [`docs/pipeline_spec.md`](pipeline_spec.md) §3.4 (direction / lifecycle weekly) + §3.5 (score 공식 weekly)**.
> 본 §9.2~§9.4 의 daily 기준은 **2026-04-27 phase 에서 weekly 단위로 전면 치환**. §9.1 가중치 (Social 40 / YouTube 25 / Cultural 15 / Momentum 20) 와 §9.5 축제 캘린더는 그대로 유효.

**핵심 차이 요약**:

| 항목 | 본 §9.2~§9.4 (옛 daily) | pipeline_spec.md (canonical, weekly) |
|---|---|---|
| 적재 cadence | 매일 실행, daily score | **weekly 만 실행** (월요일 IST 기준 주간 합산) — `pipeline_spec §3.2` |
| YouTube score | `V × 0.3 + views × 0.4 + view_growth × 0.3` | **`V × 0.3 + views × 0.7`** (view_growth 제외, 크롤링 미대응) — `pipeline_spec §3.5` |
| Direction | `daily_direction` + `weekly_direction` 둘 다 산출 | **`weekly_direction` 만** (±5% 임계, ±5% 미만 flat) — `pipeline_spec §3.4` |
| Trajectory | spec.md 미정의 | **최근 12주 score 시계열** (부족분=0 패딩) — `pipeline_spec §3.4` |
| Lifecycle 기준 | "3일 연속 상승/하락" daily | **"3주 연속 상승/하락"** weekly + "주간 변동 ±5% 이내" — `pipeline_spec §3.4` |
| Momentum 지표 | "오늘 vs 7일 일평균" daily | **"이번 주 vs 지난 주"** weekly (정의 동일, window 만 변경) — `pipeline_spec §3.5` |

<details>
<summary>옛 §9.2~§9.4 본문 (daily 기준, 참조용 보존)</summary>

### 9.2 계산식 (옛 daily 공식)

```
=== 매일 실행. 각 클러스터별로 당일 스코어 산출 ===

1. Social Score (40점)
  influencer_weight: mega=3.0 / macro=2.0 / mid=1.5 / micro=1.0
  weighted_engagement = (likes + comments×2 + saves×3) × influencer_weight
  cluster_social_raw = 모든 포스트의 weighted_engagement 합산
  social_score = (cluster_social_raw / max) × 40

2. YouTube Score (25점) — 옛 공식, view_growth 포함
  V = 최근 7일 영상 수
  view_growth = (이번 주 조회수 - 지난 주) / 지난 주
  youtube_raw = V × 0.3 + normalize(views) × 0.4 + normalize(view_growth) × 0.3
  youtube_score = normalize(youtube_raw) × 25

3. Cultural Fit Score (15점)
  festival_match: Akshaya Tritiya ±2주 내 #akshayatritiya 포함 시 ×1.5
  bollywood_presence: bollywood_decode 1건 이상 시 +0.3
  cultural_raw = festival × 0.6 + bollywood × 0.4
  cultural_score = normalize(cultural_raw) × 15

4. Momentum Score (20점)
  post_growth = (오늘 - 7일 일평균) / 7일 일평균
  hashtag_velocity = 주간 포스트 수 증가율
  new_account_ratio = 이번 주 first-seen 계정 수 / 전체
  momentum_raw = post_growth × 0.4 + hashtag_velocity × 0.3 + new_account_ratio × 0.3
  momentum_score = normalize(momentum_raw) × 20

5. Total = social + youtube + cultural + momentum (0~100)
```

### 9.3 방향성 지표 — 옛 daily/weekly 양쪽 산출

```
daily_change_pct = (오늘 Score - 어제 Score) / 어제 Score × 100
weekly_change_pct = (이번 주 일평균 - 지난 주 일평균) / 지난 주 일평균 × 100
±5% 임계로 up / flat / down
```

### 9.4 라이프사이클 — 옛 daily 기준

```
Early:    score < 30 + 고유 계정 < 10 + hashtag volume 낮음
Growth:   score 30~65 또는 3일 연속 상승 + mega/macro 1+ + 주간 +20%+
Maturity: score ≥ 65 + 일별 변동 ±5% 이내 + mega 다수
Decline:  3일 연속 하락 + hashtag 감소 + engagement 하락
```

</details>

### 9.5 축제 캘린더 — Cultural Fit 스코어링에 사용

수집 기간(4/21~5/4)에 해당하는 인도 축제:

```
Akshaya Tritiya: 4/26~27
  인도에서 금(gold) 구매와 ethnic wear 수요가 급증하는 축제.

  스코어링 반영:
  - 4/20~27 기간에 #akshayatritiya, #goldethnic 등 축제 해시태그가
    포함된 포스트에 Cultural Fit 1.5배 부스트 적용

  수집 반영:
  - 4/21 수집 시작 후 4/27까지 연속 수집 (축제 전 구매 + 당일 착장)
```

---

## 10. 파이프라인 실행 순서 — REDIRECT

> ⚠️ **REDIRECT**: 이 절은 v0 시점의 daily cadence + outdated 매칭(`trend_cluster_key`) 표현을 포함하므로 무효.
> 현재 canonical 은 [`docs/pipeline_spec.md`](pipeline_spec.md) 의 §3 (weekly cadence) + §2.4 (다대다 매칭 + multiplier) + §6 (Object palette / VLM).

### 10.0 변경 요약 (v0 → 현재)

| 항목 | spec v0 (아래) | 현재 (pipeline_spec.md) |
| --- | --- | --- |
| 실행 주기 | daily | **weekly** (월요일 일괄, §3 / §3.5) |
| 매칭 단위 | post → 단일 `trend_cluster_key` | **item ↔ representative 다대다** (§2.4, multiplier 1/2.5/5x) |
| 스코어 산출 | 일별 (§9.2 — 폐기됨) | weekly (§3.5, social/youtube/cultural/momentum) |
| direction | 전일/전주 대비 ▲▼→ | 주간 ratio (§3.4) |
| Lifecycle | 일별 갱신 | **3주 연속** 룰 (§3.4) |
| VLM 보강 | "선택적" optional | **canonical path 기본 경로** (§6.5 Object palette, M3.A 완료) |
| YouTube cadence | 주 2~3회 별도 step | weekly 통합, factor 가중치 25 (§3.5) |

### 10.1 현재 weekly 파이프라인 단계 (요약)

상세는 pipeline_spec.md §3 / §6 / §7 참조.

```
Step 1: 수집 (크롤러 레포)
  └── Instagram + YouTube raw → png DB / Azure Blob

Step 2: enrichment
  ├── 텍스트 속성 (Section 6.2 매핑 + LLM fallback)
  ├── Vision: Phase 1 SceneFilter → Phase 2 Gemini outfit 추출 → Phase 3 canonical_extractor
  │   → Phase 4 dynamic palette → Phase 5 adapter swap (β-hybrid)
  └── enriched JSON 적재

Step 3: 매칭 + multiplier (pipeline_spec §2.4)
  └── item × representative 다대다 cross-product, N=일치 결정 필드 수 → 1/2.5/5x

Step 4: weekly 스코어링 (pipeline_spec §3.5)
  ├── factor raw → max-normalize → weight (40/25/15/20)
  └── weekly_change_pct, lifecycle 3주 연속 룰 (§3.4)

Step 5: 적재 (pipeline_spec §5)
  └── StarRocks DUPLICATE KEY append-only + _latest view (4 base table)
```

<details>
<summary>📜 v0 본문 (참조용, 무효)</summary>

### v0 §10.1 매일 자동 실행

아래 5단계가 매일 순차적으로 실행. 각 단계는 이전 단계가 완료된 후 시작. 실행 시각은 기술팀 인프라에 맞춤 (새벽 권장).

```
Step 1: 수집
  ├── Instagram Top 10 인플루언서 → 신규 포스트만 수집 → DB 저장
  ├── Instagram 해시태그 15개 × Top 50 → 수집 → 중복 제거(dedup) → DB 저장
  ├── Instagram 볼리우드 디코딩 5계정 → 수집 → DB 저장
  └── 수집 완료되면 다음 Step 트리거

Step 2: 텍스트 속성 추출
  ├── 오늘 새로 수집된 전체 포스트에 대해 → Section 6.2 매핑 테이블로 룰 기반 매칭
  │   → garment_type, fabric, technique, occasion, styling_combo 등 속성 필드 채움
  ├── 룰 기반으로 garment_type 또는 technique이 null인 포스트만 → LLM 배치 분류
  ├── 매핑 테이블에 없는 해시태그 빈도 카운트 → unknown_attributes 테이블에 기록
  └── 각 포스트에 trend_cluster_key 배정 (Section 5.2 규칙에 따라)

Step 3: 스코어링
  ├── 각 클러스터별 당일 스코어 산출 (Section 9.2 공식)
  ├── 전일/전주 대비 direction 계산 (▲▼→)
  └── 라이프사이클 태그 업데이트 (Early/Growth/Maturity/Decline)

Step 4: VLM 보강 (선택적)
  ├── 각 클러스터의 상위 인게이지먼트 Instagram 포스트 중 vlm_processed=false인 것 선별
  ├── 클러스터당 최대 5개 → VLM으로 color RGB + silhouette 추출
  └── 클러스터의 color_palette 및 속성 분포 테이블 업데이트

Step 5: 결과 저장 → 대시보드 갱신
  └── 스코어보드와 drill-down 화면이 최신 데이터로 업데이트됨
```

### v0 §10.2 주 2~3회 실행 (화/목/토)

```
Step 6: YouTube 수집
  ├── 10개 채널에서 신규 영상 확인 → 수집 → DB 저장
  ├── 영상 제목+설명+태그로 텍스트 속성 추출 (룰 기반 + LLM)
  └── trend_cluster_key 배정 → 다음날 일별 스코어링에 반영됨
```

</details>

---

## 11. 비용

인건비 제외, 순수 수집·분석 비용만. 4주 기준.

| 항목 | 설명 | 4주 추정 |
| --- | --- | --- |
| 수집 인프라 | 기존 Instagram/YouTube 수집기 활용 전제. 프록시, 서버, 스토리지. | 50~100만원 |
| VLM | 일 150~250건 이미지 분석 | 10~17만원 |
| LLM 배치 | 일 100~200건 텍스트 분류 | 3~5만원 |
| YouTube Data API | 일일 quota 10,000 unit 이내 | ~0원 |
| **합계** |  | **~65~120만원** |

500만원 예산 대비 충분히 여유 있음.

---

## 12. 타임라인

| 기간 | 작업 | 담당 | 상태 |
| --- | --- | --- | --- |
| ~~4/14~15~~ | 킥오프 + 수집기 세팅 | — | ✅ 완료 |
| 4/16~20 | 수집 파이프라인 구축 + 테스트. 해시태그 변경 전달. | 기술팀 | 진행중 |
| **4/21 (월)** | **Daily 수집 시작.** 룰 기반 추출기 가동. | 기술팀 |  |
| 4/21~23 | 3일간 daily 수집 + 데이터 품질·매핑 검증 | 기술팀+종휘 |  |
| **4/24 (목)** | **1차 싱크 + 과거 1개월 벌크 수집 (3/20~4/20).** 베이스라인 확보. | 전원 |  |
| 4/24~28 | 듀얼 트랙: ① 데이터 계속 수집+스코어링 구현 ② 프론트+백엔드 개발 (COS 없이) | 기술팀 |  |
| **4/28 (월)** | **2차 싱크 — 최종 컨셉 확정.** 스코어보드 데이터+화면 리뷰. | 전원 |  |
| 4/28~30 | 최종 개발 + 데이터 마무리. 시간 여유 시 프론트 COS 이관. | 기술팀 |  |
| **4/30** | 개발 완료. 내부 리뷰. | 전원 |  |
| **5/1** | **데모 영상 촬영·편집 완료 (1차 산출물).** | 전원 |  |
| **5/2** | **대시보드 개발 완료.** | 기술팀 |  |
| 5/2~3 | 석연님 발표 연습 + 인도 출국 | 석연 |  |
| 5/4~8 | 인도 현장 (영상 기본 + 라이브 가능 시 추가) | 석연 |  |

### 4/24 1차 싱크 체크리스트 (3일치 daily 데이터 기준)

- [ ]  해시태그 Top 50에서 everyday ethnic 포스트가 충분한가?
- [ ]  활성 트렌드 클러스터가 15~20개 나오는가? (5개 미만이면 Section 5.3에 따라 granularity 조정)
- [ ]  인게이지먼트 데이터(좋아요, 댓글, 저장)가 정상적으로 수집되는가?
- [ ]  스코어링 공식 적용했을 때 클러스터 간 의미 있는 점수 차이가 나는가? (전부 비슷하면 가중치 조정)
- [ ]  unknown_attributes 테이블에 뭐가 잡혔는가? → 매핑에 추가할지, noise로 무시할지 결정
- [ ]  VLM 컬러 추출 결과 확인 → RGB 값이 정상적으로 나오는지, Plan B(전통 CV) 전환 필요한지 판단
- [ ]  과거 1개월 벌크 수집 범위·비용 최종 확인 → 실행
- [ ]  안 되는 게 있으면: 해시태그 교체, 인플루언서 추가, 클러스터 키 granularity 조정 — **4/24 안에 완료**

---

## 13. 참고

- **Feasibility 확인**: 260410_AI_Fashion_Tour_Feasibility_v2.xlsx 기준 전 항목 Go. 예외 2건 (ver.in 접속불가 1건, Instagram 댓글 1page 제한) — 현재 스코프에 영향 없음.
- **해시태그 매핑 테이블** (Section 6.2): 리서치 기반 초안. 4/21~23에 실제 수집 데이터로 1차 검증. 안 맞는 것 있으면 즉시 조정.
- **인플루언서/해시태그/채널 리스트**: 이 명세서의 리스트가 확정판. 변경 필요하면 종휘에게 사전 확인.
- 5/1 데모 영상 완성이 1차 산출물. 라이브 시연은 영상 확보 전제 하에 추가.
- COS 없이 자체 개발 우선. 시간 여유 시 프론트만 COS 이관.
- 수집 담당은 기술팀 내부에서 배정.
- VLM 파이프라인 가이드는 도균님이 별도 작성하여 기술팀 전달 예정.
