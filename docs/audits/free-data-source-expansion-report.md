# 무료 데이터 소스 확장 감사

작성일: 2026-08-27
목적: 미국 주식시장 3국면의 다음 주 예측력을 높이되, 정답지 정의·시점 인과성·데이터 권리를 훼손하지 않는 무료 데이터 확장 순서를 정한다.

## 결론

- 무료 데이터 확장은 유효하다. 다만 첫 구현은 Cboe VIX가 아니라 **OFR Financial Stress Index(FSI)** prospective shadow다.
- VIX와 VIX term structure는 경제적으로 좋은 volatility challenger지만, 무료 열람 권리와 자동 저장·모델 파생 사용 권리는 같지 않다. 서면 이용 승인을 받기 전까지 `blocked_pending_written_license`로 둔다.
- 이후 우선순위는 `OFR repo → Fed Commercial Paper → Treasury curve → NY Fed reference rates → Fed H.4.1 → ADS/RTDSM`이다.
- 새 시계열은 모두 predictor 또는 `macro_context`로만 사용한다. 현재 canonical 시장 레이블에는 섞지 않는다.
- 공식 모델에는 자동 반영하지 않는다. 같은 origin의 기존 control과 log loss·Brier·calibration·transition delay를 비교한 뒤에만 승격 심사를 시작한다.

## 선정 방법

운영 데이터 소스는 다음 다섯 조건을 함께 평가했다.

1. 원기관 또는 원기관이 공개한 aggregate인가
2. 당시 실제 이용 가능했던 시각과 revision을 재구성할 수 있는가
3. 자동 수집·로컬 저장·모델 파생 사용·파생 결과 공개 권리가 명확한가
4. 현재 ETF·ALFRED 피처와 다른 경제적 가설을 제공하는가
5. 기존 운영 build가 새 소스 장애에 종속되지 않도록 shadow로 격리할 수 있는가

논문 인용 수가 운용 계약을 대신할 수 없으므로, 데이터 가용시각·스키마·권리는 각 원기관의 공식 문서와 약관을 우선 확인했다. 예측 가설은 성능 사실이 아니며 matched-origin 실험으로 별도 검증한다.

## 즉시 구현: OFR FSI prospective shadow

[OFR FSI](https://www.financialresearch.gov/financial-stress-index/)는 신용, 주식 가치평가, funding, safe assets, volatility의 공통 스트레스를 하나의 일별 aggregate로 제공한다. 공개값은 통상 관측일보다 2영업일 늦으므로, 같은 주 금요일 16:00 결정에 관측일만 보고 소급해서 넣어서는 안 된다.

수집 계약은 다음과 같다.

- 원기관 aggregate와 공개 category/region contribution만 허용한다.
- 33개 underlying input을 역산·재수집하거나 공개하지 않는다.
- `source_released_at`, `provider_first_seen_at`, `system_retrieved_at`을 별도로 저장한다.
- raw SHA-256과 `revision_seq`를 보존하고, 같은 관측일의 값이 달라지면 새 revision으로 append한다.
- 수집 실패는 기존 Alpha Vantage·ALFRED·H.10 운영 build를 막지 않는다.
- raw response와 로컬 DB는 public package에서 제외한다.

첫 shadow feature는 `level`, `1/4/13주 변화`, `52주 train-only z-score`로 제한한다. 이후 category contribution은 aggregate가 incremental value를 보인 경우에만 확장한다. OFR은 과거값 정정 사례를 공지하므로 current-history backfill과 operational first-seen history를 동일한 OOS로 취급하지 않는다. 권리 경계는 [OFR legal notices](https://www.financialresearch.gov/legal-notices/)를 따른다.

### 첫 실수집 검증

2026-08-27 00:52 UTC의 첫 시도는 fixture보다 공식 CSV에 공개 지역 기여도 두 열이 더 있어 `schema_changed`로 차단됐고, 관측값은 저장되지 않았다. 공식 헤더를 다시 확인한 뒤 `Other advanced economies`와 `Emerging markets`만 명시 allowlist에 추가하고 관련 회귀를 재실행했다.

2026-08-27 00:54 UTC의 두 번째 시도는 성공했다. 9개 공개 aggregate/category/region 시리즈의 2000-01-03~2026-08-24 current history 60,687행을 private SQLite에 처음 본 시각으로 동결했다. 이 행들은 과거 시점의 운영 빈티지가 아니며, 앞으로의 prospective shadow에서만 실제 first-seen 근거가 된다. 실패 provenance 1건과 성공 provenance 1건, 값 없는 receipt는 로컬에 보존했고 public package에는 포함하지 않았다.

## VIX 판정

[Cboe VIX 방법론](https://cdn.cboe.com/api/global/us_indices/governance/Volatility_Index_Methodology_Selected_SPX_Target_Expected_Volatility_Term_Indices.pdf)은 VIX 계열을 정규 거래일 16:15 ET까지 계산한다고 설명한다. 따라서 무료 일별 종가를 Friday 16:00 ET 운영 결정에 당일 값으로 넣으면 시점 인과성을 위반한다.

[Cboe 약관](https://www.cboe.com/terms)과 [Cboe Content 이용 절차](https://www.cboe.com/use-of-content)는 개인 비상업 열람과 자동 축적·저장·파생 사용을 구분한다. [FRED VIXCLS](https://fred.stlouisfed.org/series/VIXCLS)도 Cboe 저작권 자료이며, FRED API가 원소유자의 권리를 대신 허가하지 않는다. 사용자의 프로젝트 승인만으로 제3자 권리가 생기지는 않는다.

서면 이용 승인을 받으면 실험을 다음처럼 분리한다.

- 전일 공식 종가: 현재 Friday 16:00 정책과 호환
- 16:00 이전 timestamp snapshot: 그 시각의 feed 권리가 있을 때만
- 16:15 official close: decision time을 옮긴 sensitivity
- VIX tenor/futures curve: roll과 expiry 정의를 고정한 별도 sensitivity

현재 상태는 `blocked_pending_written_license`이며, FRED를 통한 우회 수집도 하지 않는다.

## 다음 확장 순서

| 순서 | 공식 소스 | 핵심 신호 | 시점·빈티지 핵심 |
|---:|---|---|---|
| 1 | [OFR U.S. Repo Markets](https://www.financialresearch.gov/short-term-funding-monitor/datasets/repo/) | repo 금리·거래량·담보·tenor funding stress | preliminary와 분기 final을 별도 revision으로 저장 |
| 2 | [Fed Commercial Paper](https://www.federalreserve.gov/releases/cp/about.htm) | 금융·비금융 CP spread, A2/P2 stress, 발행량 | aggregate만 사용하고 outstanding 수정 이력을 보존 |
| 3 | [Treasury daily curve](https://home.treasury.gov/treasury-daily-interest-rate-xml-feed) | level·slope·curvature·breakeven | 15:30 quote 시각과 실제 first-seen을 구분 |
| 4 | [NY Fed reference rates](https://www.newyorkfed.org/markets/reference-rates) | SOFR/EFFR/TGCR/BGCR funding pressure | 정정 마감 이후 first-seen을 사용하고 필수 고지 보존 |
| 5 | [Fed H.4.1](https://www.federalreserve.gov/releases/h41/default.htm) | reserves·TGA·RRP·primary credit·swap | dated release를 보존하며 목요일 16:30은 금요일 결정에만 사용 |
| 6 | [Philadelphia Fed ADS](https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/ads) | mixed-frequency 실물 활동 | all-vintages와 current history를 분리 |
| 7 | [Philadelphia Fed RTDSM](https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/real-time-data-set-for-macroeconomists) | GDP·물가·노동의 당시 빈티지 | errata와 vintage 전체 history를 그대로 보존 |
| 8 | [Fed H.8](https://www.federalreserve.gov/releases/h8/about.htm)·[SLOOS](https://www.federalreserve.gov/data/sloos.htm) | 은행대출·예금·대출기준 | H.8 Friday 16:15 발표는 같은 날 16:00 결정에서 제외 |

SEC MIDAS·Fails-to-Deliver는 공개 지연이 길어 운영 신호가 아니라 사후 liquidity validation으로만 사용한다. FINRA와 Nasdaq 웹 콘텐츠는 현재 약관이 database·ML·predictive analytics 사용을 제한하므로 별도 허가 전에는 제외한다.

## 평가 계약

새 그룹은 다음 matched-origin 트랙으로 비교한다.

1. 기존 control
2. `+ volatility`
3. `+ funding/credit`
4. `+ rates/liquidity`
5. `+ macro-vintage`
6. full

시작일이 다른 데이터는 전체 장기 표본에 결측 대치를 해서 억지로 끼우지 않는다. 예를 들어 2014년 이후 repo 후보는 같은 2014년 이후 origin의 control과 비교한다. 1차 지표는 multiclass log loss, 2차는 Brier·calibration·상태별 recall·transition delay·false alarms/year다. 기존 실현변동성·NFCI와 새 stress 지표의 중복 ablation도 반드시 포함하며, 불리한 결과도 연구 artifact에 남긴다.
