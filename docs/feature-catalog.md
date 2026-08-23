# Structural feature catalog

구조 피처는 기존 v3 입력에 더해지며 기존 이름·라벨·PIT 계약을 바꾸지 않는다.

| Block | Inputs | Weekly features | Causal contract |
|---|---|---|---|
| Sector breadth | XLB, XLC, XLE, XLF, XLI, XLK, XLP, XLRE, XLU, XLV, XLY | 1/4주 상승·하락 비율, 중앙값·MAD 분산, 상승 리더 HHI, breadth 가속도, coverage | 해당 주까지 관측된 조정종가만 사용 |
| Broad / size / style | SPY, QQQ, IWM, DIA, RSP | 1/4주 상승 비율, coverage | sector와 별도 단면 |
| Cross-asset | SHY, IEF, TLT, HYG, LQD, GLD, UUP | 1/4주 상승 비율, coverage | equity breadth와 별도 단면 |
| Treasury curve | DGS3MO, DGS1, DGS2, DGS5, DGS7, DGS10, DGS20, DGS30 | Nelson–Siegel level, slope, curvature, coverage | 고정 λ=0.0609/month, 최소 4개 만기, 매주 독립 적합 |
| Bank credit | TOTBKCR, TOTCI, DPSACBW027SBOG, H8B3094NCBA | 총대출·C&I 4/13주 로그증가, C&I 비중, 예금조달 비율, 차입 비율, coverage | H8 차입금만 millions→billions 변환 |
| Release innovation | UNRATE, PAYEMS, INDPRO, RSAFS, HOUST, CPIAUCSL, PCEPI, GDPC1 | 신규 기간·revision event, delta, 과거 event 중앙값·MAD 표준화, coverage | 현재 event 이전 최대 12개 event만 사용, 최소 4개, non-event signal=0 |

`WeeklyDataset.feature_group_manifest`는 모델 입력의 각 열을 정확히 한 블록에 배정한다. 이를 기준으로 common-origin block ablation을 수행할 수 있다.

## v5 H.10 FX 블록

H.10 FX는 v4 champion 입력을 바꾸지 않는 prospective shadow 블록이다.

| 구분 | 고정 입력 | 주별 파생값 | 가용성 계약 |
|---|---|---|---|
| 달러 지수 | Broad, Advanced Foreign Economies, Emerging Market Economies | 1·4·13주 로그 변화, 13·26주 실현 변동성, Broad/AFE/EME divergence와 index MAD | 공식 일정은 월요일 16:15 ET(연방 공휴일이면 다음 영업일); 실제 `first_seen` 이후 첫 모델 cutoff부터 사용 |
| Bilateral panel | EUR, JPY, GBP, CHF, CAD, AUD, CNY, MXN, BRL | 통화별 1·4·13주 로그 변화와 13·26주 변동성, 단면 median·양(+) breadth·MAD, pair coverage | 고정 9개 중 최소 6개가 있을 때 단면값 산출; 없는 주는 forward-fill하지 않음 |

AUD·EUR·GBP처럼 `USD per foreign currency`로 공시되는 series는 로그 변화의
부호를 뒤집고, CAD·CHF·CNY·JPY·MXN·BRL처럼 `foreign currency per USD`인
series와 세 달러 지수는 그대로 둔다. 따라서 모든 FX 변화는 **양수일수록 USD 강세**다.
KRW는 parser metadata 검증 대상일 수 있지만 고정 bilateral panel에는 포함하지 않는다.

각 행은 `observation_week`, `feature_available_at`, coverage와 source status를 함께
가진다. 최초 전체 XML 수집은 과거 공개시점을 복원하지 않으며 모든 기존 관측을 실제
`first_seen` 이후에만 열어 준다. 이후 새 값과 revision도 수집·발견 시점부터만
prospective하게 사용한다.

별도 archive 민감도는 2022년 이후 공식 dated release를 각 model cutoff에서
replay한다. 정상 release는 16:15 ET, 선언 정정 release는 다음 날 00:00 ET부터
가용하다. Quarantine event는 선언 정정, 직전 page 대비 material revision,
신규 series-date 0개이면서 직전 page key와 정확히 일치하는 완전 재발행의 OR로
판정한다. 직전 event와 3 calendar days 이내인지는 보조 경보일 뿐 분류 조건이
아니다. 각 event 이후 27개 origin의 FX feature는 `correction_quarantine`로
표시하고 ablation 공통표본에서 제외한다.

FX 평가는 다음 네 열 집합을 모든 필요한 피처가 존재하는 동일 origin에서 비교한다.

1. `v4_control`
2. `v4_plus_broad_index`
3. `v4_plus_bilateral_panel`
4. `v4_plus_all_fx`

화면용 단면 context는 6/9 pair부터 계산하지만 ablation의 common origin은 9/9
pair와 모든 변형 피처를 요구한다. 공통 표본 156주가 쌓이면 다음 주 국면을 대상으로
104주 이상 expanding 학습과 1주 target purge를 적용한 고정 L2 multinomial
logistic을 네 변형에 동일하게 적합한다. 동일 origin의 log loss·Brier와 paired
13주 circular block bootstrap/Holm gate를 기록한다. 비교는 shadow diagnostic이며
gate 통과도 champion 자동 승격을 일으키지 않는다.

## v5 결과 필드

| 결과 | 의미 | 사용 규칙 |
|---|---|---|
| `current.memberships` | 현재 risk-score anchor에 대한 세 국면 소속도 | hard state의 설명값이며 미래 예측 확률로 표기하지 않음 |
| `next_week.probabilities` | 다음 주 세 국면 예측 분포 | membership과 별도 평가·표시 |
| `directional_risk` | 1·4·13주 안의 첫 이탈 도착 국면 또는 `no_departure` | 목적지 합계를 기존 any-departure 확률에 맞춤 |
| `duration_context` | 현재 spell 경과주, 상태별 조건부 생존·이탈, median과 52주 RMST | 현재 spell 우측 검열; 완료 spell 5개 미만이면 `insufficient_history` |
| `conditional_asset_stats` | state `t` 뒤 `t+1`부터 측정한 SPY·QQQ·IWM·TLT·HYG·UUP의 1·4·13주 성과 분포 | 관측 20개·episode 5개부터 `ok`와 bootstrap CI를 부여; descriptive-only |

공개 JSON에는 위 파생 결과와 출처 상태만 들어간다. H.10 raw observation,
XML payload, snapshot DB와 provider request는 공개하지 않는다.
