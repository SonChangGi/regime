# 구조 v5 사전등록

사전등록 당시 v5는 v4 운영 계약 위에 방향성 이탈, 상태 잔여기간, H.10 FX
context와 국면별 자산 설명 통계를 추가하는 opt-in 연구 계약이었다. 현재 수동
CLI의 기본 계약은 V5이며, V4 재현은 `--contract v4`를 명시해야 한다.

> **운영 개정:** 이 문서의 0.05 수치는 최초 V5 스냅샷을 재현하는 동결
> 사전등록값이다. 새 V5 공식 모델은 동일 selection 표본에서 0.01 개선 기준과 기존
> Holm·Brier·zero-fallback gate를 통과한 단일 champion으로 결정하며, 모델명은
> 고정하지 않는다. 최초 공개 결정과 V4 비교값은 변경하지 않는다.

## 1. 판정 의미

| 출력 | 고정 의미 |
|---|---|
| `current.state` | 현재 시점까지의 시장 자료로 만든 `causal_hysteresis_state` hard label |
| `current.memberships` | 현재 risk-score anchor에 대한 국면별 소속도. hard state의 membership이 `primary_membership` |
| `next_week.probabilities` | 다음 주 `risk-on / transition / risk-off` 예측 확률 |
| `transition_risk` | 1·4·13주 안에 현재 국면을 한 번이라도 떠날 확률 |
| `directional_risk` | 같은 기간의 첫 이탈 도착 국면과 `no_departure` 분포 |

Membership은 현재 판정의 연속적 설명값이고 forecast는 미래 확률이다. Hysteresis를
적용한 hard state가 membership 최댓값과 다를 수 있으며 hard state가 권위 판정이다.

## 2. 방향성 first-departure

origin `t`와 horizon `h`에 대해 `t+1 … t+h`에서 처음 현재 상태와 달라지는
주의 상태를 target으로 삼는다. 기간 안에 이탈하지 않으면 `no_departure`다.
중간에 다른 상태를 거쳐도 horizon 마지막 상태로 다시 표기하지 않는다.

후보는 다음 네 개다.

1. empirical first-passage
2. homogeneous Markov first-passage
3. regularized multinomial
4. shallow multiclass XGBoost

각 horizon은 target이 완전히 관측된 origin만 사용하고 경계에서 purge한다. 2023년
이전 origin으로 선정하며 2023년 이후는 retrospective diagnostic이다. 방향 모델은
배포 때 이탈확률을 만들지 않으므로, 실제 이탈 origin에서 목적지 확률을 재정규화한
conditional destination log loss와 Brier로 선정한다. 이탈 8건, 목적지 class 2개,
이탈을 포함한 13주 block 3개를 충족하지 못하면 empirical baseline을 유지한다.
지원 기준을 충족한 후보의 gate는 log loss 개선 0.05 이상, Brier 악화 0.01 이하,
13주 block bootstrap과 one-sided Holm `α=0.05`다.

최종 분포는 기존 `transition_risk`와 결합한다. `no_departure = 1 - p(change)`이고,
현재 상태의 목적지 확률은 0이며 나머지 목적지 확률의 합은 `p(change)`다.

### Causal multiscale ensemble amendment

`causal_multiscale_ensemble`은 결과를 확인하기 전에 고정한 v5-only 후보다. Markov,
XGBoost, XGBoost hazard-destination의 완료된 OOS log score만 사용하고
`target_date < origin`인 행으로 26·52·104주 half-life pool을 만든다. 52주 중심의
geometric grid이며 세 scale 확률은 정확히 1/3씩 고정 평균한다. 각 scale에서
fallback expert의 가중치는 0으로 둔 뒤 나머지를 재정규화한다.

이는 DMA가 아니라 causal discounted log-score multiscale pool이다. 결과 확인 뒤
half-life·outer weight를 조정하지 않는다. 기존 0.05 log-loss, Holm, Brier,
zero-fallback gate를 그대로 적용하며 우회하지 않는다. 사전등록 시점에는 v4 공개
경로를 유지했고 자동 승격을 금지했다.

## 3. 상태 잔여기간

국면 spell은 `as_of`까지의 causal hard label로만 만든다. 현재 spell은 항상 우측
검열하며 완료 spell로 세지 않는다. 상태별 Kaplan–Meier 생존함수
`S_s(t)=P(T>t)`를 적합한다. 현재 상태를 `d`개 주간 관측에서 확인했다면 조건
시점은 다음 관측의 이탈 사건 직전인 `d-1`이다.

```text
conditional_survival(h | d, s) = S_s(d - 1 + h) / S_s(d - 1)
departure_probability(h | d, s) = 1 - conditional_survival(h | d, s)
```

4·13주 조건부 생존·이탈 확률, 조건부 중앙 잔여주와 52주 제한 평균 잔여기간을
보고한다. 이산시간 RMST는 `k=0 … 51`의 조건부 생존율 합이다. 상태별 완료 spell이
5개 미만이면 `insufficient_history`로 표시한다. 불확실성은 episode 단위 bootstrap
1,999회로 계산한다.

## 4. Federal Reserve H.10 FX

공식 [H.10 release](https://www.federalreserve.gov/releases/h10/)와
[XML ZIP](https://www.federalreserve.gov/releases/h10/data/FRB_h10_xml.zip)을 사용한다.
입력은 달러 지수 3개와 고정 bilateral 9개다.

- 지수: Broad, Advanced Foreign Economies, Emerging Market Economies
- 통화쌍: EUR, JPY, GBP, CHF, CAD, AUD, CNY, MXN, BRL

주별 마지막 유효 관측을 사용하고 누락 주는 forward-fill하지 않는다. EUR·GBP·AUD의
`USD per foreign currency` quote는 로그 변화 부호를 뒤집고, 나머지 6개
`foreign currency per USD` quote와 세 지수는 그대로 둔다. 모든 파생값은
**양수 = USD 강세**다.

- 통화·지수별 1·4·13주 로그 변화
- 13·26주 실현 변동성
- Broad·AFE·EME divergence와 index MAD
- bilateral median, 양(+) breadth, MAD와 coverage

Bilateral 단면값은 고정 9개 중 최소 6개가 가용할 때 산출한다. H.10은 통상
월요일 16:15 `America/New_York`에 앞선 영업주의 값을 공개하며, 연방 공휴일이면
다음 영업일로 이월된다. observation week와 `feature_available_at`을 분리한다.

### Point-in-time 규칙

Board XML 전체 이력은 historical-vintage archive로 사용하지 않는다. 최초 수집 때
발견한 기존 관측의 availability는 실제 `first_seen`이며 과거 공개시점을 소급
채우지 않는다. 이후 새 관측과 변경값도 수집·발견 시점부터 prospective하게 연다.
수집 실패 시 last-good 파생 context가 있으면 `degraded`와 `last_good_used=true`,
없으면 `unavailable`로 기록한다.

### 공식 release archive 민감도 amendment

`v5-fx-h10-official-release-archive-20260822`는 완료된 v5 결과를 확인하기 전에
공식 `releaseDates.json`과 과거 release page의 불변 archive를 확인해 고정한
research-only amendment다. XML의 `historical_availability_backfill=false`와
prospective first-seen 결과는 primary로 보존한다.

Archive 민감도는 2022-01-01 이후 release event만 사용한다. 정상 release는 본문
`Release Date = Last Update = URL date`를 확인한 뒤 16:15 ET, 정정 release는
`URL date = Last Update > Release Date`를 확인한 뒤 다음날 00:00 ET부터 연다.
정정은 새 vintage로 보존한다.

`v5-fx-h10-correction-equivalent-lineage-20260822`는 archive page lineage를
quarantine 판정에 결속한다. 주 판정은 (1) 본문이 선언한 정정,
(2) 직전 page와 겹치는 series-date의 material revision 1개 이상,
(3) 신규 series-date가 0이고 직전 page key와 정확히 100% 일치하는 완전 재발행의
OR다. 직전 release와 3 calendar days 이내라는 사실은 validation warning과 보조
근거로만 기록하며 주 판정에 넣지 않는다. 매 refresh는 공식 index SHA-256, 고정
policy/components, event별 trigger와 overlap/revision row 수를 provenance에 묶는다.

2026-08-21까지 245 page regression에서 quarantine event는 2024-08-07,
2025-01-07, 2026-08-12 세 건뿐이다. 이 중 본문 선언 정정은 1건이고, material
revision은 1 event·12 rows이며, 세 건 모두 완전 재발행이다. 정정 상당 page가 표
범위 밖 과거값까지 바꿀 수 있으므로 각 event 이후 최초 영향 model cutoff부터
27개 origin을 평가에서 제외한다. 겹치는 window를 합치면 quarantine origin은
51개이고, 공통 eligible 표본은 165주(2022-07-08~2026-08-07)다. 기존 DB는 private
page cache로 lineage를 재구성하며 zero-delta refresh도 observation을 다시 쓰지
않고 provenance만 안전하게 승격한다.

Archive 결과는 FX context와 별도 shadow ablation에만 쓰며
`official_archive_sensitivity=true`, `promotion_allowed=false`다. core champion 자동
승격은 금지하고 별도 promotion decision을 요구한다.

## 5. FX shadow ablation

다음 네 변형을 동일한 origin과 기존 gate에서 비교한다.

1. `v4_control`
2. `v4_plus_broad_index`
3. `v4_plus_bilateral_panel`
4. `v4_plus_all_fx`

화면용 bilateral context는 고정 9개 중 6개부터 계산하지만 모델 비교에는 9개를
모두 요구한다. 모든 변형에 필요한 FX 피처가 실제로 가용한 공통 표본 156주가
쌓이면, 다음 주 국면을 target으로 최소 104주 expanding window를 학습한다. 직전
origin의 target이 평가 origin과 겹치지 않도록 1주를 purge하고
`last_train_target < evaluation_origin`을 각 fold에서 확인한다.

모델은 train-median 결측 대치와 train-standardization을 포함한 고정 L2
multinomial logistic(`C=0.1`) 하나다. 네 변형을 동일 origin에서 log loss와
multiclass Brier로 비교한다. v4 control 대비 log loss가 0.05 이상 개선되고 Brier
악화가 0.01 이하이며, 13주 paired circular block bootstrap 1,999회의 Holm 보정
유의확률이 0.05 이하일 때만 gate 통과로 기록한다. 이 gate도 shadow diagnostic이며
자동 승격은 금지한다. 별도 승격 심사 전에는 champion core feature set에 들어가지
않는다.

평가가 실행되면 `fx-ablation-oos.csv`에 네 변형의 공통 origin·target, 실제 state,
3-state 확률, train size, 마지막 학습 target, purge와 fallback 증거를 기록한다.
원 FX·core feature 값은 넣지 않는다. payload는 이 sidecar의 행 수와 SHA-256을
고정하며 audit은 여기서 variant metric, paired bootstrap, Holm과 gate를 다시 계산한다.

## 6. 국면별 자산 설명 통계

state `t`를 관측한 뒤 다음 주 종가 `t+1`부터 SPY·QQQ·IWM·TLT·HYG·UUP의
1·4·13주 USD 성과를 측정한다. 각 state·asset·horizon에서 평균·중앙값 수익률,
상승 비율, 연율 변동성, downside 변동성, historical CVaR 5%, 평균 최대낙폭과
신뢰구간을 보고한다.

표본 20개와 고유 episode 5개 이상이면 `ok`로 판정하고 신뢰구간을 제공한다.
지원 기준 미만의 점추정치는 `insufficient_support`로 구분한다. 신뢰구간은 episode
경계를 넘지 않고 주간 origin 가중치를 보존하는 13주 circular-block bootstrap
1,999회로 계산한다. 이 표는
descriptive-only이며 전략수익, allocation, weight, position, signal을 출력하지 않는다.

## 7. 사전등록 당시 결과·공개 계약

- v4가 기본·운영·live publication 계약이었다.
- v5는 로컬 opt-in 연구 결과와 별도 evidence artifact를 생성하도록 정했다.
- Membership과 forecast probability를 같은 필드나 용어로 합치지 않는다.
- 2023년 이후 retrospective 구간으로 후보·threshold를 다시 선택하지 않는다.
- 공개 패키지는 파생 JSON, source status와 검증된 정적 자산만 포함한다.
- H.10 raw observation·XML, snapshot DB, provider payload와 request 정보는 공개하지 않는다.
- 실행 profile·표본 상한·bootstrap 횟수와 사전등록 override는 hash-bound
  `model.execution_parameters`에 기록한다.

## 8. 사후 공개 결정

2026-08-23의 검토 결과와 운영 전환은
[`v5-release-decision.md`](v5-release-decision.md)에 별도로 기록한다. 사전등록 gate와
비승격 결과는 변경하지 않는다.

통계·모델 원형과 H.10 공식 문서는 [연구 참고문헌](references.md), 각 파생 피처는
[피처 카탈로그](feature-catalog.md), 공통 PIT·walk-forward 규칙은
[방법론](methodology.md)을 따른다.
