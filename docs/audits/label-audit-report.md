# Regime 정답지 감사 보고서

- 감사 기준일: 2026-08-27
- 범위: canonical 3상태 정답지, PIT 총수익·broad-equity challenger, 외부 타당성 평가 계약
- 판정 단위: **구현됨**, **synthetic/계약 검증됨**, **기존 실데이터 증거**, **미실행·미수집**을 구분한다.
- 승격 원칙: 이 문서는 모델 또는 정답지 승격을 승인하지 않는다. 불리한 결과도 같은 산출물에 남긴다.

## 결론

현재 canonical 정답지는 미국 경기순환 정답지가 아니라 **SPY 조정종가 한 종목군의 추세·스트레스에 기반한 미국 대형주 시장 국면**이다. RSP·IWM·섹터 breadth, 금리, 신용, 거시는 예측 피처이거나 연구 challenger이며 canonical label의 입력은 아니다. 따라서 현 정의는 투명하고 인과적인 frozen control로는 유효하지만, “미국 주식시장 전체 흐름”을 대표한다고 단정하기에는 좁다.

이번 변경은 기존 `market-causal-3state-v1` 결과를 바꾸지 않고 정답지 정의를 단일 hash-locked 계약으로 옮겼다. 이후 최종 source-bound [`label-bakeoff-final-20260827-r3`](../../build/label-bakeoff-final-20260827-r3/runs/20260826T160035.092517Z-80388cd7/label-audit-report.json)에서 2006-01-06–2026-08-21의 1,077개 공통 origin으로 PIT-composite·broad-equity challenger와 Pagan–Sossounov 사후 chronology를 실제 역사 자료에 실행했다. 다만 증거는 `reconstructed_oos`이고, composite는 provider-current adjusted close에서 implied split을 구성해 adjusted return을 대수적으로 재현한다. exact split series를 요구한 최종 [`strict` 실행](../../build/label-bakeoff-strict-final-20260827-r3/runs/20260826T155922.233456Z-92120dc9/label-audit-report.json)은 12개 ETF 모두 입력 누락으로 fail-closed됐다. 두 generation은 같은 research source fingerprint `9791b77434f60d27f5ff8c3e7aa62e3b34c3998fad46f9498b4534ea869541c6`에 결속된다. 따라서 공식 정답지는 계속 `v1_spy_hysteresis`이고, 새 정답지의 자동 승격은 금지한다.

## 1. Canonical 정답지의 정확한 정의

단일 source of truth는 [`config/label-spec.json`](../../config/label-spec.json)이고 typed loader는 [`src/regime_lab/analysis/label_spec.py`](../../src/regime_lab/analysis/label_spec.py)다. 현재 repo-local reviewed publication payload가 가리키는 정답지 identity는 다음과 같다.

| 항목 | 현재 계약 |
|---|---|
| spec id | `v1_spy_hysteresis` |
| version | `market-causal-3state-v1` |
| status | `official_frozen` |
| spec SHA-256 | `bec1600aea104985405d0d5c2b3706088b885ef7aa788c57c9575aa884c5f3a7` |
| 입력 | `SPY` provider-current adjusted close 한 개 |
| 상태 순서 | `risk_on`, `transition`, `risk_off` |
| 초기 상태 | `transition` |

산식은 [`src/regime_lab/analysis/labels.py`](../../src/regime_lab/analysis/labels.py)에 구현되어 있다.

1. SPY 가격을 로그 변환하고 주간 로그수익률을 만든다.
2. 방향 블록은 13·26주 가격변화를 같은 구간 변동성으로 나눈 위험조정 추세다.
3. 스트레스 블록은 4·13주 연율화 실현변동성과 13·52주 낙폭이다. 낙폭은 스트레스가 클수록 양수가 되도록 부호를 바꾼다.
4. 각 성분과 방향·스트레스 블록을 train-only median/IQR로 표준화한다. IQR이 퇴화하면 MAD를 사용하고, 그래도 퇴화하면 고정 scale 1을 쓴다.
5. `risk_score = direction_score - stress_score`로 결합한다.
6. 초기 520주 fit prefix에서 30%·70% 분위 임계값을 동결한다.
7. 임계 폭의 15%를 hysteresis margin으로 두고 순차적으로 상태를 갱신한다. 미래 관측을 보고 과거 상태를 다시 쓰지 않는다.

`risk_on ↔ risk_off` 직접 전환은 현행 hard-label 규칙에서 허용된다. 따라서 이를 무조건 중간 `transition`으로 라우팅하는 예측모형은 label의 실제 전환공간보다 좁다.

화면의 “관측 소속도”는 posterior probability가 아니다. 각 상태의 임계 anchor와 risk score 간 제곱거리, temperature `0.75`를 softmax로 정규화한 **거리 기반 membership**이다. 계약과 UI 모두 `distance_to_anchor_not_posterior`로 명시한다.

## 2. SPY와 미국 주식시장 전체 흐름의 관계

### 확인된 사실

- canonical label은 SPY adjusted close만 사용한다.
- RSP, IWM, QQQ, sector breadth, 금리, 신용, 거시, FX는 feature pipeline에 존재할 수 있으나 canonical label에는 들어가지 않는다.
- macro context를 canonical 3상태와 곱해 8상태 hard label로 만들지 않는다.

### 경제적 해석

SPY는 미국 대형주 시가총액 가중 시장의 유용한 대표값이지만, 동일가중·소형주 참여도와 업종 확산을 직접 측정하지 않는다. 따라서 현재 label을 “미국 주식시장 전체”라고 부를 때는 다음 오차가 생길 수 있다.

- 소수 초대형주가 SPY를 끌어올리지만 RSP·IWM·다수 섹터가 약한 좁은 상승을 `risk_on`으로 볼 수 있다.
- 대형주 방어가 강한 동안 소형주·equal-weight 스트레스가 먼저 확대되는 국면을 늦게 포착할 수 있다.
- 현재 조정종가는 제공자의 **현재 조정계수로 재작성된 과거**이므로, 과거 시점에 알려진 배당·분할 정보만으로 만든 PIT 총수익과 같지 않다.

이 한계는 정의가 틀렸다는 의미가 아니다. canonical 목적을 “SPY/미국 대형주 추세·스트레스 국면”으로 정확히 표시하면 투명한 control이다. 더 넓은 시장 정의가 우수한지는 별도 외부 타당성으로 검증해야 한다. 금융시장 regime에 단일 객관적 정답지가 없고 평균수익뿐 아니라 변동성·상관·지속성이 함께 달라질 수 있다는 점은 [Ang–Timmermann](https://www.annualreviews.org/content/journals/10.1146/annurev-financial-110311-101808)의 정리와 일치한다. ex-post bull/bear chronology는 [Pagan–Sossounov](https://onlinelibrary.wiley.com/doi/full/10.1002/jae.664)를 연구 기준으로만 사용한다. NBER 경기순환은 목적과 정보집합이 다르므로 canonical label을 대체하지 않는다([NBER 절차](https://www.nber.org/research/business-cycle-dating/business-cycle-dating-procedure-frequently-asked-questions)).

## 3. 구현 상태

| 대상 | 상태 | 현재 증거 | 해석 |
|---|---|---|---|
| `v1_spy_hysteresis` | 구현·동결 | hash lock, 기존 출력 bit-for-bit 회귀 테스트 | 공식 control 유지 |
| `v2_spy_pit_total_return` | 구현·synthetic 검증·reconstructed composite 실행 | 1,077 matched origin, latest state `transition`; v1과 0주 변경·agreement 1.0 | 현재 adjusted-return을 대수적으로 재현한 composite 민감도일 뿐 historical PIT가 아님; exact split strict run은 12/12 series 누락으로 차단 |
| `v2_broad_equity` | 구현·synthetic 검증·reconstructed composite 실행 | 1,077 matched origin, latest state `transition`; v1 대비 192주 변경·agreement 0.821727 | flip 135→285, transition-date Jaccard 0.242604로 더 불안정한 결과를 보존; 우월성 증거 아님 |
| label 품질 평가 | 구현·reconstructed 실행 | 세 label의 점유율, 체류기간, flip, crash/recovery lag, SPY·RSP·IWM 1·4·13주 외부 성과, prefix 안정성 생성 | 전체 window·quantile·hysteresis sensitivity grid는 비어 있고 operational vintage 비교도 미완료 |
| filtered HSMM | causal shadow 코드·synthetic 및 reconstructed 실데이터 shadow 실행 | 1,064주 causal forward filter, backward smoothing·supervised target 미사용, MAP direct jump 1건 | canonical target 아님; latest-revised market backfill이라 operational OOS 아님 |
| Pagan–Sossounov chronology | ex-post 라이브러리·synthetic 및 reconstructed 실데이터 실행 | 247 monthly proxy rows, turning point 8건; `uses_future_observations=true`, canonical/promotion false | exact month-end daily close가 아닌 주간 raw-close 월말 proxy이며 미래 확인이 필요한 사후 감사만 가능 |

관련 검증은 [`tests/test_analysis_label_research.py`](../../tests/test_analysis_label_research.py), [`tests/test_pit_total_return.py`](../../tests/test_pit_total_return.py), [`tests/test_shadow_regimes.py`](../../tests/test_shadow_regimes.py), [`tests/test_pagan_sossounov.py`](../../tests/test_pagan_sossounov.py)에 있다. 2026-08-26 현재 focused 계약 검증에 이어 전체 1,054건을 수집해 `1,053 passed, 1 skipped`로 완료했다. skip은 명시적 환경변수가 필요한 장시간 frozen V4 offline E2E다. 이는 synthetic/contract 검증을 실데이터 승격 증거로 바꾸지 않는다.

## 4. 정답지 challenger 계약

### `v2_spy_pit_total_return`

raw close와 그 시점까지 알려진 배당·분할 이벤트로 재구성한 SPY PIT total-return index를 요구한다. 현행과 같은 13/26·4/13·13/52주 구조를 사용해 **조정 방식만 바꾼 효과**를 먼저 분리한다.

구현 계약은 주간 gross return을 `(close_t × split_t + dividend_t) / close_(t-1)`로 정의하며, `dividend_t`는 pre-split share 기준으로 정규화된 현금배당이다. 입력마다 source release, provider first-seen, system retrieval, revision sequence, raw SHA-256을 요구한다. reconstructed track에서도 source release가 decision 이후인 행은 거부하고, symbol별 결과는 evidence track·decision index·snapshot hash가 일치하는 typed panel로만 labeler에 전달한다. 최종 composite run에는 SPY·RSP·IWM·9개 섹터 ETF의 `dividend_amount`가 각각 1,077개 finite row로 실제 수집되어 close·adjusted close와 같은 1,077주에 정렬됐다. 그러나 exact split coefficient는 Alpha Vantage weekly 응답에 없었다([Alpha Vantage documentation](https://www.alphavantage.co/documentation/)). strict 경로는 이를 단위값으로 채우거나 adjusted close로 대체하지 않고 차단한다. 별도 composite sensitivity만 current adjusted close에서 implied coefficient를 구성하며, 이 경로는 `adjusted_close_composite_is_historical_pit=false`, `operational_oos_eligible_for_promotion=false`로 명시된다.

### `v2_broad_equity`

- 방향: SPY·RSP·IWM 13·26주 위험조정 추세
- 스트레스: SPY 4·13주 변동성, 13·52주 낙폭
- breadth: RSP·IWM과 XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY의 양(+) 1주 수익률 비율과 13주 양(+) 추세 비율
- 결합: 블록 안 동일가중, train-only robust scaling, `direction + breadth - stress`

고정 9개 섹터를 사용한 이유는 장기 공통분모를 유지하기 위해서다. 한 구성요소가 빠진 주에는 분모를 줄여 낙관적으로 계산하지 않고 해당 breadth 값을 결측으로 둔다.

### 2026-08-27 reconstructed bake-off 결과

[`label-audit-report.json`](../../build/label-bakeoff-final-20260827-r3/runs/20260826T160035.092517Z-80388cd7/label-audit-report.json)은 fit prefix 520주와 공통 origin hash `f7a577b8f36f1a2f6d59d2325c3f4e6a48654abdf2bcb69b7ec8032f6f54cba7`를 결속한다. 세 label 모두 prefix 520·798·1,077행 재계산에서 과거 mismatch 0을 기록했다.

- PIT composite는 v1과 1,077주 모두 같았다. 이는 새로운 PIT 정보가 v1을 확인했다기보다 composite 정의가 current adjusted return을 그대로 재현한다는 결과다.
- broad equity는 v1과 192주가 달랐고 agreement는 `0.821727`, transition-date Jaccard는 `0.242604`였다. 연환산 flip도 `6.524`에서 `13.773`으로 늘어, breadth 추가가 현재 설정에서 잦은 상태전환을 유발했다.
- 2008, 2020, 2022 drawdown episode에서 broad label의 Risk-off 탐지 지연은 0주였지만, 이것만으로 정답지 우월성을 판단하지 않는다. 1·4·13주 outcome은 close-to-close 구성타당성 통계이며 거래 성과가 아니다.
- Pagan–Sossounov은 247개 월별 proxy 관측에서 8개 turning point를 냈다. 각 turning point는 8개월 미래 확인을 사용하므로 운영 label·예측 target으로 사용할 수 없다.

## 5. 실데이터 승격 전 필수 감사

아래 항목 중 matched-origin reconstructed 비교와 일부 품질표는 생성됐다. 그러나 exact corporate-action PIT, 전체 민감도 grid, operational vintage, 독립 승격 감사까지 모두 통과한 것은 아니다.

1. **PIT 입력 인증**: 각 ETF의 raw close, 배당, 분할, `source_released_at`, `provider_first_seen_at`, `system_retrieved_at`, revision, raw hash를 보존한다.
2. **동일 origin 비교**: v1, SPY PIT, broad equity가 같은 weekly origin과 같은 가용정보 범위에서 비교되어야 한다.
3. **상태 품질**: 시대별 점유율, 체류기간, 연간 flip, direct jump를 보고 특정 시대·상태의 붕괴를 확인한다.
4. **외부 구성타당성**: SPY·RSP·IWM의 다음 1·4·13주 수익률, 변동성, 최대낙폭 분리가 경제적으로 일관적인지 확인한다. 이 값은 close-to-close 기술 통계이며 실행 가능한 매매수익이 아니다.
5. **급락·회복 지연**: drawdown episode별 Risk-off 및 Risk-on 포착 지연과 미탐을 함께 보고한다.
6. **민감도**: 8/13/26/52주, 20/30/40 분위, 0/10/15/20% hysteresis 전 조합에서 결론이 유지되는지 확인한다.
7. **prefix·vintage 안정성**: 미래 행 또는 늦은 revision을 추가해도 과거 label·score·membership가 바뀌지 않아야 한다.
8. **모형결론 안정성**: 정답지를 바꿔도 데이터·모델 ablation 결론이 유지되는지 확인한다. “예측하기 쉬움”만으로 label을 고르지 않는다.
9. **독립 감사**: label spec hash, 입력 snapshot hash, code/execution hash와 결과를 별도 프로세스가 재계산한다.

## 6. 현재 판정

- **공식 유지**: `v1_spy_hysteresis`.
- **현행 publication**: generation `20260826T184946.198911Z`의 `operational_oos`·dynamic champion·manifest/2이며, label spec은 여전히 `market-causal-3state-v1`이다. [`build/pages-workflow-package-final-20260827`](../../build/pages-workflow-package-final-20260827)은 local browser QA와 공개 8파일 byte readback을 통과했다. CI·Pages·브라우저·자동화 증거는 [`release-evidence.md`](release-evidence.md)에 분리해 기록한다.
- **연구 진행 가능**: `v2_spy_pit_total_return`, `v2_broad_equity`.
- **승격 불가 사유**: matched-origin reconstructed 표는 생성됐지만 exact split/당시 first-seen을 포함한 operational PIT panel은 없다. composite PIT는 v1 adjusted return을 대수적으로 재현했고, broad label은 flip이 두 배 이상 늘었으며 전체 sensitivity grid·정답지별 모델 ablation도 미완료다.
- **표현 원칙**: canonical을 “SPY 추세·변동성·낙폭 기반 3국면”으로 표시하고, broad market·거시는 predictor/challenger로 표시한다.
- **투자 해석**: 상태별 1·4·13주 통계는 정의의 외부 타당성 진단일 뿐 매매전략, 미래수익 또는 원금보장을 뜻하지 않는다.
