# US Equity Regime Lab

미국 증시의 **현재 주간 국면**과 **다음 주 국면 확률**을 point-in-time
정보로 산출하고, 날짜를 선택해 결과를 탐색하는 로컬 연구 프로젝트입니다.
실제 투자 주문·자산배분 백테스트는 범위에 포함하지 않습니다. 현재 공개 페이지는
2026-08-21까지의 검토된 V5 스냅샷입니다. 로컬 수집·학습과 원자료를 제외한
개인·비상업 파생 결과 공개는 확인된 프로젝트 승인 범위에서 실행합니다.

## 무엇을 보여주나

- `risk-on / transition / risk-off` 현재 국면과 세 상태의 확률
- 같은 시점에서 계산한 다음 주 국면 확률
- 다음 주 권위 예측과 분리해 비교하는 1·4·13주 내 `any-departure` 위험
- 예측 entropy·confidence와 별도로 표시하는 explicit-duration shadow nowcast
- 추세·스트레스·거시·금융여건 score와 주요 driver
- 동일한 walk-forward 조건에서 비교한 next-state·전환위험 모델 leaderboard
- `dataAsOf`, 미국 동부시간 cutoff, source freshness·권리·장애 상태

대시보드는 Python 결과를 다시 계산하지 않습니다. Python 파이프라인이 계약에 맞는
결과를 만들고, dependency-free 정적 웹 앱이 그 결과를 읽습니다. 로컬 `serve`의
기본 입력은 검토된 `publication/live/regime-results.json`이다. GitHub Pages는
원자료나 DB가 아니라 이 payload, hash-bound V5/V4 비교 sidecar와 정적 자산
allowlist만 배포합니다.

공개 live 파생 결과: [sonchanggi.github.io/regime](https://sonchanggi.github.io/regime/)

## V5 운영·연구 계약

최신 공개 페이지와 주간 자동화의 운영 계약은 V5다. V4는 이전 계산과의 회귀를
검증하는 동결 비교 기준이며 현재 서비스 버전이 아니다. 첫 V5 표준 실행에서는
당시 gate 결과에 따라 Markov가 champion으로 선정됐다. 이후 V5 공식 모델은 이름으로
고정하지 않고 동일 OOS 표본의 selection gate를 통과한 단일 champion을 사용한다.
공개 payload에는 선정 근거와 검토 결정을 기록하고, V4 비교 결과는 파생값 전용
sidecar로 함께 배포한다.

```bash
.venv/bin/regime-lab demo --contract v5 \
  --output build/v5-demo/regime-results.json \
  --artifacts build/v5-demo/artifacts
# live build는 수집·저장·학습 권한이 affirmative일 때만 실행
```

H.10 prospective history는 모델 학습과 분리해 가볍게 축적할 수 있다. 이 명령은
`--contract v5`를 반드시 요구하고 live build와 같은 DB lock을 사용한다. H.10
private snapshot DB와 파생 상태 receipt만 갱신하며 공개 payload·artifacts·자동화
health와 V5 모델 결과는 변경하지 않는다.

```bash
.venv/bin/regime-lab collect-h10 --contract v5
# 기본 receipt: build/v5-h10/collection-receipt.json
# 필요할 때만 명시적인 model cutoff와 격리된 receipt 경로를 지정
.venv/bin/regime-lab collect-h10 --contract v5 \
  --as-of 2026-08-21T20:00:00+00:00 \
  --receipt build/v5-h10/collection-receipt.json

# 공식 release archive 민감도 bootstrap/증분 갱신
.venv/bin/regime-lab collect-h10 --contract v5 \
  --official-release-archive-ingest \
  --archive-start 2022-01-01 \
  --archive-through 2026-08-21 \
  --as-of 2026-08-21T20:00:00+00:00
# 기본 receipt: build/v5-h10/archive-collection-receipt.json
```

월요일 H.10 공개 이후 매주 한 번 실행하면 실제 first-seen 기준 표본이 누적된다.
Receipt에는 원자료 없이 snapshot 증감, source/FX 상태, last-good 사용 여부,
156주 공통표본 readiness와 archive의 파생 lineage 개수·정책·인덱스 SHA-256만
기록한다. Archive 명령은 공식 JSON 인덱스를 매회 검증하고 새 release event만
가져오며, 중단된 page는 비공개 DB cache에서 재개한다. 기존 DB도 zero-delta
refresh에서 관측값을 다시 쓰지 않고 cache 기반 lineage provenance를 승격한다.
스케줄 설치는 별도 opt-in 작업이다.

실제 profile·표본 상한·bootstrap 횟수와 사전등록 override는
`model.execution_parameters`에 SHA-256으로 결속한다. quick 데모의 199회와
standard/full의 1,999회를 같은 실행으로 오인하지 않는다.

v5는 현재 hard state에 대한 `membership`과 다음 주 `forecast probability`를
분리한다. 1·4·13주 방향성 위험은 horizon 끝 상태가 아니라 **처음 현재 상태를
떠날 때 도착하는 상태**를 예측하며, 기존 any-departure 확률과 합계가 일치한다.
현재 spell은 우측 검열해 상태별 Kaplan–Meier 조건부 생존율과 52주 제한
평균 잔여기간(RMST)을 산출한다.

Federal Reserve H.10의 Broad·AFE·EME 달러 지수와 고정 9개 통화쌍을
`양수 = USD 강세`로 정규화한다. 최초 수집 이전의 공개시점을 소급 복원하지 않고,
관측·수정값은 실제 `first_seen` 이후 origin에만 투입한다. FX는 v4 control,
Broad 추가, bilateral 추가, 전체 FX 추가의 네 shadow ablation은 공통 156주가
쌓인 뒤 9/9 pair, 104주 expanding 학습, 1주 target purge, 고정 L2 multinomial
logistic으로 동일 origin에서 평가한다. 화면용 context의 6/9 기준과 모델 평가의
9/9 기준은 다르다. gate 통과도 자동 승격하지 않고 별도 승격 심사를 거친다.
평가 행은 원 FX 값 없이 `fx-ablation-oos.csv`에 남겨 metric과 gate를 재검산한다.
국면별 자산 통계는 state `t`를
관측한 뒤 `t+1`부터 측정하는 설명 통계이며 매매·배분 신호가 아니다.

세부 계약은 [v5 사전등록](docs/structural-v5-preregistration.md),
[V5 공개 결정](docs/v5-release-decision.md), [방법론](docs/methodology.md),
[피처 카탈로그](docs/feature-catalog.md),
[연구 참고문헌](docs/references.md)에 고정한다. 공개물에는 파생 결과와 상태만
포함하며 H.10 원자료·로컬 DB·provider payload는 포함하지 않는다.

## 빠른 실행: 격리된 연구용 데모

```bash
python3 -m venv .venv
# Apple Silicon에서 XGBoost가 요구하는 OpenMP runtime
brew install libomp
.venv/bin/pip install -e '.[test,hmm]'
.venv/bin/regime-lab demo --contract v5 \
  --output build/public-demo-source/regime-results.json \
  --artifacts build/public-demo-artifacts
.venv/bin/regime-lab serve \
  --payload build/public-demo-source/regime-results.json
```

브라우저에서 `http://127.0.0.1:8765/`를 엽니다. 데모는 UI와 전체 분석
경로를 재현하기 위한 고정 seed 모의 자료이며, 실제 미국 시장 판단으로
표시되거나 사용되지 않습니다. `build`와 `demo`의 기본 계약은 현재 운영 계약인
V5다. 동결 V4 회귀 결과가 필요한 경우에만 `--contract v4`를 명시합니다.

## 공개 live 파생 결과 갱신

공개 패키지는 정적 앱 3개 파일, 파생 payload, V5/V4 sidecar와 파일별 SHA-256
manifest만 허용합니다. `config/provider_rights.json`이 수집·저장·학습·파생 공개를
모두 허용하지 않으면 live 패키징과 Pages 갱신은 시작되지 않습니다. FRED/ALFRED와
Alpha Vantage의 프로젝트별 수집·저장·학습 승인은 2026-08-24, 원자료를 제외한
개인·비상업 파생 결과 공개 범위는 2026-08-25 사용자 확인으로 기록했습니다.
합성 데모 패키징은 이 경계와 분리됩니다.

## 주간 수집·재학습·배포 자동화

로컬 LaunchAgent와 Pages의 분리 구조를 유지하고, 원자료 또는 승인 범위를 벗어난
공개는 계속 차단합니다. `automation status`는
단순 due-check와 전체 수집→공개 성공 시각을 분리해 기록합니다.

```bash
.venv/bin/regime-lab automation status
```

자동화 해제는 다음과 같습니다.

```bash
.venv/bin/regime-lab automation uninstall
```

## 실데이터 실행

실데이터 로컬 연구 경로는 승인 범위 안에서 실행할 수 있습니다. 사용자 확인 flag만으로
권리를 새로 만들지는 않으며, live CLI는 네트워크·DB 변경 전에 기계 판독 가능한
`config/provider_rights.json`을 검사합니다.

- FRED/ALFRED·Alpha Vantage: 프로젝트별 수집·저장·학습 및 개인·비상업 파생 결과 공개 승인 기록
- 두 provider의 원자료·재구성 가능한 관측값 및 상업적 공개: 금지
- Federal Reserve Board 직접 자료: public-domain 고지와 출처 표기를 전제로 허용

V6의 역사적 matched OOS 후보는 공식 전체 빈티지가 있는 Philadelphia Fed
ADS/RTDSM으로 제한합니다. Board H.4.1/H.8은 dated archive parser와 520주 공통표본을
검증한 뒤 합류할 수 있고, H.15/OFR FSI는 retrospective sensitivity, H.10/CP는
prospective shadow로 분리합니다. 시장 가격·수익률 label은 승인된 Alpha Vantage를
비공개 수집·저장·학습에 사용합니다. 최초 수집 때 받은 과거 조정 이력은 당시의
역사적 빈티지를 재현하지 않으므로 retrospective sensitivity로만 평가하고, 이후
동결한 스냅샷부터 prospective 근거를 축적합니다. 검토를 통과한 파생 결과만
개인·비상업 공개 경로에 승격할 수 있습니다.

OFR FSI는 운영 V5 DB·학습·공개 build와 분리된 V6 prospective shadow로만
수집합니다. 아래 명령은 공식 OFR aggregate CSV와 공개 category/region contribution만
private SQLite에 append-only로 보존하고, 값이 없는 로컬 receipt만 작성합니다.
공식 intraday 발표시각은 알려져 있지 않으므로 `source_released_at`은 비워 두고 실제
`provider_first_seen_at`부터만 사용할 수 있습니다. 과거 current-history를 PIT 빈티지로
소급하지 않으며, raw CSV와 관측값은 public package에 포함하지 않습니다.

```bash
.venv/bin/regime-lab collect-ofr-fsi --contract v6 \
  --database data/ofr-fsi-shadow.sqlite3 \
  --receipt build/v6-ofr-fsi/collection-receipt.json
```

provider 실패나 schema drift가 발생하면 OFR shadow의 last-good만 유지하며 주간 운영
build의 성공/실패에는 영향을 주지 않습니다. 저장 전에
`config/structural_v6_research.json`, `config/release-source-catalog.json`,
`config/provider_rights.json`의 OFR aggregate 계약을 모두 대조합니다.

## 정보 시점 계약

- 기준 시각: 완료된 미국 거래 주의 금요일 16:00 `America/New_York`
- 사용 조건: `available_at <= cutoff`
- 발표시각을 모르는 당일 거시자료: 보수적으로 다음 주부터 사용
- 월·분기 자료: 실제 발표 뒤 forward-fill하며 `age_days`, `is_filled`,
  `release_lag_days`를 함께 보존
- 금지: backward-fill, centered smoothing, full-sample scaling/PCA,
  smoothed HMM/Viterbi 상태를 실시간 label로 사용
- 가격 label: 시장 자료만 사용; 거시·금리·FX·유동성은 예측 input으로만 사용

자세한 정의는 [methodology](docs/methodology.md)를 참고하세요.

## 입력 자료와 구조 실험

V5의 Alpha Vantage·ALFRED feature는 승인된 로컬 연구 경로에서 새 학습에도 사용할
수 있습니다. V6 preregistration은 ADS/RTDSM만 역사적 PIT core 후보로 두고,
나머지 공식 자료는 빈티지 복원 가능성에 따라 archive-certification,
retrospective-sensitivity, prospective-shadow로 나눕니다. 단순 `/USD` 통화쌍 확대는
기존 H.10 FX ablation과 중복되므로 primary 후보가 아닙니다. CP spread는
funding-liquidity 후보, DOL/BLS/Census/BEA direct 자료는 archive parser 검증 전까지
shadow/reconciliation입니다.

구조 v4 실험은 기존 label과 v3 기준선을 그대로 두고, 11개 GICS sector
breadth, H.15 Treasury curve level/slope/curvature, H.8 bank credit·funding,
ANFCI, 과거 release만 이용한 macro innovation을 추가합니다. 모델 쪽에서는
1주 이탈 hazard와 조건부 destination을 결합하고, 완료된 과거 OOS loss만으로
Markov·XGBoost·hazard-destination 구조의 비중을 갱신하는 causal ensemble을
비교합니다. 한 주 hazard를 1·4·13주로 누적한 monotone survival 값은
geometric-duration coherence shadow일 뿐 선정 후보나 운영 예측을 대체하지
않습니다. 사전 계약과 v3 SHA 기준선은
[structural v4 preregistration](docs/structural-v4-preregistration.md)에 고정했습니다.

## 모델 비교

v3 기준 next-state 비교군은 14개입니다. Persistence·smoothed Markov·majority 기준선에
elastic-net 및 ridge multinomial logistic, 현재 국면 dummy를 추가한
transition logistic, duration-aware TVTP stay/switch/destination hurdle,
shrinkage LDA, calibrated linear SVM, PCA-spline logistic, random forest,
extra trees, histogram gradient boosting, shallow regularized XGBoost를 같은
feature와 expanding walk-forward split으로 비교합니다. Gaussian HMM은 `full`
프로필에서만 추가합니다. v4는 사전등록한 `xgb_hazard_destination`과
`causal_dynamic_ensemble`을 더해 standard 16개, full 17개를 같은
walk-forward gate에서 비교합니다. V5 standard는
`causal_multiscale_ensemble`을 추가한 17개이며, V5 full은 Gaussian HMM을 더한
18개입니다. 공식 champion은 이 후보군의 검증된 selection 성과로 결정합니다. Deep learning은 현재 범위에서 제외하며,
HMM의 full-sample smoothed path는 label이나 실시간 판정에 쓰지 않습니다.

V6 opt-in 연구 후보는 최근 정보에 더 큰 가중치를 두는
`recency_weighted_xgboost_208w`, fold 내부에서만 PCA를 적합하는
`pca_ridge_logistic`, 전이행렬에 지수감쇠를 적용하는
`discounted_markov_208w`입니다. 기존 공개 후보 목록과 champion은 바꾸지 않으며,
2026-08-21 이후 prospective 표본이 쌓여야 승격 심사를 시작합니다.

2023년 이전의 이용 가능한 OOS origin 전체만으로 provisional champion을
고정합니다. 각 challenger는 가장 좋은 probabilistic baseline보다 selection
Log loss가 최소 0.01 낮고, fallback이 없고, Brier가 0.01보다 더 악화되지
않아야 합니다. 주별 paired loss에는 13주 circular moving-block bootstrap
(seed 17, 1,999회)과 one-sided Holm 다중비교 보정을 적용합니다. 통과 모델이
없으면 가장 좋은 probabilistic baseline을 유지합니다. V4 동결 비교에는 당시의
0.05 기준을 그대로 보존합니다.

이 프로젝트에서 2023년 이후 결과는 이전 후보군을 대상으로 이미 확인한
적이 있으므로, v3 후보군에 대해 더 이상 `untouched holdout`이라고
부르지 않습니다. 화면과 산출물에서는 `retrospective external-period
diagnostic`으로 명시하며, 이 구간의 결과로 champion을 교체하지 않습니다.
진단 구간에서 champion의 Log loss가 최우수 모델보다 0.05 초과 뒤처지면
`weak_generalization`과 `degraded` 상태는 계속 표시합니다. 후보·고정
hyperparameter·계산 budget은 `candidate-manifest.json`과 SHA-256으로 남깁니다.

next-state champion과 별도로 1·4·13주 전환위험 연구군은 empirical hazard,
homogeneous Markov hazard, duration-aware TVTP hurdle, regularized binary
logistic을 비교하고, 표준·full 프로필에서는 가용한 경우 shallow binary
XGBoost도 포함합니다. 여기서 사건은 **기준 국면과 다른 상태가 `t+1..t+h`
중 한 번이라도 등장하는 것**입니다. `t+h`의 마지막 상태나 특정 destination을
맞히는 문제가 아니므로, 떠났다가 돌아온 경로도 사건으로 셉니다.

각 horizon `h`에서 학습 label의 `target_end`는 예측 origin보다 반드시 앞서며
`h`개 origin을 purge합니다. 모델 family, Platt-logit calibration, 경보 threshold는
2023년 이전에 `target_end`까지 끝난 selection OOS만으로 정합니다. threshold는
그 과거 행의 balanced accuracy를 최대화하고 근거가 부족하면 0.5를 사용합니다.
진단 origin은 2023-01-01 이후부터 시작하며, horizon이 cutoff를 가로지르는 중간
`h-1`개 origin은 embargo로 양쪽 평가에서 제외합니다. 이후에는 family,
calibrator, threshold를 고정하되 각 estimator 자체는 해당 origin 전에 결과가
확정된 행까지 expanding 방식으로 다시 적합합니다. 이 구간은 계속
transition 산출물에서 `retrospective_diagnostic`으로 표시합니다. 전체 dashboard의
next-state 기간 역할은 `retrospective_external_period_diagnostic`으로 유지됩니다.

화면의 1주 `transition_probability`는 권위 있는 next-state 확률에서 계산한
`1 - P(S[t+1] = S[t])`이며 `transition_risk.1w`와 같은 값입니다. 4·13주 값은
별도 전환위험 benchmark가 제공합니다. 또한 고정 parameter explicit-duration
filter를 **shadow nowcast**로 병기하지만, 이는 민감도 점검일 뿐 canonical 3국면
label이나 다음 주 정답을 바꾸지 않습니다.

각 build는 payload와 모든 supporting artifact를 비공개 sibling transaction
directory에 먼저 완성합니다. 기존 payload와 artifact를 함께 recovery 위치로
옮긴 뒤 새 generation을 승격하며, 실패하면 둘 다 이전 generation으로
rollback합니다. 핵심 추가 artifact는 다음과 같습니다.

- `transition-oos-predictions.csv`
- `transition-model-leaderboard.csv`
- `transition-walk-forward-splits.csv`
- `nested-selection.csv`
- `transition-forecasts.csv`
- `transition-candidate-forecasts.csv`
- `transition-candidate-status.csv`
- `feature-manifest.json`
- `feature-quality.json`
- `feature-ablation-manifest.json`
- `feature-ablation-oos-predictions.csv`
- `feature-ablation-leaderboard.csv`
- `stacking-weights.csv`
- `structural-forecasts.csv`
- `joint-survival-forecasts.csv`
- `state-label-history.csv`
- `weekly-state-forecasts.csv`
- `multiscale-ensemble-scales.csv`
- `directional-oos-predictions.csv`
- `directional-model-leaderboard.csv`
- `directional-walk-forward-splits.csv`
- `directional-selection-diagnostics.csv`
- `directional-forecasts.csv`
- `conditional-asset-outcomes.csv`
- `conditional-asset-statistics.csv`
- `fx-features.csv`
- `fx-coverage.csv`
- `fx-ablation-oos.csv`

v2 비교 기준은 로컬 ignored snapshot으로 고정했습니다. payload SHA-256은
`50ab693b15f5100b1e39d98356c88455b76a4a2c4a4c335e5882509568c5fe98`,
snapshot inventory SHA-256은
`09603aca14244fc00ee56f0d75a45192fc29a77c8f1a47b9927aef32d4fcbf0f`입니다.
v4의 직접 비교 기준인 frozen v3 payload SHA-256은
`de93c585117b2784750f586a4f84ad99964c63081b252ad7affd7a75bd797095`, inventory
SHA-256은 `8ef3778cc8c36faff0c80e2bf094f1f11bd6966ab3b7b2d6edb84ba292aff6b9`입니다.
독립 audit는 상수뿐 아니라 snapshot 전 파일과 preregistration 원본을 직접
해시하므로 이후 결과가 좋아지거나 나빠져도 기준선을 소급 변경할 수 없습니다.

## 주요 명령

```bash
.venv/bin/regime-lab demo --profile quick
.venv/bin/regime-lab validate publication/live/regime-results.json
.venv/bin/python scripts/audit_outputs.py --target publication-live
.venv/bin/pytest
```

명령과 화면이 `demo`, `degraded`, `live`를 명시적으로 구분합니다. 핵심
source가 stale·차단·누락이면 마지막 값을 현재 값처럼 조용히 연장하지 않고
결과를 `unavailable` 또는 `degraded`로 표시합니다.

실데이터 날짜 선택 범위는 2023+ 사후 진단 구간부터입니다.
2006–2022 자료는 label threshold, feature history, 모델 선정에 쓰이지만 당시
날짜에 오늘 고른 champion을 소급 표시하지 않습니다. 최초 `standard` 실행은
이 MacBook Air 환경에서 1시간 이상 걸릴 수 있으며, CLI는 약 5% 간격으로
walk-forward 진행률을 출력합니다. `quick`은 개발·UI 점검용입니다.

## 디렉터리

```text
config/                 source universe와 cutoff/model 기본값
src/regime_lab/data/    provider, PIT contract, snapshot store, as-of join
src/regime_lab/analysis feature, label, walk-forward models, evaluation
src/regime_lab/         CLI, payload contract, orchestration, local server
web/                    정적 동적 dashboard
tests/                  누수·계약·모델·UI 회귀 테스트
```

## 연구 한계

Regime은 관측 가능한 정답이 아니라 잠재적 시장 요약입니다. 높은 accuracy는
지속성 baseline만 반복해도 나올 수 있으므로 확률 score와 전환 성능을 함께
봐야 합니다. 현재 공개 산출물은 개인·비상업 연구용 파생 결과입니다.

개별 기업 재무·실적은 현재 무료 source에서 장기간의 발표 당시 값과 수정 이력을
일관되게 재현하기 어려워 직접 feature로 넣지 않았습니다. 대신 규모·동일가중·섹터
ETF와 회사채 ETF를 기업·breadth proxy로 사용합니다. 신뢰할 수 있는
point-in-time fundamentals source가 확보되기 전까지 오늘의 재무 이력을 과거
fold에 소급하는 방식은 허용하지 않습니다.
