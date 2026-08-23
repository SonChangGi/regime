# 방법론

## 구조 v5 운영 계약

주간 자동화와 publication은 현재 v5 계약이다. 수동 build/demo의 기본값은 V4
재현을 위해 유지하며 raw V5 후보는 명시적으로 opt-in한다. 첫 표준 실데이터 실행에서 Markov는 frozen v4
Markov와 matched OOS 확률·평가값이 정확히 일치해 champion을 유지했다. 다중 기억
앙상블과 FX ablation은 사전등록 gate를 통과하지 못해 core champion에 승격하지
않았다. 후보 생성과 공개 검토 표식 추가는 분리하며, 공개 payload는 파생 비교
sidecar와 byte 단위로 결속한다. 전체 사전등록값은
[`structural-v5-preregistration.md`](structural-v5-preregistration.md)에 고정한다.

- **현재 국면:** `causal_hysteresis_state`가 hard label이다. `memberships`는 같은
  시점 risk-score anchor에 대한 소속도이며 미래 확률이 아니다. 다음 주 예측은
  별도 `next_week.probabilities`에 둔다.
- **방향성 이탈:** 1·4·13주 안에 현재 국면을 처음 떠날 때의 도착 상태를
  `no_departure`와 함께 예측한다. horizon 말의 상태를 예측하는 target이 아니다.
  목적지 확률의 합은 기존 any-departure 확률에 맞춘다. 방향 모델은 실제 이탈
  origin의 conditional destination loss로 고르고 8 events·2 classes·3 blocks
  미만이면 empirical baseline을 유지한다.
- **다중 기억 앙상블:** Markov·XGBoost·hazard-destination의 완료된
  OOS log score를 26·52·104주 half-life로 각각 할인한 후 세 확률을
  정확히 1/3씩 평균한다. 해당 origin의 fallback expert는 가중치 0이며,
  기존 common-origin log-loss·Brier·Holm gate를 통과해야만 v5 후보로
  선택된다.
- **잔여기간:** `as_of`까지 확정된 spell만 구성하고 현재 spell은 우측 검열한다.
  현재 상태를 `d`개 주간 관측에서 확인했을 때 이탈 직전 시점 `d-1`을 조건으로
  `S(d-1+h)/S(d-1)`을 계산하고, 52주까지의 조건부 생존율 합을 제한 평균
  잔여기간(RMST)으로 보고한다.
- **FX:** Federal Reserve H.10의 Broad·AFE·EME 지수와 EUR·JPY·GBP·CHF·CAD·
  AUD·CNY·MXN·BRL 고정 패널을 사용한다. 로그 변화의 부호는 모두
  `양수 = USD 강세`로 통일한다. 1·4·13주 변화, 13·26주 변동성, AFE–EME
  divergence와 9개 통화쌍 median·상승 breadth·MAD를 만든다.
- **FX 시점:** Board XML 전체 이력은 historical-vintage archive로 간주하지 않는다.
  최초 발견값의 가용시점은 `first_seen`이며 이후 발견한 과거값 변경도 발견 시점부터
  prospective revision이다. 최초 수집 이전 availability를 소급 채우지 않는다.
- **FX archive 민감도:** 2022-01-01 이후 공식 `releaseDates.json`과 dated
  release page를 XML과 별도 chain으로 replay한다. 정상 발표는 16:15 ET,
  시각이 없는 선언 정정 페이지는 다음 날 00:00 ET부터 가용하다. 선언 정정,
  직전 page 대비 material revision, 신규 series-date가 없고 직전 page key와
  정확히 일치하는 완전 재발행을 correction-equivalent event로 판정한다. 3일 이내
  page gap은 보조 경보만 맡는다. 해당 page가 표 범위 밖 과거값에 미친 영향을
  알 수 없으므로 각 event 이후 27개 model origin을 평가에서 제외한다. 이 chain은
  민감도이며 core 자동 승격은 금지된다.
- **조건부 자산 통계:** state `t`를 관측한 뒤 다음 주 종가 `t+1`에 진입한 것으로
  정렬해 SPY·QQQ·IWM·TLT·HYG·UUP의 1·4·13주 성과를 요약한다. 결과는
  descriptive-only이며 allocation·position·signal을 만들지 않는다. 신뢰구간은
  episode 경계를 넘지 않는 circular block으로 주간 origin 가중치를 보존한다.

FX의 champion 투입은 잠겨 있다. `v4_control`, `v4_plus_broad_index`,
`v4_plus_bilateral_panel`, `v4_plus_all_fx` 네 shadow ablation을 모든 필요한
FX 피처와 9개 bilateral pair가 실제 가용한 공통 156주 이상에서 비교한다. 화면용
context의 6/9 coverage 기준은 모델 평가의 9/9 기준과 분리한다. 다음 주 국면에 대해
최소 104주 expanding 학습, 1주 target purge, 고정 L2 multinomial logistic을
사용한다. 네 변형은 동일 origin의 log loss·multiclass Brier로 평가하며, control
대비 log loss 0.05 이상 개선, Brier 악화 0.01 이하, 13주 paired circular block
bootstrap 1,999회와 Holm 5% gate를 모두 충족해야 통과로 기록한다. 통과 여부와
관계없이 core champion 자동 승격은 금지한다.
공개 계약은 파생값만 허용하며 H.10 raw observation과 로컬 snapshot DB를 제외한다.
`fx-ablation-oos.csv`는 공통 origin별 실제 state·확률·purge·fallback만 담는
derived-only 감사 자료이며, aggregate metric과 paired bootstrap/Holm gate는 이
자료에서 독립 재계산한다.

## Frozen V4 기준선·비교 계약

The structural-v4 experiment is frozen in
[`config/structural_v4.json`](../config/structural_v4.json) and summarized in
[`structural-v4-preregistration.md`](structural-v4-preregistration.md). It keeps
the canonical three-state label and every v3 point-in-time rule fixed. New
data blocks and structural forecasts must pass the existing pre-2023 gate on
common origins; 2023+ remains a development diagnostic and cannot tune or
select them.

## Structural v4 interpretation contract

The structural names are implementation labels, not claims of paper
replication. Only the fixed Nelson--Siegel loadings, `lambda=0.0609`,
cross-sectional OLS, and the conditional-survival product identity closely
match their cited formulas. The weekly H.15 application, robust ALFRED
`release_innovation` block (interpreted as a release-change statistic),
reduced-form H.8/ANFCI and sector features, XGBoost
departure/destination splice, and discounted completed-OOS log-score pool are
motivated adaptations. In particular, the release statistic is not a
model-conditioned economic surprise, the bank block is not an identified
credit-supply shock, and the score pool is not Dynamic Model Averaging.

ANFCI features use the independent `anfci__*` namespace and the
`legacy_plus_financial_conditions` selection-only ablation, so their pre-2023
incremental forecast contribution is not conflated with `legacy_v3`. Ensemble
weights use only completed origins strictly before the current origin and score
all three experts on the identical all-nonfallback origin intersection. These
guards improve attribution and score comparability; they do not turn either
adaptation into a causal or Bayesian identification result.

The joint-survival projection currently repeats the same scalar one-week
departure hazard along its 13-step path. This guarantees internally monotone
1/4/13-week risks but implies a geometric duration law; after the origin hazard
is supplied, incremented duration is metadata and does not recalculate future
hazards. It is therefore a `shadow_only` coherence comparison, while the
separately fitted, horizon-purged 4/13-week
transition forecasts remain operational. The source-by-source audit and
failure conditions are in [references](references.md).

## Decision timestamp

The canonical observation is the last completed US trading week. A result for
week `t` uses only records whose `available_at` is no later than Friday 16:00
America/New_York. When an official source supplies a release date but no release
time, a same-day release becomes usable in the following week.

Monthly and quarterly values are never interpolated backward or centered. They
are forward-filled only after release, with age and fill indicators retained as
features. Historical model evaluation queries the vintage that existed at each
cutoff rather than today's revised history.

The first ALFRED snapshot is a full `output_type=1` real-time history. Daily
series are split into four-year real-time ranges to remain below the JSON limit
of 2,000 vintage dates. Later weekly snapshots request only new and revised
observations with `output_type=3`, including a one-day overlap with the prior
successful cutoff. The store reads the successful full base plus subsequent
deltas, removes overlap by the natural revision-event key, and excludes failed
or partial attempts. The local join then selects only events available by each
weekly cutoff. `T10Y2Y` and `T10YIE` have an explicit 2014 ALFRED archive start,
so their earlier feature values remain unavailable instead of being backfilled
from today's history.

For each weekly type-3 window, the adapter first queries
`/series/vintagedates` and sends only the returned dates for that series to the
observations endpoint. This is a conservative implementation rule based on a
2026-08-12 live JSON check in which an UNRATE type-3 request containing dates
not returned by vintage discovery received HTTP 400; it is not stated as a
universal guarantee about every series or future provider behavior. If FRED
returns a 5xx for a narrow window, the adapter does not infer an empty delta
from that error. It repeats discovery over the bounded observation-history
range and accepts an empty weekly intersection only after that wider response
passes schema and range validation. A failed or malformed wider response keeps
the source degraded and blocks training and publication.

Alpha Vantage's free weekly endpoint returns full symbol histories. After the
first successful full snapshot, the collector stores only new or changed rows;
a changed historical adjusted value is appended as a new revision whose
availability begins at collection/discovery time, so it cannot rewrite an
older fold. Because the endpoint supplies no explicit deletion event, an
omitted prior row degrades the response and preserves the last-good chain
rather than interpreting a transient omission as a tombstone. Provider
failures retain request provenance and health issues but never retain partial
observation rows or replace the last-good chain.

The provider documents 25 requests per day for its standard free service but
does not publish the reset timezone. The local guard therefore enforces a
provider-wide rolling 24-hour event ledger. Its configuration contract is the
literal integer `25`; alternate types or values fail before client construction
or event reservation. Legacy date counters are reconciled
bucket by bucket against immutable successful and failed-batch provenance;
only calls with auditable post-response timestamps receive those timestamps,
while unmatched calls are conservatively anchored at migration time. Multiple
standard-free credentials share this aggregate cap and are never rotated after
a quota response. A premium or provider-verified open-source/educational
credential requires explicit entitlement and rate metadata before it can use a
different limit; such quota entitlement does not grant public redistribution
rights.

Before any Alpha transport, the collector atomically reserves the complete
requested-symbol batch in SQLite. The transport spends only those prepaid
credits, and unused credits remain charged after a provider failure or process
crash. Standard-free requests are not retried, so the reserved batch size is
also the maximum number of transport attempts. Every reservation transaction
also absorbs any concurrent legacy calendar-counter write before testing the
aggregate cap.

A spent prepaid credit is timestamped again immediately before its actual
transport attempt, so a later symbol cannot expire earlier than its real call.
An upgrade requires an operational barrier that stops all legacy collectors
before the first v3 reservation. The new binary reconciles legacy rows already
written before each reservation, but two independently running old/new binaries
cannot mutually exclude a legacy call written after the new transaction commits;
that call is detected and charged on the next new-ledger access.

The initial Alpha Vantage baseline is the adjusted history visible on the first
collection date, not a reconstruction of the adjusted values that the provider
showed on every historical date. The endpoint does not expose those original
vintages. Therefore the pre-initialization market history is an explicit
current-adjusted backfill limitation; only changes discovered after project
initialization receive prospective revision timestamps.

The endpoint can also expose an in-progress current-week row. Alpha rows whose
observation period is later than the completed-week cutoff of the snapshot are
excluded permanently from the last-good chain; advancing a later cutoff cannot
make a formerly partial row reappear as completed history.

An ALFRED missing-value revision invalidates the earlier revision for that same
observation period. Weekly model input may still forward-fill the latest older
period with a valid value (for example across a market holiday), while the
audit view can expose the tombstoned period itself.

## Regime definition

The operational target has three market-only states:

- `risk_on`: positive risk-adjusted 13/26-week trend without extreme stress.
- `transition`: mixed direction/stress evidence or a recent state boundary.
- `risk_off`: negative trend or severe downside volatility/drawdown.

Thresholds are estimated on the training history only. Entry and exit thresholds
use causal hysteresis and never move a transition backward after observing the
future. Macro, rates, FX and liquidity data are predictors, not ingredients of
the supervised target, which avoids circular accuracy.

The canonical state label and forecast horizon remain unchanged in v3. The
primary supervised problem is still the three-class state at `t+1`; the
multi-horizon departure models described below are additive diagnostics and do
not redefine the label.

## Feature construction

Every transformation is one-sided and uses the current as-of row and rows to
its left. Missing values remain missing until fold-local model imputation; the
feature layer performs no backward fill, centered smoothing, or full-sample
standardization.

The 20-ETF market panel contributes adjusted-price returns, risk-adjusted
trends, realized/downside volatility, drawdowns, recovery, and relative
performance versus SPY. Equity-only market internals summarize positive-return
breadth over 1/4 weeks, participation above trailing 13/26-week means,
cross-sectional 1/4-week return dispersion, one-week directional
synchronization, and average pairwise correlation over trailing 13/26-week
windows. Bond, commodity, and currency ETFs are explicitly excluded from the
equity breadth denominator.

Five economically interpretable ETF spreads are measured over 4/13 weeks:
XLY/XLP (cyclical versus defensive), XLK/XLU (growth versus defensive), HYG/LQD
(high yield versus investment grade), HYG/IEF (credit versus Treasury), and
TLT/SHY (long versus short Treasury). All configured ETF volumes contribute
past-only log changes and rolling z-scores, plus rising-volume breadth and a
net price-volume confirmation statistic.

For SPY, IWM, RSP, HYG, and TLT, the same-row ratio of adjusted close to raw
close adjusts open, high, and low. The resulting features are the weekly
log high-low range, close location within that adjusted range, and the log gap
from the previous adjusted close to the adjusted open. The collector requires
complete `open/high/low/close/adjusted_close/volume` coverage and fails closed
instead of combining unmatched periods or synthesizing a missing OHLC field.

ALFRED inputs include rates, inflation, labor, growth, the broad dollar,
liquidity, NFCI, and the four official NFCI contributions: risk, credit,
financial leverage, and nonfinancial leverage. Besides levels and ordinary
changes, each macro series can contribute a 4-week change standardized only
against its trailing 52-week history. Age, release-lag, forward-fill, and
missingness indicators remain explicit inputs.

## Next-state model comparison

Every next-state candidate receives the same point-in-time rows and one-week
target. The outer evaluation is expanding walk-forward with a one-week gap.
Missing-value handling, scaling, class weights, feature selection, calibration
and fitting are performed inside each training window. The primary score is
multiclass log loss; Brier score, balanced accuracy, macro F1, transition
recall/precision and calibration error are guardrails.

Model-family choice and later-period reporting use different time segments.
Every available OOS forecast whose target precedes 2023-01-01 is used for
selection. Because results from 2023 onward had already been inspected for an
older model suite before the v3 candidate expansion, that period is honestly
labelled a `retrospective_external_period_diagnostic`, not an untouched
confirmatory holdout. It never enters the v3 champion-selection function. The
latest t+1 forecast refits the selected family on the expanding information
set. Any fit or probability-contract failure is surfaced as a class-prior
fallback and degrades result health.

Selection is probability-first and conservative. Each learned challenger is
paired with the best probabilistic baseline on identical weeks. Promotion
requires at least 0.05 absolute multiclass-log-loss improvement, zero fallback
rows, no more than 0.01 Brier degradation, and a one-sided Holm-adjusted result
from a deterministic 13-week circular moving-block bootstrap (1,999 resamples,
seed 17). If no challenger clears every gate, the best baseline remains the
provisional champion. Candidate definitions, hyperparameters, profile budgets,
seed and a SHA-256 are serialized in `candidate-manifest.json`.

The payload also reports the provisional champion's 2023+ diagnostic. If its
multiclass-log-loss regret versus that period's best model exceeds 0.05, model
health is marked `weak_generalization` and live metadata is degraded. This
diagnostic can motivate a future pre-registered suite, but it does not replace
the current champion.

The comparison intentionally excludes deep learning. The default fourteen
models comprise three simple baselines; elastic-net, ridge and state-augmented
transition logistic regressions; a duration-aware time-varying-transition-
probability (TVTP) stay/switch/destination hurdle; shrinkage LDA; calibrated
linear SVM; a fold-local PCA/spline logistic approximation to additive smooth
effects; random forest; extremely randomized trees; histogram gradient
boosting; and shallow regularized XGBoost. The hurdle first estimates whether
the current state will be left, then restricts destination mass to adjacent
states. Duration, backward score changes, acceleration, and distances to the
train-fitted regime boundaries enter this structural candidate.

An optional Gaussian HMM is added only by the `full` profile as a fifteenth
latent-state diagnostic. It never supplies full-sample smoothed/Viterbi states
to the canonical label or a historical real-time row. LightGBM and CatBoost are
deferred because their incremental information is limited in this numeric,
roughly one-thousand-row setting; Gaussian NB, kNN and raw QDA are deferred
because correlated high-dimensional inputs can make their probabilities
unstable.

## Multi-horizon departure risk

The separate transition benchmark estimates the probability of at least one
departure from the origin state during `t+1 .. t+h` for `h = 1, 4, 13`. It is
not a terminal-state or destination forecast: a path that leaves and returns
before `t+h` is still a positive event. The first departed state is retained
only as an internal pseudo-destination for the hurdle decomposition.

The default binary comparison contains an empirical smoothed hazard, a
homogeneous Markov hazard, the duration-aware TVTP hurdle, and regularized
logistic regression. Shallow binary XGBoost is included when requested and its
runtime is available (standard/full pipeline profiles request it). Binary log
loss is the family-selection score; Brier score is a guardrail. Average
precision, precision, recall, false alarms per year, event/non-event counts,
and fallback counts are reported rather than collapsing the result into
accuracy alone.

Purging follows the forecast horizon, not a fixed one-week convention. For an
origin at `t`, every training event label must have `target_end < t`, so the
last `h` possible origins are excluded and both recorded `gap` and
`purged_origin_count` equal `h`. This prevents a 13-week event whose outcome is
still unfolding from entering a 13-week forecast fit.

Family selection, Platt-logit calibration, and alert-threshold selection use
only earlier pre-2023 selection OOS rows whose own `target_end` precedes the
current origin. The threshold maximizes balanced accuracy over the deterministic
0.05--0.95 grid and falls back to 0.5 when there are too few rows or only one
event class. The post-2023 family, calibrator, and threshold are therefore
locked from selection data. Estimator parameters can still be refit at each
origin on the expanding training sample, including post-2023 outcomes that
have become fully observed before that origin; this is causal adaptive fitting,
not post-period model-family selection. All post-2023 scored rows retain the
`retrospective_diagnostic` role.

The transition split also applies a horizon-specific boundary embargo.
Selection requires `target_end < 2023-01-01`, while a retrospective diagnostic
requires `origin >= 2023-01-01`. The intervening `h-1` origins whose event
windows cross the boundary belong to neither segment. Thus selection and the
external-period diagnostic never share weeks from the same event window.

For the dashboard weekly result, the authoritative one-week
`transition_probability` is calculated from the selected three-class
next-state forecast as `1 - P(S[t+1] = S[t])` and is copied to
`transition_risk.1w`. The separate binary one-week benchmark remains in the
research artifacts and leaderboard. The displayed 4/13-week values come from
their horizon-specific selected transition models. Neither quantity is the
probability of entering the middle `transition` state.

## Shadow explicit-duration nowcast

A fixed-parameter explicit-duration filter consumes the canonical three-state
evidence probabilities sequentially. Exit hazards depend on the current state
and elapsed duration; a one-week path cannot jump directly between `risk_on`
and `risk_off`, so such display changes are routed through `transition`. The
filter has no fitting method and each row uses only the previous posterior and
the current evidence row.

This output is deliberately marked `shadow_only` and
`canonical_target=false`. It is a sensitivity view for persistence and routing,
not another ground-truth label, not a replacement for the canonical current
state, and not an input that can rewrite next-state evaluation.

## Reproducible artifacts and frozen anchors

Alongside the main leaderboard, OOS forecasts, walk-forward splits, selection
diagnostics, and candidate manifest, a v4 run stages one complete generation.
Transition OOS, leaderboard, split, nested-selection, prospective and
candidate-forecast files are joined by the feature manifest, seven fixed
feature-family ablations, causal stacking weights, latest structural forecasts,
and the non-selectable survival-coherence shadow. `state-label-history.csv`
stores the full sequential three-state evidence surface, while
`weekly-state-forecasts.csv` mirrors every published current and next-week row.
Their exact columns, row counts and raw CSV SHA-256 values are linked from the
payload so the independent auditor can rebuild hysteresis states, soft
probabilities, any-departure targets and dashboard forecast parity.

Payload and artifacts are serialized in private sibling transaction
directories. Both previous active outputs are moved to recovery locations
before either new output is installed; a failed cutover rolls both sides back.
The prospective transition file contains the final `h` origins whose full
`t+1..t+h` outcomes are not yet observable, and these rows never enter scored
metrics.

The frozen v2 comparison anchor has result version
`weekly-regime-result-v2`, label version `market-causal-3state-v1`, model version
`weekly-nondl-walkforward-v2`, and champion `markov`. Its payload SHA-256 is
`50ab693b15f5100b1e39d98356c88455b76a4a2c4a4c335e5882509568c5fe98`; its
snapshot inventory SHA-256 is
`09603aca14244fc00ee56f0d75a45192fc29a77c8f1a47b9927aef32d4fcbf0f`.
The ignored local snapshot is an audit anchor, not training input and not a
claim that v3 improves on it.

The direct v4 comparison anchor is the ignored frozen v3 generation. Its
payload SHA-256 is
`de93c585117b2784750f586a4f84ad99964c63081b252ad7affd7a75bd797095` and its
inventory SHA-256 is
`8ef3778cc8c36faff0c80e2bf094f1f11bd6966ab3b7b2d6edb84ba292aff6b9`.
The auditor hashes the materialized snapshot members and
`config/structural_v4.json`, rather than trusting copied metadata alone.

## Interpretation

Regimes are latent summaries, not objective truths or investment advice. The
dashboard therefore shows probability vectors, entropy/confidence,
multi-horizon departure risk, factor scores, model identity, data cutoff and
source health rather than only a hard label.

## Research lineage

The design follows the discrete-state regime tradition of Hamilton (1989), the
time-varying transition formulation of Filardo (1994), explicit-duration
modeling summarized by Chiappa, the real-time-vintage warning of Croushore and
Stark (2001), real-time turning-point evaluation in Chauvet and Piger (2008),
and the comparative out-of-sample ML discipline in Gu, Kelly, and Xiu (2020).
Chicago Fed's official NFCI documentation supplies the interpretation of the
four contribution series. See [references](references.md) for the implementation
choices derived from each source.
