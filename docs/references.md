# 연구 참고문헌과 설계 반영

## 구조 v5 연구 및 공식 데이터

아래 자료의 추정 대상과 시간축을 v5 계약에 맞춰 사용한다. 논문의 이름을 모델명으로
차용하지 않고, 실제 구현한 target·검열·검증 규칙을 결과에 기록한다.

| 주제 | 1차 출처 | v5 설계 반영 |
|---|---|---|
| 상태 전환 | Hamilton (1989), [Markov switching](https://doi.org/10.2307/1912559); Filardo (1994), [time-varying transition probabilities](https://doi.org/10.1080/07350015.1994.10524545) | empirical·Markov first-passage와 PIT 공변량 기반 정규화 multinomial·shallow XGBoost를 같은 first-departure target과 purged origin에서 비교한다. |
| 우측 검열과 생존 | Kaplan & Meier (1958), [product-limit estimator](https://doi.org/10.1080/01621459.1958.10501452); Andersen et al. (1993), [*Statistical Models Based on Counting Processes*](https://link.springer.com/book/10.1007/978-1-4612-4348-9) | 현재 spell을 완료 사건으로 세지 않고 상태별 생존함수와 경과기간 조건부 생존율을 계산한다. |
| 제한 평균 잔여기간 | Royston & Parmar (2013), [RMST](https://doi.org/10.1186/1471-2288-13-152) | 생존곡선 면적 개념을 주별 이산시간에 적용해 52주 제한 조건부 평균 잔여기간을 산출한다. |
| 의존 자료 bootstrap | Künsch (1989), [block bootstrap](https://doi.org/10.1214/aos/1176347265) | 주별 예측 손실에는 13주 block을, spell·조건부 통계에는 episode 경계를 보존하는 재표본을 사용한다. |
| H.10 공식 자료 | Federal Reserve Board [H.10 release](https://www.federalreserve.gov/releases/h10/), [Broad·AFE·EME 지수 설명](https://www.federalreserve.gov/releases/h10/summary/), [technical Q&A](https://www.federalreserve.gov/releases/h10/h10_technical_qa.htm), [달러 지수 개편 방법론](https://www.federalreserve.gov/econres/notes/feds-notes/revisions-to-the-federal-reserve-dollar-indexes-20190115.html), [공식 XML ZIP](https://www.federalreserve.gov/releases/h10/data/FRB_h10_xml.zip) | 공식 월요일 16:15 ET 일정과 observation week를 기록하되 실제 `feature_available_at`은 수집 `first_seen`으로 둔다. Broad·AFE·EME와 고정 9개 통화쌍을 동일한 USD 방향으로 정규화한다. |
| 달러 공통요인과 단면 | Lustig, Roussanov & Verdelhan (2011), [common currency factors](https://doi.org/10.1093/rfs/hhr068); Avdjiev et al. (2018), [BIS Working Paper 695](https://www.bis.org/publ/work695.htm) | 달러 지수의 공통 움직임과 bilateral median·breadth·MAD·AFE–EME divergence를 분리한다. 이는 FX를 equity label로 쓰는 근거가 아니라 shadow predictor/context를 시험하는 근거다. |

H.10 현재 XML은 과거 공개 빈티지의 시각별 archive가 아니다. 따라서 초기 전체 이력의
가용시점은 최초 수집 `first_seen`으로 두고, 이후 발견한 revision만 발견 시점부터
사용한다. 지수 가중치 개편으로 과거 지수값이 바뀔 수 있다는 Board 설명도 같은
prospective revision 규칙으로 처리한다.

Board 공시 시각은 월요일 16:15 ET이며 연방 공휴일이면 다음 영업일이다. 현재 adapter는
Board release-page XML을 직접 사용한다. Board 페이지가 안내하는 FRED 전환은 단순 URL
교체로 처리하지 않는다. FRED는 [별도 이용·귀속 조건](https://fred.stlouisfed.org/legal/)을
적용하므로 provider, PIT 시각과 attribution 계약을 다시 확인한 뒤 전환한다.

## 구조 v4 참고문헌

- James D. Hamilton (1989), *A New Approach to the Economic Analysis of
  Nonstationary Time Series and the Business Cycle*, Econometrica 57(2),
  357–384, DOI [10.2307/1912559](https://doi.org/10.2307/1912559). This motivates
  Markov/HMM challengers and persistent discrete states, but the project does
  not use full-sample smoothed states as real-time labels.
- Dean Croushore and Tom Stark (2001), *A Real-Time Data Set for
  Macroeconomists*, Journal of Econometrics 105(1), 111–130,
  [Federal Reserve Bank of Philadelphia overview](https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/real-time-data-set-for-macroeconomists).
  Their central revision problem motivates ALFRED event vintages, `available_at`
  cutoffs, and the ban on using today's revised macro history in older folds.
- Marcelle Chauvet and Jeremy Piger (2008), *A Comparison of the Real-Time
  Performance of Business Cycle Dating Methods*, Journal of Business & Economic
  Statistics 26(1), DOI
  [10.1198/073500107000000296](https://doi.org/10.1198/073500107000000296).
  This supports probability-first real-time evaluation rather than treating a
  final retrospective chronology as immediately observable truth.
- Shihao Gu, Bryan Kelly, and Dacheng Xiu (2020), *Empirical Asset Pricing via
  Machine Learning*, Review of Financial Studies 33(5), 2223–2273, DOI
  [10.1093/rfs/hhaa009](https://doi.org/10.1093/rfs/hhaa009). Its comparative
  emphasis on regularized linear and nonlinear models motivates the common
  feature/split benchmark; this project adds a strict chronological selection /
  retrospective-diagnostic boundary because the sample is much smaller and the
  later period has already been inspected.
- Andrew J. Filardo (1994), *Business-Cycle Phases and Their Transitional
  Dynamics*, Journal of Business & Economic Statistics 12(3), DOI
  [10.1080/07350015.1994.10524545](https://doi.org/10.1080/07350015.1994.10524545).
  This motivates both the state-augmented transition logistic challenger and
  time-varying transition probabilities: departure risk may depend on the
  current state and point-in-time covariates instead of a constant matrix.
- Silvia Chiappa, *Explicit-Duration Markov Switching Models*, Foundations and
  Trends in Machine Learning, DOI
  [10.1561/2200000054](https://doi.org/10.1561/2200000054). This motivates
  retaining causal state duration explicitly rather than relying on the
  geometric-duration implication of an ordinary homogeneous Markov chain. The
  implementation uses that idea in a stay/switch/destination hurdle and a
  separate fixed-parameter shadow filter; it does not claim to reproduce the
  paper's full model.
- Jerome H. Friedman (1989), *Regularized Discriminant Analysis*, Journal of
  the American Statistical Association 84(405), DOI
  [10.1080/01621459.1989.10478752](https://doi.org/10.1080/01621459.1989.10478752).
  Shrinkage LDA is included as a low-variance generative benchmark for a weekly
  sample with many correlated predictors; raw QDA is excluded because its
  class-specific covariance estimates are too unstable here.
- Trevor Hastie and Robert Tibshirani (1986), *Generalized Additive Models*,
  Statistical Science 1(3), DOI
  [10.1214/ss/1177013604](https://doi.org/10.1214/ss/1177013604). A fold-local
  dimension-reduction and spline pipeline tests smooth nonlinear effects
  without expanding all input columns into an unmanageably large basis.
- Tianqi Chen and Carlos Guestrin (2016), *XGBoost: A Scalable Tree Boosting
  System*, KDD '16, DOI
  [10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785). A shallow,
  strongly regularized CPU configuration is included as the modern nonlinear
  interaction challenger; no broad hyperparameter search is performed.
- Michael Puglia and Adam Tucker (2020), *Machine Learning, the Treasury Yield
  Curve and Recession Forecasting*, Federal Reserve FEDS 2020-038,
  [official paper page](https://www.federalreserve.gov/econres/feds/machine-learning-the-treasury-yield-curve-and-recession-forecasting.htm).
  Their documented ranking reversals between ordinary k-fold and time-aware
  validation reinforce the purged expanding walk-forward contract.
- Rehim Kilic (2025), *Linear and nonlinear econometric models against machine
  learning models: realized volatility prediction*, Federal Reserve FEDS
  2025-061,
  [official paper page](https://www.federalreserve.gov/econres/feds/linear-and-nonlinear-econometric-models-against-machine-learning-models.htm).
  The finding that transparent regime-switching models can outperform more
  complex ML methods is why Markov remains a first-class probabilistic baseline
  rather than a weak strawman.
- [Chicago Fed National Financial Conditions Index: About the NFCI](https://www.chicagofed.org/research/data/nfci/about).
  The official description defines the broad NFCI and its risk, credit,
  financial-leverage, and nonfinancial-leverage contributions. Those five
  weekly ALFRED series enter as point-in-time predictors; they do not define the
  market-only regime label.
- [FRED series observations API](https://fred.stlouisfed.org/docs/api/fred/series_observations.html).
  The provider documents `output_type=1` real-time periods, `output_type=2/3`
  vintage views, and file-type-specific vintage limits. The implementation uses
  a type-1 full base, type-3 new/revised deltas, and a local weekly as-of join.
- [FRED series vintage dates API](https://fred.stlouisfed.org/docs/api/fred/series_vintagedates.html).
  Weekly delta collection resolves its calendar overlap window through this
  endpoint and sends only returned series vintages to `output_type=3`. This
  filtering is a conservative response to a 2026-08-12 live JSON check where an
  UNRATE request containing non-discovered dates received HTTP 400, rather than
  a claim that the documentation guarantees identical rejection behavior for
  every series or future API version. A narrow 5xx is recovered only through a
  successful bounded wider discovery; it is never treated directly as proof of
  an empty vintage window.
- [FRED real-time periods](https://fred.stlouisfed.org/docs/api/fred/realtime_period.html).
  This defines the real-time start/end interval used for the initial history;
  the weekly delta chain retains its own vintage dates and availability times.

## Structural v4 research

- Andrew Ang and Allan Timmermann (2012), *Regime Changes and Financial
  Markets*, Annual Review of Financial Economics 4, DOI
  [10.1146/annurev-financial-110311-101808](https://doi.org/10.1146/annurev-financial-110311-101808).
  The review motivates treating persistence, changing correlations and
  state-dependent distributions as structural objects rather than adding more
  unrestricted classifiers.
- Adrian E. Raftery, Miroslav Kárný and Pavel Ettler (2010), *Online Prediction
  Under Model Uncertainty via Dynamic Model Averaging*, Technometrics 52(1),
  DOI [10.1198/TECH.2009.08104](https://doi.org/10.1198/TECH.2009.08104).
  Their Sections 3.1--3.2 combine per-model dynamic linear models with a hidden
  model indicator and predictive-likelihood forgetting. The project instead
  discounts completed OOS log losses and applies a softmax. It is therefore a
  causal predictive-score pool motivated by DMA, not a DMA replication, and it
  may not choose weights from the 2023+ diagnostic period.
- Francis X. Diebold and Canlin Li (2006), *Forecasting the Term Structure of
  Government Bond Yields*, Journal of Econometrics 130(2), DOI
  [10.1016/j.jeconom.2005.03.005](https://doi.org/10.1016/j.jeconom.2005.03.005).
  A fixed-loading Nelson–Siegel projection compresses eight point-in-time H.15
  maturities into level, slope and curvature instead of searching many spreads.
- Eric C. Engstrom and Steven A. Sharpe (2018), *The Near-Term Forward Yield
  Spread as a Leading Indicator*, Federal Reserve FEDS 2018-055,
  [official paper](https://www.federalreserve.gov/econres/feds/files/2018055r1pap.pdf),
  and Peter Johansson and Andrew Meldrum (2018), *Predicting Recession
  Probabilities Using the Slope of the Yield Curve*,
  [FEDS Note](https://www.federalreserve.gov/econres/notes/feds-notes/predicting-recession-probabilities-using-the-slope-of-the-yield-curve-20180301.html).
  These results motivate testing richer front-end and whole-curve information;
  they are not claims that any single spread must dominate in this market-state
  target.
- Domenico Giannone, Lucrezia Reichlin and David Small (2008), *Nowcasting:
  The Real-Time Informational Content of Macroeconomic Data*, Journal of
  Monetary Economics 55(4), DOI
  [10.1016/j.jmoneco.2008.05.010](https://doi.org/10.1016/j.jmoneco.2008.05.010).
  Their Sections 2 and 4 define news through a model-conditioned change in a
  nowcast as the information set expands. The project does not estimate that
  object: it computes a robust, prior-event-only release change. The paper
  supports the ragged-edge timing discipline, not an `economic surprise` claim.
- Ryan Prescott Adams and David J. C. MacKay (2007), *Bayesian Online
  Changepoint Detection*, [arXiv:0710.3742](https://arxiv.org/abs/0710.3742).
  A future boundary detector may use its causal run-length filter, but such a
  signal remains shadow evidence until it independently clears the same gate.
- [Federal Reserve H.8 release description](https://www.federalreserve.gov/releases/h8/about.htm)
  and the FRED pages for
  [bank credit](https://fred.stlouisfed.org/series/TOTBKCR),
  [C&I loans](https://fred.stlouisfed.org/series/TOTCI),
  [deposits](https://fred.stlouisfed.org/series/DPSACBW027SBOG), and
  [borrowings](https://fred.stlouisfed.org/series/H8B3094NCBA).
  These official weekly estimates add lending and funding information. They are
  reduced-form balance-sheet states, not identified credit-supply shocks.
  Because the 16:00 project cutoff precedes the usual 16:15 Friday publication,
  a value from that release is deferred to the next completed week.
- [Chicago Fed NFCI documentation](https://www.chicagofed.org/research/data/nfci/current-data)
  defines ANFCI as the financial-conditions component adjusted for prevailing
  economic activity and inflation and documents its weekly revisions.

These references guide methodology; they do not imply that the three operational
labels are an objective or uniquely correct market taxonomy. The label is a
transparent, frozen market-behaviour definition designed for forecast comparison.

### Evidence-selection audit

The sources below were retained after reading their method, measurement and
limitations sections. Search position and citation counts were not selection
criteria. A source can support a block only when its estimand and mechanism are
traceable, its release/revision timing can be reconciled with the point-in-time
cutoff, its sample and target are stated, and a frozen common-origin ablation can
falsify the proposed use. Reviews were used for orientation; the decisions below
rest on primary papers and official measurement documentation.

| Block | Primary logic actually checked | Scope, contradiction or failure condition | Decision and replication status |
| --- | --- | --- | --- |
| Treasury curve | Diebold and Li, [Section 2 and equations around the Nelson--Siegel loadings](https://www.nber.org/papers/w10048.pdf), fix `lambda=0.0609` so curvature peaks at 30 months and estimate each cross-section by OLS. Federal Reserve work compares [near-term forwards](https://www.federalreserve.gov/econres/feds/files/2018055r1pap.pdf), [three yield PCs](https://www.federalreserve.gov/econres/notes/feds-notes/predicting-recession-probabilities-using-the-slope-of-the-yield-curve-20180301.html), and [sample/horizon-dependent spreads](https://www.federalreserve.gov/econres/notes/feds-notes/there-is-no-single-best-predictor-of-recessions-20190521.html). | Diebold--Li use monthly Fama--Bliss zero yields and forecast yields; their one-month forecasts do not beat a random walk although one-year forecasts improve. The Fed studies target monthly/quarterly recessions, and Johansson--Meldrum report an early-2008 miss from information not spanned by yields. Weekly H.15 constant-maturity inputs and a next-week equity-regime target are outside all of those evidence scopes. | **Retain.** The fixed loadings, `lambda` and row OLS are a close formula replication. The data, frequency and predictive use are a motivated adaptation that must win the frozen ablation; no single-spread or recession-prediction claim is imported. |
| ALFRED release change | Croushore--Stark's [real-time data paper](https://www.philadelphiafed.org/-/media/FRBP/Assets/working-papers/2000/wp00-6.pdf?sc_lang=en) shows why current revised histories can change real-time conclusions. Official [ALFRED](https://fred.stlouisfed.org/docs/api/fred/alfred.html) stores original releases and revisions. Giannone--Reichlin--Small [Sections 2 and 4](https://www.ecb.europa.eu/pub/pdf/scpwps/ecbwp633.pdf) define news as a release-induced change in a model nowcast relative to the model's pre-release prediction. | Their news depends on a large ragged-edge factor model, release order and a GDP/inflation nowcast; the paper explicitly studies news and uncertainty rather than an OOS forecast contest. A raw revision or release delta is not the same estimand. | **Retain with corrected interpretation.** PIT event timing follows the real-time-data logic. The configured `release_innovation` statistic is a robust univariate release change, not a replication of model news, a survey surprise, or an identified macro shock. |
| H.8 bank credit/funding | The Federal Reserve [H.8 description](https://www.federalreserve.gov/releases/h8/About.HTM) documents an estimated Wednesday aggregate from a voluntary weekly panel plus Call Reports, normally published Friday at 16:15. Its [technical Q&A](https://www.federalreserve.gov/releases/h8/h8_technical_qa.html) documents quarterly reweighting, panel-shift adjustments and historical revisions. Dalal, Dias and Uysal [Sections 2.1--2.2](https://www.federalreserve.gov/econres/feds/files/2025055pap.pdf) require an external monetary-policy instrument, sign restrictions and lending-standards information to separate policy, credit-demand and credit-supply shocks. | Loan growth and funding ratios mix supply, demand, monetary policy, mergers, benchmarking and seasonal adjustment. A fall in credit growth alone cannot identify bank tightening. The normal release is after this project's Friday cutoff. | **Retain as reduced-form state only.** Use ALFRED vintages and defer the Friday release to the next week. Exclude causal `credit supply` or `lending standards` language unless a separately identified design is added. |
| ANFCI | The Chicago Fed's [methodology](https://www.chicagofed.org/publications/chicago-fed-letter/2017/386) constructs a mixed-frequency dynamic factor and obtains ANFCI after removing variation associated with activity and inflation. Its [current-data documentation](https://www.chicagofed.org/research/data/nfci/current-data) records weekly publication and revisions. | ANFCI is revised and already contains equity, credit and banking indicators, so it overlaps other inputs. Residualization is not causal identification and historical factor weights can change. | **Retain as a broad PIT financial-conditions feature**, not an independent structural shock. The `anfci__*` namespace and pre-2023 `legacy_plus_financial_conditions` common-origin ablation now isolate its incremental predictive contribution; even a pass would not identify a causal shock. |
| Sector breadth | Moskowitz and Grinblatt's [industry-momentum paper](https://onlinelibrary.wiley.com/doi/pdf/10.1111/0022-1082.00146) studies 20 industry portfolios and finds effects at one and 6--12 months, but its conclusion does not identify why they exist. Duffee's [industry/firm dispersion study](https://www.frbsf.org/wp-content/uploads/wp00-18bk.pdf) finds short-lived asymmetry and reports that non-market volatility does not forecast next-day market returns. | Neither source tests 11 sector ETFs, weekly breadth/HHI, or three-state probabilities. Related dispersion evidence does not establish a regime mechanism or short-horizon market-return predictability. | **Retain measurement separation; treat predictability as unproven.** Separate sector, broad/size-style and cross-asset denominators to avoid compositional contamination. Breadth, acceleration and HHI remain motivated features whose value is decided only by ablation. |
| H.10 dollar indexes and bilateral FX | The Board's [H.10 description](https://www.federalreserve.gov/releases/h10/about.htm) defines the official dollar indexes and bilateral exchange rates. Obstfeld and Zhou's [Global Dollar Cycle](https://www.nber.org/papers/w31004) links broad dollar movements to global financial conditions and risk appetite, while Hofmann and Park's [BIS study](https://www.bis.org/publ/qtrpdf/r_qt2012b.htm) finds the broad index more informative for EME risk than isolated bilateral rates. Bruno, Shim and Shin's [BIS Working Paper 1000](https://www.bis.org/publ/work1000.htm) identifies broad-dollar exposure as a priced risk factor in EME stock returns and finds the broad dollar more informative than bilateral rates in its EME regressions. Federal Reserve evidence also connects dollar appreciation to U.S. leveraged-loan risk compensation through a [dollar transmission channel](https://www.federalreserve.gov/econres/feds/the-dollar-channel-of-monetary-policy-transmission.htm). | These studies do not forecast this project's next-week U.S. equity-regime label. In particular, the EME stock result cannot be extrapolated into U.S. weekly regime predictability. Bilateral exchange rates mix relative monetary policy, local growth and country shocks; their direction and units also differ. The official archive is not a complete revision ledger, so dated correction-equivalent pages cannot identify every affected historical value. | **Evaluate as a separate PIT shadow block.** Use the literature only to support a broad-first design with a fixed nine-currency bilateral shadow panel. Normalize every series so positive means USD appreciation, compare only common eligible origins, and quarantine 27 model origins after every correction-equivalent event. No FX variant can enter the champion without a separate promotion decision. |
| XGBoost departure plus destination | Filardo's [equations (1)--(5) and Sections 2--4](https://doi.org/10.1080/07350015.1994.10524545) jointly estimate the state-dependent observation likelihood and logistic, covariate-dependent transition probabilities. The [readable primary-text copy](https://studylib.net/doc/28410229/tvtp-ms-var) also stresses that significant transition coefficients alone are insufficient; inferred states and turning-point content must be checked. | Filardo uses a two-state monthly industrial-production model and revised data. The project estimates a binary one-week XGBoost departure hazard, then allocates its mass according to the two non-current probabilities of a separately fitted multiclass XGBoost. The destination component is not trained conditionally on departure. | **Retain as a candidate splice, not TVTP replication.** It must be labeled `hazard_destination` rather than a joint Filardo model, compared on identical OOS origins, and rejected if probability or calibration gains do not survive the frozen gate. |
| Discounted causal ensemble | Raftery, Kárný and Ettler [Sections 3.1--3.2](https://sites.stat.washington.edu/raftery/Research/PDF/Karny2010.pdf) use dynamic linear-model parameter states, a hidden model indicator, predictive likelihoods and forgetting. Gneiting and Raftery [Sections 2--4](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf) show why the logarithmic score is strictly proper for probability forecasts. | The project has neither DMA parameter-state evolution nor a model-indicator posterior. It discounts realized, completed-OOS log losses with a fixed 52-week half-life and softmaxes them. Correlated experts, two XGBoost-derived experts, short score histories and abrupt structural breaks can still make weights unstable or over-concentrated. | **Retain as a discounted predictive-score pool.** Completed-target timing and log score are justified; the pooling rule and half-life are a preregistered adaptation, not DMA. Every expert is now scored on the identical intersection of completed origins where all three experts are nonfallback, and equal weights remain in force until that common history reaches 26 rows. |
| Causal multiscale score pool | Pesaran and Pick's [forecast combination across estimation windows](https://repub.eur.nl/pub/26871) shows why averaging forecasts formed with different memory lengths can reduce window-choice risk under structural breaks. McAdam and Warne's [real-time density-combination study](https://www.ecb.europa.eu/pub/pdf/scpwps/ecb.wp2378~a9b3837817.en.pdf) evaluates log-score pools with real-time information lags and reports stronger results for combinations whose weights vary moderately and remain positive across models. Ranjan and Gneiting's [probability-pooling result](https://rss.onlinelibrary.wiley.com/doi/pdf/10.1111/j.1467-9868.2009.00726.x) shows that linear pooling does not by itself guarantee calibration. | The cited window result concerns point forecasts from the same model, and the ECB application uses macro density forecasts rather than three-state weekly equity probabilities. None establishes 26, 52 and 104 weeks as optimal here. Averaging three pools built from correlated experts may smooth score noise but can also dilute a genuinely faster adaptation. | **Admit one frozen v5-only candidate.** Form three completed-OOS discounted-log-score pools at 26, 52 and 104 week half-lives, average their probabilities with exact one-third weights, and leave calibration and predictive value to the unchanged common-origin log-loss, Brier, zero-fallback and Holm gate. |
| Joint survival hazard | Chiappa's [Chapter 2, equation for regime duration](https://arxiv.org/pdf/1909.05800) shows that a constant Markov stay probability implies the geometric duration law `pi_ii^(d-1) * (1-pi_ii)`; explicit-duration models require duration variables or a non-geometric duration distribution. Singer and Willett's [discrete-time survival formulation](https://doi.org/10.3102/10769986018002155) defines the period hazard conditional on remaining at risk and recovers a survival function from those hazards. Thus cumulative departure risk is `1 - product(1-h_j)` only when each `h_j` is conditional on survival through prior weeks. | The current forecast path repeats one scalar one-week hazard for 13 steps. It is therefore geometric: advancing a displayed duration without recomputing future conditional hazards does not make the forecast duration-aware. Covariates are also not projected along the future path. | **Shadow/coherence benchmark only.** The product identity and monotonic 1/4/13-week risks are exact; an explicit-duration or shared-hazard model is not. Horizon-specific purged 4/13-week models remain operational unless genuinely duration-aware hazards are estimated and pass the same gate. |
