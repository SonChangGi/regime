# Structural v4 preregistration

`config/structural_v4.json` is the machine-readable specification frozen before
the first v4 live collection or fit.  The v3 payload and every supporting
artifact are preserved under `artifacts/baselines/v3-20260813/`; its
`SHA256SUMS` file is the comparison inventory.

## Fixed question

Can information that is structurally absent from v3 improve the frozen
three-state, next-week probability forecast?  The label remains
`market-causal-3state-v1`; changing the target to make accuracy easier is not a
valid improvement.

The candidate blocks are:

1. a point-in-time Treasury-curve factor and weekly bank-credit/funding block;
2. complete 11-sector breadth separated from broad and cross-asset ETFs;
3. real-time macro release innovations derived only from prior releases;
4. a joint departure-hazard and conditional-destination probability;
5. a causal probability ensemble updated only after an OOS target is known;
6. a shared weekly survival hazard whose 1/4/13-week risks are monotone.

## Time and promotion contract

- The minimum training history is 520 weeks.
- Model and feature-family decisions use only OOS targets ending before
  2023-01-01.
- Results from 2023 onward are development diagnostics. They cannot change a
  family, feature block, ensemble half-life, threshold, or gate.
- Every forecast, ensemble weight, calibration row and release innovation must
  use information available before its origin. Horizon-specific purge and the
  2023 boundary embargo remain mandatory.
- The primary metric is multiclass log loss on common origins. Promotion still
  requires at least 0.05 absolute improvement, no more than 0.01 Brier
  degradation, zero fallback rows, and the existing 13-week block-bootstrap
  Holm gate.
- If no structural candidate passes every rule, v3 remains the operational
  champion. Diagnostic value alone is not promotion evidence.

## Source and rights boundary

New market data stay inside the existing Alpha Vantage free 25-request plan;
23 symbols leave two requests of reserve. New macro/rates/credit series use the
existing confirmed ALFRED research and derived-output scope. Raw provider
history is never added to a public package. Unknown or post-cutoff release
times become usable in the following completed week.

All standard-free Alpha credentials, if more than one exists, share the same
project-wide rolling 24-hour cap; credential rotation is not a quota expansion
mechanism. Only a provider-verified open-source/educational entitlement or a
premium plan with explicit rate metadata may define a separate limit, and
neither changes the raw-data redistribution boundary.

The complete requested-symbol batch is atomically reserved before transport;
spent credits move to their actual attempt timestamps, while unused credits
remain charged from reservation. Standard-free calls use one attempt. During
collection, the standard-free cap is the literal integer `25`; another type or
value fails before client construction or event reservation. During
the one-time ledger upgrade, every legacy collector must be stopped before the
first new reservation because independently running old and new binaries cannot
mutually lock two different ledgers.

## Research basis

- Hamilton and Filardo motivate persistent states and covariate-dependent
  transitions.
- Chiappa motivates explicit duration and a shared exit-time structure.
- Raftery, Kárný and Ettler motivate causal model averaging when the best model
  changes over time.
- Diebold and Li motivate parsimonious yield-curve level, slope and curvature.
- Giannone, Reichlin and Small motivate a ragged-edge real-time factor/news
  treatment rather than revised macro levels.
- Engstrom and Sharpe, and Johansson and Meldrum, motivate richer front-end and
  whole-curve information instead of relying only on the 10y–2y spread.
- Federal Reserve H.8 and Chicago Fed NFCI documentation motivate the bank
  credit, funding and adjusted financial-conditions blocks.

See `docs/references.md` for primary links and the exact implementation scope.

## Evidence audit and final freeze (2026-08-13)

The evidence audit occurred before the first v4 live collection or fit. It used
no outcome comparison or post-2023 result. The final machine-readable freeze
corrects two auditability defects: ANFCI has its own `financial_conditions`
feature family and selection-only ablation, and ensemble losses use the identical
completed-origin intersection on which all three experts are nonfallback. The
target, raw feature calculations, model hyperparameters, ensemble half-life and
promotion gate are unchanged. The final `config/structural_v4.json` SHA-256 is
`2f53ada564efca770261f16ce6eb16ec9c9782bde014de7a7d85b7b24dbe407b`.

Sources were selected from direct reading of primary methods and official
measurement documentation, not from search rank or citation count.

- The fixed Nelson--Siegel loadings, `lambda=0.0609`, and cross-sectional OLS
  closely reproduce Diebold--Li's factor extraction. Weekly H.15 inputs and an
  equity-regime target are an adaptation.
- The configured `release_innovation` block is a prior-event-only robust
  release-change statistic. It is not Giannone--Reichlin--Small model news, a
  survey surprise, or an identified macro shock.
- H.8 ratios, ANFCI and sector breadth are reduced-form candidate measurements.
  They carry no causal credit-supply, financial-shock or sector-leadership
  claim. ANFCI's incremental forecast value is isolated by
  `legacy_plus_financial_conditions` on pre-2023 common origins.
- `xgb_hazard_destination` combines a separately fitted departure hazard with
  non-current multiclass probabilities; it is not Filardo's jointly estimated
  TVTP likelihood.
- The dynamic ensemble is a discounted completed-OOS log-score pool, not
  Raftery--Kárný--Ettler DMA. All three experts are scored on the same
  all-nonfallback completed-origin intersection.
- `1 - product(1-h_j)` is an exact conditional-survival identity. The current
  13-step path repeats one scalar hazard and is consequently geometric, not an
  explicit-duration model. It remains `shadow_only`; the purged horizon-specific
  4/13-week forecasts remain operational.

The evidence-quality rubric, scope limits and failure conditions are recorded in
`docs/references.md`. These interpretation boundaries cannot be used to reopen
the final freeze or tune against post-2023 results.
