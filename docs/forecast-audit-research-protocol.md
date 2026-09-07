# Forecast audit research protocol v1

Frozen 2026-09-07, before running these candidates. Earlier 2023+ audit diagnostics
are already known. This is retrospective research, not a new holdout or prospective
issuance record. The official label and operating champion remain unchanged.

## Fixed candidate family

1. `boundary_asymmetric_ewma`: one zero-mean asymmetric EWMA volatility challenger,
   alpha=2/(13+1), negative/positive squared-shock weights 1.5/0.5, floor=0.003
   weekly log-return standard deviation. Reuse the boundary model's frozen label,
   trailing 520 standardized residuals, clipping at +/-8, and 1% contamination.
   Historical shocks use the previous week's volatility. Scalar temperature uses
   only completed earlier raw OOS predictions under the existing fixed rule.
2. `directional_duration_hazard`: two regularized cause-specific binary logits
   (C=0.1, no class weights), comparing worsening versus stay and recovery versus
   stay. Exclude the opposite event when fitting each conditional logit. Convert
   their odds versus stay into one competing-risk distribution. Route each
   direction with add-one historical destination counts, retaining direct jumps.
   Inputs: current-state indicators, frozen lower/upper boundary distances,
   log(1+spell age), positive and negative latest standardized shocks. Fit the
   scaler/imputer and classifier on at most 520 past observations, with training
   target strictly before the origin. No fitted probability calibration.
3. Project the second candidate through one deterministic multistate recursion
   for 1/4/13 weeks. Update state and duration on every path, resetting age to one
   after transitions. Hold observed boundary distances and shock covariates fixed
   at origin; this is a conditional scenario, not a simulated future price path.
   A matched smoothed Markov path is the baseline. No separate horizon fitting.

No hyperparameter search, feature selection, discretionary threshold tuning, automatic champion
promotion, or label changes are part of this protocol. Expanding prequential
evidence may train the fixed estimators; diagnostic results cannot change the
recipe. Any recipe change requires a new protocol version and separate output.

## Samples and outcomes

- Use the exact complete baseline one-week origin set, with no silent intersection.
- Before 2023-01-01: selection; from that date: retrospective diagnostic. Purge
  origins whose target reaches across the selection boundary; do this independently
  for each horizon. Never report the diagnostic as a fresh holdout.
- Proper scores: multiclass Log loss/Brier for next state, endpoint and first
  departure; binary Log loss/Brier for worsening, recovery and any risk-off entry.
- `any_risk_off_entry`: any future **non-risk-off to risk-off transition**. For a
  risk-off origin, staying risk-off is not a new entry; leaving and returning is.
  Also expose `any_risk_off_occupancy` to answer future occupancy separately.
- Worsening/recovery capture and false alarms use next-state argmax. No tuned
  decision threshold. Display event counts and exposure, not just rates.
- Episode validation: distinct contiguous risk-off spells, exact-entry alerts,
  maximum four-week advance alerts, right-censoring and eligibility counts. Count
  each spell once. The four-week alert is any-entry probability >=0.5, fixed here.
- Worsening budget policies: calibrate the lowest threshold satisfying at most
  four false alerts per 52.1775 observed weeks. One threshold is frozen using only
  pre-2023 selection; the second uses at most 156 strictly completed past OOS rows,
  at least 52, under a fixed prequential update rule. No diagnostic-specific
  policy choice. Report actual future budget exceedance, not a promised cap.
- Persistence: whether a worsening/recovery remains on that side of the origin
  state for at least two future observations; first-week reversals separately.
- Label-independent outcomes: future 4/13-week annualized realized volatility,
  cumulative return, minimum cumulative price return from origin, and -5% downside
  event. These validate economic associations of the risk score, not investment
  performance or calibrated probabilities of price losses. No transaction claims.
- Report full selection and diagnostic, latest 52 mature origins, and current-state
  strata. Paired 13-week block uncertainty is descriptive and does not undo prior
  research selection. No winner is selected from these outputs.

## Parent/UI JSON contract

Producer: `scripts/run_forecast_audit_research.py`; full evidence: `audit-research.json`.
The following full audit object is adapted by `forecast_research_extension` to
`forecast_research.json`, schema `regime-forecast-research/1`. Parent attaches that
adapter as the optional `research.forecast_research` extension.
The producer does not edit payload, pipeline, publication or operating ledgers.

```json
{
  "schema_version": "regime-forecast-audit-research/1",
  "status": "research_only",
  "data_as_of": "ISO timestamp",
  "generated_at": "actual UTC timestamp at completed artifact construction",
  "protocol": {"version": "forecast-audit-v1", "sha256": "..."},
  "evidence_track": "reconstructed_market",
  "automatic_promotion": false,
  "models": [{"id": "directional_duration_hazard", "label": "..."}],
  "latest": [{"model": "...", "origin_date": "...", "current_state": "risk_on",
    "next_state": {"risk_on": 0.6, "transition": 0.3, "risk_off": 0.1},
    "paths": [{"horizon_weeks": 4,
      "first_departure": {"no_departure": 0.3, "risk_on": 0.0, "transition": 0.5, "risk_off": 0.2},
      "endpoint": {"risk_on": 0.4, "transition": 0.3, "risk_off": 0.3},
      "any_risk_off_entry": 0.4, "any_risk_off_occupancy": 0.4}]}],
  "weekly": [{"origin_date": "...", "models": []}],
  "metrics": [],
  "paired_comparisons": [],
  "episodes": [],
  "persistence": [],
  "economic_validation": [],
  "residual_diagnostics": {},
  "source_hashes": {},
  "limitations": []
}
```

The UI adapter preserves the audit evidence, adding `selected_model` as a fixed
research display default (no performance selection) and
`models:[{id,label,history:[row],latest:row}]`. Each row contains `origin_date`,
`current_state`, `next_state`, and `horizons:{1w,4w,13w}`. Horizon values use the
native path fields above plus `target_date` and `horizon_weeks`. One-week-only
models use `horizons:{}`; models without a current forecast omit `latest`.
`validate_forecast_research_extension` enforces the matching Python/UI semantics.
`generated_at` is set by the running producer and is never inferred from data_as_of.

All probabilities are unit fractions; missing statistics are null, never NaN.
`latest` has no realized outcome. `weekly` includes dated predictions and path
probabilities for selectable origins. `metrics` contains model, split, horizon,
target, sample count, period dates and scores. Horizon comparisons require the
same model-origin sets. Full per-origin predictions and episode details are also
saved as CSV companions for independent re-evaluation. A protocol content hash
and input hashes bind every run to this recipe and the saved derived inputs.
