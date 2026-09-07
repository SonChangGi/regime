# Optional forecast research UI contracts

The dashboard accepts these blocks under `payload.research`. They never replace
`weekly.next_week`, `weekly.transition_risk` or the official champion. The parent
builds a separate local preview bundle; reviewed publication files stay intact.

## Paths and new candidates

`forecast_research.schema_version = "regime-forecast-research/1"`.
Producer: `analysis.forecast_audit_research.forecast_research_extension`.

Required display fields: `data_as_of` (same instant as payload cutoff),
`selected_model` (fixed research display default), `automatic_promotion: false`,
`models: [{id, label, role, history, latest?}]`. History is chronological with the
same origin sample across models. A latest row can repeat the latest history
origin; history itself cannot contain duplicate origins.

Each row has `origin_date` (zoned timestamp), `current_state`, `next_state`
(three-state probability distribution), and `horizons`. One-week-only models use
`horizons: {}`. Path models have all `1w`, `4w`, `13w` objects, each with:

- `horizon_weeks`, `target_date` (week-aligned; UTC duration allows ±1h for DST)
- `endpoint: {risk_on, transition, risk_off}`
- `first_departure: {no_departure, risk_on, transition, risk_off}`
- `any_risk_off_entry`, `any_risk_off_occupancy`

Entry is a change from another state into risk-off. If currently risk-off, entry
requires leaving and returning. Occupancy includes any risk-off week from t+1
through t+h. Neither includes the observed origin. First departure, cumulative
entry/occupancy, and endpoint are distinct targets. No extrapolation or aliasing
is allowed when evidence for a selected origin/model is missing.

The dedicated research selector changes the row used by the displayed one-week
and path probabilities. `research_model` in the view URL preserves this selection.
Candidate score rows use native `metrics` with `target: "next_state"`,
`horizon_weeks: 1`, `period: "retrospective_2023_2026"`, no `stratum`, and fields
`model`, `weeks`, `log_loss`, `brier`, `worsening_brier`, `worsening_hits/events`,
`recovery_hits/events`, `departure_false_alarms_per_year`. This table is explicitly
full research diagnostic scope, separate from the selected prediction origin.

## Calibration audit

`calibration_audit.schema_version = "regime-calibration-audit/1"`, `data_as_of`,
`selection_protocol` (text), `rows`, optional `latest_rows`.
Rows: `model`, `horizon_weeks`, `evaluation_split`, `n_predictions`,
`previous_published_log_loss/brier` (OLD issued), `previous_calibrated_*` (OLD
pre-projection), `raw_log_loss/brier` (identity), `calibrated_log_loss/brier` (NEW
selection before projection), `final_log_loss/brier` (NEW after projection),
`selected_method`, `final_available`. Missing final values stay null and display
as unavailable. The table labels old publication and new research distinctly.

Latest rows with `previous_published_probability` display that OLD probability
beside `identity_final_probability` and NEW `final_probability`, with
`horizon_weeks` and `target_end`. This comparison does not rewrite the official
outlook card or imply operational adoption.

## Additional information

`forecast_information.schema_version = "regime-forecast-information/1"`,
`data_as_of`, `status`, `rows`, optional `notes` (strings).
Rows: `feature_block`, `model`, `baseline_model`, `evaluation_split` or
`evidence_track`, `matched_n`, `baseline_log_loss`, `candidate_log_loss`,
`delta_log_loss`, `delta_brier`, `delta_worsening_brier`.
Empty rows render unavailable mature comparisons, never fabricated zero scores.
`prospective_only` means evidence collection; it is not historical backfill.

## Full evidence files

Each block may include `artifacts: [{label, url}]` for complete JSON/CSV reports.
URLs can be `./data/...`, `/data/...` or HTTPS. The parent must bundle those files
and choose routes that exist in the preview; the UI does not infer filenames.

## Operational denominators

`operational_diagnostics.probability_scores` supports `completed_weeks` and
`completed_entries`, `prospective_completed_weeks`,
`excluded_late_or_unverified_weeks`, `duplicate_target_entries`, `log_loss`,
`brier`, `evaluation_basis: "independent_of_investment_execution"`.
Display mature unique weeks / entries separately, and score only the verified
prospective denominator. Duplicate target entries do not inflate the week count.
