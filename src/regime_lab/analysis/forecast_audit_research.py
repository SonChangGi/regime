"""Frozen audit experiments and economic/episode validation on saved inputs.

This module has no collector, database or publication side effects. The caller
owns input loading and output assembly. Post-2022 is always diagnostic evidence.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from regime_lab.analysis.boundary_forecast import (
    ASYMMETRIC_MODEL, DEFAULT_BOUNDARY_CONFIG, BoundaryConfig, asymmetric_volatility,
    build_boundary_inputs, forecast_boundary_latest, run_boundary_walk_forward,
)
from regime_lab.analysis.forecast_paths import (
    HORIZONS, FIRST_DEPARTURE_ORDER, PATH_MODEL, MARKOV_PATH_MODEL,
    DirectionHazardConfig, fit_direction_hazard, fit_markov_path,
    hazard_features, path_outcomes, project_multistate_paths,
)
from regime_lab.analysis.forecast_research_evaluation import (
    match_forecasts, paired_comparisons, per_week_scores, summarize_scores,
)
from regime_lab.schema import STATE_ORDER

SCHEMA_VERSION = "regime-forecast-audit-research/1"
BOUNDARY_BASELINE = "boundary_filtered_history"
BASELINES = ("causal_dynamic_ensemble", "recency_weighted_xgboost_208w")
PROBABILITIES = [f"p_{s}" for s in STATE_ORDER]
CUTOFF = pd.Timestamp("2023-01-01", tz="UTC")


@dataclass(frozen=True)
class AuditResearchProtocol:
    version: str = "forecast-audit-v1"
    frozen_at: str = "2026-09-07"
    boundary: BoundaryConfig = DEFAULT_BOUNDARY_CONFIG
    hazard: DirectionHazardConfig = DirectionHazardConfig()
    annual_false_alert_budget: float = 4.
    policy_window_weeks: int = 156
    policy_minimum_rows: int = 52
    episode_lead_weeks: int = 4
    episode_probability_threshold: float = .5
    downside_return_threshold: float = -.05
    bootstrap_resamples: int = 999
    random_seed: int = 20260907

    def __post_init__(self):
        if self.version != "forecast-audit-v1" or self.frozen_at != "2026-09-07":
            raise ValueError("unsupported audit protocol")
        if not np.isfinite(self.annual_false_alert_budget) or self.annual_false_alert_budget < 0:
            raise ValueError("false alert budget must be nonnegative")
        if not 0 < self.episode_probability_threshold < 1 or not -1 < self.downside_return_threshold < 0:
            raise ValueError("invalid economic/episode thresholds")
        for key in ("policy_window_weeks", "policy_minimum_rows", "episode_lead_weeks", "bootstrap_resamples"):
            if isinstance(getattr(self, key), bool) or not isinstance(getattr(self, key), int) or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.policy_minimum_rows > self.policy_window_weeks:
            raise ValueError("minimum policy rows exceed window")

    def record(self) -> dict:
        content = asdict(self)
        content.update({"selection_end_exclusive": CUTOFF.isoformat(),
                        "diagnostic_role": "already_inspected_retrospective_diagnostic",
                        "diagnostic_hyperparameter_tuning": False,
                        "training_rule": "fixed_prequential_recipe_target_strictly_before_origin",
                        "horizons_weeks": list(HORIZONS), "automatic_promotion": False})
        content["sha256"] = hashlib.sha256(json.dumps(content, sort_keys=True, allow_nan=False).encode()).hexdigest()
        return content


def json_safe(value):
    if value is None or value is pd.NaT or value is pd.NA:
        return None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def split_for_dates(origin: pd.Timestamp, target: pd.Timestamp) -> str | None:
    """Independently purge every horizon that straddles the frozen boundary."""
    if target <= origin:
        raise ValueError("target must follow origin")
    if origin < CUTOFF <= target:
        return None
    return "selection" if target < CUTOFF else "retrospective_diagnostic"


def residual_sign_size_diagnostics(returns: pd.Series, sigma: pd.Series) -> dict:
    """Descriptive Engle-Ng-style lagged sign/size regression, not causal proof.

    z[t] uses sigma[t-1]. Regress z[t]^2 on a constant, negative sign at t-1,
    negative size and positive size at t-1. No significance-based model choice.
    """
    z = (returns / sigma.shift(1)).replace([np.inf, -np.inf], np.nan)
    lag = z.shift(1)
    table = pd.DataFrame({"squared_residual": z.pow(2), "negative_sign": (lag < 0).astype(float),
                          "negative_size": -lag.clip(upper=0), "positive_size": lag.clip(lower=0)}).dropna()
    if len(table) < 8:
        return {"status": "insufficient_residuals", "rows": len(table)}
    x = np.column_stack([np.ones(len(table)), table.iloc[:, 1:].to_numpy()])
    y = table.squared_residual.to_numpy()
    coefficients, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    residual = y - x @ coefficients
    total = float(((y - y.mean()) ** 2).sum())
    groups = []
    for name, mask in (("negative", table.negative_sign.eq(1)), ("nonnegative", table.negative_sign.eq(0))):
        values = table.loc[mask, "squared_residual"]
        groups.append({"previous_shock": name, "rows": len(values), "mean_next_squared_residual": values.mean()})
    return json_safe({"status": "descriptive", "rows": len(table), "design_rank": int(rank),
                      "start": table.index[0], "end": table.index[-1],
                      "coefficients": dict(zip(("intercept", *table.columns[1:]), coefficients)),
                      "r_squared": 1 - float((residual ** 2).sum()) / total if total else None,
                      "groups": groups, "used_to_select_or_tune_candidate": False})


def budget_threshold(probabilities: np.ndarray, worsening: np.ndarray, *, annual_budget: float) -> float:
    """Lowest threshold respecting the fixed historical false-alarm budget.

    Exposure is all observed weeks, including true worsening events. Thresholds
    are fitted from completed evidence; they do not assert a future budget cap.
    """
    p = np.asarray(probabilities, float)
    y = np.asarray(worsening, bool)
    if p.ndim != 1 or p.shape != y.shape or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("invalid alert calibration probabilities")
    if not np.isfinite(annual_budget) or annual_budget < 0:
        raise ValueError("invalid annual budget")
    if not len(p):
        return float(np.nextafter(1., 2.))
    allowed = annual_budget * len(p) / 52.1775
    candidates = np.unique(np.r_[0., p, np.nextafter(p[~y], np.inf), np.nextafter(1., 2.)])
    for threshold in candidates:
        if int(((p >= threshold) & ~y).sum()) <= allowed + 1e-12:
            return float(threshold)
    raise RuntimeError("no feasible alert threshold")


def evaluate_alert_policies(scored: pd.DataFrame, protocol: AuditResearchProtocol) -> tuple[pd.DataFrame, list[dict]]:
    rows = []
    for model, frame in scored.groupby("model"):
        frame = frame.sort_values("origin_date")
        selection = frame.loc[frame.target_date < CUTOFF]
        frozen = budget_threshold(selection.worsening_probability.to_numpy(), selection.worsening_event.to_numpy(), annual_budget=protocol.annual_false_alert_budget)
        for record in frame.itertuples():
            # Both policies are reported only after the frozen selection period.
            if record.origin_date < CUTOFF:
                continue
            past = frame.loc[frame.target_date < record.origin_date].tail(protocol.policy_window_weeks)
            ready = len(past) >= protocol.policy_minimum_rows
            rolling = budget_threshold(past.worsening_probability.to_numpy(), past.worsening_event.to_numpy(), annual_budget=protocol.annual_false_alert_budget) if ready else float(np.nextafter(1., 2.))
            for policy, threshold, history in (("frozen_selection_budget", frozen, selection), ("prequential_budget", rolling, past)):
                rows.append({"model": model, "policy": policy, "origin_date": record.origin_date,
                             "target_date": record.target_date, "probability": record.worsening_probability,
                             "threshold": threshold, "alert": record.worsening_probability >= threshold,
                             "event": bool(record.worsening_event), "calibration_rows": len(history),
                             "last_policy_target": history.target_date.max() if len(history) else None,
                             "fallback": policy == "prequential_budget" and not ready})
    table = pd.DataFrame(rows)
    summary = []
    if table.empty:
        return table, summary
    for (model, policy), group in table.groupby(["model", "policy"]):
        for period, part in (("retrospective_diagnostic", group), ("recent_52_mature_origins", group.tail(52))):
            events, hits = int(part.event.sum()), int((part.event & part.alert).sum())
            false = int((~part.event & part.alert).sum())
            rate = false / len(part) * 52.1775
            summary.append({"model": model, "policy": policy, "period": period, "weeks": len(part),
                            "events": events, "hits": hits, "recall": hits / events if events else None,
                            "false_alarms": false, "false_alarms_per_year": rate,
                            "annual_false_alarm_budget": protocol.annual_false_alert_budget,
                            "realized_budget_exceeded": rate > protocol.annual_false_alert_budget,
                            "origin_start": part.origin_date.min(), "origin_end": part.origin_date.max(),
                            "threshold_latest": part.threshold.iloc[-1]})
    return table, json_safe(summary)


def economic_outcomes(canonical: pd.DataFrame, horizons=(4, 13), *, downside_threshold: float = -.05) -> pd.DataFrame:
    price = canonical.spy_close.to_numpy(float)
    if not np.isfinite(price).all() or (price <= 0).any():
        raise ValueError("economic outcomes require finite positive prices")
    returns = np.diff(np.log(price))
    rows = []
    for horizon in horizons:
        for origin in range(len(price) - horizon):
            future = price[origin + 1:origin + horizon + 1] / price[origin] - 1
            minimum = float(min(0., future.min()))
            rows.append({"origin_date": canonical.index[origin], "target_date": canonical.index[origin + horizon],
                         "horizon_weeks": horizon, "forward_return": float(future[-1]),
                         "minimum_cumulative_return": minimum, "downside_event": minimum <= downside_threshold,
                         "annualized_realized_volatility": float(returns[origin:origin + horizon].std(ddof=0) * np.sqrt(52))})
    return pd.DataFrame(rows)


def economic_validation(scored: pd.DataFrame, canonical: pd.DataFrame, protocol: AuditResearchProtocol) -> list[dict]:
    outcomes = economic_outcomes(canonical, downside_threshold=protocol.downside_return_threshold)
    # Worsening is impossible in the worst ordinal state. Preserve the current
    # state so a zero probability there is not interpreted as low absolute risk.
    fields = ["model", "origin_date", "worsening_probability"]
    if "current_state" in scored:
        fields.append("current_state")
    joined = scored[fields].merge(outcomes, on="origin_date", validate="many_to_many")
    joined["split"] = [split_for_dates(o, t) for o, t in zip(joined.origin_date, joined.target_date)]
    summaries = []
    samples = [("all", joined)]
    if "current_state" in joined:
        samples.extend((state, joined.loc[joined.current_state.eq(state)]) for state in STATE_ORDER)
        samples.append(("at_risk", joined.loc[joined.current_state.ne("risk_off")]))
    for stratum, sample in samples:
      for (model, horizon, split), group in sample.dropna(subset=["split"]).groupby(["model", "horizon_weeks", "split"]):
        p = group.worsening_probability
        metrics = {"model": model, "horizon_weeks": int(horizon), "split": split,
                   "weeks": len(group), "origin_start": group.origin_date.min(), "origin_end": group.origin_date.max(),
                   "stratum": stratum,
                   "interpretation": ("pooled_descriptive_state_mix_not_incremental_skill" if stratum == "all" else "worsening_not_applicable_already_risk_off" if stratum == "risk_off" else "within_current_state_or_at_risk_association"),
                   "score_definition": "one_week_worsening_probability_as_risk_indicator",
                   "downside_threshold": protocol.downside_return_threshold,
                   "downside_events": int(group.downside_event.sum()), "risk_bins": []}
        for column in ("annualized_realized_volatility", "minimum_cumulative_return", "forward_return"):
            metrics[f"spearman_{column}"] = float(spearmanr(p, group[column]).statistic) if p.nunique() > 1 and group[column].nunique() > 1 else None
        for low, high in ((0, .1), (.1, .25), (.25, .5), (.5, 1.0000001)):
            part = group.loc[(p >= low) & (p < high)]
            metrics["risk_bins"].append({"lower_inclusive": low, "upper": min(high, 1.), "weeks": len(part),
                                         "downside_event_rate": part.downside_event.mean() if len(part) else None,
                                         "mean_realized_volatility": part.annualized_realized_volatility.mean(),
                                         "mean_forward_return": part.forward_return.mean(),
                                         "mean_minimum_cumulative_return": part.minimum_cumulative_return.mean()})
        summaries.append(metrics)
    return json_safe(summaries)


def persistence_validation(scored: pd.DataFrame, states: pd.Series) -> list[dict]:
    rows = []
    for record in scored.itertuples():
        position = int(states.index.get_loc(record.origin_date))
        if position + 2 >= len(states):
            continue
        split = split_for_dates(record.origin_date, states.index[position + 2])
        if split is None:
            continue
        current = STATE_ORDER.index(record.current_state)
        first = STATE_ORDER.index(record.actual)
        second = STATE_ORDER.index(states.iloc[position + 2])
        for name, sign in (("worsening", 1), ("recovery", -1)):
            if (first - current) * sign <= 0:
                continue
            rows.append({"model": record.model, "split": split, "direction": name,
                         "persistent_two_weeks": (second - current) * sign > 0,
                         "hit": bool(getattr(record, f"{name}_hit"))})
    result = []
    if not rows:
        return result
    for (model, split, direction), group in pd.DataFrame(rows).groupby(["model", "split", "direction"]):
        for persistent in (True, False):
            part = group.loc[group.persistent_two_weeks.eq(persistent)]
            result.append({"model": model, "split": split, "direction": direction,
                           "event_type": "persistent_two_weeks" if persistent else "reversed_by_second_week",
                           "events": len(part), "hits": int(part.hit.sum()),
                           "recall": float(part.hit.mean()) if len(part) else None})
    return result


def episode_validation(path_predictions: list[dict], states: pd.Series, protocol: AuditResearchProtocol) -> tuple[pd.DataFrame, list[dict]]:
    lookup = {(row["model"], pd.Timestamp(row["origin_date"]), row["horizon_weeks"]): row for row in path_predictions}
    models = sorted({row["model"] for row in path_predictions})
    starts = np.flatnonzero(states.eq("risk_off").to_numpy() & states.shift().ne("risk_off").to_numpy())
    rows = []
    for start in starts:
        if start == 0:  # left-censored first spell has no known entry
            continue
        end = int(start)
        while end + 1 < len(states) and states.iloc[end + 1] == "risk_off":
            end += 1
        split = "selection" if states.index[start] < CUTOFF else "retrospective_diagnostic"
        for model in models:
            eligible, leads = [], []
            for lead in range(1, protocol.episode_lead_weeks + 1):
                if start - lead < 0:
                    continue
                origin = states.index[start - lead]
                if split_for_dates(origin, states.index[start]) != split:
                    continue
                row = lookup.get((model, origin, protocol.episode_lead_weeks))
                if row is not None:
                    eligible.append(lead)
                    if row["any_risk_off_entry"] >= protocol.episode_probability_threshold:
                        leads.append(lead)
            exact = lookup.get((model, states.index[start - 1], 1))
            rows.append({"model": model, "split": split, "episode_id": states.index[start].isoformat(),
                         "start": states.index[start], "last_observed": states.index[end],
                         "duration_weeks": end - start + 1, "right_censored": end == len(states) - 1,
                         "eligible_lead_origins": len(eligible), "complete_lead_window": len(eligible) == protocol.episode_lead_weeks,
                         "detected_within_lead_window": bool(leads), "lead_weeks": max(leads) if leads else None,
                         "exact_entry_alert": exact["any_risk_off_entry"] >= protocol.episode_probability_threshold if exact else None})
    frame = pd.DataFrame(rows)
    summary = []
    if frame.empty:
        return frame, summary
    for (model, split), group in frame.groupby(["model", "split"]):
        complete = group.loc[group.complete_lead_window]
        detected = complete.loc[complete.detected_within_lead_window]
        summary.append({"model": model, "split": split, "episodes_total": len(group),
                        "eligible_episodes": len(complete), "excluded_incomplete_lead_window": len(group) - len(complete),
                        "detected_episodes": len(detected), "missed_episodes": len(complete) - len(detected),
                        "recall": len(detected) / len(complete) if len(complete) else None,
                        "mean_lead_weeks_conditional_on_detection": detected.lead_weeks.mean(),
                        "right_censored_episodes": int(complete.right_censored.sum()),
                        "completed_episodes": int((~complete.right_censored).sum()),
                        "event_definition": "distinct_risk_off_spells", "threshold": protocol.episode_probability_threshold,
                        "lead_window_weeks": protocol.episode_lead_weeks})
    return frame, json_safe(summary)


def evaluate_paths(predictions: list[dict], states: pd.Series) -> tuple[pd.DataFrame, list[dict]]:
    truth = path_outcomes(states).set_index(["origin_date", "horizon_weeks"])
    rows = []
    for row in predictions:
        key = (pd.Timestamp(row["origin_date"]), row["horizon_weeks"])
        if key not in truth.index:
            continue
        actual = truth.loc[key]
        split = split_for_dates(key[0], actual.target_date)
        if split is None:
            continue
        base = {"model": row["model"], "origin_date": key[0], "target_date": actual.target_date,
                "horizon_weeks": key[1], "split": split, "current_state": actual.current_state}
        for target, order in (("endpoint", STATE_ORDER), ("first_departure", FIRST_DEPARTURE_ORDER)):
            probabilities = np.asarray([row[target][state] for state in order])
            label = order.index(actual[target])
            rows.append({**base, "target": target, "log_loss": -np.log(max(probabilities[label], 1e-9)),
                         "brier": float(((probabilities - np.eye(len(order))[label]) ** 2).sum()),
                         "actual": actual[target], "probability_actual": probabilities[label]})
        for target in ("any_risk_off_entry", "any_risk_off_occupancy"):
            p, y = row[target], int(actual[target])
            clipped = np.clip(p, 1e-9, 1 - 1e-9)
            rows.append({**base, "target": target, "log_loss": -(y * np.log(clipped) + (1 - y) * np.log1p(-clipped)),
                         "brier": (p - y) ** 2, "actual": bool(y), "probability_actual": p if y else 1 - p,
                         "event": bool(y), "probability": p})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame, []
    result = []
    for (split, horizon, target), comparison in frame.groupby(["split", "horizon_weeks", "target"]):
        groups = list(comparison.groupby("model"))
        reference = set(groups[0][1].origin_date)
        if any(set(group.origin_date) != reference for _, group in groups):
            raise ValueError("path models do not share identical mature origins")
        for model, group in groups:
            parts = [("all", group), ("recent_52_mature_origins", group.sort_values("origin_date").tail(52))]
            parts.extend((state, group.loc[group.current_state.eq(state)]) for state in STATE_ORDER)
            for stratum, part in parts:
                if part.empty:
                    continue
                result.append({"model": model, "split": split, "horizon_weeks": horizon, "target": target,
                               "stratum": stratum, "weeks": len(part), "origin_start": part.origin_date.min(),
                               "origin_end": part.origin_date.max(), "log_loss": part.log_loss.mean(), "brier": part.brier.mean(),
                               "events": int(part.event.sum()) if target.startswith("any_") else None})
    return frame, json_safe(result)


@dataclass
class AuditResearchResult:
    document: dict
    predictions: pd.DataFrame
    path_predictions: list[dict]
    path_scores: pd.DataFrame
    alert_predictions: pd.DataFrame
    episode_details: pd.DataFrame


def run_audit_research(canonical: pd.DataFrame, states: pd.Series, baseline: pd.DataFrame, *,
                       protocol: AuditResearchProtocol = AuditResearchProtocol(),
                       progress: Callable[[str], None] | None = None) -> AuditResearchResult:
    baseline = baseline.loc[baseline.model.isin(BASELINES)].copy()
    if set(baseline.model) != set(BASELINES):
        raise ValueError("both fixed baseline models are required")
    for key in ("origin_date", "target_date"):
        baseline[key] = pd.to_datetime(baseline[key], utc=True, format="ISO8601")
    for row in baseline.itertuples():
        split = split_for_dates(row.origin_date, row.target_date)
        expected = "selection" if split == "selection" else "holdout" if split else None
        if row.evaluation_split != expected:
            raise ValueError("baseline split disagrees with frozen date boundary")
    source_origins = sorted(baseline.loc[baseline.model.eq(BASELINES[0]), "origin_date"].unique())
    positions = [int(states.index.get_loc(origin)) for origin in source_origins]
    if not positions:
        raise ValueError("baseline has no origins")
    # Reject missing common observations independently of string split helpers.
    expected_positions = [i for i in range(521, len(states) - 1) if split_for_dates(states.index[i], states.index[i + 1]) is not None]
    if positions != expected_positions:
        raise ValueError("baseline must cover every eligible weekly origin; no silent intersection")
    inputs = build_boundary_inputs(canonical, states, config=protocol.boundary, include_asymmetric=True)
    boundary = run_boundary_walk_forward(inputs, origin_positions=positions,
                                         models=(BOUNDARY_BASELINE, ASYMMETRIC_MODEL), progress=progress)
    features = hazard_features(inputs)
    ages = states.groupby(states.ne(states.shift()).cumsum()).cumcount() + 1
    forecast_rows, paths, latest = [], [], []
    for count, position in enumerate([*positions, len(states) - 1]):
        fitted = fit_direction_hazard(inputs, position, config=protocol.hazard, features=features)
        current, age = str(states.iloc[position]), int(ages.iloc[position])
        origin = states.index[position]
        matrix = fit_markov_path(states, position, train_window_weeks=protocol.hazard.train_window_weeks)
        for model, projections in ((PATH_MODEL, fitted.paths(current, age)),
                                   (MARKOV_PATH_MODEL, project_multistate_paths(lambda s, a: matrix[STATE_ORDER.index(s)], current, age))):
            for path in projections:
                target_position = position + path["horizon_weeks"]
                target = states.index[target_position] if target_position < len(states) else (origin.tz_convert("America/New_York") + pd.DateOffset(weeks=path["horizon_weeks"])).tz_convert("UTC")
                path["target_date"] = target.isoformat()
            entry = {"model": model, "origin_date": origin.isoformat(), "current_state": current,
                     "current_duration_weeks": age, "next_state": projections[0]["endpoint"], "paths": projections,
                     "training": fitted.audit if model == PATH_MODEL else {"last_train_target": states.index[position - 1].isoformat()}}
            paths.extend({"model": model, "origin_date": origin.isoformat(), "current_state": current, **path} for path in projections)
            if position == len(states) - 1:
                latest.append(entry)
                continue
            fallback = model == PATH_MODEL and any(x["fallback"] for x in fitted.audit["directions"].values())
            forecast_rows.append({"model": model, "origin_date": origin, "target_date": states.index[position + 1],
                                  "evaluation_split": "selection" if states.index[position + 1] < CUTOFF else "holdout",
                                  "current_state": current, "actual": str(states.iloc[position + 1]),
                                  **{f"p_{s}": p for s, p in projections[0]["endpoint"].items()},
                                  "fallback": fallback, "fallback_reason": "insufficient_completed_direction_events" if fallback else "",
                                  "last_train_target": states.index[position - 1].isoformat()})
        if progress and (count % 25 == 0 or count == len(positions)):
            progress(f"multistate origin {count + 1}/{len(positions) + 1} {origin.date()}")
    candidates = pd.concat([boundary, pd.DataFrame(forecast_rows)], ignore_index=True)
    matched = match_forecasts(baseline, candidates, states)
    scored = per_week_scores(matched)
    one_week = summarize_scores(scored).to_dict("records")
    # One-week current-state strata, with proper-score denominators visible.
    for (model, split, state), group in scored.groupby(["model", "evaluation_split", "current_state"]):
        one_week.append({"model": model, "period": "selection" if split == "selection" else "retrospective_diagnostic",
                         "stratum": state, "weeks": len(group), "log_loss": group.loss.mean(), "brier": group.brier.mean(),
                         "worsening_events": int(group.worsening_event.sum()), "recovery_events": int(group.recovery_event.sum())})
    comparisons = paired_comparisons(scored, [ASYMMETRIC_MODEL, PATH_MODEL], [BOUNDARY_BASELINE, *BASELINES],
                                    resamples=protocol.bootstrap_resamples, seed=protocol.random_seed)
    path_scores, path_metrics = evaluate_paths(paths, states)
    alert_rows, alerts = evaluate_alert_policies(scored, protocol)
    episode_details, episodes = episode_validation(paths, states, protocol)
    for model in (BOUNDARY_BASELINE, ASYMMETRIC_MODEL):
        row = forecast_boundary_latest(inputs, boundary, model=model)
        latest.append({"model": model, "origin_date": row["origin_date"], "current_state": row["current_state"],
                       "next_state": {s: row[f"p_{s}"] for s in STATE_ORDER}, "paths": [],
                       "training": {key: row[key] for key in ("last_train_target", "temperature", "calibration_rows")}})
    weekly = []
    path_lookup = defaultdict(list)
    for row in paths:
        path_lookup[row["model"], row["origin_date"]].append({k: v for k, v in row.items() if k not in ("model", "origin_date", "current_state")})
    for origin, group in matched.groupby("origin_date"):
        weekly.append({"origin_date": origin.isoformat(), "models": [
            {"model": row.model, "current_state": row.current_state, "target_date": row.target_date.isoformat(),
             "next_state": {s: getattr(row, f"p_{s}") for s in STATE_ORDER},
             "paths": path_lookup[row.model, origin.isoformat()]} for row in group.itertuples()]})
    returns = np.log(canonical.spy_close).diff()
    sigma = returns.ewm(span=protocol.boundary.volatility_span_weeks, adjust=False).std(bias=True).clip(lower=protocol.boundary.volatility_floor)
    residuals = {}
    for name, volatility in (("symmetric_ewma", sigma), (ASYMMETRIC_MODEL, asymmetric_volatility(returns, protocol.boundary))):
        residuals[name] = {"selection": residual_sign_size_diagnostics(returns.loc[returns.index < CUTOFF], volatility.loc[volatility.index < CUTOFF]),
                           "all_retrospective": residual_sign_size_diagnostics(returns, volatility)}
    document = {"schema_version": SCHEMA_VERSION, "status": "research_only", "data_as_of": states.index[-1].isoformat(),
                "protocol": protocol.record(), "evidence_track": "reconstructed_market", "automatic_promotion": False,
                "models": [{"id": model, "role": "challenger" if model in (ASYMMETRIC_MODEL, PATH_MODEL) else "baseline"} for model in sorted(matched.model.unique())],
                "latest": latest, "weekly": weekly, "metrics": [{**row, "horizon_weeks": 1, "target": "next_state"} for row in one_week] + path_metrics,
                "paired_comparisons": comparisons.to_dict("records"), "alert_policies": alerts,
                "episodes": episodes, "persistence": persistence_validation(scored, states),
                "economic_validation": economic_validation(scored, canonical, protocol), "residual_diagnostics": residuals,
                "source_hashes": {}, "limitations": [
                    "2023+ is previously inspected retrospective diagnostic evidence, not a fresh holdout.",
                    "The protocol is fixed; completed diagnostic observations may update estimators but never choose hyperparameters.",
                    "Path state/duration evolves; observed boundary and shock covariates stay fixed at origin.",
                    "Risk-off entry differs from occupancy, especially for an already risk-off origin.",
                    "Economic associations and overlapping horizons do not establish investable performance or causality.",
                    "Historical false-alert budgets are calibration constraints, not future guarantees.",
                    "Paired block intervals describe these candidates and do not remove prior research selection bias."]}
    document["generated_at"] = datetime.now(timezone.utc).isoformat()
    return AuditResearchResult(json_safe(document), matched, json_safe(paths), path_scores, alert_rows, episode_details)


def forecast_research_extension(document: dict) -> dict:
    """UI adapter; a display model is not a selected or promoted champion."""
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported research document")
    labels = {PATH_MODEL: "방향·지속기간 hazard", MARKOV_PATH_MODEL: "다상태 Markov 기준선",
              ASYMMETRIC_MODEL: "경계 전환 · 비대칭 변동성", BOUNDARY_BASELINE: "경계 전환 · 과거 충격",
              "causal_dynamic_ensemble": "공식 동적 앙상블", "recency_weighted_xgboost_208w": "최근 가중 XGBoost"}

    def row_view(row):
        return {"origin_date": row["origin_date"], "current_state": row["current_state"],
                "next_state": row["next_state"], "horizons": {f"{path['horizon_weeks']}w": path for path in row["paths"]}}

    latest = {row["model"]: row for row in document["latest"]}
    models = []
    for model in document["models"]:
        model_id = model["id"]
        history = [row_view({"origin_date": week["origin_date"], **row})
                   for week in document["weekly"] for row in week["models"] if row["model"] == model_id]
        item = {**model, "label": labels[model_id], "history": history}
        if model_id in latest:
            item["latest"] = row_view(latest[model_id])
        models.append(item)
    extension = {**document, "schema_version": "regime-forecast-research/1", "audit_schema_version": SCHEMA_VERSION,
                 "selected_model": PATH_MODEL, "display_selection_rule": "fixed_protocol_default_not_performance_selection",
                 "models": models}
    validate_forecast_research_extension(extension)
    return extension


def validate_forecast_research_extension(document: dict) -> None:
    """Strict optional extension contract shared with the JavaScript UI.

    Validation asserts probabilities and dates; it does not silently repair them.
    Entry means a transition into risk-off, occupancy means any future risk-off.
    """
    if not isinstance(document, dict) or document.get("schema_version") != "regime-forecast-research/1":
        raise ValueError("unsupported forecast research schema")

    def timestamp(value):
        if not isinstance(value, str):
            raise ValueError("research timestamps must be explicit zoned strings")
        result = pd.Timestamp(value)
        if pd.isna(result) or result.tzinfo is None:
            raise ValueError("research timestamps require a timezone")
        return result.tz_convert("UTC")

    def scalar(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("research probabilities must be finite unit fractions")
        return float(value)

    def distribution(value, keys):
        if not isinstance(value, dict) or set(value) != set(keys):
            raise ValueError("research probability keys must match the target")
        probabilities = [scalar(value[key]) for key in keys]
        if not np.isclose(sum(probabilities), 1, atol=1e-8, rtol=0):
            raise ValueError("research probability mass must sum to one")

    as_of = timestamp(document.get("data_as_of"))
    models = document.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("research models are required")
    ids = [model.get("id") for model in models if isinstance(model, dict)]
    if len(ids) != len(models) or any(not isinstance(x, str) or not x for x in ids) or len(set(ids)) != len(ids):
        raise ValueError("research model IDs must be unique nonempty strings")
    if document.get("selected_model") not in ids:
        raise ValueError("research display model is absent")

    def validate_row(row, *, latest):
        if not isinstance(row, dict) or row.get("current_state") not in STATE_ORDER:
            raise ValueError("invalid research current state")
        origin = timestamp(row.get("origin_date"))
        if origin > as_of or (latest and origin != as_of):
            raise ValueError("research origin disagrees with data_as_of")
        current = row["current_state"]
        if "next_state" in row:
            distribution(row["next_state"], STATE_ORDER)
        horizons = row.get("horizons")
        if not isinstance(horizons, dict) or set(horizons) not in (set(), {"1w", "4w", "13w"}):
            raise ValueError("research paths require all 1/4/13 horizons or none")
        previous = None
        for horizon in HORIZONS if horizons else ():
            path = horizons[f"{horizon}w"]
            if not isinstance(path, dict) or type(path.get("horizon_weeks")) is not int or path["horizon_weeks"] != horizon:
                raise ValueError("research horizon disagrees with its key")
            target = timestamp(path.get("target_date"))
            hours = (target - origin).total_seconds() / 3600
            if not horizon * 168 - 1 <= hours <= horizon * 168 + 1:
                raise ValueError("research target must be horizon-aligned (+/-1h DST)")
            distribution(path.get("endpoint"), STATE_ORDER)
            distribution(path.get("first_departure"), FIRST_DEPARTURE_ORDER)
            endpoint, first = path["endpoint"], path["first_departure"]
            entry, occupancy = scalar(path.get("any_risk_off_entry")), scalar(path.get("any_risk_off_occupancy"))
            if abs(first[current]) > 1e-8 or first["no_departure"] > endpoint[current] + 1e-8:
                raise ValueError("first departure contradicts the current/endpoint state")
            if entry > occupancy + 1e-8 or endpoint["risk_off"] > occupancy + 1e-8:
                raise ValueError("risk-off entry/occupancy probabilities are incoherent")
            if current != "risk_off" and (abs(entry - occupancy) > 1e-8 or first["risk_off"] > entry + 1e-8):
                raise ValueError("risk-off entry and occupancy must agree outside risk-off")
            if current == "risk_off" and occupancy + 1e-8 < first["no_departure"]:
                raise ValueError("risk-off no-departure mass must be occupied")
            if horizon == 1:
                expected = {s: first["no_departure"] if s == current else first[s] for s in STATE_ORDER}
                if any(abs(endpoint[s] - expected[s]) > 1e-8 for s in STATE_ORDER):
                    raise ValueError("one-week first departure must equal next-state probabilities")
                if "next_state" in row and any(abs(endpoint[s] - row["next_state"][s]) > 1e-8 for s in STATE_ORDER):
                    raise ValueError("one-week endpoint disagrees with next_state")
                if abs(occupancy - endpoint["risk_off"]) > 1e-8 or abs(entry - (0 if current == "risk_off" else occupancy)) > 1e-8:
                    raise ValueError("one-week entry/occupancy semantics disagree")
            if previous is not None:
                if entry + 1e-8 < previous["any_risk_off_entry"] or occupancy + 1e-8 < previous["any_risk_off_occupancy"]:
                    raise ValueError("cumulative risk must not decrease with horizon")
                if first["no_departure"] > previous["first_departure"]["no_departure"] + 1e-8 or any(first[s] + 1e-8 < previous["first_departure"][s] for s in STATE_ORDER):
                    raise ValueError("first-departure cumulative mass is not monotone")
            previous = path
        return origin

    reference = None
    for model in models:
        if not isinstance(model.get("label"), str) or not isinstance(model.get("history"), list):
            raise ValueError("research models require label and history")
        origins = [validate_row(row, latest=False) for row in model["history"]]
        if origins != sorted(set(origins)):
            raise ValueError("research history must be unique and chronological")
        if reference is not None and origins != reference:
            raise ValueError("research histories require identical origin samples")
        reference = origins
        if "latest" in model:
            validate_row(model["latest"], latest=True)
    if document.get("automatic_promotion") is not False:
        raise ValueError("audit research cannot automatically promote a model")
    json.dumps(document, allow_nan=False)
