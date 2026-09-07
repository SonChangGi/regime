"""Preview-only probability decoder and separately budgeted risk/alpha trading.

The issued v1 ledger is deliberately outside this API. Parameters are fixed in
the versioned specification; reconstructed post-selection returns never choose
an operating policy. Four-week alpha payoffs end at the next scheduled open,
including the exit gap, exactly when the alpha decision can be replaced.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from regime_lab.integrity import canonical_json_sha256_v1
from regime_lab.analysis.decision_shadow import STATE_ORDER, _scheduled_nyse_entry_at
from .shadow import (
    _composite_momentum_at,
    _date_lookup,
    _metrics,
    _signal_frame,
    _turnover,
    load_allocation_shadow_spec,
    split_safe_asset_return_frames,
)


def _block_indices(n: int, spec: Mapping[str, Any]) -> np.ndarray:
    block = min(int(spec["bootstrap_block_weeks"]), max(1, n // 2))
    rng = np.random.default_rng(int(spec["bootstrap_seed"]))
    starts = rng.integers(
        0, n, (int(spec["bootstrap_resamples"]), math.ceil(n / block))
    )
    return ((starts[..., None] + np.arange(block)) % n).reshape(len(starts), -1)[:, :n]


def _mean_interval(values: np.ndarray, spec: Mapping[str, Any]) -> list[float] | None:
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        return None
    return np.quantile(
        values[_block_indices(len(values), spec)].mean(axis=1), [0.025, 0.975]
    ).tolist()


def _probability_matrix(frame: pd.DataFrame) -> np.ndarray:
    matrix = frame[[f"p_{s}" for s in STATE_ORDER]].to_numpy(dtype=float)
    if (
        not np.isfinite(matrix).all()
        or (matrix < 0).any()
        or not np.allclose(matrix.sum(axis=1), 1, atol=1e-6)
    ):
        raise ValueError("v2 selection probabilities must be finite simplex rows")
    return matrix / matrix.sum(axis=1, keepdims=True)


def _selection_sample(
    predictions: pd.DataFrame, model: str, cutoff: pd.Timestamp
) -> pd.DataFrame:
    required = {
        "model",
        "evaluation_split",
        "origin_date",
        "target_date",
        "actual",
        *[f"p_{s}" for s in STATE_ORDER],
    }
    if not required <= set(predictions):
        raise ValueError("v2 decoder requires stored selection OOS probability rows")
    sample = predictions.loc[
        (predictions.model == model) & (predictions.evaluation_split == "selection")
    ].copy()
    sample["origin_date"] = pd.to_datetime(sample.origin_date, utc=True)
    sample["target_date"] = pd.to_datetime(sample.target_date, utc=True)
    if sample.empty or sample.origin_date.duplicated().any():
        raise ValueError("v2 selection OOS origins must be unique and nonempty")
    if (sample.target_date > cutoff).any() or (
        sample.origin_date >= sample.target_date
    ).any():
        raise ValueError("v2 selection targets cross the frozen boundary")
    if not sample.actual.isin(STATE_ORDER).all():
        raise ValueError("v2 selection actual states are invalid")
    sample = sample.sort_values("origin_date").reset_index(drop=True)
    _probability_matrix(sample)
    return sample


def _probability_diagnostics(
    sample: pd.DataFrame, benchmark: pd.DataFrame, spec: Mapping[str, Any]
) -> dict[str, Any]:
    aligned = sample.merge(
        benchmark,
        on=["origin_date", "target_date", "actual"],
        suffixes=("", "_benchmark"),
        validate="one_to_one",
    )
    if len(aligned) != len(sample) or len(aligned) != len(benchmark):
        raise ValueError(
            "v2 Markov skill comparison requires exactly matched origins/actuals"
        )
    p = _probability_matrix(aligned)
    q = aligned[[f"p_{s}_benchmark" for s in STATE_ORDER]].to_numpy(dtype=float)
    actual = np.array([STATE_ORDER.index(s) for s in aligned.actual])
    truth = np.eye(3)[actual]
    loss = -np.log(np.maximum(p[np.arange(len(p)), actual], 1e-9))
    base_loss = -np.log(np.maximum(q[np.arange(len(p)), actual], 1e-9))
    interval = _mean_interval(base_loss - loss, spec)
    bins = []
    for j, state in enumerate(STATE_ORDER):
        for low in np.arange(0, 1, 0.2):
            mask = (p[:, j] >= low) & (p[:, j] < low + 0.2 + (1e-9 if low > 0.7 else 0))
            if mask.any():
                bins.append(
                    {
                        "state": state,
                        "lower": float(low),
                        "upper": min(1.0, float(low + 0.2)),
                        "n": int(mask.sum()),
                        "mean_probability": float(p[mask, j].mean()),
                        "observed_rate": float(truth[mask, j].mean()),
                    }
                )
    passed = (
        len(p) >= int(spec["minimum_probability_skill_origins"])
        and interval is not None
        and interval[0] > 0
    )
    return {
        "benchmark": "markov",
        "n": len(p),
        "log_loss": float(loss.mean()),
        "benchmark_log_loss": float(base_loss.mean()),
        "brier": float(((p - truth) ** 2).sum(axis=1).mean()),
        "log_loss_improvement": float((base_loss - loss).mean()),
        "log_loss_improvement_ci95": interval,
        "skill_gate_passed": passed,
        "calibration": {
            "status": "diagnostic_not_certification",
            "bins": bins,
            "method": "one_vs_rest_fixed_probability_bins",
        },
    }


def _forward_open_return(
    gaps: pd.DataFrame, intraday: pd.DataFrame, start: int, end: int
) -> np.ndarray:
    """From open at start to open at end; end's intraday return is excluded."""
    if end <= start or end >= len(gaps):
        raise ValueError("invalid open-to-open holding window")
    factors = (
        intraday.iloc[start:end].to_numpy() * gaps.iloc[start + 1 : end + 1].to_numpy()
    )
    if not np.isfinite(factors).all() or (factors <= 0).any():
        raise ValueError("v2 holding window has invalid price relatives")
    return factors.prod(axis=0) - 1


def _fit_decoder(
    sample: pd.DataFrame,
    prices: pd.DataFrame,
    gaps: pd.DataFrame,
    intraday: pd.DataFrame,
    cutoff: pd.Timestamp,
    spec: Mapping[str, Any],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    lookup = _date_lookup(prices.index)
    p, outcomes, origin_times, exits = [], [], [], []
    horizon = int(spec["holding_weeks"])
    for row in sample.itertuples(index=False):
        origin = lookup.get(row.origin_date.date().isoformat())
        if origin is None:
            continue
        position = int(prices.index.get_loc(origin))
        start, end = position + 1, position + 1 + horizon
        if end >= len(prices) or prices.index[end].tz_convert("UTC") > cutoff:
            continue
        # Weekly index continuity is a required contract, not a row-count proxy.
        if (prices.index[end].date() - prices.index[start].date()).days != 7 * horizon:
            raise ValueError("v2 decoder requires consecutive market weeks")
        returns = _forward_open_return(gaps, intraday, start, end)
        p.append([getattr(row, f"p_{s}") for s in STATE_ORDER])
        outcomes.append(returns[0] - returns[1])
        origin_times.append(row.origin_date)
        exits.append(prices.index[end].tz_convert("UTC"))
    X, y = np.asarray(p), np.asarray(outcomes)
    minimum = int(spec["selection_minimum_origins"])
    if len(y) < minimum:
        raise ValueError(
            f"v2 decoder needs {minimum} matured selection windows, found {len(y)}"
        )
    penalty = float(spec["ridge_pseudo_observations"])

    def fit(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.linalg.solve(a.T @ a + penalty * np.eye(3), a.T @ b)

    coefficients = fit(X, y)
    draws = np.asarray([fit(X[ix], y[ix]) for ix in _block_indices(len(y), spec)])
    improvements, validation_rows = [], []
    for k, origin in enumerate(origin_times):
        eligible = np.array([i for i in range(k) if exits[i] < origin], dtype=int)
        if len(eligible) < minimum:
            continue
        estimate = float(X[k] @ fit(X[eligible], y[eligible]))
        prior_mean = float(y[eligible].mean())
        improvements.append((y[k] - prior_mean) ** 2 - (y[k] - estimate) ** 2)
        validation_rows.append(
            {
                "origin": origin.isoformat(),
                "last_training_exit": exits[eligible[-1]].isoformat(),
                "estimate": estimate,
                "actual": float(y[k]),
            }
        )
    validation_interval = _mean_interval(np.asarray(improvements), spec)
    return (
        {
            "method": "ridge_on_completed_oos_state_probabilities_zero_prior",
            "holding_weeks": horizon,
            "return_window": "next_open_to_open_four_weeks_later",
            "exit_gap_included": True,
            "selection_n": len(y),
            "selection_end": cutoff.isoformat(),
            "last_training_exit": max(exits).isoformat(),
            "coefficients": dict(zip(STATE_ORDER, coefficients.tolist())),
            "coefficient_ci95": {
                s: np.quantile(draws[:, i], [0.025, 0.975]).tolist()
                for i, s in enumerate(STATE_ORDER)
            },
            "bootstrap_method": "circular_moving_block_full_decoder_refit",
            "bootstrap_block_weeks": int(spec["bootstrap_block_weeks"]),
            "bootstrap_resamples": int(spec["bootstrap_resamples"]),
            "validation": {
                "method": "expanding_maturity_purged_selection_only",
                "n": len(improvements),
                "mse_improvement_vs_past_mean": (
                    float(np.mean(improvements)) if improvements else None
                ),
                "mse_improvement_ci95": validation_interval,
                "skill_gate_passed": validation_interval is not None
                and validation_interval[0] > 0,
                "rows": validation_rows,
            },
            "relative_return_scale": float(y.std(ddof=1)),
        },
        coefficients,
        draws,
    )


def _monthly_sector_evaluation(
    prices: pd.DataFrame,
    cutoff: pd.Timestamp,
    spec: Mapping[str, Any],
    *,
    evaluation_split: str = "selection",
) -> dict[str, Any]:
    legacy = load_allocation_shadow_spec()
    sectors = tuple(legacy["assets"]["sectors"])
    gaps, intra = split_safe_asset_return_frames(prices, ("SPY", *sectors))
    schedule = []
    last_month = None
    for position in range(1, len(prices)):
        entry = _scheduled_nyse_entry_at(prices.index[position].date().isoformat())
        if entry.month != last_month:
            schedule.append(position)
            last_month = entry.month
    rows = []
    prior = {"SPY": 1.0}
    for start, end in zip(schedule, schedule[1:]):
        origin = prices.index[start - 1]
        if evaluation_split == "selection":
            if prices.index[end].tz_convert("UTC") > cutoff:
                continue
        elif origin.tz_convert("UTC") <= cutoff:
            continue
        scores, _ = _composite_momentum_at(
            prices, origin, sectors, legacy["sector_rotation"]
        )
        selected = sorted(scores, key=scores.get, reverse=True)[:3]
        if len(selected) < 3:
            continue
        selected_columns = ["SPY", *selected]
        realized = _forward_open_return(
            gaps[selected_columns], intra[selected_columns], start, end
        )
        targets = {s: 1 / 3 for s in selected}
        turnover = sum(
            abs(targets.get(s, 0) - prior.get(s, 0)) for s in set(prior) | set(targets)
        )
        gross_factor = 1 + float(realized[1:].mean())
        rows.append(
            {
                "origin": origin.date().isoformat(),
                "entry_week": prices.index[start].date().isoformat(),
                "exit_week": prices.index[end].date().isoformat(),
                "holding_weeks": end - start,
                "selected": selected,
                "gross_relative_return": float(realized[1:].mean() - realized[0]),
                "full_l1_turnover": turnover,
                "net_relative_return_by_cost": {
                    f"{cost:g}bps": (1 - cost / 10000 * turnover) * gross_factor
                    - 1
                    - float(realized[0])
                    for cost in spec["cost_bps_per_traded_notional"]
                },
            }
        )
        prior = {
            s: (1 / 3) * (1 + float(realized[i + 1])) / gross_factor
            for i, s in enumerate(selected)
        }
    returns = np.asarray([r["gross_relative_return"] for r in rows])
    result = {
        "status": "diagnostic_no_sleeve_promotion",
        "evaluation_split": evaluation_split,
        "holding_contract": "first_monthly_open_to_next_first_monthly_open",
        "rule_contract": "unchanged_legacy_momentum_lookbacks_equal_weight_top_three_no_holdout_retuning",
        "initial_funding": "SPY_position_conversion_costed_at_first_monthly_trade",
        "selection_end": cutoff.isoformat(),
        "n_months": len(rows),
        "mean_relative_return": float(returns.mean()) if len(returns) else None,
        "mean_relative_return_ci95": _mean_interval(
            returns, {**spec, "bootstrap_block_weeks": 3}
        ),
        "net_mean_relative_return_by_cost": {
            f"{cost:g}bps": (
                float(
                    np.mean(
                        [r["net_relative_return_by_cost"][f"{cost:g}bps"] for r in rows]
                    )
                )
                if rows
                else None
            )
            for cost in spec["cost_bps_per_traded_notional"]
        },
        "rows": rows,
    }
    if evaluation_split == "selection":
        result["holdout"] = _monthly_sector_evaluation(
            prices, cutoff, spec, evaluation_split="holdout"
        )
    return result


def _simulate(
    index: pd.DatetimeIndex,
    gaps: pd.DataFrame,
    intra: pd.DataFrame,
    cash: pd.Series,
    signals: Sequence[Mapping[str, Any]],
    spec: Mapping[str, Any],
    cost_bps: float,
    mode: str,
    initial_weight: float | None = None,
) -> list[dict[str, Any]]:
    anchor, cap = float(spec["anchor_spy_weight"]), float(spec["one_way_trade_cap"])
    prior, prior_cash, alpha = np.zeros(2), 1.0, 0.0
    wealth = gross_wealth = peak = 1.0
    rows = []
    for k, t in enumerate(index):
        gap = gaps.loc[t].to_numpy(float)
        holding = intra.loc[t].to_numpy(float)
        if (
            not np.isfinite(gap).all()
            or not np.isfinite(holding).all()
            or (gap <= 0).any()
            or (holding <= 0).any()
        ):
            raise ValueError("v2 evaluation has missing/nonpositive returns")
        factor = 1.0 if k == 0 else float(prior @ gap + prior_cash)
        pre = prior * gap / factor
        pre_cash = prior_cash / factor
        signal = signals[k]
        alpha_due = k % int(spec["holding_weeks"]) == 0
        alpha_allowed = False
        previous_alpha = alpha
        if alpha_due and mode == "regime_alpha":
            alpha = float(signal["eligible_tilt"])
            alpha_allowed = abs(alpha) > 0
        weight = (anchor + alpha) if mode == "regime_alpha" else anchor
        if mode == "spy_buy_and_hold":
            weight = 1.0
        if mode == "initial_weight_buy_and_hold":
            weight = float(initial_weight)
        aim = np.array([weight, 1 - weight])
        if mode == "risk_matched_60_40":
            aim *= min(1.0, float(signal["risk_match_scale"]))
        aim_cash = 1 - float(aim.sum())
        distance, _ = _turnover(pre, pre_cash, aim, aim_cash)
        buy_hold = mode in {
            "spy_buy_and_hold",
            "static_60_40_buy_and_hold",
            "initial_weight_buy_and_hold",
        }
        if k == 0:
            target, action = aim, "initial_allocate"
        elif buy_hold:
            target, action = pre, "buy_and_hold"
        elif (
            mode == "regime_alpha"
            and alpha_due
            and (abs(previous_alpha) > 1e-12 or alpha_allowed)
        ):
            # The old horizon expires here. Returning to core is risk-budget
            # maintenance, not an alpha order that needs a positive return edge.
            move = aim - pre
            target = pre + move * min(1.0, cap / max(distance, 1e-12))
            action = "alpha_rebalance" if alpha_allowed else "alpha_expiry_core_reset"
        elif distance >= float(spec["core_one_way_band"]):
            move = float(spec["core_partial_adjustment"]) * (aim - pre)
            one_way = 0.5 * (np.abs(move).sum() + abs(float(move.sum())))
            target = pre + move * min(1.0, cap / max(one_way, 1e-12))
            action = "core_risk_rebalance"
        else:
            target, action = pre, "band_hold"
        target_cash = max(0.0, 1 - float(target.sum()))
        one_way, full = _turnover(pre, pre_cash, target, target_cash)
        if k and full < 1e-12:
            action = "band_hold" if not buy_hold else "buy_and_hold"
        charge = cost_bps / 10000 * full
        cash_factor = float(cash.loc[t])
        held = float(target @ holding + target_cash * cash_factor)
        gross = factor * held - 1
        net = factor * (1 - charge) * held - 1
        gross_wealth *= 1 + gross
        wealth *= 1 + net
        peak = max(peak, wealth)
        rows.append(
            {
                "date": t.date().isoformat(),
                "week": t.date().isoformat(),
                "gross_return": gross,
                "net_return": net,
                "gross_wealth": gross_wealth,
                "net_wealth": wealth,
                "drawdown": wealth / peak - 1,
                "one_way_turnover": one_way,
                "full_l1_turnover": full,
                "transaction_cost_rate": charge,
                "action": action,
                "alpha_tilt": alpha if mode == "regime_alpha" else 0.0,
                "target_weights": {"SPY": float(target[0]), "TLT": float(target[1])},
                "cash": target_cash,
                "expected_holding_relative_return": signal["expected_relative_return"],
                "alpha_screen": (
                    signal["alpha_screen"]
                    if alpha_due and mode == "regime_alpha"
                    else "not_alpha_decision"
                ),
            }
        )
        prior = target * holding / held
        prior_cash = target_cash * cash_factor / held
    return rows


def build_allocation_shadow_v2(
    weekly: Sequence[Mapping[str, Any]],
    prices: pd.DataFrame,
    selection_predictions: pd.DataFrame,
    *,
    forecast_model: str,
    selection_end: str,
    current_signal: Mapping[str, Any],
    spec_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate a versioned 4-week research policy without mutating v1 intent.

    prices is the same PIT-aligned canonical panel accepted by v1; predictions
    may contain all splits, but only matured selection rows are read by fit.
    """
    path = (
        Path(spec_path)
        if spec_path
        else Path(__file__).resolve().parents[3] / "config/allocation-shadow-v2.json"
    )
    spec = json.loads(path.read_text())
    if spec.get("schema_version") != "regime-allocation-research-spec/2":
        raise ValueError("invalid allocation v2 specification")
    if (
        int(spec["holding_weeks"]) != 4
        or min(spec["cost_bps_per_traded_notional"]) < 10
        or float(spec["benefit_cost_multiple"]) < 2
    ):
        raise ValueError(
            "v2 requires the frozen four-week horizon and unreduced cost hurdle"
        )
    if (
        not isinstance(prices.index, pd.DatetimeIndex)
        or prices.index.tz is None
        or prices.index.has_duplicates
        or not prices.index.is_monotonic_increasing
    ):
        raise ValueError("v2 canonical prices require a sorted aware unique index")
    cutoff = pd.Timestamp(selection_end)
    cutoff = (
        cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    )
    sample = _selection_sample(selection_predictions, forecast_model, cutoff)
    benchmark = _selection_sample(selection_predictions, "markov", cutoff)
    probability = _probability_diagnostics(sample, benchmark, spec)
    gaps, intra = split_safe_asset_return_frames(prices, ("SPY", "TLT"))
    decoder, coefficients, draws = _fit_decoder(
        sample, prices, gaps, intra, cutoff, spec
    )
    source_signals = _signal_frame(weekly, prices, forecast_model=forecast_model)
    realized = [r for r in source_signals if r["target"] is not None]
    if not realized:
        raise ValueError("v2 requires realized post-selection forecasts")
    index = pd.DatetimeIndex([r["target"] for r in realized])
    if index.has_duplicates or any(
        r["origin"].tz_convert("UTC") <= cutoff for r in realized
    ):
        raise ValueError("v2 evaluation origins must be strictly post-selection")
    if any((b.date() - a.date()).days != 7 for a, b in zip(index, index[1:])):
        raise ValueError("v2 evaluation requires consecutive weekly origins")
    cash = (1 + pd.to_numeric(prices["dgs3mo"]).shift(1) / 100 / 52).reindex(index)
    if not np.isfinite(cash).all() or (cash <= 0).any():
        raise ValueError("v2 cash accrual is unavailable")
    weekly_returns = gaps * intra - 1
    signals, covariances = [], []
    hurdle = (
        2
        * float(spec["benefit_cost_multiple"])
        * min(spec["cost_bps_per_traded_notional"])
        / 10000
    )
    for row in source_signals:
        p = np.array([row["probabilities"][s] for s in STATE_ORDER])
        expected = float(p @ coefficients)
        ci = np.quantile(draws @ p, [0.025, 0.975]).tolist()
        failures = []
        if (
            spec["require_positive_skill_interval"]
            and not probability["skill_gate_passed"]
        ):
            failures.append("probability_skill_unconfirmed")
        if (
            spec["require_positive_decoder_validation_interval"]
            and not decoder["validation"]["skill_gate_passed"]
        ):
            failures.append("decoder_skill_unconfirmed")
        conservative = ci[0] if expected >= 0 else -ci[1]
        if spec["require_positive_alpha_payoff_interval"] and conservative <= hurdle:
            failures.append("payoff_interval_below_cost")
        if abs(expected) <= hurdle:
            failures.append("mean_payoff_below_cost")
        tilt = (
            float(spec["maximum_alpha_tilt"])
            * math.tanh(expected / max(decoder["relative_return_scale"], 1e-12))
            if not failures
            else 0.0
        )
        history = weekly_returns.loc[: row["origin"]].tail(26)
        covariance = history.cov().to_numpy()
        covariances.append(covariance if len(history) >= 13 else None)
        base = np.array(
            [float(spec["anchor_spy_weight"]), 1 - float(spec["anchor_spy_weight"])]
        )
        candidate = base + np.array([tilt, -tilt])
        base_var = float(base @ covariance @ base)
        scale = (
            math.sqrt(max(0, float(candidate @ covariance @ candidate)) / base_var)
            if len(history) >= 13 and base_var > 0
            else 1.0
        )
        signals.append(
            {
                "origin": row["origin_date"],
                "target_week": row["target_date"],
                "expected_relative_return": expected,
                "payoff_ci95": ci,
                "eligible_tilt": tilt,
                "alpha_screen": "passed" if not failures else ";".join(failures),
                "risk_match_scale": scale,
            }
        )
    modes = [
        "core_60_40",
        "spy_buy_and_hold",
        "static_60_40_buy_and_hold",
        "initial_weight_buy_and_hold",
        "regime_alpha",
        "risk_matched_60_40",
    ]
    runs, metrics = {}, {}
    for cost in spec["cost_bps_per_traded_notional"]:
        key = f"{cost:g}bps"
        # Stress costs screen alpha at the stress hurdle, not only execution.
        cost_signals = [dict(s) for s in signals[: len(realized)]]
        threshold = 2 * float(spec["benefit_cost_multiple"]) * float(cost) / 10000
        for s in cost_signals:
            conservative = (
                s["payoff_ci95"][0]
                if s["expected_relative_return"] >= 0
                else -s["payoff_ci95"][1]
            )
            if conservative <= threshold:
                s["eligible_tilt"] = 0.0
                s["alpha_screen"] += ";stress_cost_not_covered"
        initial = float(spec["anchor_spy_weight"]) + float(
            cost_signals[0]["eligible_tilt"]
        )
        runs[key] = {
            mode: _simulate(
                index, gaps, intra, cash, cost_signals, spec, float(cost), mode, initial
            )
            for mode in modes
            if mode != "risk_matched_60_40"
        }
        # Match the candidate's actual opening holdings after its four-week
        # schedule, risk bands and cost screen. Weekly proposed tilts would
        # describe a different portfolio, especially under the stress cost.
        base_weight = np.array(
            [float(spec["anchor_spy_weight"]), 1 - float(spec["anchor_spy_weight"])]
        )
        risk_signals = [dict(s) for s in cost_signals]
        for k, row in enumerate(runs[key]["regime_alpha"]):
            covariance = covariances[k]
            candidate = np.array(
                [row["target_weights"]["SPY"], row["target_weights"]["TLT"]]
            )
            base_var = (
                float(base_weight @ covariance @ base_weight)
                if covariance is not None
                else 0.0
            )
            risk_signals[k]["risk_match_scale"] = (
                math.sqrt(
                    max(0.0, float(candidate @ covariance @ candidate)) / base_var
                )
                if base_var > 0
                else 1.0
            )
        runs[key]["risk_matched_60_40"] = _simulate(
            index,
            gaps,
            intra,
            cash,
            risk_signals,
            spec,
            float(cost),
            "risk_matched_60_40",
            initial,
        )
        metrics[key] = {}
        for mode, rows in runs[key].items():
            m = _metrics(
                rows,
                annualization=float(spec["annualization_weeks"]),
                gamma=float(spec["certainty_equivalent_risk_aversion"]),
            )
            net = np.array([r["net_return"] for r in rows])
            excess = net - (cash.to_numpy() - 1)
            m["sharpe"] = (
                float(
                    excess.mean()
                    / excess.std(ddof=1)
                    * math.sqrt(float(spec["annualization_weeks"]))
                )
                if len(excess) > 1 and excess.std(ddof=1) > 0
                else None
            )
            m["sharpe_basis"] = "weekly_excess_return_over_aligned_cash_proxy"
            m["action_counts"] = dict(Counter(r["action"] for r in rows))
            metrics[key][mode] = m
    primary = f"{min(spec['cost_bps_per_traded_notional']):g}bps"
    base_rows, alpha_rows = runs[primary]["core_60_40"], runs[primary]["regime_alpha"]
    paired = np.array(
        [a["net_return"] - b["net_return"] for a, b in zip(alpha_rows, base_rows)]
    )
    m = metrics[primary]
    return {
        "schema_version": "regime-allocation-research/2",
        "role": "research_only_no_promotion",
        "policy_status": "research_only_no_promotion",
        "affects_official_forecast": False,
        "affects_champion_selection": False,
        "affects_issued_ledger": False,
        "spec": {
            "path": str(path.name),
            "sha256": canonical_json_sha256_v1(spec),
            "parameters": spec,
        },
        "decoder": decoder,
        "probability_skill": probability,
        "sector_rotation": _monthly_sector_evaluation(prices, cutoff, spec),
        "current_signal": dict(current_signal),
        "current_research_signal": signals[-1],
        "execution_contract": {
            "core": "weekly_risk_weight_band_independent_of_alpha_return_hurdle",
            "alpha": "four_week_expiry_and_reassessment",
            "benefit_cost_multiple": float(spec["benefit_cost_multiple"]),
            "alpha_cost_threshold_relative_return": hurdle,
            "late_signal_policy": "no_trade",
            "reconstructed_timing_assumption": "available_before_scheduled_open",
            "return_accounting": "split_safe_price_only",
            "cash_return": "aligned_DGS3MO_weekly_proxy",
        },
        "benchmark_contracts": {
            "risk_matched_60_40": {
                "method": "cash_scale_60_40_to_candidate_opening_weight_volatility",
                "covariance": "26_prior_completed_weeks_minimum_13",
                "candidate_basis": "actual_alpha_weights_after_holding_schedule_cost_screen_and_trade_cap",
                "maximum_risky_scale": 1.0,
                "leverage_allowed": False,
                "rebalancing": "same_core_risk_band_and_trade_cap",
                "exact_risk_match": False,
                "interpretation": "constrained_ex_ante_reference_realized_volatility_may_differ",
            },
            "initial_weight_buy_and_hold": "same_first_alpha_weights_no_subsequent_trades",
            "static_60_40_buy_and_hold": "initial_60_40_no_subsequent_trades",
        },
        "performance": {
            "evidence_track": "reconstructed_oos",
            "evaluation_start_week": index.min().date().isoformat(),
            "evaluation_end_week": index.max().date().isoformat(),
            "weeks": len(index),
            "strategies": m,
            "cost_sensitivity": metrics,
            "paired_mean_weekly_return_difference": float(paired.mean()),
            "paired_mean_weekly_return_difference_ci95": _mean_interval(paired, spec),
            "confidence_method": "matched_week_circular_block_bootstrap",
            "automatic_promotion_eligible": False,
        },
        "performance_path": [
            {
                "date": index[k].date().isoformat(),
                "week": index[k].date().isoformat(),
                "strategies": {mode: runs[primary][mode][k] for mode in modes},
            }
            for k in range(len(index))
        ],
        "economic_diagnostics": {
            "action_counts": {mode: m[mode]["action_counts"] for mode in modes},
            "alpha_screen_counts": dict(
                Counter(
                    s["alpha_screen"] for s in signals[:: int(spec["holding_weeks"])]
                )
            ),
            "attribution": {
                "initial_allocation_difference": m["initial_weight_buy_and_hold"][
                    "cumulative_return"
                ]
                - m["static_60_40_buy_and_hold"]["cumulative_return"],
                "dynamic_adjustment_vs_initial_buy_hold": m["regime_alpha"][
                    "cumulative_return"
                ]
                - m["initial_weight_buy_and_hold"]["cumulative_return"],
                "core_rebalancing_vs_buy_hold": m["core_60_40"]["cumulative_return"]
                - m["static_60_40_buy_and_hold"]["cumulative_return"],
                "alpha_policy_vs_core": m["regime_alpha"]["cumulative_return"]
                - m["core_60_40"]["cumulative_return"],
                "basis": "differences_of_net_cumulative_returns_not_additive_log_attribution",
            },
        },
    }
