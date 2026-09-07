"""Frozen grid evaluation of label construction, never automatic selection.

Entry quantiles are symmetric (q, 1-q); exit quantiles are (e, 1-e).
An exit quantile above one half intentionally permits state-dependent overlap.
Minimum duration is a minimum number of observed weeks before another switch.
The operating control keeps its own, unchanged label specification.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from regime_lab.analysis.labels import CausalRegimeLabeler
from regime_lab.analysis.label_spec import load_label_spec
from regime_lab.schema import STATE_ORDER

DIMENSIONS = (
    "trend_window_weeks",
    "stress_window_weeks",
    "entry_quantile",
    "exit_quantile",
    "hysteresis_minimum_weeks",
)
# Registered before any outcome is inspected; no best-result representative.
REPRESENTATIVES = ((13, 8, 0.25, 0.5, 2), (8, 4, 0.3, 0.4, 1), (26, 13, 0.2, 0.6, 4))


@dataclass(frozen=True)
class LabelSensitivityResult:
    summary: dict[str, Any]
    spec_metrics: pd.DataFrame
    representative_oos: pd.DataFrame


def _z(values: pd.Series, fit_weeks: int) -> pd.Series:
    train = values.iloc[:fit_weeks].dropna()
    centre = float(train.median())
    scale = float((train.quantile(0.75) - train.quantile(0.25)) / 1.349)
    if not np.isfinite(scale) or scale < 1e-12:
        scale = float((train - centre).abs().median() * 1.4826)
    return (values - centre) / (scale if np.isfinite(scale) and scale > 1e-12 else 1.0)


def _grid_labels(
    price: pd.Series, fit_weeks: int, params: tuple[Any, ...]
) -> pd.Series:
    trend_window, stress_window, entry, exit_quantile, minimum = params
    returns = np.log(price).diff()
    trend = np.log(price).diff(trend_window) / (
        returns.rolling(trend_window, min_periods=trend_window).std(ddof=0)
        * np.sqrt(trend_window)
    ).replace(0, np.nan)
    volatility = returns.rolling(stress_window, min_periods=stress_window).std(
        ddof=0
    ) * np.sqrt(52)
    drawdown = 1 - price / price.rolling(52, min_periods=52).max()
    stress = _z((_z(volatility, fit_weeks) + _z(drawdown, fit_weeks)) / 2, fit_weeks)
    score = _z(trend, fit_weeks) - stress
    train = score.iloc[:fit_weeks].dropna()
    lower, upper = float(train.quantile(entry)), float(train.quantile(1 - entry))
    off_exit, on_exit = (
        float(train.quantile(exit_quantile)),
        float(train.quantile(1 - exit_quantile)),
    )
    state, age, labels = "transition", 0, []
    for value in score:
        target = state
        if pd.notna(value) and age >= minimum:
            if value <= lower:
                target = "risk_off"
            elif value >= upper:
                target = "risk_on"
            elif state == "risk_off" and value > off_exit:
                target = "transition"
            elif state == "risk_on" and value < on_exit:
                target = "transition"
        age = age + 1 if target == state else 1
        state = target
        labels.append(state)
    return pd.Series(labels, index=price.index, dtype="object")


def _spec_id(params: tuple[Any, ...]) -> str:
    return "grid-" + "-".join(str(v).replace(".", "p") for v in params)


def _model_predictions(
    price: pd.Series, labels_by_spec: dict[str, pd.Series], positions: np.ndarray
) -> pd.DataFrame:
    returns = np.log(price).diff()
    features = pd.DataFrame(
        {
            "return_1w": returns,
            "return_4w": np.log(price).diff(4),
            "return_13w": np.log(price).diff(13),
            "volatility_13w": returns.rolling(13).std(ddof=0),
            "drawdown_52w": 1 - price / price.rolling(52).max(),
        }
    ).replace([np.inf, -np.inf], np.nan)
    output = []
    for spec, states in labels_by_spec.items():
        numeric_states = np.asarray([STATE_ORDER.index(s) for s in states])
        x = np.column_stack([features.to_numpy(float), np.eye(3)[numeric_states]])
        for t in positions:
            # Same strict one-week gap as the operating benchmark: last target t-1.
            stop = int(t) - 1
            target = numeric_states[1 : stop + 1]
            current = numeric_states[:stop]
            counts = np.ones((3, 3))
            np.add.at(counts, (current, target), 1)
            markov = counts[numeric_states[t]] / counts[numeric_states[t]].sum()
            train = x[:stop].copy()
            median = np.nanmedian(train, axis=0)
            median = np.nan_to_num(median)
            train = np.where(np.isnan(train), median, train)
            test = np.where(np.isnan(x[[t]]), median, x[[t]])
            scaler = StandardScaler().fit(train)
            ridge = np.full(3, 1e-9)
            fallback = len(np.unique(target)) < 2
            if fallback:
                ridge = (np.bincount(target, minlength=3) + 1) / (len(target) + 3)
            else:
                estimator = LogisticRegression(C=0.1, max_iter=500, tol=1e-6).fit(
                    scaler.transform(train), target
                )
                ridge[estimator.classes_.astype(int)] = estimator.predict_proba(
                    scaler.transform(test)
                )[0]
                ridge /= ridge.sum()
            persistence = np.full(3, 1e-9)
            persistence[numeric_states[t]] = 1 - 2e-9
            for name, probability in (
                ("markov", markov),
                ("current_state_ridge", ridge),
                ("persistence", persistence),
            ):
                output.append(
                    {
                        "spec_id": spec,
                        "model": name,
                        "origin_date": price.index[t].date().isoformat(),
                        "target_date": price.index[t + 1].date().isoformat(),
                        "last_train_target": price.index[t - 1].date().isoformat(),
                        "actual_state": states.iloc[t + 1],
                        "fallback": bool(fallback and name == "current_state_ridge"),
                        **{
                            f"p_{s}": float(probability[i])
                            for i, s in enumerate(STATE_ORDER)
                        },
                    }
                )
    return pd.DataFrame(output)


def run_label_sensitivity(
    canonical: pd.DataFrame,
    *,
    states: pd.Series | None = None,
    fit_weeks: int = 520,
    selection_end: str | pd.Timestamp = "2023-01-01",
    evaluate_model_ranks: bool = True,
    grid_path: str | Path | None = None,
) -> LabelSensitivityResult:
    """Execute all registered label variants and fixed representative models.

    Only selection-period outcomes are scored.  Appending holdout prices cannot
    alter these results.  The result is a research artifact, not a new label.
    """
    if grid_path is None:
        grid_path = (
            Path(__file__).resolve().parents[3] / "config/label-sensitivity-grid.json"
        )
    body = Path(grid_path).read_bytes()
    document = json.loads(body)
    if (
        not isinstance(canonical.index, pd.DatetimeIndex)
        or canonical.index.has_duplicates
        or not canonical.index.is_monotonic_increasing
    ):
        raise ValueError("canonical data require a unique ordered DatetimeIndex")
    price = pd.to_numeric(canonical["spy_close"], errors="coerce").astype(float)
    if len(price) <= fit_weeks + 13 or price.isna().any() or (price <= 0).any():
        raise ValueError(
            "label sensitivity requires positive prices and enough post-fit history"
        )
    cutoff = pd.Timestamp(selection_end)
    if price.index.tz is not None and cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize(price.index.tz)
    elif price.index.tz is None and cutoff.tzinfo is not None:
        cutoff = cutoff.tz_localize(None)
    # Cut before feature construction so no post-selection values enter hashes
    # or evaluation.  Training statistics stay frozen at the fit prefix.
    price = price.loc[price.index < cutoff]
    if len(price) <= fit_weeks + 13:
        raise ValueError("selection history is too short after the frozen label fit")
    if states is None:
        frame = canonical.loc[price.index]
        control = CausalRegimeLabeler().fit(frame.iloc[:fit_weeks]).transform(frame)
    else:
        control = states.reindex(price.index)
        if control.isna().any():
            raise ValueError("control labels must cover selection history")
    all_params = list(product(*(document["grid"][k] for k in DIMENSIONS)))
    labels_by_spec = {"operating_control": control}
    for params in all_params:
        labels_by_spec[_spec_id(params)] = _grid_labels(price, fit_weeks, params)
    positions = np.arange(fit_weeks, len(price))
    control_changes = set(
        np.flatnonzero(
            control.iloc[positions].ne(control.shift().iloc[positions]).to_numpy()
        ).tolist()
    )
    rows = []
    for spec, labels in labels_by_spec.items():
        evaluated = labels.iloc[positions]
        changes = set(
            np.flatnonzero(
                evaluated.ne(labels.shift().iloc[positions]).to_numpy()
            ).tolist()
        )
        occupancy = evaluated.value_counts(normalize=True)
        row: dict[str, Any] = {
            "spec_id": spec,
            "n_origins": len(evaluated),
            "episode_count": len(changes) + int(0 not in changes),
            "weekly_flip_rate": len(changes) / len(evaluated),
            "transition_jaccard": len(changes & control_changes)
            / len(changes | control_changes)
            if changes | control_changes
            else 1.0,
            "label_sha256": sha256("|".join(labels).encode()).hexdigest(),
            **{f"occupancy_{s}": float(occupancy.get(s, 0)) for s in STATE_ORDER},
        }
        for horizon in (1, 4, 13):
            future = price.shift(-horizon) / price - 1
            grouped = future.iloc[positions].groupby(evaluated).mean()
            for state in STATE_ORDER:
                values = future.iloc[positions][evaluated.eq(state)].dropna()
                row[f"{state}_mean_return_{horizon}w"] = (
                    float(grouped.get(state))
                    if state in grouped and pd.notna(grouped[state])
                    else None
                )
                row[f"{state}_downside_q10_{horizon}w"] = (
                    float(values.quantile(0.1)) if len(values) else None
                )
                row[f"{state}_return_n_{horizon}w"] = len(values)
        rows.append(row)
    representatives = {
        "operating_control": control,
        **{_spec_id(p): labels_by_spec[_spec_id(p)] for p in REPRESENTATIVES},
    }
    model_positions = np.arange(fit_weeks + 1, len(price) - 1)
    predictions = (
        _model_predictions(price, representatives, model_positions)
        if evaluate_model_ranks
        else pd.DataFrame()
    )
    rank_rows = []
    if not predictions.empty:
        for (spec, model), group in predictions.groupby(["spec_id", "model"]):
            matrix = group[[f"p_{s}" for s in STATE_ORDER]].to_numpy(float)
            actual = np.asarray([STATE_ORDER.index(s) for s in group["actual_state"]])
            rank_rows.append(
                {
                    "spec_id": spec,
                    "model": model,
                    "n_predictions": len(group),
                    "log_loss": float(
                        -np.log(
                            np.maximum(matrix[np.arange(len(matrix)), actual], 1e-12)
                        ).mean()
                    ),
                    "brier": float(
                        np.square(matrix - np.eye(3)[actual]).sum(axis=1).mean()
                    ),
                    "fallback_count": int(group["fallback"].sum()),
                }
            )
        ranks = pd.DataFrame(rank_rows)
        ranks["rank"] = (
            ranks.groupby("spec_id")["log_loss"].rank(method="min").astype(int)
        )
        rank_rows = ranks.to_dict(orient="records")
    specification = load_label_spec()
    metrics = pd.DataFrame(rows)
    summary = {
        "schema_version": "regime-label-sensitivity-summary/2",
        "status": "evaluated" if evaluate_model_ranks else "label_metrics_evaluated",
        "evidence_track": "reconstructed_oos",
        "evaluation_split": "selection_only",
        "control": {
            "spec_id": specification.spec_id,
            "spec_version": specification.version,
            "spec_sha256": specification.spec_sha256,
            "remains_operating_control": True,
        },
        "grid": {
            "path": "config/label-sensitivity-grid.json",
            "sha256": sha256(
                json.dumps(
                    document, sort_keys=True, separators=(",", ":"), ensure_ascii=False
                ).encode()
            ).hexdigest(),
            "dimensions": document["grid"],
        },
        "execution_summary": {
            "evaluated_spec_count": len(all_params),
            "state_occupancy": [
                {k: r[k] for k in ("spec_id", *[f"occupancy_{s}" for s in STATE_ORDER])}
                for r in rows
            ],
            "episode_count": [
                {"spec_id": r["spec_id"], "value": r["episode_count"]} for r in rows
            ],
            "weekly_flip_rate": [
                {"spec_id": r["spec_id"], "value": r["weekly_flip_rate"]} for r in rows
            ],
            "transition_jaccard": [
                {"spec_id": r["spec_id"], "value": r["transition_jaccard"]}
                for r in rows
            ],
            "forward_return_separation": [
                {
                    k: v
                    for k, v in r.items()
                    if k == "spec_id" or "return_" in k or "downside_" in k
                }
                for r in rows
            ],
            "model_rank_robustness": {
                "status": "evaluated" if evaluate_model_ranks else "not_executed",
                "representative_selection": "fixed_central_fast_slow_before_outcome_evaluation",
                "representative_specs": [
                    dict(zip(DIMENSIONS, p, strict=True)) for p in REPRESENTATIVES
                ],
                "models": ["markov", "current_state_ridge", "persistence"],
                "scope": "small_fixed_common_feature_benchmark_not_operating_champion_reselection",
                "rows": rank_rows,
            },
        },
        "execution": {
            "method_version": "symmetric_entry_exit_frozen_robust_score/1",
            "fit_weeks": fit_weeks,
            "fit_end": price.index[fit_weeks - 1].date().isoformat(),
            "selection_end": cutoff.date().isoformat(),
            "first_evaluation_origin": price.index[fit_weeks].date().isoformat(),
            "last_evaluation_origin": price.index[-1].date().isoformat(),
            "source_price_sha256": sha256(
                pd.util.hash_pandas_object(price).values.tobytes()
            ).hexdigest(),
            "control_state_sha256": sha256(
                pd.util.hash_pandas_object(control).values.tobytes()
            ).hexdigest(),
            "implementation_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
            "return_semantics": "close_to_close_label_validity_not_executable_strategy",
            "model_gap_weeks": 1,
        },
        "automatic_promotion_eligible": False,
    }
    return LabelSensitivityResult(summary, metrics, predictions)
