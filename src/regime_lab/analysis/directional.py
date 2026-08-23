"""Causal first-departure direction benchmark for the v5 research contract."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Iterable, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


STATE_ORDER = ("risk_on", "transition", "risk_off")
OUTCOME_ORDER = ("no_departure", *STATE_ORDER)
BASELINE_MODELS = ("empirical_first_passage", "markov_first_passage")
EPSILON = 1e-8
SELECTION_BLOCK_WEEKS = 13
MINIMUM_SELECTION_EVENTS = 8
MINIMUM_DESTINATION_CLASSES = 2
MINIMUM_EVENT_BLOCKS = 3


@dataclass(frozen=True)
class DirectionalBenchmarkResult:
    leaderboard: pd.DataFrame
    predictions: pd.DataFrame
    split_audit: pd.DataFrame
    selection_diagnostics: pd.DataFrame
    champions_by_horizon: Mapping[int, str]
    latest_forecasts: pd.DataFrame
    selection_end: pd.Timestamp


def _validate_states(states: pd.Series) -> pd.Series:
    if not isinstance(states, pd.Series) or not isinstance(
        states.index, pd.DatetimeIndex
    ):
        raise TypeError("states must be a Series with a DatetimeIndex")
    if states.empty or not states.index.is_monotonic_increasing or states.index.has_duplicates:
        raise ValueError("states must be non-empty, unique, and increasing")
    values = states.astype(str)
    unknown = sorted(set(values).difference(STATE_ORDER))
    if unknown:
        raise ValueError(f"unsupported states: {unknown}")
    return values


def first_departure_targets(states: pd.Series, horizon: int) -> pd.DataFrame:
    """Return first-departure destinations or no-departure for each known origin."""

    states = _validate_states(states)
    if isinstance(horizon, bool) or int(horizon) != horizon or int(horizon) < 1:
        raise ValueError("horizon must be a positive integer")
    horizon = int(horizon)
    if horizon >= len(states):
        raise ValueError("horizon must be shorter than state history")
    values = states.to_numpy(dtype=object)
    rows: list[dict[str, object]] = []
    for origin in range(len(states) - horizon):
        current = str(values[origin])
        future = values[origin + 1 : origin + horizon + 1]
        departures = np.flatnonzero(future != current)
        outcome = (
            str(future[int(departures[0])]) if len(departures) else "no_departure"
        )
        rows.append(
            {
                "origin_position": origin,
                "origin_date": pd.Timestamp(states.index[origin]),
                "target_end": pd.Timestamp(states.index[origin + horizon]),
                "current_state": current,
                "outcome": outcome,
                "actual_change": outcome != "no_departure",
            }
        )
    return pd.DataFrame(rows).set_index("origin_position", drop=False)


def _normalized(values: Mapping[str, float], current_state: str) -> dict[str, float]:
    output = {name: max(0.0, float(values.get(name, 0.0))) for name in OUTCOME_ORDER}
    output[current_state] = 0.0
    total = float(sum(output.values()))
    if not np.isfinite(total) or total <= 0.0:
        output = {name: 0.0 for name in OUTCOME_ORDER}
        output["no_departure"] = 1.0
        return output
    return {name: float(value / total) for name, value in output.items()}


def empirical_first_passage_probabilities(
    targets: pd.DataFrame,
    current_state: str,
    *,
    alpha: float = 1.0,
) -> dict[str, float]:
    """Smoothed state-conditioned first-departure outcome probabilities."""

    if current_state not in STATE_ORDER:
        raise ValueError("current_state is invalid")
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    eligible = targets.loc[targets["current_state"].astype(str).eq(current_state)]
    allowed = ("no_departure", *(state for state in STATE_ORDER if state != current_state))
    counts = eligible["outcome"].astype(str).value_counts()
    values = {
        outcome: float(counts.get(outcome, 0)) + float(alpha)
        for outcome in allowed
    }
    return _normalized(values, current_state)


def markov_first_passage_probabilities(
    states: pd.Series,
    current_state: str,
    horizon: int,
    *,
    alpha: float = 1.0,
) -> dict[str, float]:
    """Homogeneous Markov first-exit distribution through a fixed horizon."""

    states = _validate_states(states)
    if current_state not in STATE_ORDER:
        raise ValueError("current_state is invalid")
    if isinstance(horizon, bool) or int(horizon) != horizon or int(horizon) < 1:
        raise ValueError("horizon must be a positive integer")
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    current = states.iloc[:-1].astype(str)
    following = states.iloc[1:].astype(str).to_numpy()
    mask = current.eq(current_state).to_numpy()
    counts = {
        destination: float(np.count_nonzero(following[mask] == destination)) + alpha
        for destination in STATE_ORDER
    }
    total = float(sum(counts.values()))
    one_step = {state: counts[state] / total for state in STATE_ORDER}
    stay = float(one_step[current_state]) ** int(horizon)
    exit_mass = 1.0 - stay
    one_step_exit = 1.0 - float(one_step[current_state])
    values: dict[str, float] = {"no_departure": stay, current_state: 0.0}
    for destination in STATE_ORDER:
        if destination == current_state:
            continue
        values[destination] = exit_mass * float(one_step[destination]) / one_step_exit
    return _normalized(values, current_state)


def _design_frame(features: pd.DataFrame, states: pd.Series) -> pd.DataFrame:
    numeric = features.astype(float).replace([np.inf, -np.inf], np.nan).copy()
    for state in STATE_ORDER:
        numeric[f"current_state__{state}"] = states.astype(str).eq(state).astype(float)
    durations: list[int] = []
    previous: str | None = None
    elapsed = 0
    for value in states.astype(str):
        elapsed = elapsed + 1 if value == previous else 1
        durations.append(elapsed)
        previous = value
    numeric["current_duration_weeks"] = np.asarray(durations, dtype=float)
    return numeric


def _regularized_logistic(random_state: int) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median", add_indicator=True, keep_empty_features=True
                ),
            ),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=0.05,
                    solver="lbfgs",
                    max_iter=2_000,
                    tol=1e-7,
                    random_state=random_state,
                ),
            ),
        ]
    )


def _shallow_xgboost(random_state: int):
    from xgboost import XGBClassifier

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median", add_indicator=True, keep_empty_features=True
                ),
            ),
            (
                "classifier",
                XGBClassifier(
                    objective="multi:softprob",
                    num_class=len(OUTCOME_ORDER),
                    eval_metric="mlogloss",
                    n_estimators=80,
                    max_depth=2,
                    learning_rate=0.04,
                    min_child_weight=8.0,
                    subsample=0.85,
                    colsample_bytree=0.65,
                    reg_alpha=1.0,
                    reg_lambda=8.0,
                    tree_method="hist",
                    n_jobs=1,
                    random_state=random_state,
                ),
            ),
        ]
    )


def _fit_candidate(
    name: str,
    design: pd.DataFrame,
    states: pd.Series,
    targets: pd.DataFrame,
    *,
    current_state: str,
    horizon: int,
    train_stop: int,
    test_position: int,
    random_state: int,
) -> tuple[dict[str, float], bool, str]:
    train_targets = targets.loc[targets["origin_position"] < train_stop]
    fallback = empirical_first_passage_probabilities(train_targets, current_state)
    try:
        if name == "empirical_first_passage":
            return fallback, False, ""
        if name == "markov_first_passage":
            return (
                markov_first_passage_probabilities(
                    states.iloc[:train_stop], current_state, horizon
                ),
                False,
                "",
            )
        if name not in {"regularized_multinomial", "shallow_multiclass_xgboost"}:
            raise ValueError(f"unknown directional model: {name}")
        y = train_targets["outcome"].astype(str)
        if y.nunique() < 2:
            raise ValueError("directional target has fewer than two classes")
        x_train = design.iloc[train_targets["origin_position"].to_numpy(dtype=int)]
        x_test = design.iloc[[test_position]]
        if name == "regularized_multinomial":
            estimator = _regularized_logistic(random_state)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                estimator.fit(x_train, y)
            classes = tuple(str(value) for value in estimator[-1].classes_)
        else:
            encoded = y.map({name: index for index, name in enumerate(OUTCOME_ORDER)})
            if encoded.isna().any() or set(encoded.astype(int)) != set(range(len(OUTCOME_ORDER))):
                raise ValueError("xgboost requires every directional class in training")
            estimator = _shallow_xgboost(random_state)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                estimator.fit(x_train, encoded.astype(int))
            classes = OUTCOME_ORDER
        vector = estimator.predict_proba(x_test)[0]
        values = {outcome: 0.0 for outcome in OUTCOME_ORDER}
        for class_name, value in zip(classes, vector, strict=True):
            values[str(class_name)] = float(value)
        return _normalized(values, current_state), False, ""
    except (ImportError, OSError, ValueError, RuntimeError, FloatingPointError) as exc:
        return fallback, True, f"{type(exc).__name__}: {exc}"


def _conditional_destination_matrix(rows: pd.DataFrame) -> np.ndarray:
    """Return the deployed destination distribution conditional on departure."""

    matrix = rows[[f"p_{name}" for name in STATE_ORDER]].to_numpy(float)
    currents = rows["current_state"].astype(str).to_numpy()
    positions = {name: index for index, name in enumerate(STATE_ORDER)}
    output = np.zeros_like(matrix)
    for row_index, current_state in enumerate(currents):
        vector = np.maximum(matrix[row_index], 0.0)
        vector[positions[current_state]] = 0.0
        total = float(vector.sum())
        if not np.isfinite(total) or total <= 0.0:
            vector = np.asarray(
                [0.0 if state == current_state else 0.5 for state in STATE_ORDER],
                dtype=float,
            )
        else:
            vector = vector / total
        output[row_index] = vector
    return output


def _event_support(rows: pd.DataFrame) -> dict[str, int]:
    ordered = rows.sort_values("origin_date").reset_index(drop=True)
    events = ordered["actual_change"].astype(bool).to_numpy()
    event_blocks = {
        position // SELECTION_BLOCK_WEEKS
        for position in np.flatnonzero(events)
    }
    destinations = ordered.loc[events, "actual_outcome"].astype(str)
    return {
        "event_count": int(events.sum()),
        "destination_class_count": int(destinations.nunique()),
        "effective_event_blocks": int(len(event_blocks)),
    }


def _conditional_losses(rows: pd.DataFrame) -> np.ndarray:
    ordered = rows.sort_values("origin_date").reset_index(drop=True)
    losses = np.full(len(ordered), np.nan, dtype=float)
    event_mask = ordered["actual_change"].astype(bool).to_numpy()
    if not event_mask.any():
        return losses
    matrix = _conditional_destination_matrix(ordered)
    positions = {name: index for index, name in enumerate(STATE_ORDER)}
    actual = ordered["actual_outcome"].astype(str).to_numpy()
    event_positions = np.flatnonzero(event_mask)
    actual_p = np.asarray(
        [matrix[row, positions[actual[row]]] for row in event_positions],
        dtype=float,
    )
    losses[event_positions] = -np.log(np.clip(actual_p, EPSILON, 1.0))
    return losses


def _evaluate(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (horizon, split, model), group in predictions.groupby(
        ["horizon_weeks", "evaluation_split", "model"], sort=False
    ):
        scored = group.loc[group["actual_change"].astype(bool)].copy()
        support = _event_support(group)
        if scored.empty:
            log_loss = float("nan")
            brier = float("nan")
        else:
            matrix = _conditional_destination_matrix(scored)
            actual = scored["actual_outcome"].astype(str).to_numpy()
            positions = {name: index for index, name in enumerate(STATE_ORDER)}
            actual_p = np.asarray(
                [matrix[row, positions[name]] for row, name in enumerate(actual)]
            )
            one_hot = np.zeros_like(matrix)
            one_hot[
                np.arange(len(scored)), [positions[name] for name in actual]
            ] = 1.0
            log_loss = float(-np.log(np.clip(actual_p, EPSILON, 1.0)).mean())
            brier = float(np.square(matrix - one_hot).sum(axis=1).mean())
        rows.append(
            {
                "horizon_weeks": int(horizon),
                "evaluation_split": str(split),
                "model": str(model),
                "score_target": "first_destination_given_departure",
                "log_loss": log_loss,
                "brier": brier,
                "n_predictions": int(len(group)),
                **support,
                "fallback_count": int(group["fallback"].astype(bool).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["horizon_weeks", "evaluation_split", "log_loss", "brier", "model"],
        ignore_index=True,
    )


def _bootstrap_pvalue(
    values: np.ndarray,
    *,
    seed: int,
    resamples: int = 999,
) -> float | None:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return None
    observed = float(values[finite].mean())
    centered = values.copy()
    centered[finite] -= observed
    block = min(SELECTION_BLOCK_WEEKS, max(1, len(values) // 2))
    blocks = int(np.ceil(len(values) / block))
    generator = np.random.default_rng(seed)
    starts = generator.integers(0, len(values), size=(resamples, blocks))
    offsets = np.arange(block)
    indexes = (starts[..., None] + offsets) % len(values)
    indexes = indexes.reshape(resamples, -1)[:, : len(values)]
    sampled = centered[indexes]
    counts = np.isfinite(sampled).sum(axis=1)
    valid = counts > 0
    if int(valid.sum()) < int(np.ceil(resamples * 0.8)):
        return None
    null = np.nansum(sampled[valid], axis=1) / counts[valid]
    return float((1 + np.count_nonzero(null >= observed)) / (len(null) + 1))


def _select_horizon(
    predictions: pd.DataFrame,
    horizon: int,
    *,
    minimum_selection_events: int,
    minimum_destination_classes: int,
    minimum_event_blocks: int,
) -> tuple[str, list[dict[str, object]]]:
    frame = predictions.loc[
        predictions["horizon_weeks"].eq(horizon)
        & predictions["evaluation_split"].eq("selection")
    ].copy()
    table = _evaluate(frame).set_index("model", drop=False)
    baseline_rows = (
        table.loc[[name for name in BASELINE_MODELS if name in table.index]]
        .reset_index(drop=True)
    )
    reference_name = (
        "empirical_first_passage"
        if "empirical_first_passage" in table.index
        else str(baseline_rows.sort_values("model").iloc[0]["model"])
    )
    support = _event_support(
        frame.loc[frame["model"].eq(reference_name)]
    )
    support_failures: list[str] = []
    if support["event_count"] < minimum_selection_events:
        support_failures.append("insufficient_departure_events")
    if support["destination_class_count"] < minimum_destination_classes:
        support_failures.append("insufficient_destination_classes")
    if support["effective_event_blocks"] < minimum_event_blocks:
        support_failures.append("insufficient_event_blocks")
    if support_failures:
        diagnostics = []
        reason = ";".join(support_failures)
        for model, row in table.iterrows():
            diagnostics.append(
                {
                    "horizon_weeks": horizon,
                    "model": str(model),
                    "reference_model": reference_name,
                    "selected": str(model) == reference_name,
                    "gate_passed": False,
                    "gate_reason": reason,
                    "score_target": "first_destination_given_departure",
                    "selection_event_count": support["event_count"],
                    "selection_destination_class_count": support[
                        "destination_class_count"
                    ],
                    "selection_effective_event_blocks": support[
                        "effective_event_blocks"
                    ],
                    "minimum_selection_events": minimum_selection_events,
                    "minimum_destination_classes": minimum_destination_classes,
                    "minimum_event_blocks": minimum_event_blocks,
                    "log_loss": None
                    if not np.isfinite(float(row["log_loss"]))
                    else float(row["log_loss"]),
                    "brier": None
                    if not np.isfinite(float(row["brier"]))
                    else float(row["brier"]),
                    "absolute_log_loss_improvement": None,
                    "holm_adjusted_p_value": None,
                    "fallback_count": int(row["fallback_count"]),
                }
            )
        return reference_name, diagnostics

    baseline = baseline_rows.sort_values(
        ["log_loss", "brier", "model"], na_position="last"
    ).iloc[0]
    reference = str(baseline["model"])
    reference_rows = frame.loc[frame["model"].eq(reference)].sort_values("origin_date")

    def losses(rows: pd.DataFrame) -> np.ndarray:
        return _conditional_losses(rows)

    reference_loss = losses(reference_rows)
    raw_pvalues: dict[str, float] = {}
    improvements: dict[str, float] = {}
    for model in table.index:
        if model in BASELINE_MODELS:
            continue
        candidate_loss = losses(frame.loc[frame["model"].eq(model)])
        differential = reference_loss - candidate_loss
        improvements[str(model)] = float(np.nanmean(differential))
        pvalue = _bootstrap_pvalue(differential, seed=17)
        if pvalue is not None:
            raw_pvalues[str(model)] = pvalue
    ordered_p = sorted(raw_pvalues.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (model, value) in enumerate(ordered_p):
        running = max(running, min(1.0, value * (len(ordered_p) - rank)))
        adjusted[model] = running

    diagnostics: list[dict[str, object]] = []
    passing: list[str] = []
    for model, row in table.iterrows():
        failures: list[str] = []
        if model not in BASELINE_MODELS:
            if int(row["fallback_count"]) != 0:
                failures.append("fallback_present")
            if improvements[model] < 0.05:
                failures.append("insufficient_log_loss_improvement")
            if float(row["brier"]) > float(baseline["brier"]) + 0.01:
                failures.append("brier_degradation")
            if model not in adjusted:
                failures.append("bootstrap_insufficient")
            elif adjusted[model] > 0.05:
                failures.append("holm_not_significant")
            if not failures:
                passing.append(str(model))
        elif model != reference:
            failures.append("non_reference_baseline")
        diagnostics.append(
            {
                "horizon_weeks": horizon,
                "model": str(model),
                "reference_model": reference,
                "selected": False,
                "gate_passed": not failures,
                "gate_reason": "passed" if not failures else ";".join(failures),
                "score_target": "first_destination_given_departure",
                "selection_event_count": support["event_count"],
                "selection_destination_class_count": support[
                    "destination_class_count"
                ],
                "selection_effective_event_blocks": support[
                    "effective_event_blocks"
                ],
                "minimum_selection_events": minimum_selection_events,
                "minimum_destination_classes": minimum_destination_classes,
                "minimum_event_blocks": minimum_event_blocks,
                "log_loss": float(row["log_loss"]),
                "brier": float(row["brier"]),
                "absolute_log_loss_improvement": float(
                    float(baseline["log_loss"]) - float(row["log_loss"])
                ),
                "holm_adjusted_p_value": adjusted.get(str(model)),
                "fallback_count": int(row["fallback_count"]),
            }
        )
    champion = reference
    if passing:
        champion = str(
            table.loc[passing]
            .reset_index(drop=True)
            .sort_values(["log_loss", "brier", "model"])
            .iloc[0]["model"]
        )
    for row in diagnostics:
        row["selected"] = row["model"] == champion
    return champion, diagnostics


def run_directional_transition_benchmark(
    features: pd.DataFrame,
    states: pd.Series,
    *,
    horizons: Iterable[int] = (1, 4, 13),
    models: Sequence[str] = (
        "empirical_first_passage",
        "markov_first_passage",
        "regularized_multinomial",
        "shallow_multiclass_xgboost",
    ),
    minimum_train_weeks: int = 520,
    selection_end: str | pd.Timestamp = "2023-01-01",
    minimum_selection_predictions: int = 12,
    minimum_diagnostic_predictions: int = 12,
    maximum_diagnostic_origins: int | None = None,
    selection_max_origins: int | None = None,
    minimum_selection_events: int = MINIMUM_SELECTION_EVENTS,
    minimum_destination_classes: int = MINIMUM_DESTINATION_CLASSES,
    minimum_event_blocks: int = MINIMUM_EVENT_BLOCKS,
    random_state: int = 17,
) -> DirectionalBenchmarkResult:
    """Evaluate destination/no-departure outcomes with strict horizon purging."""

    states = _validate_states(states)
    if not features.index.equals(states.index):
        raise ValueError("features and states must have identical indexes")
    horizons = tuple(dict.fromkeys(int(value) for value in horizons))
    if not horizons or any(value < 1 for value in horizons):
        raise ValueError("horizons must contain positive integers")
    if minimum_train_weeks < 12:
        raise ValueError("minimum_train_weeks must be at least 12")
    if minimum_selection_events < 1:
        raise ValueError("minimum_selection_events must be positive")
    if not 1 <= minimum_destination_classes <= len(STATE_ORDER):
        raise ValueError("minimum_destination_classes is invalid")
    if minimum_event_blocks < 1:
        raise ValueError("minimum_event_blocks must be positive")
    unknown = sorted(
        set(models).difference(
            {
                "empirical_first_passage",
                "markov_first_passage",
                "regularized_multinomial",
                "shallow_multiclass_xgboost",
            }
        )
    )
    if unknown:
        raise ValueError(f"unknown directional models: {unknown}")
    cutoff = pd.Timestamp(selection_end)
    if states.index.tz is None and cutoff.tzinfo is not None:
        cutoff = cutoff.tz_localize(None)
    elif states.index.tz is not None and cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize(states.index.tz)
    elif states.index.tz is not None and cutoff.tzinfo is not None:
        cutoff = cutoff.tz_convert(states.index.tz)
    design = _design_frame(features, states)
    rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    target_cache = {horizon: first_departure_targets(states, horizon) for horizon in horizons}
    for horizon in horizons:
        targets = target_cache[horizon]
        available = list(range(minimum_train_weeks + horizon, len(states) - horizon))
        selection = [
            position
            for position in available
            if pd.Timestamp(states.index[position + horizon]) < cutoff
        ]
        diagnostic = [
            position
            for position in available
            if pd.Timestamp(states.index[position]) >= cutoff
        ]
        if len(selection) < minimum_selection_predictions:
            raise ValueError(f"insufficient horizon-{horizon} selection origins")
        if len(diagnostic) < minimum_diagnostic_predictions:
            raise ValueError(f"insufficient horizon-{horizon} diagnostic origins")
        if selection_max_origins is not None:
            selection = selection[-max(selection_max_origins, minimum_selection_predictions) :]
        if maximum_diagnostic_origins is not None:
            diagnostic = diagnostic[-maximum_diagnostic_origins:]
        positions = sorted((*selection, *diagnostic))
        selection_set = set(selection)
        for position in positions:
            train_stop = position - horizon
            target = targets.loc[position]
            origin = pd.Timestamp(states.index[position])
            split = "selection" if position in selection_set else "retrospective_diagnostic"
            split_rows.append(
                {
                    "horizon_weeks": horizon,
                    "origin_date": origin,
                    "target_end": pd.Timestamp(states.index[position + horizon]),
                    "evaluation_split": split,
                    "train_size": train_stop,
                    "last_train_origin": pd.Timestamp(states.index[train_stop - 1]),
                    "last_train_target_end": pd.Timestamp(states.index[position - 1]),
                    "purged_origin_count": horizon,
                }
            )
            for name in models:
                probability, fallback, reason = _fit_candidate(
                    name,
                    design,
                    states,
                    targets,
                    current_state=str(states.iloc[position]),
                    horizon=horizon,
                    train_stop=train_stop,
                    test_position=position,
                    random_state=random_state,
                )
                rows.append(
                    {
                        "horizon_weeks": horizon,
                        "origin_date": origin,
                        "target_end": pd.Timestamp(states.index[position + horizon]),
                        "evaluation_split": split,
                        "model": name,
                        "current_state": str(states.iloc[position]),
                        "actual_outcome": str(target["outcome"]),
                        "actual_change": bool(target["actual_change"]),
                        **{f"p_{key}": probability[key] for key in OUTCOME_ORDER},
                        "fallback": fallback,
                        "fallback_reason": reason,
                    }
                )
    predictions = pd.DataFrame(rows).sort_values(
        ["horizon_weeks", "origin_date", "model"], ignore_index=True
    )
    champions: dict[int, str] = {}
    diagnostics: list[dict[str, object]] = []
    for horizon in horizons:
        champion, horizon_diagnostics = _select_horizon(
            predictions,
            horizon,
            minimum_selection_events=minimum_selection_events,
            minimum_destination_classes=minimum_destination_classes,
            minimum_event_blocks=minimum_event_blocks,
        )
        champions[horizon] = champion
        diagnostics.extend(horizon_diagnostics)
    leaderboard = _evaluate(predictions)
    leaderboard.insert(
        3,
        "selected",
        [
            str(model) == champions[int(horizon)]
            for horizon, model in zip(
                leaderboard["horizon_weeks"], leaderboard["model"], strict=True
            )
        ],
    )

    latest_rows: list[dict[str, object]] = []
    for horizon in horizons:
        targets = target_cache[horizon]
        name = champions[horizon]
        for position in range(len(states) - horizon, len(states)):
            train_stop = position - horizon
            probability, fallback, reason = _fit_candidate(
                name,
                design,
                states,
                targets,
                current_state=str(states.iloc[position]),
                horizon=horizon,
                train_stop=train_stop,
                test_position=position,
                random_state=random_state,
            )
            beyond = position + horizon - (len(states) - 1)
            target_end = (
                pd.Timestamp(states.index[position + horizon])
                if position + horizon < len(states)
                else pd.Timestamp(states.index[-1]) + timedelta(weeks=beyond)
            )
            latest_rows.append(
                {
                    "horizon_weeks": horizon,
                    "origin_date": pd.Timestamp(states.index[position]),
                    "target_end": target_end,
                    "model": name,
                    "current_state": str(states.iloc[position]),
                    **{f"p_{key}": probability[key] for key in OUTCOME_ORDER},
                    "fallback": fallback,
                    "fallback_reason": reason,
                }
            )
    return DirectionalBenchmarkResult(
        leaderboard=leaderboard,
        predictions=predictions,
        split_audit=pd.DataFrame(split_rows).drop_duplicates().sort_values(
            ["horizon_weeks", "origin_date"], ignore_index=True
        ),
        selection_diagnostics=pd.DataFrame(diagnostics).sort_values(
            ["horizon_weeks", "model"], ignore_index=True
        ),
        champions_by_horizon=champions,
        latest_forecasts=pd.DataFrame(latest_rows).sort_values(
            ["horizon_weeks", "origin_date"], ignore_index=True
        ),
        selection_end=cutoff,
    )


def reconcile_directional_risk(
    p_change: float,
    current_state: str,
    probabilities: Mapping[str, float],
) -> dict[str, object]:
    """Keep the established departure risk and use this model only for direction."""

    if current_state not in STATE_ORDER:
        raise ValueError("current_state is invalid")
    if not np.isfinite(float(p_change)) or not 0.0 <= float(p_change) <= 1.0:
        raise ValueError("p_change must be in [0, 1]")
    destination = {
        state: 0.0 if state == current_state else max(0.0, float(probabilities.get(state, 0.0)))
        for state in STATE_ORDER
    }
    total = float(sum(destination.values()))
    if total <= 0.0:
        alternatives = [state for state in STATE_ORDER if state != current_state]
        destination = {
            state: (1.0 / len(alternatives) if state in alternatives else 0.0)
            for state in STATE_ORDER
        }
    else:
        destination = {state: value / total for state, value in destination.items()}
    scaled = {state: round(float(p_change) * value, 8) for state, value in destination.items()}
    residual = round(float(p_change) - sum(scaled.values()), 8)
    if residual:
        chosen = max((state for state in STATE_ORDER if state != current_state), key=scaled.get)
        scaled[chosen] = round(scaled[chosen] + residual, 8)
    return {
        "no_departure": round(1.0 - float(p_change), 8),
        "first_destination": scaled,
        "definition": "first_departure_state_within_h_or_no_departure",
    }


__all__ = [
    "BASELINE_MODELS",
    "DirectionalBenchmarkResult",
    "OUTCOME_ORDER",
    "STATE_ORDER",
    "empirical_first_passage_probabilities",
    "first_departure_targets",
    "markov_first_passage_probabilities",
    "reconcile_directional_risk",
    "run_directional_transition_benchmark",
]
