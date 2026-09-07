"""Next-week forecasts from the frozen label's causal price mechanics.

The official state machine has hysteresis, but no confirmation streak.  A
known rolling observation drops out next week; conditional on next week's
return, the resulting score and state are deterministic.  Integrating over a
return distribution models this mechanism directly without changing labels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.optimize import minimize_scalar
from scipy.stats import t as student_t
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from regime_lab.analysis.labels import CausalRegimeLabeler, RegimeLabelConfig
from regime_lab.analysis.models import StateEncodedXGBoostClassifier
from regime_lab.schema import STATE_ORDER

MODELS = (
    "boundary_student_t",
    "boundary_filtered_history",
    "boundary_state_logistic",
    "boundary_shallow_xgb",
    "boundary_mechanistic_blend",
)
PROBABILITY_COLUMNS = tuple(f"p_{state}" for state in STATE_ORDER)
ASYMMETRIC_MODEL = "boundary_asymmetric_ewma"


@dataclass(frozen=True)
class BoundaryConfig:
    """Versioned units/recipe; v1 defaults reproduce the existing boundary model."""

    version: str = "boundary-v1"
    volatility_span_weeks: int = 13
    volatility_floor: float = .003  # weekly log-return standard deviation
    residual_window_weeks: int = 520
    residual_clip: float = 8.
    student_df: float = 5.
    integration_points: int = 1001
    contamination: float = .01
    calibration_minimum_rows: int = 52
    calibration_maximum_rows: int = 156
    temperature_minimum: float = .5
    temperature_maximum: float = 2.
    temperature_penalty: float = .01
    asymmetric_strength: float = .5

    def __post_init__(self):
        if self.version != "boundary-v1":
            raise ValueError("unsupported boundary config version")
        for key in ("volatility_span_weeks", "residual_window_weeks", "integration_points",
                    "calibration_minimum_rows", "calibration_maximum_rows"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        scalars = (self.volatility_floor, self.residual_clip, self.student_df,
                   self.contamination, self.temperature_minimum, self.temperature_maximum,
                   self.temperature_penalty, self.asymmetric_strength)
        if not np.isfinite(scalars).all():
            raise ValueError("boundary parameters must be finite")
        if self.volatility_floor <= 0 or self.residual_clip <= 0 or self.student_df <= 2:
            raise ValueError("invalid boundary distribution scale")
        if not 0 <= self.contamination < 1 or not 0 <= self.asymmetric_strength < 1:
            raise ValueError("invalid contamination/asymmetric strength")
        if not 0 < self.temperature_minimum <= 1 <= self.temperature_maximum:
            raise ValueError("temperature interval must contain identity")
        if self.temperature_penalty < 0 or self.calibration_minimum_rows > self.calibration_maximum_rows:
            raise ValueError("invalid calibration settings")

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_BOUNDARY_CONFIG = BoundaryConfig()


def asymmetric_volatility(returns: pd.Series, config: BoundaryConfig = DEFAULT_BOUNDARY_CONFIG) -> pd.Series:
    """One fixed GJR-style EWMA: negative/positive squared-shock weights 1.5/.5.

    Zero conditional drift, no optimization; symmetric innovations have unit
    expected weight. The observed origin return updates next-week variance.
    """
    squared = returns.pow(2) * np.where(returns < 0, 1 + config.asymmetric_strength,
                                      1 - config.asymmetric_strength)
    return np.sqrt(squared.ewm(span=config.volatility_span_weeks, adjust=False).mean()).clip(lower=config.volatility_floor)


def next_scores(
    history: np.ndarray, future_log_returns: np.ndarray, labeler: CausalRegimeLabeler
) -> np.ndarray:
    """Vectorize the official next score for arbitrary hypothetical returns."""
    prices = np.asarray(history, dtype=float)
    shocks = np.asarray(future_log_returns, dtype=float)
    if len(prices) < 52 or not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("at least 52 finite positive historical prices are required")
    if shocks.ndim != 1 or not np.isfinite(shocks).all():
        raise ValueError("future returns must be a finite vector")
    labeler._require_fitted()
    assert labeler.component_stats_ is not None
    assert labeler.composite_stats_ is not None
    log_prices = np.log(prices)
    returns = np.diff(log_prices)
    new_log_price = log_prices[-1] + shocks
    raw: dict[str, np.ndarray] = {}
    for window in (13, 26):
        sample = np.column_stack(
            [np.broadcast_to(returns[-(window - 1):], (len(shocks), window - 1)), shocks]
        )
        sigma = sample.std(axis=1, ddof=0)
        raw[f"trend_{window}w"] = np.divide(
            new_log_price - log_prices[-window], sigma * np.sqrt(window),
            out=np.full(len(shocks), np.nan), where=sigma > 0,
        )
    for window in (4, 13):
        sample = np.column_stack(
            [np.broadcast_to(returns[-(window - 1):], (len(shocks), window - 1)), shocks]
        )
        raw[f"vol_{window}w"] = sample.std(axis=1, ddof=0) * np.sqrt(52)
    new_price = np.exp(new_log_price)
    for window in (13, 52):
        peak = np.maximum(prices[-(window - 1):].max(), new_price)
        raw[f"drawdown_{window}w"] = 1 - new_price / peak
    scaled = {
        key: (values - labeler.component_stats_[key][0]) / labeler.component_stats_[key][1]
        for key, values in raw.items()
    }
    trend_values = np.column_stack([scaled["trend_13w"], scaled["trend_26w"]])
    finite_trends = np.isfinite(trend_values).sum(axis=1)
    trend = np.divide(np.nansum(trend_values, axis=1), finite_trends,
                      out=np.full(len(shocks), np.nan), where=finite_trends > 0)
    stress = sum(scaled[key] for key in ("vol_4w", "vol_13w", "drawdown_13w", "drawdown_52w")) / 4
    trend_centre, trend_scale = labeler.composite_stats_["trend"]
    stress_centre, stress_scale = labeler.composite_stats_["stress"]
    return (trend - trend_centre) / trend_scale - (stress - stress_centre) / stress_scale


def next_states(scores: np.ndarray, current_state: str, labeler: CausalRegimeLabeler) -> np.ndarray:
    """Apply the exact official hysteresis inequalities from the current state."""
    if current_state not in STATE_ORDER:
        raise ValueError("unsupported current state")
    lower, upper = labeler.lower_threshold_, labeler.upper_threshold_
    assert lower is not None and upper is not None
    margin = (upper - lower) * labeler.config.hysteresis_fraction
    values = np.asarray(scores, dtype=float)
    output = np.full(len(values), current_state, dtype=object)
    if current_state == "transition":
        output[values <= lower] = "risk_off"
        output[values >= upper] = "risk_on"
    elif current_state == "risk_on":
        output[values < upper - margin] = "transition"
        output[values <= lower - margin] = "risk_off"
    else:
        output[values > lower + margin] = "transition"
        output[values >= upper + margin] = "risk_on"
    return output


def state_distribution(scores: np.ndarray, current_state: str, labeler: CausalRegimeLabeler,
                       config: BoundaryConfig = DEFAULT_BOUNDARY_CONFIG) -> np.ndarray:
    values = next_states(scores, current_state, labeler)
    # A fixed 1% uniform contamination protects finite-grid tail events.  This
    # value is specified before evaluation and is never optimized on holdout.
    return (1 - config.contamination) * np.asarray([(values == state).mean() for state in STATE_ORDER]) + config.contamination / 3


@dataclass(frozen=True)
class BoundaryInputs:
    features: pd.DataFrame
    mechanistic: dict[str, pd.DataFrame]
    labeler: CausalRegimeLabeler
    states: pd.Series
    config: BoundaryConfig = DEFAULT_BOUNDARY_CONFIG


def build_boundary_inputs(canonical: pd.DataFrame, states: pd.Series, *, fit_weeks: int = 520,
                          config: BoundaryConfig = DEFAULT_BOUNDARY_CONFIG,
                          include_asymmetric: bool = False) -> BoundaryInputs:
    if not canonical.index.equals(states.index):
        raise ValueError("canonical and state indexes must match")
    if canonical.index.has_duplicates or not canonical.index.is_monotonic_increasing:
        raise ValueError("index must be unique and increasing")
    labeler = CausalRegimeLabeler(RegimeLabelConfig(minimum_fit_observations=min(260, fit_weeks // 2)))
    labeler.fit(canonical.iloc[:fit_weeks])
    reconstructed = labeler.transform(canonical)
    if not reconstructed.equals(states.rename("regime")):
        raise ValueError("authoritative states differ from frozen official label reconstruction")
    scores = labeler.score_frame(canonical)
    price = canonical.spy_close.to_numpy(dtype=float)
    returns = np.log(canonical.spy_close).diff()
    sigma = returns.ewm(span=config.volatility_span_weeks, adjust=False).std(bias=True).clip(lower=config.volatility_floor)
    residual = (returns / sigma.shift(1)).replace([np.inf, -np.inf], np.nan)
    asym_sigma = asymmetric_volatility(returns, config) if include_asymmetric else None
    asym_residual = returns / asym_sigma.shift(1) if asym_sigma is not None else None
    lower, upper = labeler.lower_threshold_, labeler.upper_threshold_
    assert lower is not None and upper is not None
    width = upper - lower
    feature = pd.DataFrame(index=canonical.index)
    feature["score"] = scores.risk_score / width
    feature["trend"] = scores.trend_score / width
    feature["stress"] = scores.stress_score / width
    for lag in (1, 2, 4):
        feature[f"score_delta_{lag}"] = scores.risk_score.diff(lag) / width
    feature["score_acceleration"] = scores.risk_score.diff().diff() / width
    feature["distance_lower"] = (scores.risk_score - lower) / width
    feature["distance_upper"] = (scores.risk_score - upper) / width
    feature["return_1w"] = returns
    feature["volatility"] = sigma
    age = states.groupby(states.ne(states.shift()).cumsum()).cumcount() + 1
    feature["log_state_age"] = np.log1p(age)
    for lag in (3, 12, 25, 51):
        feature[f"return_rolloff_{lag}"] = returns.shift(lag)
    for symbol in ("qqq", "iwm", "rsp", "hyg", "lqd", "tlt", "uup"):
        column = f"{symbol}_close"
        if column in canonical:
            rel = np.log(canonical[column]).diff(4) - np.log(canonical.spy_close).diff(4)
            feature[f"relative_{symbol}_4w"] = rel
    probabilities = {name: np.full((len(price), 3), np.nan) for name in MODELS[:2]}
    if include_asymmetric:
        probabilities[ASYMMETRIC_MODEL] = np.full((len(price), 3), np.nan)
    quantiles = (np.arange(config.integration_points) + .5) / config.integration_points
    student_residuals = student_t.ppf(quantiles, df=config.student_df) * np.sqrt((config.student_df - 2) / config.student_df)
    for position in range(52, len(price)):
        current = str(states.iloc[position])
        history = price[max(0, position - 52):position + 1]
        scale = float(sigma.iloc[position])
        simulated_scores = next_scores(history, scale * student_residuals, labeler)
        probabilities[MODELS[0]][position] = state_distribution(simulated_scores, current, labeler, config)
        # Every return used to fit the empirical distribution completed strictly
        # before the origin.  Current volatility is an observed input at origin.
        historical = residual.iloc[max(2, position - config.residual_window_weeks):position].dropna().to_numpy()
        historical = np.clip(historical, -config.residual_clip, config.residual_clip)
        probabilities[MODELS[1]][position] = state_distribution(
            next_scores(history, scale * historical, labeler), current, labeler, config
        )
        if asym_sigma is not None and asym_residual is not None:
            shocks = asym_residual.iloc[max(2, position - config.residual_window_weeks):position].dropna().to_numpy()
            shocks = np.clip(shocks, -config.residual_clip, config.residual_clip)
            probabilities[ASYMMETRIC_MODEL][position] = state_distribution(
                next_scores(history, float(asym_sigma.iloc[position]) * shocks, labeler), current, labeler, config)
        hypothetical = next_scores(history, scale * np.asarray([-1., 0., 1.]), labeler)
        feature.loc[feature.index[position], "zero_return_next_score"] = hypothetical[1] / width
        feature.loc[feature.index[position], "down_return_next_score"] = hypothetical[0] / width
        feature.loc[feature.index[position], "up_return_next_score"] = hypothetical[2] / width
        feature.loc[feature.index[position], "known_rolloff_score_change"] = (hypothetical[1] - scores.risk_score.iloc[position]) / width
    # Explicit state interactions allow different risk-on exit, risk-off exit,
    # and transition-entry equations without confusing the boundary directions.
    interaction_columns = ["score", "distance_lower", "distance_upper", "zero_return_next_score", "down_return_next_score", "up_return_next_score", "score_delta_1"]
    for state in STATE_ORDER:
        indicator = states.eq(state).astype(float)
        feature[f"state_{state}"] = indicator
        for column in interaction_columns:
            feature[f"{state}__{column}"] = indicator * feature[column]
    mechanistic = {name: pd.DataFrame(value, index=canonical.index, columns=STATE_ORDER) for name, value in probabilities.items()}
    for state in STATE_ORDER:
        feature[f"mechanical_probability_{state}"] = mechanistic[MODELS[0]][state]
    return BoundaryInputs(feature.replace([np.inf, -np.inf], np.nan), mechanistic, labeler, states, config)


def _align(estimator, matrix: np.ndarray) -> np.ndarray:
    probability = estimator.predict_proba(matrix)[0]
    output = np.full(3, 1e-6)
    for label, value in zip(estimator.classes_, probability):
        output[STATE_ORDER.index(str(label))] = value
    return output / output.sum()


def prequential_temperature(probability: np.ndarray, history: list[tuple[int, np.ndarray, int]], origin_position: int,
                            config: BoundaryConfig = DEFAULT_BOUNDARY_CONFIG) -> tuple[np.ndarray, float, int]:
    """Fit only completed earlier OOS predictions; target must precede origin."""
    eligible = [(p, y) for target, p, y in history if target < origin_position][-config.calibration_maximum_rows:]
    if len(eligible) < config.calibration_minimum_rows:
        return probability, 1., len(eligible)
    p = np.asarray([row[0] for row in eligible]); y = np.asarray([row[1] for row in eligible])
    def loss(log_temperature):
        candidate = softmax(np.log(np.clip(p, 1e-8, 1)) / np.exp(log_temperature), axis=1)
        return -np.log(candidate[np.arange(len(y)), y]).mean() + config.temperature_penalty * log_temperature ** 2
    fitted = minimize_scalar(loss, bounds=(np.log(config.temperature_minimum), np.log(config.temperature_maximum)), method="bounded")
    temperature = float(np.exp(fitted.x))
    return softmax(np.log(np.clip(probability, 1e-8, 1)) / temperature), temperature, len(eligible)


def _raw_predictions_at_origin(inputs: BoundaryInputs, position: int, models: tuple[str, ...], train_window: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    train_positions = np.arange(max(52, position - train_window - 1), position - 1)
    x_train = inputs.features.iloc[train_positions]
    x_test = inputs.features.iloc[[position]]
    y_train = inputs.states.iloc[train_positions + 1]
    raw = {model: values.iloc[position].to_numpy() for model, values in inputs.mechanistic.items()}
    if any(model in models for model in MODELS[2:]):
        logistic = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), LogisticRegression(C=.1, max_iter=1000, tol=1e-6))
        logistic.fit(x_train, y_train)
        raw[MODELS[2]] = _align(logistic, x_test)
    if MODELS[3] in models:
        xgb = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StateEncodedXGBoostClassifier(n_estimators=120, max_depth=2, min_child_weight=8, reg_lambda=12))
        weights = np.exp2(-np.arange(len(train_positions)-1, -1, -1) / 208)
        xgb.fit(x_train, y_train, stateencodedxgboostclassifier__sample_weight=weights / weights.mean())
        raw[MODELS[3]] = _align(xgb, x_test)
    raw[MODELS[4]] = .5 * raw[MODELS[0]] + .5 * raw.get(MODELS[2], raw[MODELS[0]])
    return raw, train_positions


def forecast_boundary_latest(inputs: BoundaryInputs, historical_oos: pd.DataFrame, *, model: str = "boundary_filtered_history", train_window: int = 520) -> dict:
    """Forecast the last observed origin; an unobserved target stays ``None``.

    Calibration reuses completed raw OOS predictions, never fitted training
    probabilities. The input predictions and historical probabilities are not
    mutated by issuing this forecast.
    """
    if model not in (*MODELS, ASYMMETRIC_MODEL):
        raise ValueError("unsupported boundary model")
    position = len(inputs.states) - 1
    if position < 521:
        raise ValueError("latest origin must follow the frozen fit period")
    states = inputs.states
    origin = states.index[position]
    rows = historical_oos.loc[historical_oos.model.eq(model)].copy()
    rows["origin_date"] = pd.to_datetime(rows.origin_date, utc=True)
    if rows.origin_date.duplicated().any():
        raise ValueError("historical OOS contains duplicate origins")
    rows["target_date"] = pd.to_datetime(rows.target_date, utc=True)
    rows = rows.loc[rows.target_date < origin].sort_values("target_date")
    history = []
    for row in rows.itertuples():
        if row.origin_date not in states.index or row.target_date <= row.origin_date:
            raise ValueError("historical forecast origin is invalid")
        historical_position = states.index.get_loc(row.origin_date)
        if historical_position + 1 >= len(states) or states.index[historical_position + 1] != row.target_date:
            raise ValueError("historical forecast must target the next observed week")
        if pd.Timestamp(row.last_train_target) >= row.origin_date:
            raise ValueError("historical training target must strictly precede origin")
        if row.current_state != states.loc[row.origin_date]:
            raise ValueError("historical current state disagrees with authoritative state")
        if row.target_date not in states.index:
            raise ValueError("historical target is outside authoritative state history")
        if row.actual != states.loc[row.target_date]:
            raise ValueError("historical target disagrees with authoritative state")
        raw = np.asarray([getattr(row, f"raw_p_{state}") for state in STATE_ORDER])
        if not np.isfinite(raw).all() or (raw <= 0).any() or not np.isclose(raw.sum(), 1):
            raise ValueError("historical raw probability is invalid")
        history.append((states.index.get_loc(row.target_date), raw, STATE_ORDER.index(row.actual)))
    raw, train_positions = _raw_predictions_at_origin(inputs, position, (model,), train_window)
    p, temperature, calibration_rows = prequential_temperature(raw[model], history, position, inputs.config)
    target_date = (origin.tz_convert("America/New_York") + pd.DateOffset(weeks=1)).tz_convert("UTC")
    return {"origin_date": origin.isoformat(), "target_date": target_date.isoformat(),
            "model": model, "evaluation_split": "unobserved", "current_state": str(states.iloc[position]),
            "actual": None, **dict(zip(PROBABILITY_COLUMNS, p)),
            **{f"raw_p_{state}": float(value) for state, value in zip(STATE_ORDER, raw[model])},
            "train_size": len(train_positions), "last_train_target": states.index[train_positions[-1]+1].isoformat(),
            "fallback": False, "temperature": temperature, "calibration_rows": calibration_rows}


def run_boundary_walk_forward(inputs: BoundaryInputs, *, origin_positions: list[int] | None = None, models: tuple[str, ...] = MODELS, train_window: int = 520, progress: Callable[[str], None] | None = None) -> pd.DataFrame:
    unsupported = set(models).difference((*MODELS, ASYMMETRIC_MODEL))
    if unsupported:
        raise ValueError(f"unsupported models: {sorted(unsupported)}")
    states = inputs.states
    positions = origin_positions if origin_positions is not None else [
        position for position in range(521, len(states) - 1)
        if not (states.index[position].year < 2023 <= states.index[position + 1].year)
    ]
    if positions != sorted(set(positions)):
        raise ValueError("origins must be unique and chronological")
    histories: dict[str, list[tuple[int, np.ndarray, int]]] = {model: [] for model in models}
    rows = []
    for count, position in enumerate(positions):
        if position < 521 or position >= len(states) - 1:
            raise ValueError("origin is outside frozen-label OOS evaluation range")
        raw, train_positions = _raw_predictions_at_origin(inputs, position, models, train_window)
        for model in models:
            p, temperature, calibration_rows = prequential_temperature(raw[model], histories[model], position, inputs.config)
            actual_index = STATE_ORDER.index(str(states.iloc[position + 1]))
            histories[model].append((position + 1, raw[model].copy(), actual_index))
            rows.append({
                "origin_date": states.index[position].isoformat(),
                "target_date": states.index[position + 1].isoformat(),
                "model": model,
                "evaluation_split": "selection" if states.index[position + 1].year < 2023 else "holdout",
                "current_state": str(states.iloc[position]), "actual": str(states.iloc[position + 1]),
                **dict(zip(PROBABILITY_COLUMNS, p)),
                **{f"raw_p_{state}": float(value) for state, value in zip(STATE_ORDER, raw[model])},
                "train_size": len(train_positions),
                "last_train_target": states.index[train_positions[-1] + 1].isoformat(),
                "fallback": False, "temperature": temperature, "calibration_rows": calibration_rows,
            })
        if progress and (count % 25 == 0 or count == len(positions)-1):
            progress(f"boundary origin {count+1}/{len(positions)} {states.index[position].date()}")
    return pd.DataFrame(rows)


def summarize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, split), group in predictions.groupby(["model", "evaluation_split"]):
        probabilities = group.loc[:, PROBABILITY_COLUMNS].to_numpy()
        actual = np.asarray([STATE_ORDER.index(str(value)) for value in group.actual])
        current = np.asarray([STATE_ORDER.index(str(value)) for value in group.current_state])
        change = actual != current
        p_change = 1 - probabilities[np.arange(len(group)), current]
        predicted = probabilities.argmax(axis=1)
        alert = predicted != current
        probability_alert = p_change >= .5
        deteriorating = actual > current
        years = len(group) / 52.1775
        rows.append({"model": model, "evaluation_split": split, "weeks": len(group),
                     "log_loss": float(-np.log(np.clip(probabilities[np.arange(len(group)), actual], 1e-12, 1)).mean()),
                     "brier_score": float(((probabilities - np.eye(3)[actual])**2).sum(axis=1).mean()),
                     "accuracy": float((predicted == actual).mean()), "events": int(change.sum()),
                     "hits": int((alert & change).sum()), "event_recall": float((alert & change).sum()/max(1, change.sum())),
                     "false_alerts_per_year": float((alert & ~change).sum()/years),
                     "mean_change_probability": float(p_change.mean()),
                     "wrong_destination_alerts": int((alert & change & (predicted != actual)).sum()),
                     "probability_threshold_hits": int((probability_alert & change).sum()),
                     "probability_threshold_false_alerts": int((probability_alert & ~change).sum()),
                     "deterioration_events": int(deteriorating.sum()),
                     "deterioration_hits": int((deteriorating & (predicted > current)).sum())})
    return pd.DataFrame(rows)
