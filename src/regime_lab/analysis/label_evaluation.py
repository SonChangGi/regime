"""Matched descriptive evaluation for competing regime-label definitions."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd

from regime_lab.analysis.duration import causal_spell_table
from regime_lab.schema import STATE_ORDER


LABEL_EVALUATION_HORIZONS: tuple[int, ...] = (1, 4, 13)
EXTERNAL_ORIGIN_COLUMNS: tuple[str, ...] = (
    "origin_position",
    "origin_date",
    "exit_date",
    "state",
    "asset",
    "horizon_weeks",
    "execution_lag_weeks",
    "forward_return",
    "max_drawdown",
)


class FittedLabeler(Protocol):
    def transform(self, frame: Any) -> pd.Series: ...

    def score_frame(self, frame: Any) -> pd.DataFrame: ...

    def state_memberships(self, frame: Any) -> pd.DataFrame: ...


@dataclass(frozen=True)
class LabelEvaluationResult:
    """Auditable tables; no model-promotion conclusion is implied."""

    occupancy: pd.DataFrame
    durations: pd.DataFrame
    flips: pd.DataFrame
    external_origin_outcomes: pd.DataFrame
    external_outcomes: pd.DataFrame
    crash_recovery: pd.DataFrame
    sensitivity: pd.DataFrame
    prefix_stability: pd.DataFrame


def _validate_states(states: pd.Series) -> pd.Series:
    if not isinstance(states, pd.Series):
        raise TypeError("states must be a pandas Series")
    if not isinstance(states.index, pd.DatetimeIndex):
        raise TypeError("states must use a DatetimeIndex")
    if states.empty:
        raise ValueError("states must not be empty")
    if states.index.has_duplicates or not states.index.is_monotonic_increasing:
        raise ValueError("state dates must be unique and increasing")
    if states.isna().any():
        raise ValueError("states must not contain missing values")
    if len(states) > 1:
        dates = states.index.tz_localize(None).normalize()
        if not bool(((dates[1:] - dates[:-1]) == np.timedelta64(7, "D")).all()):
            raise ValueError("states must contain consecutive weekly observations")
    values = states.astype(str)
    unsupported = sorted(set(values).difference(STATE_ORDER))
    if unsupported:
        raise ValueError(f"unsupported states: {unsupported}")
    return values


def label_occupancy(states: pd.Series) -> pd.DataFrame:
    labels = _validate_states(states)
    counts = labels.value_counts()
    return pd.DataFrame(
        [
            {
                "state": state,
                "observations": int(counts.get(state, 0)),
                "share": float(counts.get(state, 0) / len(labels)),
            }
            for state in STATE_ORDER
        ]
    )


def label_duration_summary(states: pd.Series) -> pd.DataFrame:
    labels = _validate_states(states)
    spells = causal_spell_table(labels)
    rows: list[dict[str, Any]] = []
    for state in STATE_ORDER:
        selected = spells.loc[spells["state"].eq(state)]
        duration = selected["duration_weeks"].to_numpy(dtype=float)
        rows.append(
            {
                "state": state,
                "episodes": int(len(selected)),
                "completed_episodes": int(selected["event_observed"].sum()),
                "current_spell_weeks": int(
                    selected.loc[selected["is_current"], "duration_weeks"].iloc[0]
                )
                if bool(selected["is_current"].any())
                else 0,
                "mean_duration_weeks": float(np.mean(duration))
                if len(duration)
                else float("nan"),
                "median_duration_weeks": float(np.median(duration))
                if len(duration)
                else float("nan"),
                "p90_duration_weeks": float(np.quantile(duration, 0.90))
                if len(duration)
                else float("nan"),
                "max_duration_weeks": int(np.max(duration)) if len(duration) else 0,
            }
        )
    return pd.DataFrame(rows)


def label_flip_summary(states: pd.Series) -> pd.DataFrame:
    labels = _validate_states(states)
    previous = labels.shift(1)
    changed = labels.ne(previous) & previous.notna()
    direct = changed & (
        (previous.eq("risk_on") & labels.eq("risk_off"))
        | (previous.eq("risk_off") & labels.eq("risk_on"))
    )
    opportunities = max(len(labels) - 1, 1)
    return pd.DataFrame(
        [
            {
                "observations": int(len(labels)),
                "flip_opportunities": int(max(len(labels) - 1, 0)),
                "flips": int(changed.sum()),
                "direct_risk_on_off_flips": int(direct.sum()),
                "flip_rate_per_week": float(changed.sum() / opportunities),
                "annualized_flips": float(changed.sum() / opportunities * 52.0),
            }
        ]
    )


def _validate_external_prices(
    prices: pd.DataFrame,
    states: pd.Series,
    asset_columns: Mapping[str, str] | None,
) -> tuple[pd.DataFrame, Mapping[str, str]]:
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("external prices must be a pandas DataFrame")
    if not prices.index.equals(states.index):
        raise ValueError("external prices and states must use the same weekly index")
    configured = (
        {str(column).upper(): str(column) for column in prices.columns}
        if asset_columns is None
        else {
            str(asset).upper(): str(column)
            for asset, column in asset_columns.items()
        }
    )
    if not configured:
        raise ValueError("at least one external asset is required")
    missing = sorted(set(configured.values()).difference(prices.columns))
    if missing:
        raise KeyError(f"missing external price columns: {missing}")
    normalized: dict[str, pd.Series] = {}
    for asset, column in configured.items():
        values = pd.to_numeric(prices[column], errors="coerce").astype(float)
        finite = values[np.isfinite(values)]
        if (finite <= 0.0).any() or np.isinf(values.to_numpy(dtype=float)).any():
            raise ValueError(f"external prices for {asset} must be finite and positive")
        normalized[asset] = values
    return pd.DataFrame(normalized, index=states.index), configured


def _path_drawdown(path: np.ndarray) -> float:
    running_peak = np.maximum.accumulate(path)
    return float(np.min(path / running_peak - 1.0))


def build_external_origin_outcomes(
    prices: pd.DataFrame,
    states: pd.Series,
    *,
    asset_columns: Mapping[str, str] | None = None,
    horizons: Sequence[int] = LABEL_EVALUATION_HORIZONS,
) -> pd.DataFrame:
    """Build close-at-t to close-at-t+h descriptive outcomes.

    These are external construct-validity diagnostics for a label observed at
    the origin close.  They are not executable strategy returns and do not
    include a one-week trading lag.
    """

    labels = _validate_states(states)
    normalized, _configured = _validate_external_prices(
        prices, labels, asset_columns
    )
    raw_horizons = tuple(horizons)
    if any(
        isinstance(item, bool) or not isinstance(item, (int, np.integer))
        for item in raw_horizons
    ):
        raise ValueError(
            f"external outcome horizons must be exactly {LABEL_EVALUATION_HORIZONS}"
        )
    resolved_horizons = tuple(int(item) for item in raw_horizons)
    if resolved_horizons != LABEL_EVALUATION_HORIZONS:
        raise ValueError(
            f"external outcome horizons must be exactly {LABEL_EVALUATION_HORIZONS}"
        )
    rows: list[dict[str, Any]] = []
    for origin_position, origin_date in enumerate(labels.index):
        for horizon in resolved_horizons:
            exit_position = origin_position + horizon
            if exit_position >= len(labels):
                continue
            for asset in normalized:
                path = normalized[asset].iloc[
                    origin_position : exit_position + 1
                ].to_numpy(dtype=float)
                if len(path) != horizon + 1 or not np.isfinite(path).all():
                    continue
                rows.append(
                    {
                        "origin_position": int(origin_position),
                        "origin_date": origin_date,
                        "exit_date": labels.index[exit_position],
                        "state": str(labels.iloc[origin_position]),
                        "asset": asset,
                        "horizon_weeks": int(horizon),
                        "execution_lag_weeks": 0,
                        "forward_return": float(path[-1] / path[0] - 1.0),
                        "max_drawdown": _path_drawdown(path),
                    }
                )
    return pd.DataFrame(rows, columns=EXTERNAL_ORIGIN_COLUMNS)


def summarize_external_outcomes(outcomes: pd.DataFrame) -> pd.DataFrame:
    required = {
        "state",
        "asset",
        "horizon_weeks",
        "forward_return",
        "max_drawdown",
    }
    if not isinstance(outcomes, pd.DataFrame) or not required.issubset(outcomes.columns):
        raise ValueError("external origin outcomes have an invalid schema")
    rows: list[dict[str, Any]] = []
    assets = tuple(dict.fromkeys(outcomes["asset"].astype(str)))
    for state in STATE_ORDER:
        for asset in assets:
            for horizon in LABEL_EVALUATION_HORIZONS:
                selected = outcomes.loc[
                    outcomes["state"].eq(state)
                    & outcomes["asset"].eq(asset)
                    & outcomes["horizon_weeks"].eq(horizon)
                ]
                returns = selected["forward_return"].to_numpy(dtype=float)
                drawdowns = selected["max_drawdown"].to_numpy(dtype=float)
                rows.append(
                    {
                        "state": state,
                        "asset": asset,
                        "horizon_weeks": horizon,
                        "execution_lag_weeks": 0,
                        "n": int(len(selected)),
                        "mean_return": float(np.mean(returns))
                        if len(returns)
                        else float("nan"),
                        "median_return": float(np.median(returns))
                        if len(returns)
                        else float("nan"),
                        "positive_rate": float(np.mean(returns > 0.0))
                        if len(returns)
                        else float("nan"),
                        "annualized_volatility": float(
                            np.std(returns, ddof=1) * np.sqrt(52.0 / horizon)
                        )
                        if len(returns) > 1
                        else float("nan"),
                        "mean_max_drawdown": float(np.mean(drawdowns))
                        if len(drawdowns)
                        else float("nan"),
                        "worst_max_drawdown": float(np.min(drawdowns))
                        if len(drawdowns)
                        else float("nan"),
                    }
                )
    return pd.DataFrame(rows)


def crash_recovery_lags(
    price: pd.Series,
    states: pd.Series,
    *,
    crash_drawdown: float = -0.20,
) -> pd.DataFrame:
    """Measure state-response lags around non-overlapping drawdown episodes."""

    labels = _validate_states(states)
    if not isinstance(price, pd.Series) or not price.index.equals(labels.index):
        raise ValueError("crash price and states must use the same weekly index")
    if not -1.0 < float(crash_drawdown) < 0.0:
        raise ValueError("crash_drawdown must be between -1 and 0")
    values = pd.to_numeric(price, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise ValueError("crash price must be complete, finite, and positive")

    rows: list[dict[str, Any]] = []
    position = 0
    peak_position = 0
    peak_value = values[0]
    while position < len(values):
        if values[position] > peak_value:
            peak_value = values[position]
            peak_position = position
        drawdown = values[position] / peak_value - 1.0
        if drawdown > crash_drawdown:
            position += 1
            continue
        crash_position = position
        recovery_position: int | None = None
        for candidate in range(crash_position + 1, len(values)):
            if values[candidate] >= peak_value:
                recovery_position = candidate
                break
        search_end = (
            recovery_position if recovery_position is not None else len(values) - 1
        )
        risk_off_positions = np.flatnonzero(
            labels.iloc[crash_position : search_end + 1]
            .eq("risk_off")
            .to_numpy(dtype=bool)
        )
        risk_off_position = (
            crash_position + int(risk_off_positions[0])
            if len(risk_off_positions)
            else None
        )
        risk_on_position: int | None = None
        if recovery_position is not None:
            recovery_risk_on = np.flatnonzero(
                labels.iloc[recovery_position:].eq("risk_on").to_numpy(dtype=bool)
            )
            if len(recovery_risk_on):
                risk_on_position = recovery_position + int(recovery_risk_on[0])
        trough_slice = values[crash_position : search_end + 1]
        trough_position = crash_position + int(np.argmin(trough_slice))
        rows.append(
            {
                "event_id": len(rows),
                "peak_date": labels.index[peak_position],
                "crash_date": labels.index[crash_position],
                "trough_date": labels.index[trough_position],
                "recovery_date": labels.index[recovery_position]
                if recovery_position is not None
                else pd.NaT,
                "risk_off_date": labels.index[risk_off_position]
                if risk_off_position is not None
                else pd.NaT,
                "risk_on_after_recovery_date": labels.index[risk_on_position]
                if risk_on_position is not None
                else pd.NaT,
                "maximum_drawdown": float(values[trough_position] / peak_value - 1.0),
                "crash_detection_lag_weeks": int(risk_off_position - crash_position)
                if risk_off_position is not None
                else float("nan"),
                "recovery_detection_lag_weeks": int(
                    risk_on_position - recovery_position
                )
                if risk_on_position is not None and recovery_position is not None
                else float("nan"),
                "recovery_censored": recovery_position is None,
            }
        )
        if recovery_position is None:
            break
        position = recovery_position + 1
        peak_position = recovery_position
        peak_value = values[recovery_position]
    return pd.DataFrame(
        rows,
        columns=(
            "event_id",
            "peak_date",
            "crash_date",
            "trough_date",
            "recovery_date",
            "risk_off_date",
            "risk_on_after_recovery_date",
            "maximum_drawdown",
            "crash_detection_lag_weeks",
            "recovery_detection_lag_weeks",
            "recovery_censored",
        ),
    )


def compare_label_sensitivity(
    reference: pd.Series,
    alternatives: Mapping[str, pd.Series] | None,
) -> pd.DataFrame:
    labels = _validate_states(reference)
    if alternatives is None:
        alternatives = {}
    reference_occupancy = label_occupancy(labels).set_index("state")["share"]
    reference_flips = int(label_flip_summary(labels).iloc[0]["flips"])
    reference_changed = labels.ne(labels.shift(1))
    reference_changed.iloc[0] = False
    reference_transitions = set(labels.index[reference_changed])
    rows: list[dict[str, Any]] = []
    for variant, raw in alternatives.items():
        candidate = _validate_states(raw)
        if not candidate.index.equals(labels.index):
            raise ValueError(f"sensitivity variant {variant} index does not match")
        occupancy = label_occupancy(candidate).set_index("state")["share"]
        candidate_changed = candidate.ne(candidate.shift(1))
        candidate_changed.iloc[0] = False
        candidate_transitions = set(candidate.index[candidate_changed])
        union = reference_transitions | candidate_transitions
        intersection = reference_transitions & candidate_transitions
        rows.append(
            {
                "variant": str(variant),
                "observations": int(len(labels)),
                "changed_weeks": int(candidate.ne(labels).sum()),
                "agreement": float(candidate.eq(labels).mean()),
                "flip_delta": int(
                    int(label_flip_summary(candidate).iloc[0]["flips"])
                    - reference_flips
                ),
                "maximum_absolute_occupancy_shift": float(
                    (occupancy - reference_occupancy).abs().max()
                ),
                "transition_date_jaccard": float(len(intersection) / len(union))
                if union
                else 1.0,
            }
        )
    return pd.DataFrame(
        rows,
        columns=(
            "variant",
            "observations",
            "changed_weeks",
            "agreement",
            "flip_delta",
            "maximum_absolute_occupancy_shift",
            "transition_date_jaccard",
        ),
    )


def _maximum_absolute_difference(left: np.ndarray, right: np.ndarray) -> float:
    finite = np.isfinite(left) & np.isfinite(right)
    mismatch = np.isfinite(left) ^ np.isfinite(right)
    if mismatch.any():
        return float("inf")
    if not finite.any():
        return 0.0
    return float(np.max(np.abs(left[finite] - right[finite])))


def prefix_stability_report(
    labeler: FittedLabeler,
    frame: Any,
    *,
    prefix_lengths: Sequence[int],
) -> pd.DataFrame:
    """Compare each truncated transform with the corresponding full prefix."""

    if not prefix_lengths:
        raise ValueError("prefix_lengths must not be empty")
    full_labels = labeler.transform(frame)
    full_scores = labeler.score_frame(frame)
    full_membership = labeler.state_memberships(frame)
    rows: list[dict[str, Any]] = []
    for raw_length in prefix_lengths:
        if isinstance(raw_length, bool) or int(raw_length) != raw_length:
            raise ValueError("prefix lengths must be integers")
        length = int(raw_length)
        if length < 1 or length > len(frame):
            raise ValueError("prefix lengths must fall inside the frame")
        slice_rows = getattr(frame, "slice_rows", None)
        prefix = slice_rows(length) if callable(slice_rows) else frame.iloc[:length]
        labels = labeler.transform(prefix)
        scores = labeler.score_frame(prefix)
        membership = labeler.state_memberships(prefix)
        label_mismatches = int(labels.ne(full_labels.iloc[:length]).sum())
        score_difference = _maximum_absolute_difference(
            scores.to_numpy(dtype=float),
            full_scores.iloc[:length].to_numpy(dtype=float),
        )
        membership_difference = _maximum_absolute_difference(
            membership.to_numpy(dtype=float),
            full_membership.iloc[:length].to_numpy(dtype=float),
        )
        rows.append(
            {
                "prefix_rows": length,
                "prefix_end": prefix.index[-1],
                "label_mismatches": label_mismatches,
                "maximum_absolute_score_difference": score_difference,
                "maximum_absolute_membership_difference": membership_difference,
                "stable": bool(
                    label_mismatches == 0
                    and score_difference == 0.0
                    and membership_difference == 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def evaluate_label_definition(
    states: pd.Series,
    *,
    external_prices: pd.DataFrame | None = None,
    asset_columns: Mapping[str, str] | None = None,
    crash_asset: str | None = None,
    crash_drawdown: float = -0.20,
    sensitivity_labels: Mapping[str, pd.Series] | None = None,
    prefix_stability: pd.DataFrame | None = None,
) -> LabelEvaluationResult:
    """Run every registered descriptive label-quality dimension."""

    labels = _validate_states(states)
    external_origin = pd.DataFrame(columns=EXTERNAL_ORIGIN_COLUMNS)
    external_summary = pd.DataFrame()
    crash = pd.DataFrame()
    if external_prices is not None:
        external_origin = build_external_origin_outcomes(
            external_prices,
            labels,
            asset_columns=asset_columns,
        )
        external_summary = summarize_external_outcomes(external_origin)
        normalized, configured = _validate_external_prices(
            external_prices, labels, asset_columns
        )
        selected_asset = crash_asset or next(iter(configured))
        if selected_asset not in normalized:
            raise KeyError(f"unknown crash asset: {selected_asset}")
        crash = crash_recovery_lags(
            normalized[selected_asset],
            labels,
            crash_drawdown=crash_drawdown,
        )
    return LabelEvaluationResult(
        occupancy=label_occupancy(labels),
        durations=label_duration_summary(labels),
        flips=label_flip_summary(labels),
        external_origin_outcomes=external_origin,
        external_outcomes=external_summary,
        crash_recovery=crash,
        sensitivity=compare_label_sensitivity(labels, sensitivity_labels),
        prefix_stability=(
            prefix_stability.copy()
            if prefix_stability is not None
            else pd.DataFrame()
        ),
    )


__all__ = [
    "EXTERNAL_ORIGIN_COLUMNS",
    "LABEL_EVALUATION_HORIZONS",
    "LabelEvaluationResult",
    "build_external_origin_outcomes",
    "compare_label_sensitivity",
    "crash_recovery_lags",
    "evaluate_label_definition",
    "label_duration_summary",
    "label_flip_summary",
    "label_occupancy",
    "prefix_stability_report",
    "summarize_external_outcomes",
]
