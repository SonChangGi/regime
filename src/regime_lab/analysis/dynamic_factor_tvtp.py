"""Causal expanding-factor TVTP shadow forecasts.

This is a research shadow, not an operating candidate.  At every origin the
imputer, scaler, and PCA factor map are refit on the purged training prefix.
The transformed factor level and backward differences then feed the existing
direct-jump duration-aware TVTP hurdle.  No target-time row is used to fit a
transform or estimator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from regime_lab.integrity import canonical_json_sha256_v1

from .labels import STATE_ORDER
from .transitions import (
    CURRENT_STATE_COLUMN,
    DURATION_COLUMN,
    DurationAwareTVTPHurdleClassifier,
    causal_state_durations,
)


DYNAMIC_FACTOR_TVTP_SCHEMA_VERSION = "regime-dynamic-factor-tvtp-shadow/1"
_PROBABILITY_COLUMNS = tuple(f"p_{state}" for state in STATE_ORDER)


@dataclass(frozen=True)
class DynamicFactorTVTPConfig:
    """Frozen hyperparameters for the expanding-prefix shadow."""

    n_factors: int = 3
    factor_difference_lags: tuple[int, ...] = (1, 4, 13)
    min_train_size: int = 156
    max_origins: int | None = None
    gap_weeks: int = 1
    hazard_C: float = 0.10
    destination_C: float = 0.10
    smoothing: float = 1.0
    random_state: int = 17

    def __post_init__(self) -> None:
        if (
            isinstance(self.n_factors, bool)
            or int(self.n_factors) != self.n_factors
            or int(self.n_factors) < 1
        ):
            raise ValueError("n_factors must be a positive integer")
        if (
            isinstance(self.min_train_size, bool)
            or int(self.min_train_size) != self.min_train_size
            or int(self.min_train_size) < int(self.n_factors) + 2
        ):
            raise ValueError("min_train_size must exceed n_factors by at least two")
        if self.max_origins is not None and (
            isinstance(self.max_origins, bool)
            or int(self.max_origins) != self.max_origins
            or int(self.max_origins) < 1
        ):
            raise ValueError("max_origins must be a positive integer or None")
        if self.gap_weeks != 1:
            raise ValueError("dynamic-factor TVTP preserves the official gap=1")
        lags = tuple(self.factor_difference_lags)
        if (
            not lags
            or len(lags) != len(set(lags))
            or tuple(sorted(lags)) != lags
            or any(
                isinstance(value, bool)
                or int(value) != value
                or int(value) < 1
                for value in lags
            )
        ):
            raise ValueError("factor difference lags must be unique increasing integers")
        for field_name in ("hazard_C", "destination_C", "smoothing"):
            value = float(getattr(self, field_name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field_name} must be positive and finite")
        if (
            isinstance(self.random_state, bool)
            or int(self.random_state) != self.random_state
            or int(self.random_state) < 0
        ):
            raise ValueError("random_state must be a non-negative integer")

        # Hash the effective configuration, not incidental Python spellings
        # such as 2 versus 2.0 or a list versus a tuple.
        object.__setattr__(self, "n_factors", int(self.n_factors))
        object.__setattr__(
            self,
            "factor_difference_lags",
            tuple(int(value) for value in lags),
        )
        object.__setattr__(self, "min_train_size", int(self.min_train_size))
        object.__setattr__(
            self,
            "max_origins",
            None if self.max_origins is None else int(self.max_origins),
        )
        object.__setattr__(self, "gap_weeks", int(self.gap_weeks))
        object.__setattr__(self, "hazard_C", float(self.hazard_C))
        object.__setattr__(self, "destination_C", float(self.destination_C))
        object.__setattr__(self, "smoothing", float(self.smoothing))
        object.__setattr__(self, "random_state", int(self.random_state))

    def manifest(self) -> dict[str, Any]:
        body = asdict(self)
        body["factor_difference_lags"] = list(self.factor_difference_lags)
        return body


@dataclass(frozen=True)
class DynamicFactorTVTPShadowResult:
    predictions: pd.DataFrame
    configuration_sha256: str
    method: str = "expanding_prefix_pca_direct_jump_tvtp"
    role: str = "prospective_shadow_only"
    canonical_target: bool = False
    automatic_promotion_eligible: bool = False
    causality_scope: str = "structural_row_prefix_only"
    vintage_safety: str = "not_established_without_origin_snapshot_vintages"
    operational_oos_eligible: bool = False


def _validate_inputs(
    features: pd.DataFrame,
    states: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame")
    if not isinstance(states, pd.Series):
        raise TypeError("states must be a pandas Series")
    if not isinstance(features.index, pd.DatetimeIndex):
        raise TypeError("features must use a DatetimeIndex")
    if not states.index.equals(features.index):
        raise ValueError("states must use the exact feature index")
    if (
        features.empty
        or features.columns.empty
        or features.columns.has_duplicates
        or features.index.has_duplicates
        or not features.index.is_monotonic_increasing
    ):
        raise ValueError("features must be non-empty with unique ordered axes")
    if not all(isinstance(column, str) and column for column in features.columns):
        raise TypeError("feature column names must be non-empty strings")
    numeric = features.apply(pd.to_numeric, errors="coerce").astype(float)
    invalid = features.notna() & numeric.isna()
    if invalid.any().any():
        raise TypeError("features must be numeric where observed")
    if np.isinf(numeric.to_numpy(dtype=float)).any():
        raise ValueError("features must not contain infinities")
    labels = states.astype(str)
    invalid_states = sorted(set(labels).difference(STATE_ORDER))
    if invalid_states or states.isna().any():
        raise ValueError(f"states contain unsupported values: {invalid_states}")
    return numeric, labels


def _factor_frame(
    values: pd.DataFrame,
    *,
    imputer: SimpleImputer,
    scaler: StandardScaler,
    pca: PCA,
    lags: tuple[int, ...],
) -> pd.DataFrame:
    matrix = imputer.transform(values)
    standardized = scaler.transform(matrix)
    factors = pca.transform(standardized)
    frame = pd.DataFrame(
        factors,
        index=values.index,
        columns=[f"dynamic_factor_{position + 1}" for position in range(factors.shape[1])],
        dtype=float,
    )
    for column in tuple(frame.columns):
        for lag in lags:
            frame[f"{column}__delta_{lag}w"] = frame[column].diff(lag)
    return frame


def run_dynamic_factor_tvtp_shadow(
    features: pd.DataFrame,
    states: pd.Series,
    *,
    config: DynamicFactorTVTPConfig | None = None,
) -> DynamicFactorTVTPShadowResult:
    """Run a purged one-week walk-forward shadow without changing selection.

    With ``gap_weeks=1``, origin position ``t`` fits feature origins through
    ``t-2`` whose final target is ``S[t-1]``.  The feature transform is fit on
    exactly that same prefix, then applied through the known origin ``t``.
    """

    settings = config or DynamicFactorTVTPConfig()
    matrix, labels = _validate_inputs(features, states)
    start = int(settings.min_train_size) + int(settings.gap_weeks)
    if len(matrix) <= start:
        raise ValueError("not enough rows for a dynamic-factor TVTP forecast")
    durations = causal_state_durations(labels)
    next_states = labels.shift(-1)
    rows: list[dict[str, Any]] = []

    first_origin = start
    if settings.max_origins is not None:
        first_origin = max(first_origin, len(matrix) - 1 - settings.max_origins)

    for origin_position in range(first_origin, len(matrix) - 1):
        train_stop = origin_position - int(settings.gap_weeks)
        train_values = matrix.iloc[:train_stop]
        train_target = next_states.iloc[:train_stop]
        if train_target.isna().any():
            raise RuntimeError("purged training target unexpectedly contains missing rows")

        imputer = SimpleImputer(
            strategy="median",
            add_indicator=False,
            keep_empty_features=True,
        )
        scaler = StandardScaler()
        imputed_train = imputer.fit_transform(train_values)
        standardized_train = scaler.fit_transform(imputed_train)
        factor_count = min(
            int(settings.n_factors),
            standardized_train.shape[0],
            standardized_train.shape[1],
        )
        pca = PCA(
            n_components=factor_count,
            svd_solver="full",
            random_state=int(settings.random_state),
        ).fit(standardized_train)

        transformed = _factor_frame(
            matrix.iloc[: origin_position + 1],
            imputer=imputer,
            scaler=scaler,
            pca=pca,
            lags=tuple(settings.factor_difference_lags),
        )
        transformed[CURRENT_STATE_COLUMN] = labels.iloc[: origin_position + 1]
        transformed[DURATION_COLUMN] = durations.iloc[: origin_position + 1].astype(float)
        train_frame = transformed.iloc[:train_stop]
        forecast_frame = transformed.iloc[[origin_position]]

        estimator = DurationAwareTVTPHurdleClassifier(
            derive_duration=False,
            adjacent_only=False,
            hazard_C=float(settings.hazard_C),
            destination_C=float(settings.destination_C),
            smoothing=float(settings.smoothing),
            random_state=int(settings.random_state),
        ).fit(train_frame, train_target)
        probability = estimator.predict_proba(forecast_frame)[0]
        if not np.isfinite(probability).all() or not np.isclose(
            probability.sum(), 1.0, rtol=0.0, atol=1e-8
        ):
            raise RuntimeError("dynamic-factor TVTP emitted invalid probabilities")

        row: dict[str, Any] = {
            "origin_date": matrix.index[origin_position],
            "target_date": matrix.index[origin_position + 1],
            "last_train_origin_date": matrix.index[train_stop - 1],
            "last_train_target_date": matrix.index[train_stop],
            "current_state": labels.iloc[origin_position],
            "predicted": STATE_ORDER[int(np.argmax(probability))],
            "train_size": int(train_stop),
            "gap": int(settings.gap_weeks),
            "factor_count": int(factor_count),
            "direct_jump_allowed": True,
            "target_passed_to_prediction": False,
            "used_fallback": bool(estimator.used_fallback_),
            "fallback_reasons": list(estimator.fallback_reasons_),
        }
        row.update(
            {
                column: float(probability[position])
                for position, column in enumerate(_PROBABILITY_COLUMNS)
            }
        )
        rows.append(row)

    predictions = pd.DataFrame(rows)
    if predictions.empty:
        raise RuntimeError("dynamic-factor TVTP produced no forecast rows")
    if not (
        pd.to_datetime(predictions["last_train_target_date"])
        < pd.to_datetime(predictions["origin_date"])
    ).all():
        raise RuntimeError("dynamic-factor TVTP violated the purged target boundary")
    return DynamicFactorTVTPShadowResult(
        predictions=predictions,
        configuration_sha256=canonical_json_sha256_v1(
            {
                "schema_version": DYNAMIC_FACTOR_TVTP_SCHEMA_VERSION,
                "configuration": settings.manifest(),
            }
        ),
    )


__all__ = [
    "DYNAMIC_FACTOR_TVTP_SCHEMA_VERSION",
    "DynamicFactorTVTPConfig",
    "DynamicFactorTVTPShadowResult",
    "run_dynamic_factor_tvtp_shadow",
]
