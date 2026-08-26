"""Transparent, causal reference labels for US-equity market regimes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from regime_lab.analysis.label_spec import (
    DEFAULT_LABEL_SPEC_ID,
    LabelScalingSpec,
    LabelSpecification,
    load_label_spec,
)
from regime_lab.schema import STATE_ORDER


@dataclass(frozen=True, init=False)
class RegimeLabelConfig:
    """Compatibility view of the frozen v1 typed label specification.

    Defaults are loaded from ``config/label-spec.json`` rather than repeated
    here.  The optional keyword overrides retain the pre-registry public API;
    the production callers currently use only ``price_column`` and the stricter
    finite-observation gate.
    """

    price_column: str
    lower_quantile: float
    upper_quantile: float
    hysteresis_fraction: float
    probability_temperature: float
    minimum_fit_observations: int
    spec_id: str
    spec_sha256: str
    membership_semantics: str
    specification: LabelSpecification

    def __init__(
        self,
        price_column: str | None = None,
        lower_quantile: float | None = None,
        upper_quantile: float | None = None,
        hysteresis_fraction: float | None = None,
        probability_temperature: float | None = None,
        minimum_fit_observations: int | None = None,
        *,
        spec_id: str = DEFAULT_LABEL_SPEC_ID,
    ) -> None:
        specification = load_label_spec(spec_id)
        if specification.spec_id != DEFAULT_LABEL_SPEC_ID:
            raise ValueError(
                "RegimeLabelConfig is the frozen v1 compatibility contract; "
                "use ResearchRegimeLabeler for challenger specs"
            )
        default_price = specification.series_for_block("direction")[0].column
        resolved_price = default_price if price_column is None else str(price_column)
        resolved_lower = (
            specification.lower_quantile
            if lower_quantile is None
            else float(lower_quantile)
        )
        resolved_upper = (
            specification.upper_quantile
            if upper_quantile is None
            else float(upper_quantile)
        )
        resolved_hysteresis = (
            specification.hysteresis_fraction
            if hysteresis_fraction is None
            else float(hysteresis_fraction)
        )
        resolved_temperature = (
            specification.membership.temperature
            if probability_temperature is None
            else float(probability_temperature)
        )
        resolved_minimum = (
            specification.fit_period.minimum_finite_observations
            if minimum_fit_observations is None
            else int(minimum_fit_observations)
        )
        if not resolved_price:
            raise ValueError("price_column must not be empty")
        if not 0.0 < resolved_lower < resolved_upper < 1.0:
            raise ValueError("quantiles must satisfy 0 < lower < upper < 1")
        if not 0.0 <= resolved_hysteresis < 0.5:
            raise ValueError("hysteresis_fraction must be in [0, 0.5)")
        if resolved_temperature <= 0.0:
            raise ValueError("probability_temperature must be positive")
        if resolved_minimum < 1:
            raise ValueError("minimum_fit_observations must be positive")
        object.__setattr__(self, "price_column", resolved_price)
        object.__setattr__(self, "lower_quantile", resolved_lower)
        object.__setattr__(self, "upper_quantile", resolved_upper)
        object.__setattr__(self, "hysteresis_fraction", resolved_hysteresis)
        object.__setattr__(self, "probability_temperature", resolved_temperature)
        object.__setattr__(self, "minimum_fit_observations", resolved_minimum)
        object.__setattr__(self, "spec_id", specification.spec_id)
        object.__setattr__(self, "spec_sha256", specification.spec_sha256)
        object.__setattr__(
            self,
            "membership_semantics",
            specification.membership.semantics,
        )
        object.__setattr__(self, "specification", specification)


def _robust_location_scale(
    series: pd.Series,
    scaling: LabelScalingSpec | None = None,
) -> tuple[float, float]:
    if scaling is None:
        scaling = load_label_spec().scaling
    finite = pd.to_numeric(series, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if finite.empty:
        raise ValueError("cannot fit regime labeler without finite observations")
    centre = float(finite.median())
    q25, q75 = finite.quantile([0.25, 0.75]).to_numpy(dtype=float)
    scale = float((q75 - q25) / scaling.iqr_normalizer)
    if not np.isfinite(scale) or scale <= scaling.scale_floor:
        mad = float((finite - centre).abs().median() * scaling.mad_normalizer)
        scale = (
            mad
            if np.isfinite(mad) and mad > scaling.scale_floor
            else scaling.constant_fallback_scale
        )
    return centre, scale


def _validate_market_frame(frame: pd.DataFrame, price_column: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("frame must use a DatetimeIndex")
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError("frame index must be unique and increasing")
    if price_column not in frame:
        raise KeyError(f"missing market price column: {price_column}")


def _raw_label_components(
    frame: pd.DataFrame,
    price_column: str,
    specification: LabelSpecification | None = None,
) -> pd.DataFrame:
    resolved = specification or load_label_spec()
    price = pd.to_numeric(frame[price_column], errors="coerce").astype(float)
    log_price = np.log(price.where(price > 0.0))
    weekly_return = log_price.diff(1)

    output: dict[str, pd.Series] = {}
    for lookback in resolved.windows.direction:
        vol = weekly_return.rolling(
            lookback,
            min_periods=resolved.min_periods["direction"][lookback],
        ).std(ddof=0)
        output[f"trend_{lookback}w"] = log_price.diff(lookback) / (
            vol.replace(0.0, np.nan) * np.sqrt(float(lookback))
        )
    for window in resolved.windows.volatility:
        output[f"vol_{window}w"] = weekly_return.rolling(
            window,
            min_periods=resolved.min_periods["volatility"][window],
        ).std(ddof=0) * np.sqrt(52.0)
    for window in resolved.windows.drawdown:
        peak = price.rolling(
            window,
            min_periods=resolved.min_periods["drawdown"][window],
        ).max()
        output[f"drawdown_{window}w"] = -(price / peak - 1.0)
    return pd.DataFrame(output, index=frame.index).replace(
        [np.inf, -np.inf], np.nan
    )


class CausalRegimeLabeler:
    """Fit train-only thresholds and label observations without backdating.

    The label intentionally uses market price behaviour only.  Macro, rates,
    credit, FX, and corporate features remain eligible predictors without
    becoming circular ingredients of the supervised target.
    """

    def __init__(self, config: RegimeLabelConfig | None = None) -> None:
        self.config = config or RegimeLabelConfig()
        self.component_stats_: dict[str, tuple[float, float]] | None = None
        self.composite_stats_: dict[str, tuple[float, float]] | None = None
        self.lower_threshold_: float | None = None
        self.upper_threshold_: float | None = None
        self.train_end_: pd.Timestamp | None = None

    @property
    def is_fitted(self) -> bool:
        return self.component_stats_ is not None

    def fit(self, train_frame: pd.DataFrame) -> "CausalRegimeLabeler":
        _validate_market_frame(train_frame, self.config.price_column)
        raw = _raw_label_components(
            train_frame,
            self.config.price_column,
            self.config.specification,
        )
        component_stats: dict[str, tuple[float, float]] = {}
        standardized = pd.DataFrame(index=raw.index)
        for column in raw:
            centre, scale = _robust_location_scale(
                raw[column], self.config.specification.scaling
            )
            component_stats[column] = (centre, scale)
            standardized[column] = (raw[column] - centre) / scale

        trend_raw = standardized[["trend_13w", "trend_26w"]].mean(axis=1)
        stress_raw = standardized[
            ["vol_4w", "vol_13w", "drawdown_13w", "drawdown_52w"]
        ].mean(axis=1)
        trend_stats = _robust_location_scale(
            trend_raw, self.config.specification.scaling
        )
        stress_stats = _robust_location_scale(
            stress_raw, self.config.specification.scaling
        )
        risk_score = (
            (trend_raw - trend_stats[0]) / trend_stats[1]
            - (stress_raw - stress_stats[0]) / stress_stats[1]
        )
        finite = risk_score.dropna()
        if len(finite) < self.config.minimum_fit_observations:
            raise ValueError(
                "not enough finite training observations for reference labels: "
                f"{len(finite)} < {self.config.minimum_fit_observations}"
            )
        lower = float(finite.quantile(self.config.lower_quantile))
        upper = float(finite.quantile(self.config.upper_quantile))
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            raise ValueError("training risk score does not yield distinct thresholds")

        self.component_stats_ = component_stats
        self.composite_stats_ = {"trend": trend_stats, "stress": stress_stats}
        self.lower_threshold_ = lower
        self.upper_threshold_ = upper
        self.train_end_ = pd.Timestamp(train_frame.index[-1])
        return self

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("CausalRegimeLabeler must be fit before transform")

    def score_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return transparent trend, stress, and combined risk scores."""

        self._require_fitted()
        _validate_market_frame(frame, self.config.price_column)
        assert self.component_stats_ is not None
        assert self.composite_stats_ is not None
        raw = _raw_label_components(
            frame,
            self.config.price_column,
            self.config.specification,
        )
        standardized = pd.DataFrame(index=raw.index)
        for column, (centre, scale) in self.component_stats_.items():
            standardized[column] = (raw[column] - centre) / scale
        trend_raw = standardized[["trend_13w", "trend_26w"]].mean(axis=1)
        stress_raw = standardized[
            ["vol_4w", "vol_13w", "drawdown_13w", "drawdown_52w"]
        ].mean(axis=1)
        trend_centre, trend_scale = self.composite_stats_["trend"]
        stress_centre, stress_scale = self.composite_stats_["stress"]
        trend_score = (trend_raw - trend_centre) / trend_scale
        stress_score = (stress_raw - stress_centre) / stress_scale
        return pd.DataFrame(
            {
                "trend_score": trend_score,
                "stress_score": stress_score,
                "risk_score": trend_score - stress_score,
            },
            index=frame.index,
        )

    def transform(
        self,
        frame: pd.DataFrame,
        *,
        initial_state: str | None = None,
    ) -> pd.Series:
        """Assign causal states sequentially with hysteresis and no backdating."""

        initial_state = initial_state or self.config.specification.initial_state
        if initial_state not in STATE_ORDER:
            raise ValueError(f"initial_state must be one of {STATE_ORDER}")
        scores = self.score_frame(frame)["risk_score"]
        assert self.lower_threshold_ is not None
        assert self.upper_threshold_ is not None
        lower = self.lower_threshold_
        upper = self.upper_threshold_
        margin = (upper - lower) * self.config.hysteresis_fraction
        state = initial_state
        labels: list[str] = []
        for value in scores.to_numpy(dtype=float):
            if not np.isfinite(value):
                labels.append(state)
                continue
            if state == "transition":
                if value <= lower:
                    state = "risk_off"
                elif value >= upper:
                    state = "risk_on"
            elif state == "risk_on":
                if value <= lower - margin:
                    state = "risk_off"
                elif value < upper - margin:
                    state = "transition"
            else:  # risk_off
                if value >= upper + margin:
                    state = "risk_on"
                elif value > lower + margin:
                    state = "transition"
            labels.append(state)
        return pd.Series(labels, index=frame.index, name="regime", dtype="object")

    def state_memberships(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return distance-to-anchor membership, never a state posterior."""

        scores = self.score_frame(frame)["risk_score"].to_numpy(dtype=float)
        assert self.lower_threshold_ is not None
        assert self.upper_threshold_ is not None
        width = max(self.upper_threshold_ - self.lower_threshold_, 1e-6)
        midpoint = (self.lower_threshold_ + self.upper_threshold_) / 2.0
        references = {
            "lower": self.lower_threshold_,
            "midpoint": midpoint,
            "upper": self.upper_threshold_,
        }
        anchors = np.asarray(
            [
                references[self.config.specification.membership.anchors[state].reference]
                + width
                * self.config.specification.membership.anchors[state].width_multiplier
                for state in STATE_ORDER
            ],
            dtype=float,
        )
        scaled_distance = (scores[:, None] - anchors[None, :]) / width
        logits = -(scaled_distance**2) / self.config.probability_temperature
        missing = ~np.isfinite(scores)
        missing_logits = np.full(
            len(STATE_ORDER),
            self.config.specification.membership.missing_logit_floor,
            dtype=float,
        )
        missing_logits[
            STATE_ORDER.index(self.config.specification.membership.missing_state)
        ] = 0.0
        logits[missing] = missing_logits
        logits -= np.max(logits, axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return pd.DataFrame(probabilities, index=frame.index, columns=STATE_ORDER)

    def state_probabilities(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Compatibility alias for non-posterior distance memberships.

        The historical name is retained because it is part of the v1 output
        contract.  These values are normalized distance-to-anchor memberships,
        not HMM/Bayesian posterior probabilities.
        """

        return self.state_memberships(frame)

    def fit_transform(self, train_frame: pd.DataFrame) -> pd.Series:
        return self.fit(train_frame).transform(train_frame)


__all__ = [
    "STATE_ORDER",
    "CausalRegimeLabeler",
    "RegimeLabelConfig",
]
