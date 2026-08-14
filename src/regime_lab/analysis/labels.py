"""Transparent, causal reference labels for US-equity market regimes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from regime_lab.schema import STATE_ORDER


@dataclass(frozen=True)
class RegimeLabelConfig:
    """Configuration for the market-only three-state reference label."""

    price_column: str = "spy_close"
    lower_quantile: float = 0.30
    upper_quantile: float = 0.70
    hysteresis_fraction: float = 0.15
    probability_temperature: float = 0.75
    minimum_fit_observations: int = 26

    def __post_init__(self) -> None:
        if not 0.0 < self.lower_quantile < self.upper_quantile < 1.0:
            raise ValueError("quantiles must satisfy 0 < lower < upper < 1")
        if not 0.0 <= self.hysteresis_fraction < 0.5:
            raise ValueError("hysteresis_fraction must be in [0, 0.5)")
        if self.probability_temperature <= 0.0:
            raise ValueError("probability_temperature must be positive")


def _robust_location_scale(series: pd.Series) -> tuple[float, float]:
    finite = pd.to_numeric(series, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if finite.empty:
        raise ValueError("cannot fit regime labeler without finite observations")
    centre = float(finite.median())
    q25, q75 = finite.quantile([0.25, 0.75]).to_numpy(dtype=float)
    scale = float((q75 - q25) / 1.349)
    if not np.isfinite(scale) or scale <= 1e-12:
        mad = float((finite - centre).abs().median() * 1.4826)
        scale = mad if np.isfinite(mad) and mad > 1e-12 else 1.0
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


def _raw_label_components(frame: pd.DataFrame, price_column: str) -> pd.DataFrame:
    price = pd.to_numeric(frame[price_column], errors="coerce").astype(float)
    log_price = np.log(price.where(price > 0.0))
    weekly_return = log_price.diff(1)

    output: dict[str, pd.Series] = {}
    for lookback in (13, 26):
        vol = weekly_return.rolling(
            lookback, min_periods=max(4, lookback // 2)
        ).std(ddof=0)
        output[f"trend_{lookback}w"] = log_price.diff(lookback) / (
            vol.replace(0.0, np.nan) * np.sqrt(float(lookback))
        )
    for window in (4, 13):
        output[f"vol_{window}w"] = weekly_return.rolling(
            window, min_periods=max(2, window // 2)
        ).std(ddof=0) * np.sqrt(52.0)
    for window in (13, 52):
        peak = price.rolling(window, min_periods=max(4, window // 2)).max()
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
        raw = _raw_label_components(train_frame, self.config.price_column)
        component_stats: dict[str, tuple[float, float]] = {}
        standardized = pd.DataFrame(index=raw.index)
        for column in raw:
            centre, scale = _robust_location_scale(raw[column])
            component_stats[column] = (centre, scale)
            standardized[column] = (raw[column] - centre) / scale

        trend_raw = standardized[["trend_13w", "trend_26w"]].mean(axis=1)
        stress_raw = standardized[
            ["vol_4w", "vol_13w", "drawdown_13w", "drawdown_52w"]
        ].mean(axis=1)
        trend_stats = _robust_location_scale(trend_raw)
        stress_stats = _robust_location_scale(stress_raw)
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
        raw = _raw_label_components(frame, self.config.price_column)
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
        initial_state: str = "transition",
    ) -> pd.Series:
        """Assign causal states sequentially with hysteresis and no backdating."""

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

    def state_probabilities(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Map the continuous risk score to state-order-aligned soft evidence."""

        scores = self.score_frame(frame)["risk_score"].to_numpy(dtype=float)
        assert self.lower_threshold_ is not None
        assert self.upper_threshold_ is not None
        width = max(self.upper_threshold_ - self.lower_threshold_, 1e-6)
        anchors = np.array(
            [
                self.upper_threshold_ + width / 2.0,
                (self.lower_threshold_ + self.upper_threshold_) / 2.0,
                self.lower_threshold_ - width / 2.0,
            ]
        )
        scaled_distance = (scores[:, None] - anchors[None, :]) / width
        logits = -(scaled_distance**2) / self.config.probability_temperature
        missing = ~np.isfinite(scores)
        logits[missing] = np.array([-20.0, 0.0, -20.0])
        logits -= np.max(logits, axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return pd.DataFrame(probabilities, index=frame.index, columns=STATE_ORDER)

    def fit_transform(self, train_frame: pd.DataFrame) -> pd.Series:
        return self.fit(train_frame).transform(train_frame)


__all__ = [
    "STATE_ORDER",
    "CausalRegimeLabeler",
    "RegimeLabelConfig",
]
