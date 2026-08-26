"""Opt-in, point-in-time total-return label challengers.

Nothing in this module is selected by the operating pipeline.  The two
implementable challengers use only backward-looking transforms, freeze robust
scalers and thresholds on the configured initial prefix, and keep equal
weights inside each component block.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from regime_lab.analysis.label_spec import LabelSpecification, load_label_spec
from regime_lab.analysis.labels import _robust_location_scale
from regime_lab.analysis.pit_total_return import (
    CORPORATE_ACTION_CONTRACT,
    PITTotalReturnPanel,
    validate_pit_total_return_panel,
)
from regime_lab.schema import STATE_ORDER


IMPLEMENTABLE_LABEL_SPECS: tuple[str, ...] = (
    "v2_spy_pit_total_return",
    "v2_broad_equity",
)
MEMBERSHIP_SEMANTICS = "distance_to_anchor_not_posterior"


def _validate_research_panel(
    panel: PITTotalReturnPanel,
    specification: LabelSpecification,
) -> pd.DataFrame:
    if not isinstance(panel, PITTotalReturnPanel):
        raise TypeError(
            "v2 label input must be a provenance-bound PITTotalReturnPanel"
        )
    validate_pit_total_return_panel(panel)
    frame = panel.frame
    if panel.corporate_action_contract != CORPORATE_ACTION_CONTRACT:
        raise ValueError("PIT panel corporate-action contract is invalid")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("frame must use a DatetimeIndex")
    if frame.empty:
        raise ValueError("frame must not be empty")
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError("frame index must be unique and increasing")
    required = [item.column for item in specification.series]
    required_symbols = {item.symbol for item in specification.series}
    if set(panel.results) != required_symbols:
        raise ValueError(
            "PIT panel symbols must exactly match the selected label specification"
        )
    if set(frame.columns) != set(required):
        raise ValueError(
            "PIT panel columns must exactly match the selected label specification"
        )
    missing = [column for column in required if column not in frame]
    if missing:
        raise KeyError(f"missing label input columns: {missing}")
    numeric = frame.loc[:, required].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError("label input columns must not contain infinities")
    if (values[np.isfinite(values)] <= 0.0).any():
        raise ValueError("total-return indices must be positive where observed")
    return frame


def _series_log_levels(
    frame: pd.DataFrame,
    specification: LabelSpecification,
    *,
    block: str,
) -> dict[str, pd.Series]:
    return {
        item.symbol: np.log(
            pd.to_numeric(frame[item.column], errors="coerce").astype(float)
        )
        for item in specification.series_for_block(block)
    }


def _direction_components(
    frame: pd.DataFrame,
    specification: LabelSpecification,
) -> pd.DataFrame:
    output: dict[str, pd.Series] = {}
    for symbol, log_level in _series_log_levels(
        frame, specification, block="direction"
    ).items():
        weekly_return = log_level.diff(1)
        for window in specification.windows.direction:
            volatility = weekly_return.rolling(
                window,
                min_periods=specification.min_periods["direction"][window],
            ).std(ddof=0)
            output[f"direction__{symbol.lower()}__trend_{window}w"] = (
                log_level.diff(window)
                / (volatility.replace(0.0, np.nan) * np.sqrt(float(window)))
            )
    return pd.DataFrame(output, index=frame.index, dtype=float).replace(
        [np.inf, -np.inf], np.nan
    )


def _stress_components(
    frame: pd.DataFrame,
    specification: LabelSpecification,
) -> pd.DataFrame:
    output: dict[str, pd.Series] = {}
    for series in specification.series_for_block("stress"):
        price = pd.to_numeric(frame[series.column], errors="coerce").astype(float)
        log_level = np.log(price)
        weekly_return = log_level.diff(1)
        prefix = f"stress__{series.symbol.lower()}"
        for window in specification.windows.volatility:
            output[f"{prefix}__vol_{window}w"] = weekly_return.rolling(
                window,
                min_periods=specification.min_periods["volatility"][window],
            ).std(ddof=0) * np.sqrt(52.0)
        for window in specification.windows.drawdown:
            peak = price.rolling(
                window,
                min_periods=specification.min_periods["drawdown"][window],
            ).max()
            output[f"{prefix}__drawdown_{window}w"] = -(price / peak - 1.0)
    return pd.DataFrame(output, index=frame.index, dtype=float).replace(
        [np.inf, -np.inf], np.nan
    )


def _complete_equal_weight_share(changes: pd.DataFrame) -> pd.Series:
    """Return a fixed-denominator positive share without missing-name reweighting."""

    complete = changes.notna().all(axis=1)
    share = changes.gt(0.0).sum(axis=1).astype(float) / float(changes.shape[1])
    return share.where(complete)


def _breadth_components(
    frame: pd.DataFrame,
    specification: LabelSpecification,
) -> pd.DataFrame:
    levels = _series_log_levels(frame, specification, block="breadth")
    if not levels:
        return pd.DataFrame(index=frame.index)
    log_levels = pd.DataFrame(levels, index=frame.index, dtype=float)
    output: dict[str, pd.Series] = {}
    for window in specification.windows.breadth_return:
        changes = log_levels.diff(window)
        output[f"breadth__positive_return_share_{window}w"] = (
            _complete_equal_weight_share(changes)
        )
    for window in specification.windows.breadth_trend:
        changes = log_levels.diff(window)
        output[f"breadth__positive_trend_share_{window}w"] = (
            _complete_equal_weight_share(changes)
        )
    return pd.DataFrame(output, index=frame.index, dtype=float)


def raw_research_label_components(
    panel: PITTotalReturnPanel,
    specification: LabelSpecification,
) -> Mapping[str, pd.DataFrame]:
    """Build causal raw component matrices in deterministic block order."""

    frame = _validate_research_panel(panel, specification)
    return {
        "direction": _direction_components(frame, specification),
        "breadth": _breadth_components(frame, specification),
        "stress": _stress_components(frame, specification),
    }


class ResearchRegimeLabeler:
    """Fit and apply one registered v2 label challenger."""

    def __init__(self, spec_id: str) -> None:
        if spec_id not in IMPLEMENTABLE_LABEL_SPECS:
            raise ValueError(
                f"implementable challenger must be one of {IMPLEMENTABLE_LABEL_SPECS}"
            )
        self.specification = load_label_spec(spec_id)
        self.component_stats_: dict[str, dict[str, tuple[float, float]]] | None = None
        self.block_stats_: dict[str, tuple[float, float]] | None = None
        self.lower_threshold_: float | None = None
        self.upper_threshold_: float | None = None
        self.train_end_: pd.Timestamp | None = None
        self.fit_row_count_: int | None = None
        self.fit_input_snapshot_sha256_: str | None = None
        self.evidence_track_: str | None = None

    @property
    def is_fitted(self) -> bool:
        return self.component_stats_ is not None

    @property
    def spec_id(self) -> str:
        return self.specification.spec_id

    @property
    def spec_sha256(self) -> str:
        return self.specification.spec_sha256

    @property
    def membership_semantics(self) -> str:
        return self.specification.membership.semantics

    def _fit_prefix(self, train_panel: PITTotalReturnPanel) -> PITTotalReturnPanel:
        required = self.specification.fit_period.reference_fit_weeks
        if len(train_panel) < required:
            raise ValueError(
                f"{self.spec_id} requires at least {required} fit-prefix rows"
            )
        if self.specification.fit_period.mode == "fixed_initial_prefix_then_frozen":
            return train_panel.slice_rows(required)
        return train_panel

    def fit(self, train_panel: PITTotalReturnPanel) -> "ResearchRegimeLabeler":
        _validate_research_panel(train_panel, self.specification)
        fit_panel = self._fit_prefix(train_panel)
        fit_frame = fit_panel.frame
        raw_blocks = raw_research_label_components(fit_panel, self.specification)
        component_stats: dict[str, dict[str, tuple[float, float]]] = {}
        block_stats: dict[str, tuple[float, float]] = {}
        block_scores: dict[str, pd.Series] = {}
        for block in ("direction", "breadth", "stress"):
            weight = float(self.specification.component_weights[block])
            raw = raw_blocks[block]
            if weight == 0.0:
                if not raw.empty:
                    raise ValueError(f"inactive block {block} unexpectedly has inputs")
                continue
            if raw.empty:
                raise ValueError(f"active block {block} has no raw components")
            standardized = pd.DataFrame(index=raw.index)
            stats: dict[str, tuple[float, float]] = {}
            for column in raw:
                centre, scale = _robust_location_scale(
                    raw[column], self.specification.scaling
                )
                stats[column] = (centre, scale)
                standardized[column] = (raw[column] - centre) / scale
            component_stats[block] = stats
            raw_block_score = standardized.mean(axis=1)
            block_centre, block_scale = _robust_location_scale(
                raw_block_score, self.specification.scaling
            )
            block_stats[block] = (block_centre, block_scale)
            block_scores[block] = (
                raw_block_score - block_centre
            ) / block_scale

        risk_score = pd.Series(0.0, index=fit_frame.index, dtype=float)
        for block, score in block_scores.items():
            risk_score = risk_score + (
                float(self.specification.component_weights[block]) * score
            )
        finite = risk_score.replace([np.inf, -np.inf], np.nan).dropna()
        minimum = self.specification.fit_period.minimum_finite_observations
        if len(finite) < minimum:
            raise ValueError(
                f"not enough finite fit-prefix observations for {self.spec_id}: "
                f"{len(finite)} < {minimum}"
            )
        lower = float(finite.quantile(self.specification.lower_quantile))
        upper = float(finite.quantile(self.specification.upper_quantile))
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            raise ValueError("training risk score does not yield distinct thresholds")

        self.component_stats_ = component_stats
        self.block_stats_ = block_stats
        self.lower_threshold_ = lower
        self.upper_threshold_ = upper
        self.train_end_ = pd.Timestamp(fit_frame.index[-1])
        self.fit_row_count_ = len(fit_frame)
        self.fit_input_snapshot_sha256_ = fit_panel.input_snapshot_sha256
        self.evidence_track_ = fit_panel.evidence_track
        return self

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("ResearchRegimeLabeler must be fit before transform")

    def _validated_transform_panel(
        self,
        panel: PITTotalReturnPanel,
    ) -> pd.DataFrame:
        frame = _validate_research_panel(panel, self.specification)
        assert self.fit_row_count_ is not None
        assert self.fit_input_snapshot_sha256_ is not None
        assert self.evidence_track_ is not None
        if panel.evidence_track != self.evidence_track_:
            raise ValueError("label transform cannot change evidence track")
        if len(panel) < self.fit_row_count_:
            raise ValueError("label transform panel is shorter than the frozen fit prefix")
        observed_fit_hash = panel.slice_rows(
            self.fit_row_count_
        ).input_snapshot_sha256
        if observed_fit_hash != self.fit_input_snapshot_sha256_:
            raise ValueError("label transform revised the frozen PIT fit prefix")
        return frame

    def score_frame(self, panel: PITTotalReturnPanel) -> pd.DataFrame:
        """Return train-scaled component blocks and their signed risk score."""

        self._require_fitted()
        frame = self._validated_transform_panel(panel)
        assert self.component_stats_ is not None
        assert self.block_stats_ is not None
        raw_blocks = raw_research_label_components(panel, self.specification)
        output = pd.DataFrame(index=frame.index)
        risk_score = pd.Series(0.0, index=frame.index, dtype=float)
        for block in ("direction", "breadth", "stress"):
            weight = float(self.specification.component_weights[block])
            if weight == 0.0:
                output[f"{block}_score"] = 0.0
                continue
            stats = self.component_stats_[block]
            standardized = pd.DataFrame(index=frame.index)
            for column, (centre, scale) in stats.items():
                standardized[column] = (raw_blocks[block][column] - centre) / scale
            raw_score = standardized.mean(axis=1)
            block_centre, block_scale = self.block_stats_[block]
            score = (raw_score - block_centre) / block_scale
            output[f"{block}_score"] = score
            risk_score = risk_score + weight * score
        output["risk_score"] = risk_score
        return output.replace([np.inf, -np.inf], np.nan)

    def transform(
        self,
        panel: PITTotalReturnPanel,
        *,
        initial_state: str | None = None,
    ) -> pd.Series:
        """Apply sequential hysteresis without revising any earlier state."""

        state = initial_state or self.specification.initial_state
        if state not in STATE_ORDER:
            raise ValueError(f"initial_state must be one of {STATE_ORDER}")
        scores = self.score_frame(panel)["risk_score"].to_numpy(dtype=float)
        assert self.lower_threshold_ is not None
        assert self.upper_threshold_ is not None
        lower = self.lower_threshold_
        upper = self.upper_threshold_
        margin = (
            upper - lower
        ) * self.specification.hysteresis_fraction
        labels: list[str] = []
        for value in scores:
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
            else:
                if value >= upper + margin:
                    state = "risk_on"
                elif value > lower + margin:
                    state = "transition"
            labels.append(state)
        return pd.Series(labels, index=panel.index, name="regime", dtype="object")

    def state_memberships(self, panel: PITTotalReturnPanel) -> pd.DataFrame:
        """Map risk scores to distance memberships, not posterior probability."""

        scores = self.score_frame(panel)["risk_score"].to_numpy(dtype=float)
        assert self.lower_threshold_ is not None
        assert self.upper_threshold_ is not None
        lower = self.lower_threshold_
        upper = self.upper_threshold_
        width = max(upper - lower, 1e-6)
        references = {
            "lower": lower,
            "midpoint": (lower + upper) / 2.0,
            "upper": upper,
        }
        anchors = np.asarray(
            [
                references[self.specification.membership.anchors[state].reference]
                + width
                * self.specification.membership.anchors[state].width_multiplier
                for state in STATE_ORDER
            ],
            dtype=float,
        )
        distance = (scores[:, None] - anchors[None, :]) / width
        logits = -(distance**2) / self.specification.membership.temperature
        missing = ~np.isfinite(scores)
        missing_logits = np.full(
            len(STATE_ORDER),
            self.specification.membership.missing_logit_floor,
            dtype=float,
        )
        missing_logits[
            STATE_ORDER.index(self.specification.membership.missing_state)
        ] = 0.0
        logits[missing] = missing_logits
        logits -= np.max(logits, axis=1, keepdims=True)
        membership = np.exp(logits)
        membership /= membership.sum(axis=1, keepdims=True)
        return pd.DataFrame(membership, index=panel.index, columns=STATE_ORDER)

    def fit_transform(self, train_panel: PITTotalReturnPanel) -> pd.Series:
        return self.fit(train_panel).transform(train_panel)


def make_research_labeler(spec_id: str) -> ResearchRegimeLabeler:
    return ResearchRegimeLabeler(spec_id)


__all__ = [
    "IMPLEMENTABLE_LABEL_SPECS",
    "MEMBERSHIP_SEMANTICS",
    "ResearchRegimeLabeler",
    "make_research_labeler",
    "raw_research_label_components",
]
