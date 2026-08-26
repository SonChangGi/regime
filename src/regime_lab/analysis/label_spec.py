"""Typed, hash-locked label-definition registry for private research.

The JSON document is the sole source for label inputs and numerical choices.
The Python lock below is deliberately limited to immutable version identities:
changing a registered definition without changing its version makes loading
fail before any labels can be produced.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

from regime_lab.schema import STATE_ORDER


DEFAULT_LABEL_SPEC_ID = "v1_spy_hysteresis"
LABEL_SPEC_SCHEMA_VERSION = "1.1.0"
LABEL_SPEC_VERSION_LOCK: Mapping[str, str] = MappingProxyType(
    {
        "market-causal-3state-v1": (
            "bec1600aea104985405d0d5c2b3706088b885ef7aa788c57c9575aa884c5f3a7"
        ),
        "market-causal-3state-v2-spy-pit-total-return": (
            "84ddcaa393c171d72d96f06d3b6e67834ddfa810c421e654b75bc85e1957c94f"
        ),
        "market-causal-3state-v2-broad-equity": (
            "1784783647572b452d4437f1d822d5b4b51801e21343fb0bcba709750f70021b"
        ),
    }
)
SHADOW_METHOD_VERSION_LOCK: Mapping[str, str] = MappingProxyType(
    {
        "hsmm-explicit-duration-shadow-v1": (
            "fbc4b68e1974f823f00adb8b41a709218e8de7e9ecb191c8929f5670755676b2"
        ),
        "pagan-sossounov-ex-post-v1": (
            "6bd0c04491f72b66120eee3365fde588e8f059b07b1beb995f2f9480d35fd70c"
        ),
    }
)
BLOCK_ORDER: tuple[str, ...] = ("direction", "breadth", "stress")
WINDOW_ORDER: tuple[str, ...] = (
    "direction",
    "volatility",
    "drawdown",
    "breadth_return",
    "breadth_trend",
)
FIXED_NINE_SECTORS: tuple[str, ...] = (
    "XLB",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLU",
    "XLV",
    "XLY",
)
BROAD_EQUITY_BREADTH_SYMBOLS: tuple[str, ...] = (
    "RSP",
    "IWM",
    *FIXED_NINE_SECTORS,
)


class LabelSpecError(ValueError):
    """Raised when a label definition is incomplete or its lock is broken."""


@dataclass(frozen=True)
class LabelSeriesSpec:
    symbol: str
    column: str
    adjustment: str
    blocks: tuple[str, ...]


@dataclass(frozen=True)
class LabelWindowSpec:
    direction: tuple[int, ...]
    volatility: tuple[int, ...]
    drawdown: tuple[int, ...]
    breadth_return: tuple[int, ...]
    breadth_trend: tuple[int, ...]

    def for_name(self, name: str) -> tuple[int, ...]:
        if name not in WINDOW_ORDER:
            raise KeyError(f"unknown label window family: {name}")
        return tuple(getattr(self, name))


@dataclass(frozen=True)
class LabelFitPeriodSpec:
    mode: str
    reference_fit_weeks: int
    minimum_finite_observations: int
    production_minimum_finite_observations: int
    rolling_window_weeks: int | None


@dataclass(frozen=True)
class LabelScalingSpec:
    method: str
    iqr_normalizer: float
    mad_normalizer: float
    scale_floor: float
    constant_fallback_scale: float
    fit_scope: str


@dataclass(frozen=True)
class MembershipAnchorSpec:
    reference: str
    width_multiplier: float


@dataclass(frozen=True)
class LabelMembershipSpec:
    method: str
    semantics: str
    anchors: Mapping[str, MembershipAnchorSpec]
    temperature: float
    missing_state: str
    missing_logit_floor: float


@dataclass(frozen=True)
class LabelSpecification:
    spec_id: str
    version: str
    status: str
    series: tuple[LabelSeriesSpec, ...]
    windows: LabelWindowSpec
    min_periods: Mapping[str, Mapping[int, int]]
    fit_period: LabelFitPeriodSpec
    scaling: LabelScalingSpec
    component_weights: Mapping[str, float]
    lower_quantile: float
    upper_quantile: float
    hysteresis_fraction: float
    initial_state: str
    membership: LabelMembershipSpec
    spec_sha256: str

    def series_for_block(self, block: str) -> tuple[LabelSeriesSpec, ...]:
        if block not in BLOCK_ORDER:
            raise KeyError(f"unknown label component block: {block}")
        return tuple(item for item in self.series if block in item.blocks)


@dataclass(frozen=True)
class ShadowLabelMethod:
    method_id: str
    version: str
    status: str
    role: str
    configuration: Mapping[str, Any]
    spec_sha256: str
    results: None


@dataclass(frozen=True)
class LabelSpecRegistry:
    schema_version: str
    default_spec: str
    state_order: tuple[str, ...]
    specs: Mapping[str, LabelSpecification]
    shadow_methods: Mapping[str, ShadowLabelMethod]

    def get(self, spec_id: str | None = None) -> LabelSpecification:
        resolved = self.default_spec if spec_id is None else str(spec_id)
        try:
            return self.specs[resolved]
        except KeyError as exc:
            raise LabelSpecError(f"unknown label spec: {resolved}") from exc


def default_label_spec_path() -> Path:
    return Path(__file__).resolve().parents[3] / "config" / "label-spec.json"


def canonical_label_spec_sha256(
    spec_id: str,
    raw_spec: Mapping[str, Any],
) -> str:
    """Hash one complete spec with canonical JSON and its registry key."""

    encoded = json.dumps(
        {"spec_id": str(spec_id), "spec": raw_spec},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_shadow_method_sha256(
    method_id: str,
    raw_method: Mapping[str, Any],
) -> str:
    encoded = json.dumps(
        {"method_id": str(method_id), "method": raw_method},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mapping(value: object, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LabelSpecError(f"{context} must be an object")
    return value


def _sequence(value: object, *, context: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise LabelSpecError(f"{context} must be an array")
    return value


def _nonempty_text(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LabelSpecError(f"{context} must be a non-empty string")
    return value.strip()


def _finite_float(
    value: object,
    *,
    context: str,
    positive: bool = False,
) -> float:
    if isinstance(value, bool):
        raise LabelSpecError(f"{context} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise LabelSpecError(f"{context} must be a finite number") from exc
    if number != number or number in (float("inf"), float("-inf")):
        raise LabelSpecError(f"{context} must be finite")
    if positive and number <= 0.0:
        raise LabelSpecError(f"{context} must be positive")
    return number


def _positive_integer(value: object, *, context: str) -> int:
    if isinstance(value, bool):
        raise LabelSpecError(f"{context} must be a positive integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise LabelSpecError(f"{context} must be a positive integer") from exc
    if number != value or number < 1:
        raise LabelSpecError(f"{context} must be a positive integer")
    return number


def _parse_series(spec_id: str, raw: object) -> tuple[LabelSeriesSpec, ...]:
    rows = _sequence(raw, context=f"specs.{spec_id}.series")
    if not rows:
        raise LabelSpecError(f"specs.{spec_id}.series must not be empty")
    parsed: list[LabelSeriesSpec] = []
    for position, item in enumerate(rows):
        context = f"specs.{spec_id}.series[{position}]"
        row = _mapping(item, context=context)
        blocks = tuple(
            _nonempty_text(value, context=f"{context}.blocks")
            for value in _sequence(row.get("blocks"), context=f"{context}.blocks")
        )
        if not blocks or len(blocks) != len(set(blocks)):
            raise LabelSpecError(f"{context}.blocks must be non-empty and unique")
        unknown = sorted(set(blocks).difference(BLOCK_ORDER))
        if unknown:
            raise LabelSpecError(f"{context}.blocks contains unknown values: {unknown}")
        parsed.append(
            LabelSeriesSpec(
                symbol=_nonempty_text(row.get("symbol"), context=f"{context}.symbol"),
                column=_nonempty_text(row.get("column"), context=f"{context}.column"),
                adjustment=_nonempty_text(
                    row.get("adjustment"), context=f"{context}.adjustment"
                ),
                blocks=blocks,
            )
        )
    symbols = [item.symbol for item in parsed]
    columns = [item.column for item in parsed]
    if len(symbols) != len(set(symbols)) or len(columns) != len(set(columns)):
        raise LabelSpecError(f"specs.{spec_id}.series symbols and columns must be unique")
    return tuple(parsed)


def _parse_windows(
    spec_id: str,
    raw_windows: object,
    raw_min_periods: object,
) -> tuple[LabelWindowSpec, Mapping[str, Mapping[int, int]]]:
    windows = _mapping(raw_windows, context=f"specs.{spec_id}.windows")
    periods = _mapping(raw_min_periods, context=f"specs.{spec_id}.min_periods")
    if set(windows) != set(WINDOW_ORDER) or set(periods) != set(WINDOW_ORDER):
        raise LabelSpecError(
            f"specs.{spec_id} windows and min_periods must be exactly {WINDOW_ORDER}"
        )
    parsed_windows: dict[str, tuple[int, ...]] = {}
    parsed_periods: dict[str, Mapping[int, int]] = {}
    for name in WINDOW_ORDER:
        values = tuple(
            _positive_integer(item, context=f"specs.{spec_id}.windows.{name}")
            for item in _sequence(
                windows[name], context=f"specs.{spec_id}.windows.{name}"
            )
        )
        if len(values) != len(set(values)) or tuple(sorted(values)) != values:
            raise LabelSpecError(
                f"specs.{spec_id}.windows.{name} must be sorted and unique"
            )
        period_rows = _mapping(
            periods[name], context=f"specs.{spec_id}.min_periods.{name}"
        )
        if set(period_rows) != {str(value) for value in values}:
            raise LabelSpecError(
                f"specs.{spec_id}.min_periods.{name} must match its windows"
            )
        resolved_periods: dict[int, int] = {}
        for window in values:
            minimum = _positive_integer(
                period_rows[str(window)],
                context=f"specs.{spec_id}.min_periods.{name}.{window}",
            )
            if minimum > window:
                raise LabelSpecError(
                    f"specs.{spec_id}.min_periods.{name}.{window} exceeds window"
                )
            resolved_periods[window] = minimum
        parsed_windows[name] = values
        parsed_periods[name] = resolved_periods
    immutable_periods = MappingProxyType(
        {
            name: MappingProxyType(dict(values))
            for name, values in parsed_periods.items()
        }
    )
    return LabelWindowSpec(**parsed_windows), immutable_periods


def _parse_fit_period(spec_id: str, raw: object) -> LabelFitPeriodSpec:
    row = _mapping(raw, context=f"specs.{spec_id}.fit_period")
    expected = {
        "mode",
        "reference_fit_weeks",
        "minimum_finite_observations",
        "production_minimum_finite_observations",
        "rolling_window_weeks",
    }
    if set(row) != expected:
        raise LabelSpecError(f"specs.{spec_id}.fit_period keys must be {sorted(expected)}")
    mode = _nonempty_text(row["mode"], context=f"specs.{spec_id}.fit_period.mode")
    if mode not in {"caller_supplied_prefix", "fixed_initial_prefix_then_frozen"}:
        raise LabelSpecError(f"specs.{spec_id}.fit_period.mode is unsupported")
    rolling = row["rolling_window_weeks"]
    if rolling is not None:
        rolling = _positive_integer(
            rolling, context=f"specs.{spec_id}.fit_period.rolling_window_weeks"
        )
    result = LabelFitPeriodSpec(
        mode=mode,
        reference_fit_weeks=_positive_integer(
            row["reference_fit_weeks"],
            context=f"specs.{spec_id}.fit_period.reference_fit_weeks",
        ),
        minimum_finite_observations=_positive_integer(
            row["minimum_finite_observations"],
            context=f"specs.{spec_id}.fit_period.minimum_finite_observations",
        ),
        production_minimum_finite_observations=_positive_integer(
            row["production_minimum_finite_observations"],
            context=(
                f"specs.{spec_id}.fit_period."
                "production_minimum_finite_observations"
            ),
        ),
        rolling_window_weeks=rolling,
    )
    if result.reference_fit_weeks < result.minimum_finite_observations:
        raise LabelSpecError(f"specs.{spec_id}.fit_period cannot satisfy its minimum")
    return result


def _parse_scaling(spec_id: str, raw: object) -> LabelScalingSpec:
    row = _mapping(raw, context=f"specs.{spec_id}.scaling")
    expected = {
        "method",
        "iqr_normalizer",
        "mad_normalizer",
        "scale_floor",
        "constant_fallback_scale",
        "fit_scope",
    }
    if set(row) != expected:
        raise LabelSpecError(f"specs.{spec_id}.scaling keys must be {sorted(expected)}")
    result = LabelScalingSpec(
        method=_nonempty_text(row["method"], context=f"specs.{spec_id}.scaling.method"),
        iqr_normalizer=_finite_float(
            row["iqr_normalizer"],
            context=f"specs.{spec_id}.scaling.iqr_normalizer",
            positive=True,
        ),
        mad_normalizer=_finite_float(
            row["mad_normalizer"],
            context=f"specs.{spec_id}.scaling.mad_normalizer",
            positive=True,
        ),
        scale_floor=_finite_float(
            row["scale_floor"],
            context=f"specs.{spec_id}.scaling.scale_floor",
            positive=True,
        ),
        constant_fallback_scale=_finite_float(
            row["constant_fallback_scale"],
            context=f"specs.{spec_id}.scaling.constant_fallback_scale",
            positive=True,
        ),
        fit_scope=_nonempty_text(
            row["fit_scope"], context=f"specs.{spec_id}.scaling.fit_scope"
        ),
    )
    if result.method != "median_iqr_then_mad" or result.fit_scope != "train_only":
        raise LabelSpecError(f"specs.{spec_id}.scaling must remain train-only robust")
    return result


def _parse_membership(spec_id: str, raw: object) -> LabelMembershipSpec:
    row = _mapping(raw, context=f"specs.{spec_id}.membership")
    anchors_raw = _mapping(
        row.get("anchors"), context=f"specs.{spec_id}.membership.anchors"
    )
    if set(anchors_raw) != set(STATE_ORDER):
        raise LabelSpecError(
            f"specs.{spec_id}.membership.anchors must be exactly {STATE_ORDER}"
        )
    anchors: dict[str, MembershipAnchorSpec] = {}
    for state in STATE_ORDER:
        anchor = _mapping(
            anchors_raw[state],
            context=f"specs.{spec_id}.membership.anchors.{state}",
        )
        if set(anchor) != {"reference", "width_multiplier"}:
            raise LabelSpecError(
                f"specs.{spec_id}.membership.anchors.{state} has invalid keys"
            )
        reference = _nonempty_text(
            anchor["reference"],
            context=f"specs.{spec_id}.membership.anchors.{state}.reference",
        )
        if reference not in {"lower", "midpoint", "upper"}:
            raise LabelSpecError(
                f"specs.{spec_id}.membership.anchors.{state}.reference is invalid"
            )
        anchors[state] = MembershipAnchorSpec(
            reference=reference,
            width_multiplier=_finite_float(
                anchor["width_multiplier"],
                context=(
                    f"specs.{spec_id}.membership.anchors."
                    f"{state}.width_multiplier"
                ),
            ),
        )
    result = LabelMembershipSpec(
        method=_nonempty_text(
            row.get("method"), context=f"specs.{spec_id}.membership.method"
        ),
        semantics=_nonempty_text(
            row.get("semantics"), context=f"specs.{spec_id}.membership.semantics"
        ),
        anchors=MappingProxyType(anchors),
        temperature=_finite_float(
            row.get("temperature"),
            context=f"specs.{spec_id}.membership.temperature",
            positive=True,
        ),
        missing_state=_nonempty_text(
            row.get("missing_state"),
            context=f"specs.{spec_id}.membership.missing_state",
        ),
        missing_logit_floor=_finite_float(
            row.get("missing_logit_floor"),
            context=f"specs.{spec_id}.membership.missing_logit_floor",
        ),
    )
    if result.method != "squared_distance_to_anchor_softmax":
        raise LabelSpecError(f"specs.{spec_id}.membership method is unsupported")
    if result.semantics != "distance_to_anchor_not_posterior":
        raise LabelSpecError(
            f"specs.{spec_id}.membership must explicitly be non-posterior"
        )
    if result.missing_state not in STATE_ORDER:
        raise LabelSpecError(f"specs.{spec_id}.membership missing state is invalid")
    return result


def _parse_spec(
    spec_id: str,
    raw: Mapping[str, Any],
    *,
    verify_lock: bool,
) -> LabelSpecification:
    expected = {
        "version",
        "status",
        "series",
        "windows",
        "min_periods",
        "fit_period",
        "scaling",
        "component_weights",
        "quantiles",
        "hysteresis",
        "initial_state",
        "membership",
    }
    if set(raw) != expected:
        raise LabelSpecError(
            f"specs.{spec_id} keys differ from the typed contract: "
            f"missing={sorted(expected.difference(raw))}, "
            f"unknown={sorted(set(raw).difference(expected))}"
        )
    version = _nonempty_text(raw["version"], context=f"specs.{spec_id}.version")
    spec_hash = canonical_label_spec_sha256(spec_id, raw)
    if verify_lock:
        expected_hash = LABEL_SPEC_VERSION_LOCK.get(version)
        if expected_hash is None:
            raise LabelSpecError(
                f"unregistered label spec version {version}; bump and register it"
            )
        if expected_hash != spec_hash:
            raise LabelSpecError(
                f"label spec {version} changed without a version bump: "
                f"expected {expected_hash}, got {spec_hash}"
            )
    series = _parse_series(spec_id, raw["series"])
    windows, min_periods = _parse_windows(
        spec_id, raw["windows"], raw["min_periods"]
    )
    weights_raw = _mapping(
        raw["component_weights"], context=f"specs.{spec_id}.component_weights"
    )
    if set(weights_raw) != set(BLOCK_ORDER):
        raise LabelSpecError(
            f"specs.{spec_id}.component_weights must be exactly {BLOCK_ORDER}"
        )
    weights = {
        block: _finite_float(
            weights_raw[block], context=f"specs.{spec_id}.component_weights.{block}"
        )
        for block in BLOCK_ORDER
    }
    for block, weight in weights.items():
        if weight != 0.0 and not any(block in item.blocks for item in series):
            raise LabelSpecError(
                f"specs.{spec_id} gives {block} weight without input series"
            )
    quantiles = _mapping(raw["quantiles"], context=f"specs.{spec_id}.quantiles")
    if set(quantiles) != {"lower", "upper"}:
        raise LabelSpecError(f"specs.{spec_id}.quantiles keys are invalid")
    lower = _finite_float(
        quantiles["lower"], context=f"specs.{spec_id}.quantiles.lower"
    )
    upper = _finite_float(
        quantiles["upper"], context=f"specs.{spec_id}.quantiles.upper"
    )
    if not 0.0 < lower < upper < 1.0:
        raise LabelSpecError(
            f"specs.{spec_id}.quantiles must satisfy 0 < lower < upper < 1"
        )
    hysteresis = _mapping(
        raw["hysteresis"], context=f"specs.{spec_id}.hysteresis"
    )
    if set(hysteresis) != {"fraction_of_threshold_width"}:
        raise LabelSpecError(f"specs.{spec_id}.hysteresis keys are invalid")
    hysteresis_fraction = _finite_float(
        hysteresis["fraction_of_threshold_width"],
        context=f"specs.{spec_id}.hysteresis.fraction_of_threshold_width",
    )
    if not 0.0 <= hysteresis_fraction < 0.5:
        raise LabelSpecError(f"specs.{spec_id}.hysteresis fraction is invalid")
    initial_state = _nonempty_text(
        raw["initial_state"], context=f"specs.{spec_id}.initial_state"
    )
    if initial_state not in STATE_ORDER:
        raise LabelSpecError(f"specs.{spec_id}.initial_state is invalid")

    result = LabelSpecification(
        spec_id=spec_id,
        version=version,
        status=_nonempty_text(raw["status"], context=f"specs.{spec_id}.status"),
        series=series,
        windows=windows,
        min_periods=min_periods,
        fit_period=_parse_fit_period(spec_id, raw["fit_period"]),
        scaling=_parse_scaling(spec_id, raw["scaling"]),
        component_weights=MappingProxyType(weights),
        lower_quantile=lower,
        upper_quantile=upper,
        hysteresis_fraction=hysteresis_fraction,
        initial_state=initial_state,
        membership=_parse_membership(spec_id, raw["membership"]),
        spec_sha256=spec_hash,
    )
    if spec_id == "v2_broad_equity":
        direction_symbols = tuple(
            item.symbol for item in result.series_for_block("direction")
        )
        breadth_symbols = tuple(
            item.symbol for item in result.series_for_block("breadth")
        )
        stress_symbols = tuple(
            item.symbol for item in result.series_for_block("stress")
        )
        if direction_symbols != ("SPY", "RSP", "IWM"):
            raise LabelSpecError("v2_broad_equity direction block must be SPY/RSP/IWM")
        if stress_symbols != ("SPY",):
            raise LabelSpecError("v2_broad_equity stress block must be SPY only")
        if breadth_symbols != BROAD_EQUITY_BREADTH_SYMBOLS:
            raise LabelSpecError(
                "v2_broad_equity breadth block must be RSP/IWM plus the fixed "
                "nine-sector list"
            )
    if spec_id.startswith("v2_") and any(
        item.adjustment != "point_in_time_total_return_index" for item in result.series
    ):
        raise LabelSpecError(f"{spec_id} must use point-in-time total-return inputs")
    return result


def load_label_spec_registry(
    path: str | Path | None = None,
    *,
    verify_lock: bool = True,
) -> LabelSpecRegistry:
    config_path = Path(path) if path is not None else default_label_spec_path()
    try:
        raw_document = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise LabelSpecError(f"label spec config not found: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise LabelSpecError(f"invalid label spec JSON: {config_path}: {exc}") from exc
    document = _mapping(raw_document, context="label spec document")
    expected = {
        "schema_version",
        "default_spec",
        "state_order",
        "specs",
        "shadow_methods",
    }
    if set(document) != expected:
        raise LabelSpecError("label spec document keys do not match the schema")
    schema_version = _nonempty_text(document["schema_version"], context="schema_version")
    if schema_version != LABEL_SPEC_SCHEMA_VERSION:
        raise LabelSpecError(
            f"label spec schema must be {LABEL_SPEC_SCHEMA_VERSION}"
        )
    state_order = tuple(
        _nonempty_text(item, context="state_order")
        for item in _sequence(document["state_order"], context="state_order")
    )
    if state_order != STATE_ORDER:
        raise LabelSpecError(f"state_order must be exactly {STATE_ORDER}")
    specs_raw = _mapping(document["specs"], context="specs")
    if not specs_raw:
        raise LabelSpecError("specs must not be empty")
    specs = {
        str(spec_id): _parse_spec(
            str(spec_id),
            _mapping(raw, context=f"specs.{spec_id}"),
            verify_lock=verify_lock,
        )
        for spec_id, raw in specs_raw.items()
    }
    versions = [item.version for item in specs.values()]
    if len(versions) != len(set(versions)):
        raise LabelSpecError("each label spec must have a unique version")
    default_spec = _nonempty_text(document["default_spec"], context="default_spec")
    if default_spec not in specs:
        raise LabelSpecError("default_spec is not registered")
    if default_spec != DEFAULT_LABEL_SPEC_ID:
        raise LabelSpecError(f"default_spec must remain {DEFAULT_LABEL_SPEC_ID}")

    shadows_raw = _mapping(document["shadow_methods"], context="shadow_methods")
    shadows: dict[str, ShadowLabelMethod] = {}
    for method_id, raw in shadows_raw.items():
        row = _mapping(raw, context=f"shadow_methods.{method_id}")
        expected_shadow_fields = {
            "version",
            "status",
            "role",
            "configuration",
            "results",
        }
        if set(row) != expected_shadow_fields or row["results"] is not None:
            raise LabelSpecError(
                f"shadow method {method_id} must be explicitly unrun with null results"
            )
        version = _nonempty_text(
            row["version"], context=f"shadow_methods.{method_id}.version"
        )
        status = _nonempty_text(row["status"], context=f"shadow_methods.{method_id}.status")
        if not status.endswith("_unrun"):
            raise LabelSpecError(f"shadow method {method_id} status must end in _unrun")
        configuration = _mapping(
            row["configuration"],
            context=f"shadow_methods.{method_id}.configuration",
        )
        if method_id == "hsmm_explicit_duration" and configuration:
            raise LabelSpecError("HSMM shadow configuration must remain empty until implemented")
        if method_id == "pagan_sossounov_bull_bear":
            expected_config = {
                "window_months",
                "censor_margin_months",
                "minimum_phase_months",
                "minimum_cycle_months",
                "large_move_threshold",
            }
            if set(configuration) != expected_config:
                raise LabelSpecError("Pagan-Sossounov configuration fields are invalid")
            normalized_config: Mapping[str, Any] = MappingProxyType(
                {
                    "window_months": _positive_integer(
                        configuration["window_months"],
                        context="shadow_methods.pagan_sossounov_bull_bear.configuration.window_months",
                    ),
                    "censor_margin_months": _positive_integer(
                        configuration["censor_margin_months"],
                        context="shadow_methods.pagan_sossounov_bull_bear.configuration.censor_margin_months",
                    ),
                    "minimum_phase_months": _positive_integer(
                        configuration["minimum_phase_months"],
                        context="shadow_methods.pagan_sossounov_bull_bear.configuration.minimum_phase_months",
                    ),
                    "minimum_cycle_months": _positive_integer(
                        configuration["minimum_cycle_months"],
                        context="shadow_methods.pagan_sossounov_bull_bear.configuration.minimum_cycle_months",
                    ),
                    "large_move_threshold": _finite_float(
                        configuration["large_move_threshold"],
                        context="shadow_methods.pagan_sossounov_bull_bear.configuration.large_move_threshold",
                        positive=True,
                    ),
                }
            )
            if (
                normalized_config["minimum_cycle_months"]
                < 2 * normalized_config["minimum_phase_months"]
                or normalized_config["large_move_threshold"] >= 1.0
            ):
                raise LabelSpecError("Pagan-Sossounov configuration is inconsistent")
        else:
            normalized_config = MappingProxyType(dict(configuration))
        spec_sha256 = canonical_shadow_method_sha256(str(method_id), row)
        locked_sha256 = SHADOW_METHOD_VERSION_LOCK.get(version)
        if locked_sha256 is None:
            raise LabelSpecError(f"unregistered shadow method version: {version}")
        if verify_lock and spec_sha256 != locked_sha256:
            raise LabelSpecError(
                f"shadow method {method_id} changed without a version bump"
            )
        shadows[str(method_id)] = ShadowLabelMethod(
            method_id=str(method_id),
            version=version,
            status=status,
            role=_nonempty_text(row["role"], context=f"shadow_methods.{method_id}.role"),
            configuration=normalized_config,
            spec_sha256=spec_sha256,
            results=None,
        )
    if set(shadows) != {"hsmm_explicit_duration", "pagan_sossounov_bull_bear"}:
        raise LabelSpecError("the registered shadow label methods are incomplete")
    return LabelSpecRegistry(
        schema_version=schema_version,
        default_spec=default_spec,
        state_order=state_order,
        specs=MappingProxyType(specs),
        shadow_methods=MappingProxyType(shadows),
    )


def load_label_spec(
    spec_id: str | None = None,
    *,
    path: str | Path | None = None,
    verify_lock: bool = True,
) -> LabelSpecification:
    return load_label_spec_registry(path, verify_lock=verify_lock).get(spec_id)


def label_spec_manifest_document(
    *,
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Return a result-free manifest suitable for local research reports."""

    registry = load_label_spec_registry(path)
    return {
        "schema_version": registry.schema_version,
        "default_spec": registry.default_spec,
        "state_order": list(registry.state_order),
        "specs": [
            {
                "spec_id": item.spec_id,
                "version": item.version,
                "status": item.status,
                "spec_sha256": item.spec_sha256,
                "membership_semantics": item.membership.semantics,
            }
            for item in registry.specs.values()
        ],
        "shadow_methods": [
            {
                "method_id": item.method_id,
                "version": item.version,
                "status": item.status,
                "role": item.role,
                "configuration": dict(item.configuration),
                "spec_sha256": item.spec_sha256,
                "results": None,
            }
            for item in registry.shadow_methods.values()
        ],
    }


__all__ = [
    "BLOCK_ORDER",
    "BROAD_EQUITY_BREADTH_SYMBOLS",
    "DEFAULT_LABEL_SPEC_ID",
    "FIXED_NINE_SECTORS",
    "LABEL_SPEC_SCHEMA_VERSION",
    "LABEL_SPEC_VERSION_LOCK",
    "SHADOW_METHOD_VERSION_LOCK",
    "LabelFitPeriodSpec",
    "LabelMembershipSpec",
    "LabelScalingSpec",
    "LabelSeriesSpec",
    "LabelSpecError",
    "LabelSpecRegistry",
    "LabelSpecification",
    "LabelWindowSpec",
    "MembershipAnchorSpec",
    "ShadowLabelMethod",
    "canonical_label_spec_sha256",
    "canonical_shadow_method_sha256",
    "default_label_spec_path",
    "label_spec_manifest_document",
    "load_label_spec",
    "load_label_spec_registry",
]
