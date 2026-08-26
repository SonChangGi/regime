"""Ex-post Pagan--Sossounov bull/bear chronology.

The implementation follows the monthly raw-price rules in Appendix B of
Pagan and Sossounov (2003): centered local extrema, alternating turns, six
month edge censoring, minimum 16-month cycles, and minimum four-month phases
with a 20 percent large-move exception.  Because a centered window needs
future prices, this chronology is never an operational label or forecast
target.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd

from regime_lab.integrity import canonical_json_sha256_v1

from .label_spec import load_label_spec_registry


PAGAN_SOSSOUNOV_SCHEMA_VERSION = "pagan-sossounov-ex-post/1"
TurnKind = Literal["peak", "trough"]


@dataclass(frozen=True)
class PaganSossounovConfig:
    window_months: int
    censor_margin_months: int
    minimum_phase_months: int
    minimum_cycle_months: int
    large_move_threshold: float

    def __post_init__(self) -> None:
        for name in (
            "window_months",
            "censor_margin_months",
            "minimum_phase_months",
            "minimum_cycle_months",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or int(value) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.minimum_cycle_months < 2 * self.minimum_phase_months:
            raise ValueError("minimum cycle must span at least two minimum phases")
        if not np.isfinite(self.large_move_threshold) or not (
            0.0 < float(self.large_move_threshold) < 1.0
        ):
            raise ValueError("large_move_threshold must be in (0, 1)")
        for name in (
            "window_months",
            "censor_margin_months",
            "minimum_phase_months",
            "minimum_cycle_months",
        ):
            object.__setattr__(self, name, int(getattr(self, name)))
        object.__setattr__(
            self,
            "large_move_threshold",
            float(self.large_move_threshold),
        )

    @classmethod
    def from_label_spec(cls) -> "PaganSossounovConfig":
        method = load_label_spec_registry().shadow_methods[
            "pagan_sossounov_bull_bear"
        ]
        return cls(**dict(method.configuration))


@dataclass(frozen=True)
class _Turn:
    position: int
    kind: TurnKind


@dataclass(frozen=True)
class PaganSossounovResult:
    states: pd.Series
    turning_points: pd.DataFrame
    configuration_sha256: str
    label_method_spec_sha256: str
    configuration_origin: str
    role: str = "retrospective_ex_post_sensitivity_only"
    uses_future_observations: bool = True
    canonical_target: bool = False
    automatic_promotion_eligible: bool = False


def _monthly_prices(price: pd.Series) -> tuple[pd.Series, pd.PeriodIndex]:
    if not isinstance(price, pd.Series):
        raise TypeError("price must be a pandas Series")
    if not isinstance(price.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        raise TypeError("price must use a monthly DatetimeIndex or PeriodIndex")
    if price.empty or price.index.has_duplicates or not price.index.is_monotonic_increasing:
        raise ValueError("price index must be non-empty, unique, and increasing")
    months = (
        price.index.asfreq("M")
        if isinstance(price.index, pd.PeriodIndex)
        else price.index.to_period("M")
    )
    if months.has_duplicates:
        raise ValueError("price must contain exactly one observation per month")
    expected = pd.period_range(months[0], months[-1], freq="M")
    if not months.equals(expected):
        raise ValueError("monthly chronology cannot contain missing months")
    values = pd.to_numeric(price, errors="coerce").astype(float)
    if not np.isfinite(values.to_numpy()).all() or bool((values <= 0.0).any()):
        raise ValueError("price must be complete, finite, and positive")
    return values, months


def _choose_extreme(
    turns: list[_Turn],
    prices: np.ndarray,
) -> list[_Turn]:
    """Collapse adjacent same-kind candidates and censor incomplete edges."""

    if not turns:
        return []
    collapsed: list[_Turn] = []
    for turn in sorted(turns, key=lambda item: item.position):
        if not collapsed or turn.kind != collapsed[-1].kind:
            collapsed.append(turn)
            continue
        previous = collapsed[-1]
        prefer_new = (
            prices[turn.position] > prices[previous.position]
            if turn.kind == "peak"
            else prices[turn.position] < prices[previous.position]
        )
        if prefer_new:
            collapsed[-1] = turn

    changed = True
    while collapsed and changed:
        changed = False
        first = collapsed[0]
        if (
            first.kind == "peak" and prices[first.position] < prices[0]
        ) or (
            first.kind == "trough" and prices[first.position] > prices[0]
        ):
            collapsed.pop(0)
            changed = True
        if not collapsed:
            break
        last = collapsed[-1]
        if (
            last.kind == "peak" and prices[last.position] < prices[-1]
        ) or (
            last.kind == "trough" and prices[last.position] > prices[-1]
        ):
            collapsed.pop()
            changed = True
    if changed:
        return _choose_extreme(collapsed, prices)
    return collapsed


def _initial_turns(
    prices: np.ndarray,
    config: PaganSossounovConfig,
) -> list[_Turn]:
    width = int(config.window_months)
    turns: list[_Turn] = []
    for position in range(width, len(prices) - width):
        window = prices[position - width : position + width + 1]
        # First-occurrence tie handling is deterministic and matches the usual
        # programmed centered-window convention.  A flat plateau therefore
        # cannot create multiple identical turns.
        if int(np.argmax(window)) == width:
            turns.append(_Turn(position, "peak"))
        if int(np.argmin(window)) == width:
            turns.append(_Turn(position, "trough"))
    margin = int(config.censor_margin_months)
    turns = [
        turn
        for turn in turns
        if margin <= turn.position < len(prices) - margin
    ]
    return _choose_extreme(turns, prices)


def _censor_cycles(
    turns: list[_Turn],
    prices: np.ndarray,
    config: PaganSossounovConfig,
) -> list[_Turn]:
    output = list(turns)
    while True:
        removed = False
        for position in range(len(output) - 2):
            first, _middle, last = output[position : position + 3]
            if first.kind != last.kind:
                raise RuntimeError("turning-point alternation invariant failed")
            if last.position - first.position < int(config.minimum_cycle_months):
                del output[position]
                output = _choose_extreme(output, prices)
                removed = True
                break
        if not removed:
            return output


def _censor_phases(
    turns: list[_Turn],
    prices: np.ndarray,
    config: PaganSossounovConfig,
) -> list[_Turn]:
    output = list(turns)
    while True:
        removed = False
        for position in range(len(output) - 1):
            first, second = output[position : position + 2]
            duration = second.position - first.position
            amplitude = abs(prices[second.position] / prices[first.position] - 1.0)
            if (
                duration < int(config.minimum_phase_months)
                and amplitude < float(config.large_move_threshold)
            ):
                del output[position + 1]
                output = _choose_extreme(output, prices)
                removed = True
                break
        if not removed:
            return output


def _final_turns(
    prices: np.ndarray,
    config: PaganSossounovConfig,
) -> list[_Turn]:
    turns = _initial_turns(prices, config)
    previous: tuple[tuple[int, str], ...] | None = None
    while True:
        identity = tuple((turn.position, turn.kind) for turn in turns)
        if identity == previous:
            break
        previous = identity
        # Re-run both operations until neither can introduce a fresh violation.
        turns = _censor_cycles(turns, prices, config)
        turns = _censor_phases(turns, prices, config)
        turns = _choose_extreme(turns, prices)
    if len(turns) < 2:
        raise ValueError("price history does not contain two completed ex-post turns")
    return turns


def pagan_sossounov_chronology(
    price: pd.Series,
    *,
    config: PaganSossounovConfig | None = None,
) -> PaganSossounovResult:
    """Return an explicitly future-confirmed monthly bull/bear chronology."""

    method_spec = load_label_spec_registry().shadow_methods[
        "pagan_sossounov_bull_bear"
    ]
    settings = config or PaganSossounovConfig.from_label_spec()
    configuration_origin = (
        "label_spec_default" if config is None else "explicit_sensitivity_override"
    )
    values, months = _monthly_prices(price)
    minimum_rows = 2 * max(
        int(settings.window_months), int(settings.censor_margin_months)
    ) + int(settings.minimum_cycle_months) + 1
    if len(values) < minimum_rows:
        raise ValueError(
            f"Pagan-Sossounov chronology requires at least {minimum_rows} months"
        )
    raw = values.to_numpy(dtype=float)
    turns = _final_turns(raw, settings)

    states = np.full(len(values), "bear", dtype=object)
    first = turns[0]
    in_bull = first.kind == "peak"
    if in_bull:
        states[: first.position + 1] = "bull"
    for left, right in zip(turns[:-1], turns[1:], strict=True):
        in_bull = left.kind == "trough"
        states[left.position + 1 : right.position + 1] = "bull" if in_bull else "bear"
    final = turns[-1]
    in_bull = final.kind == "trough"
    states[final.position + 1 :] = "bull" if in_bull else "bear"

    rows = []
    for turn in turns:
        confirmation_position = turn.position + int(settings.window_months)
        if confirmation_position >= len(months):
            raise RuntimeError("turn lacks its required future confirmation window")
        rows.append(
            {
                "at": months[turn.position].to_timestamp("M"),
                "kind": turn.kind,
                "price": float(raw[turn.position]),
                "confirmed_at": months[confirmation_position].to_timestamp("M"),
                "future_confirmation_months": int(settings.window_months),
            }
        )
    turning_points = pd.DataFrame(rows)
    configuration_hash = canonical_json_sha256_v1(
        {
            "schema_version": PAGAN_SOSSOUNOV_SCHEMA_VERSION,
            "configuration": asdict(settings),
        }
    )
    return PaganSossounovResult(
        states=pd.Series(states, index=price.index, name="ex_post_bull_bear"),
        turning_points=turning_points,
        configuration_sha256=configuration_hash,
        label_method_spec_sha256=method_spec.spec_sha256,
        configuration_origin=configuration_origin,
    )


__all__ = [
    "PAGAN_SOSSOUNOV_SCHEMA_VERSION",
    "PaganSossounovConfig",
    "PaganSossounovResult",
    "pagan_sossounov_chronology",
]
