"""Private, matched-origin research runner for regime-label challengers.

The runner is deliberately separate from the operating forecast pipeline.  It
reads one successful Alpha Vantage snapshot chain, reconstructs a derived-only
total-return panel, and compares the frozen v1 label with the two registered v2
challengers.  Every completed run is ``reconstructed_oos``; this module never
promotes a label/model or claims that a current-adjusted backfill was available
to the historical operating system.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, time, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import tempfile
from typing import Any, Literal
import uuid
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from regime_lab.analysis.label_evaluation import (
    LabelEvaluationResult,
    compare_label_sensitivity,
    evaluate_label_definition,
    prefix_stability_report,
)
from regime_lab.analysis.label_research import ResearchRegimeLabeler
from regime_lab.analysis.label_spec import LabelSpecification, load_label_spec
from regime_lab.analysis.labels import CausalRegimeLabeler, RegimeLabelConfig
from regime_lab.analysis.pagan_sossounov import pagan_sossounov_chronology
from regime_lab.analysis.pit_total_return import (
    CORPORATE_ACTION_CONTRACT,
    PITTotalReturnPanel,
    build_pit_total_return_panel,
    reconstruct_pit_total_return,
)
from regime_lab.data import Observation, SQLiteSnapshotStore
from regime_lab.config import project_root
from regime_lab.integrity import canonical_json_sha256_v1
from regime_lab.io import write_json_atomic


LABEL_BAKEOFF_SCHEMA_VERSION = "regime-private-label-bakeoff/1"
PIT_REPLAY_REPORT_SCHEMA_VERSION = "regime-private-pit-replay/1"
GENERATION_MANIFEST_SCHEMA_VERSION = "regime-private-label-bakeoff-generation/1"
DEFAULT_SOURCE = "alpha_vantage"
DEFAULT_DATASET = "weekly_adjusted_etf"
EVIDENCE_TRACK = "reconstructed_oos"
SplitPolicy = Literal["exact_split_series", "adjusted_close_composite"]
_SPLIT_POLICIES: tuple[str, ...] = (
    "exact_split_series",
    "adjusted_close_composite",
)
_EASTERN = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class LabelBakeoffResult:
    """Serializable research reports and their fail-closed completion state."""

    status: str
    label_audit_report: Mapping[str, Any]
    pit_replay_report: Mapping[str, Any]

    @property
    def complete(self) -> bool:
        return self.status == "complete"


@dataclass(frozen=True)
class _SymbolAssembly:
    frame: pd.DataFrame
    price_only_index: pd.Series
    component_series: tuple[str, ...]
    lineage_kind: str
    composite_input_sha256: str
    nonunit_implied_coefficient_periods: int


def _aware_utc(value: datetime, *, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone")
    return value.astimezone(timezone.utc)


def _iso(value: object) -> str | None:
    if value is None or value is pd.NaT:
        return None
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        return None
    return timestamp.isoformat()


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert pandas values, including NaN/NaT, to strict JSON records."""

    return json.loads(
        frame.to_json(
            orient="records",
            date_format="iso",
            date_unit="us",
            double_precision=15,
        )
    )


def _hashed_document(body: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(body)
    return {**payload, "sha256": canonical_json_sha256_v1(payload)}


def label_bakeoff_source_fingerprint(root: str | Path | None = None) -> str:
    """Bind a run to every Python implementation and typed input contract."""

    selected_root = (Path(root) if root is not None else project_root()).resolve()
    paths = sorted(
        [
            *(
                path
                for path in (selected_root / "src/regime_lab").rglob("*.py")
                if "__pycache__" not in path.parts
            ),
            selected_root / "scripts/run_label_bakeoff.py",
            selected_root / "config/label-spec.json",
            selected_root / "config/operating-contract.json",
            selected_root / "config/series.json",
            selected_root / "config/provider_rights.json",
            selected_root / "pyproject.toml",
            selected_root / "requirements-ci.lock",
        ],
        key=lambda path: path.relative_to(selected_root).as_posix(),
    )
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"label bake-off source input is missing or unsafe: {path}")
        relative = path.relative_to(selected_root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _record_order(record: Observation) -> tuple[object, ...]:
    assert record.system_retrieved_at is not None
    return (
        record.available_at,
        record.vintage_date,
        record.revision_seq,
        record.system_retrieved_at,
        record.raw_sha256,
    )


def _latest_by_series_period(
    records: Iterable[Observation],
    *,
    as_of: datetime,
) -> tuple[
    dict[str, dict[pd.Timestamp, Observation]],
    dict[str, dict[str, int | str]],
]:
    latest_raw: dict[tuple[str, pd.Timestamp], Observation] = {}
    for record in records:
        retrieved = record.system_retrieved_at
        if retrieved is None or retrieved > as_of:
            continue
        period = pd.Timestamp(record.observed_period_end)
        key = (record.series_id, period)
        current = latest_raw.get(key)
        if current is None or _record_order(record) > _record_order(current):
            latest_raw[key] = record
    grouped: dict[str, dict[pd.Timestamp, Observation]] = {}
    candidates: dict[tuple[str, pd.Timestamp], list[Observation]] = {}
    shifted: dict[str, int] = {}
    for (series_id, raw_period), record in sorted(latest_raw.items()):
        weekday = raw_period.weekday()
        if weekday > 4:
            raise ValueError(
                f"{series_id} has a weekend weekly observation: {raw_period.date()}"
            )
        canonical = raw_period + pd.offsets.Day(int(4 - weekday))
        shifted[series_id] = shifted.get(series_id, 0) + int(canonical != raw_period)
        candidates.setdefault((series_id, canonical), []).append(record)
    collisions: dict[str, int] = {}
    for (series_id, canonical), rows in sorted(candidates.items()):
        # A provider artifact can leave both a non-Friday observation and the
        # true Friday row in one last-good chain.  Prefer the observation
        # nearest the canonical Friday and report the discarded collision.
        selected = max(
            rows,
            key=lambda item: (item.observed_period_end, _record_order(item)),
        )
        grouped.setdefault(series_id, {})[canonical] = selected
        collisions[series_id] = collisions.get(series_id, 0) + max(0, len(rows) - 1)
    normalization = {
        series_id: {
            "calendar_contract": "source_trading_date_to_week_ending_friday_v1",
            "shifted_source_period_rows": shifted.get(series_id, 0),
            "discarded_same_week_collisions": collisions.get(series_id, 0),
        }
        for series_id in grouped
    }
    return grouped, normalization


def _logical_series_hash(
    series_id: str,
    rows: Mapping[pd.Timestamp, Observation],
) -> str:
    return canonical_json_sha256_v1(
        {
            "schema_version": "label-bakeoff-observation-series/v1",
            "series_id": series_id,
            "records": [
                {
                    "observed_period_end": period.isoformat(),
                    "value": record.value,
                    "source_released_at": _iso(record.source_released_at),
                    "provider_first_seen_at": _iso(
                        record.provider_first_seen_at
                    ),
                    "system_retrieved_at": _iso(record.system_retrieved_at),
                    "vintage_date": record.vintage_date.isoformat(),
                    "revision_seq": record.revision_seq,
                    "raw_sha256": record.raw_sha256,
                }
                for period, record in sorted(rows.items())
            ],
        }
    )


def _coverage_record(
    series_id: str,
    rows: Mapping[pd.Timestamp, Observation],
    *,
    normalization: Mapping[str, int | str] | None = None,
) -> dict[str, Any]:
    ordered = sorted(rows)
    non_null = [
        period
        for period in ordered
        if rows[period].value is not None
        and np.isfinite(float(rows[period].value))
    ]
    missing_release = sum(
        rows[period].source_released_at is None for period in ordered
    )
    return {
        "series_id": series_id,
        "latest_period_rows": len(ordered),
        "finite_value_rows": len(non_null),
        "first_period": _iso(non_null[0]) if non_null else None,
        "last_period": _iso(non_null[-1]) if non_null else None,
        "missing_source_release_rows": int(missing_release),
        "logical_series_sha256": _logical_series_hash(series_id, rows),
        "weekly_calendar_normalization": dict(normalization or {}),
    }


def _required_series(
    symbols: tuple[str, ...],
    *,
    split_policy: SplitPolicy,
) -> tuple[str, ...]:
    component_fields = (
        ("close", "dividend_amount", "split_coefficient")
        if split_policy == "exact_split_series"
        else ("close", "dividend_amount", "adjusted_close")
    )
    series = [
        f"{symbol}.{field}"
        for symbol in symbols
        for field in component_fields
    ]
    # The frozen v1 control is always constructed from provider-current
    # adjusted SPY history, including in exact-split challenger runs.
    series.append("SPY.adjusted_close")
    return tuple(dict.fromkeys(series))


def _preflight(
    grouped: Mapping[str, Mapping[pd.Timestamp, Observation]],
    *,
    required_series: tuple[str, ...],
    minimum_rows: int,
    normalization: Mapping[str, Mapping[str, int | str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], pd.DatetimeIndex | None]:
    issues: list[dict[str, Any]] = []
    coverage = [
        _coverage_record(
            series_id,
            grouped.get(series_id, {}),
            normalization=normalization.get(series_id),
        )
        for series_id in required_series
    ]
    valid_periods: dict[str, set[pd.Timestamp]] = {}
    for series_id in required_series:
        rows = grouped.get(series_id, {})
        valid = {
            period
            for period, record in rows.items()
            if record.value is not None and np.isfinite(float(record.value))
        }
        valid_periods[series_id] = valid
        if not valid:
            issues.append(
                {
                    "code": "missing_required_series",
                    "series_id": series_id,
                }
            )
        missing_release = sum(
            record.source_released_at is None for record in rows.values()
        )
        if missing_release:
            issues.append(
                {
                    "code": "missing_source_release_clock",
                    "series_id": series_id,
                    "rows": int(missing_release),
                }
            )

    if issues:
        return coverage, issues, None

    union = set().union(*valid_periods.values())
    common = set.intersection(*valid_periods.values())
    coverage_by_series = {item["series_id"]: item for item in coverage}
    for series_id, periods in valid_periods.items():
        # Non-common observations are never consumed, but are counted rather
        # than hidden.  A missing interior week still makes the common index
        # non-consecutive below and therefore blocks the run.
        coverage_by_series[series_id]["missing_vs_union"] = len(
            union.difference(periods)
        )
        coverage_by_series[series_id]["excluded_outside_common"] = len(
            periods.difference(common)
        )

    index = pd.DatetimeIndex(sorted(common), name="observed_period_end")
    if len(index) < minimum_rows:
        issues.append(
            {
                "code": "insufficient_matched_history",
                "rows": len(index),
                "minimum_rows": minimum_rows,
            }
        )
    if len(index) > 1:
        normalized = index.tz_localize(None).normalize()
        consecutive = bool(
            ((normalized[1:] - normalized[:-1]) == np.timedelta64(7, "D")).all()
        )
        if not consecutive:
            issues.append(
                {
                    "code": "nonconsecutive_weekly_history",
                    "rows": len(index),
                }
            )
    return coverage, issues, index if not issues else None


def _component_lineage(
    records: tuple[Observation, ...],
    *,
    observed_period: pd.Timestamp,
    split_policy: SplitPolicy,
) -> str:
    return canonical_json_sha256_v1(
        {
            "schema_version": "pit-corporate-action-composite-row/v1",
            "observed_period_end": observed_period.isoformat(),
            "split_policy": split_policy,
            "components": [
                {
                    "series_id": item.series_id,
                    "source_observed_period_end": (
                        item.observed_period_end.isoformat()
                    ),
                    "value": item.value,
                    "source_released_at": _iso(item.source_released_at),
                    "provider_first_seen_at": _iso(
                        item.provider_first_seen_at
                    ),
                    "system_retrieved_at": _iso(item.system_retrieved_at),
                    "vintage_date": item.vintage_date.isoformat(),
                    "revision_seq": item.revision_seq,
                    "raw_sha256": item.raw_sha256,
                }
                for item in records
            ],
        }
    )


def _assemble_symbol(
    symbol: str,
    index: pd.DatetimeIndex,
    grouped: Mapping[str, Mapping[pd.Timestamp, Observation]],
    *,
    split_policy: SplitPolicy,
) -> _SymbolAssembly:
    close_id = f"{symbol}.close"
    dividend_id = f"{symbol}.dividend_amount"
    split_input_id = (
        f"{symbol}.split_coefficient"
        if split_policy == "exact_split_series"
        else f"{symbol}.adjusted_close"
    )
    close_rows = grouped[close_id]
    dividend_rows = grouped[dividend_id]
    split_input_rows = grouped[split_input_id]
    closes = np.asarray(
        [float(close_rows[period].value) for period in index], dtype=float
    )
    dividends = np.asarray(
        [float(dividend_rows[period].value) for period in index], dtype=float
    )
    if split_policy == "exact_split_series":
        splits = np.asarray(
            [float(split_input_rows[period].value) for period in index],
            dtype=float,
        )
        lineage_kind = "exact_dated_raw_close_dividend_split_composite"
        nonunit_implied_coefficient_periods = 0
    else:
        adjusted = np.asarray(
            [float(split_input_rows[period].value) for period in index],
            dtype=float,
        )
        if not np.isfinite(adjusted).all() or bool((adjusted <= 0.0).any()):
            raise ValueError(f"{symbol} adjusted close must be positive")
        splits = np.ones(len(index), dtype=float)
        adjusted_gross = adjusted[1:] / adjusted[:-1]
        splits[1:] = (
            adjusted_gross * closes[:-1] - dividends[1:]
        ) / closes[1:]
        lineage_kind = (
            "reconstructed_current_adjusted_close_implied_split_composite"
        )
        nonunit_implied_coefficient_periods = int(
            (~np.isclose(splits[1:], 1.0, rtol=1e-6, atol=1e-8)).sum()
        )
    if (
        not np.isfinite(closes).all()
        or not np.isfinite(dividends).all()
        or not np.isfinite(splits).all()
        or bool((closes <= 0.0).any())
        or bool((dividends < 0.0).any())
        or bool((splits <= 0.0).any())
    ):
        raise ValueError(
            f"{symbol} corporate-action inputs imply an invalid reconstruction"
        )

    rows: list[dict[str, Any]] = []
    lineage_hashes: list[str] = []
    for position, period in enumerate(index):
        components = (
            close_rows[period],
            dividend_rows[period],
            split_input_rows[period],
        )
        releases = [item.source_released_at for item in components]
        if any(item is None for item in releases):
            raise ValueError(f"{symbol} corporate-action source release is missing")
        first_seen = [item.provider_first_seen_at for item in components]
        retrieved = [item.system_retrieved_at for item in components]
        if any(item is None for item in first_seen + retrieved):
            raise ValueError(f"{symbol} corporate-action retrieval clock is missing")
        row_hash = _component_lineage(
            components,
            observed_period=period,
            split_policy=split_policy,
        )
        lineage_hashes.append(row_hash)
        rows.append(
            {
                "raw_close": float(closes[position]),
                "dividend_amount": float(dividends[position]),
                "split_coefficient": float(splits[position]),
                "corporate_action_contract": CORPORATE_ACTION_CONTRACT,
                "source_released_at": max(releases),
                "provider_first_seen_at": max(first_seen),
                "system_retrieved_at": max(retrieved),
                "revision_seq": max(item.revision_seq for item in components),
                "raw_sha256": row_hash,
            }
        )
    frame = pd.DataFrame(rows, index=index)
    price_gross = closes[1:] * splits[1:] / closes[:-1]
    if not np.isfinite(price_gross).all() or bool((price_gross <= 0.0).any()):
        raise ValueError(f"{symbol} split-adjusted price path is invalid")
    price_only = pd.Series(100.0, index=index, name=symbol, dtype=float)
    price_only.iloc[1:] = 100.0 * np.cumprod(price_gross)
    return _SymbolAssembly(
        frame=frame,
        price_only_index=price_only,
        component_series=(close_id, dividend_id, split_input_id),
        lineage_kind=lineage_kind,
        composite_input_sha256=canonical_json_sha256_v1(
            {
                "schema_version": "pit-corporate-action-symbol-composite/v1",
                "symbol": symbol,
                "split_policy": split_policy,
                "row_lineage_sha256": lineage_hashes,
            }
        ),
        nonunit_implied_coefficient_periods=(
            nonunit_implied_coefficient_periods
        ),
    )


def _build_panel(
    symbols: tuple[str, ...],
    index: pd.DatetimeIndex,
    grouped: Mapping[str, Mapping[pd.Timestamp, Observation]],
    *,
    split_policy: SplitPolicy,
) -> tuple[PITTotalReturnPanel, Mapping[str, _SymbolAssembly]]:
    assemblies = {
        symbol: _assemble_symbol(
            symbol,
            index,
            grouped,
            split_policy=split_policy,
        )
        for symbol in symbols
    }
    decision_at = pd.Series(
        [
            max(
                (
                    pd.Timestamp(
                        datetime.combine(
                            period.date(),
                            time(16, 15),
                            tzinfo=_EASTERN,
                        )
                    ).tz_convert("UTC"),
                    *(
                        pd.Timestamp(
                            assemblies[symbol].frame.loc[
                                period, "source_released_at"
                            ]
                        )
                        for symbol in symbols
                    ),
                )
            )
            for period in index
        ],
        index=index,
        dtype="datetime64[ns, UTC]",
        name="decision_at",
    )
    results = {
        symbol: reconstruct_pit_total_return(
            assembly.frame,
            decision_at=decision_at,
            evidence_track=EVIDENCE_TRACK,
        )
        for symbol, assembly in assemblies.items()
    }
    return build_pit_total_return_panel(results), assemblies


def _prefix_lengths(total_rows: int, fit_rows: int) -> tuple[int, ...]:
    candidates = (fit_rows, fit_rows + (total_rows - fit_rows) // 2, total_rows)
    return tuple(dict.fromkeys(item for item in candidates if 1 <= item <= total_rows))


def _evaluation_payload(result: LabelEvaluationResult) -> dict[str, Any]:
    return {
        "occupancy": _json_records(result.occupancy),
        "durations": _json_records(result.durations),
        "flips": _json_records(result.flips),
        "external_origin_outcomes": _json_records(
            result.external_origin_outcomes
        ),
        "external_outcomes": _json_records(result.external_outcomes),
        "crash_recovery": _json_records(result.crash_recovery),
        "sensitivity": _json_records(result.sensitivity),
        "prefix_stability": _json_records(result.prefix_stability),
    }


def _label_payload(
    *,
    specification: LabelSpecification,
    labeler: CausalRegimeLabeler | ResearchRegimeLabeler,
    frame: pd.DataFrame | PITTotalReturnPanel,
    states: pd.Series,
    evaluation: LabelEvaluationResult,
    fit_rows: int,
    input_lineage: str,
) -> dict[str, Any]:
    scores = labeler.score_frame(frame)
    memberships = labeler.state_memberships(frame)
    origin_records = pd.DataFrame(
        {
            "origin_week": states.index,
            "state": states.astype(str).to_numpy(),
            "risk_score": scores["risk_score"].to_numpy(dtype=float),
            **{
                f"membership_{state}": memberships[state].to_numpy(dtype=float)
                for state in memberships
            },
        }
    )
    lower = getattr(labeler, "lower_threshold_", None)
    upper = getattr(labeler, "upper_threshold_", None)
    return {
        "spec_id": specification.spec_id,
        "version": specification.version,
        "status": specification.status,
        "spec_sha256": specification.spec_sha256,
        "evidence_track": EVIDENCE_TRACK,
        "input_lineage": input_lineage,
        "fit_period": {
            "mode": specification.fit_period.mode,
            "fit_rows": fit_rows,
            "fit_start": _iso(states.index[0]),
            "fit_end": _iso(states.index[fit_rows - 1]),
            "lower_threshold": float(lower),
            "upper_threshold": float(upper),
        },
        "membership_semantics": specification.membership.semantics,
        "latest_state": str(states.iloc[-1]),
        "origin_records": _json_records(origin_records),
        "evaluation": _evaluation_payload(evaluation),
    }


def _pagan_payload(
    price: pd.Series | None,
    *,
    as_of: datetime,
    input_lineage: str,
) -> dict[str, Any]:
    base = {
        "role": "retrospective_ex_post_sensitivity_only",
        "uses_future_observations": True,
        "canonical_target": False,
        "automatic_promotion_eligible": False,
        "input_lineage": input_lineage,
    }
    if price is None or price.empty:
        return {**base, "status": "blocked_missing_spy_price"}
    periods = price.index.to_period("M")
    completed_before = pd.Timestamp(as_of).tz_convert("UTC").tz_localize(None).to_period("M")
    eligible = periods < completed_before
    monthly = price.loc[eligible].groupby(periods[eligible]).last()
    monthly.index = monthly.index.to_timestamp("M")
    try:
        result = pagan_sossounov_chronology(monthly)
    except ValueError as exc:
        return {
            **base,
            "status": "blocked_insufficient_completed_turns",
            "reason": str(exc),
            "monthly_rows": len(monthly),
        }
    turns = result.turning_points.drop(columns=["price"], errors="ignore")
    states = pd.DataFrame(
        {
            "month_end": result.states.index,
            "state": result.states.astype(str).to_numpy(),
        }
    )
    return {
        **base,
        "status": "complete_ex_post_only",
        "configuration_sha256": result.configuration_sha256,
        "label_method_spec_sha256": result.label_method_spec_sha256,
        "configuration_origin": result.configuration_origin,
        "monthly_rows": len(monthly),
        "turning_point_count": len(turns),
        "turning_points": _json_records(turns),
        "monthly_states": _json_records(states),
    }


def _blocked_reports(
    *,
    generated_at: datetime,
    as_of: datetime,
    source: str,
    dataset: str,
    split_policy: SplitPolicy,
    coverage: list[dict[str, Any]],
    issues: list[dict[str, Any]],
    pagan: Mapping[str, Any],
    source_fingerprint_sha256: str,
) -> LabelBakeoffResult:
    common = {
        "generated_at": generated_at.isoformat(),
        "data_as_of": as_of.isoformat(),
        "evidence_track": EVIDENCE_TRACK,
        "status": "blocked_input_contract",
        "source": source,
        "dataset": dataset,
        "split_policy": split_policy,
        "automatic_promotion_eligible": False,
        "operating_pipeline_mutated": False,
        "official_label_unchanged": True,
        "official_model_unchanged": True,
        "research_source_fingerprint_sha256": source_fingerprint_sha256,
        "runtime_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    label_body = {
        "schema_version": LABEL_BAKEOFF_SCHEMA_VERSION,
        **common,
        "matched_origin_comparison_completed": False,
        "input_issues": issues,
        "pagan_sossounov": dict(pagan),
        "labels": {},
        "interpretation": {
            "missing_inputs_are_not_imputed": True,
            "adjusted_close_is_not_silently_substituted": True,
            "result_is_not_a_label_or_model_promotion": True,
        },
    }
    pit_body = {
        "schema_version": PIT_REPLAY_REPORT_SCHEMA_VERSION,
        **common,
        "replay_completed": False,
        "operational_oos_claimed": False,
        "component_coverage": coverage,
        "input_issues": issues,
        "lineage_contract": {
            "corporate_action_contract": CORPORATE_ACTION_CONTRACT,
            "split_policy": split_policy,
            "exact_split_required_when_policy_is_exact": True,
            "unit_split_fill_allowed": False,
        },
        "symbols": {},
    }
    return LabelBakeoffResult(
        status="blocked_input_contract",
        label_audit_report=_hashed_document(label_body),
        pit_replay_report=_hashed_document(pit_body),
    )


def run_label_bakeoff(
    database: str | Path,
    *,
    as_of: datetime,
    split_policy: SplitPolicy,
    source: str = DEFAULT_SOURCE,
    dataset: str = DEFAULT_DATASET,
    generated_at: datetime | None = None,
) -> LabelBakeoffResult:
    """Run the private v1/v2 comparison from one read-only SQLite snapshot.

    ``split_policy`` is mandatory so an adjusted-close-derived split cannot be
    mistaken for an observed split event.  Both policies remain reconstructed
    research evidence and are ineligible for automatic promotion.
    """

    if split_policy not in _SPLIT_POLICIES:
        raise ValueError(f"split_policy must be one of {_SPLIT_POLICIES}")
    source_fingerprint = label_bakeoff_source_fingerprint()
    cutoff = _aware_utc(as_of, field="as_of")
    created = _aware_utc(
        generated_at or datetime.now(timezone.utc), field="generated_at"
    )
    broad_spec = load_label_spec("v2_broad_equity")
    symbols = tuple(item.symbol for item in broad_spec.series)
    required = _required_series(symbols, split_policy=split_policy)
    fit_rows = max(
        load_label_spec(spec_id).fit_period.reference_fit_weeks
        for spec_id in (
            "v1_spy_hysteresis",
            "v2_spy_pit_total_return",
            "v2_broad_equity",
        )
    )
    with SQLiteSnapshotStore(database, read_only=True) as store:
        # One explicit SQLite read transaction keeps a concurrent successful
        # collection from changing the selected snapshot chain mid-report.
        store._connection.execute("BEGIN")
        try:
            records = store.read_last_good_observations(
                source=source,
                dataset=dataset,
                series_ids=required,
                available_as_of=cutoff,
            )
        finally:
            store._connection.rollback()
    grouped, normalization = _latest_by_series_period(records, as_of=cutoff)
    coverage, issues, index = _preflight(
        grouped,
        required_series=required,
        minimum_rows=fit_rows,
        normalization=normalization,
    )

    raw_spy_rows = grouped.get("SPY.close", {})
    partial_spy = None
    if raw_spy_rows:
        partial_index = pd.DatetimeIndex(sorted(raw_spy_rows))
        partial_spy = pd.Series(
            [float(raw_spy_rows[item].value) for item in partial_index],
            index=partial_index,
            dtype=float,
            name="SPY",
        )
    partial_pagan = _pagan_payload(
        partial_spy,
        as_of=cutoff,
        input_lineage=(
            "partial_unadjusted_spy_raw_close_ex_post_only"
            if partial_spy is not None
            else "unavailable"
        ),
    )
    if issues or index is None:
        if label_bakeoff_source_fingerprint() != source_fingerprint:
            raise RuntimeError("label bake-off source changed during input audit")
        return _blocked_reports(
            generated_at=created,
            as_of=cutoff,
            source=source,
            dataset=dataset,
            split_policy=split_policy,
            coverage=coverage,
            issues=issues,
            pagan=partial_pagan,
            source_fingerprint_sha256=source_fingerprint,
        )

    panel, assemblies = _build_panel(
        symbols,
        index,
        grouped,
        split_policy=split_policy,
    )
    spy_panel = build_pit_total_return_panel({"SPY": panel.results["SPY"]})
    adjusted_rows = grouped["SPY.adjusted_close"]
    v1_frame = pd.DataFrame(
        {
            "spy_close": [
                float(adjusted_rows[period].value) for period in index
            ]
        },
        index=index,
    )
    external_prices = panel.frame.loc[
        :,
        [
            "spy_pit_total_return",
            "rsp_pit_total_return",
            "iwm_pit_total_return",
        ],
    ].rename(
        columns={
            "spy_pit_total_return": "SPY",
            "rsp_pit_total_return": "RSP",
            "iwm_pit_total_return": "IWM",
        }
    )

    v1_spec = load_label_spec("v1_spy_hysteresis")
    v1_labeler = CausalRegimeLabeler(
        RegimeLabelConfig(
            price_column="spy_close",
            minimum_fit_observations=(
                v1_spec.fit_period.production_minimum_finite_observations
            ),
        )
    ).fit(v1_frame.iloc[:fit_rows])
    spy_labeler = ResearchRegimeLabeler("v2_spy_pit_total_return").fit(spy_panel)
    broad_labeler = ResearchRegimeLabeler("v2_broad_equity").fit(panel)
    label_inputs: dict[
        str,
        tuple[
            LabelSpecification,
            CausalRegimeLabeler | ResearchRegimeLabeler,
            pd.DataFrame | PITTotalReturnPanel,
            str,
        ],
    ] = {
        "v1_spy_hysteresis": (
            v1_spec,
            v1_labeler,
            v1_frame,
            "provider_current_adjusted_close_frozen_control_not_historical_pit",
        ),
        "v2_spy_pit_total_return": (
            load_label_spec("v2_spy_pit_total_return"),
            spy_labeler,
            spy_panel,
            assemblies["SPY"].lineage_kind,
        ),
        "v2_broad_equity": (
            broad_spec,
            broad_labeler,
            panel,
            assemblies["SPY"].lineage_kind,
        ),
    }
    states = {
        spec_id: labeler.transform(frame)
        for spec_id, (_spec, labeler, frame, _lineage) in label_inputs.items()
    }
    labels_payload: dict[str, Any] = {}
    prefix_lengths = _prefix_lengths(len(index), fit_rows)
    for spec_id, (specification, labeler, frame, input_lineage) in label_inputs.items():
        stability = prefix_stability_report(
            labeler,
            frame,
            prefix_lengths=prefix_lengths,
        )
        evaluation = evaluate_label_definition(
            states[spec_id],
            external_prices=external_prices,
            crash_asset="SPY",
            prefix_stability=stability,
        )
        labels_payload[spec_id] = _label_payload(
            specification=specification,
            labeler=labeler,
            frame=frame,
            states=states[spec_id],
            evaluation=evaluation,
            fit_rows=fit_rows,
            input_lineage=input_lineage,
        )

    comparisons = compare_label_sensitivity(
        states["v1_spy_hysteresis"],
        {
            "v2_spy_pit_total_return": states["v2_spy_pit_total_return"],
            "v2_broad_equity": states["v2_broad_equity"],
        },
    )
    pagan = _pagan_payload(
        assemblies["SPY"].price_only_index,
        as_of=cutoff,
        input_lineage=(
            f"{assemblies['SPY'].lineage_kind}; dividends excluded from price chronology"
        ),
    )
    origin_hash = canonical_json_sha256_v1(
        [period.isoformat() for period in index]
    )
    common = {
        "generated_at": created.isoformat(),
        "data_as_of": cutoff.isoformat(),
        "evidence_track": EVIDENCE_TRACK,
        "status": "complete",
        "source": source,
        "dataset": dataset,
        "split_policy": split_policy,
        "automatic_promotion_eligible": False,
        "operating_pipeline_mutated": False,
        "official_label_unchanged": True,
        "official_model_unchanged": True,
        "research_source_fingerprint_sha256": source_fingerprint,
        "runtime_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    label_body = {
        "schema_version": LABEL_BAKEOFF_SCHEMA_VERSION,
        **common,
        "matched_origin_comparison_completed": True,
        "matched_origins": {
            "rows": len(index),
            "first_origin": _iso(index[0]),
            "last_origin": _iso(index[-1]),
            "origin_sha256": origin_hash,
            "fit_prefix_rows": fit_rows,
        },
        "labels": labels_payload,
        "v1_reference_comparisons": _json_records(comparisons),
        "pagan_sossounov": pagan,
        "interpretation": {
            "matched_origin_descriptive_comparison": True,
            "external_outcomes_are_construct_validity_not_trading_returns": True,
            "observed_memberships_are_not_posteriors": True,
            "pagan_chronology_is_future_confirmed_ex_post_only": True,
            "result_is_not_a_label_or_model_promotion": True,
        },
    }
    symbol_reports: dict[str, Any] = {}
    for symbol in symbols:
        assembly = assemblies[symbol]
        result = panel.results[symbol]
        symbol_reports[symbol] = {
            "rows": len(result.audit),
            "first_period": _iso(result.audit.index[0]),
            "last_period": _iso(result.audit.index[-1]),
            "component_series": list(assembly.component_series),
            "lineage_kind": assembly.lineage_kind,
            "composite_input_sha256": assembly.composite_input_sha256,
            "pit_input_snapshot_sha256": result.input_snapshot_sha256,
            "nonunit_implied_coefficient_periods": (
                assembly.nonunit_implied_coefficient_periods
            ),
            "reconstructed_eligible_rows": int(
                result.audit["reconstructed_eligible"].sum()
            ),
            "operational_eligible_rows_at_reconstructed_decision_clock": int(
                result.audit["operational_eligible"].sum()
            ),
        }
    pit_body = {
        "schema_version": PIT_REPLAY_REPORT_SCHEMA_VERSION,
        **common,
        "replay_completed": True,
        "operational_oos_claimed": False,
        "operational_oos_eligible_for_promotion": False,
        "component_coverage": coverage,
        "input_issues": [],
        "matched_origins": {
            "rows": len(index),
            "first_origin": _iso(index[0]),
            "last_origin": _iso(index[-1]),
            "origin_sha256": origin_hash,
        },
        "lineage_contract": {
            "corporate_action_contract": CORPORATE_ACTION_CONTRACT,
            "split_policy": split_policy,
            "decision_clock": (
                "max(cross_symbol_source_released_at, canonical Friday 16:15 ET)"
            ),
            "decision_clock_role": "retrospective_economic_availability_only",
            "provider_first_seen_relaxed": True,
            "revision_seq_semantics": "maximum component revision; composite SHA binds every component",
            "unit_split_fill_allowed": False,
            "adjusted_close_composite_is_historical_pit": False,
            "adjusted_close_composite_algebraically_reproduces_adjusted_returns": (
                split_policy == "adjusted_close_composite"
            ),
        },
        "panel_input_snapshot_sha256": panel.input_snapshot_sha256,
        "symbols": symbol_reports,
    }
    if label_bakeoff_source_fingerprint() != source_fingerprint:
        raise RuntimeError("label bake-off source changed during computation")
    return LabelBakeoffResult(
        status="complete",
        label_audit_report=_hashed_document(label_body),
        pit_replay_report=_hashed_document(pit_body),
    )


def write_label_bakeoff_generation(
    output_root: str | Path,
    result: LabelBakeoffResult,
) -> Path:
    """Atomically expose one immutable private generation and latest pointer."""

    expected_source_fingerprint = str(
        result.label_audit_report["research_source_fingerprint_sha256"]
    )
    if label_bakeoff_source_fingerprint() != expected_source_fingerprint:
        raise RuntimeError("label bake-off source changed before generation write")
    root = Path(output_root).resolve()
    runs = root / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    generation_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        + "-"
        + uuid.uuid4().hex[:8]
    )
    staging = Path(tempfile.mkdtemp(prefix=".label-bakeoff-", dir=runs))
    final = runs / generation_id
    try:
        label_path = write_json_atomic(
            staging / "label-audit-report.json",
            result.label_audit_report,
        )
        pit_path = write_json_atomic(
            staging / "pit-replay-report.json",
            result.pit_replay_report,
        )
        artifacts = {}
        for name, path in (
            ("label_audit_report", label_path),
            ("pit_replay_report", pit_path),
        ):
            payload = path.read_bytes()
            artifacts[name] = {
                "path": path.name,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        manifest_body = {
            "schema_version": GENERATION_MANIFEST_SCHEMA_VERSION,
            "status": result.status,
            "evidence_track": EVIDENCE_TRACK,
            "derived_only": True,
            "public_release_eligible": False,
            "automatic_promotion_eligible": False,
            "artifacts": artifacts,
        }
        manifest = _hashed_document(manifest_body)
        manifest_path = write_json_atomic(
            staging / "generation-manifest.json",
            manifest,
        )
        if label_bakeoff_source_fingerprint() != expected_source_fingerprint:
            raise RuntimeError("label bake-off source changed before generation cutover")
        os.replace(staging, final)
        write_json_atomic(
            root / "latest.json",
            {
                "schema_version": 1,
                "generation": f"runs/{generation_id}",
                "status": result.status,
                "evidence_track": EVIDENCE_TRACK,
                "generation_manifest_sha256": manifest["sha256"],
                "generation_manifest_file_sha256": hashlib.sha256(
                    (final / manifest_path.name).read_bytes()
                ).hexdigest(),
            },
        )
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return final


__all__ = [
    "DEFAULT_DATASET",
    "DEFAULT_SOURCE",
    "EVIDENCE_TRACK",
    "GENERATION_MANIFEST_SCHEMA_VERSION",
    "LABEL_BAKEOFF_SCHEMA_VERSION",
    "LabelBakeoffResult",
    "PIT_REPLAY_REPORT_SCHEMA_VERSION",
    "SplitPolicy",
    "label_bakeoff_source_fingerprint",
    "run_label_bakeoff",
    "write_label_bakeoff_generation",
]
