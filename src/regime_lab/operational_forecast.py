"""Prepare one locked-model forecast without rerunning historical research.

Preparation never appends an issued forecast, changes publication, or schedules
a job. The output expires at the target week's first market open. Historical
parity replays are permanently marked ineligible for operational use.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from regime_lab.analysis.decision_shadow import STATE_ORDER, _scheduled_nyse_entry_at
from regime_lab.analysis.models import BenchmarkProfile
from regime_lab.analysis.structural_models import forecast_structural_probabilities
from regime_lab.analysis.transitions import (
    causal_state_durations,
    derive_causal_transition_features,
)
from regime_lab.analysis.validation import (
    _calibrate_transition_probability,
    _fit_transition_candidate,
    _transition_numeric_frame,
    _transition_targets,
    forecast_next_regime,
    transition_calibration_version,
)
from regime_lab.integrity import canonical_json_sha256_v1

EXPERTS = ("markov", "xgboost", "xgb_hazard_destination")


class OperationalPreparationError(ValueError):
    pass


def frame_sha256(frame: pd.DataFrame | pd.Series) -> str:
    """Bind ordered index, ordered columns, dtypes and values of an input cache."""
    table = frame.to_frame() if isinstance(frame, pd.Series) else frame
    header = canonical_json_sha256_v1(
        {
            "columns": [str(c) for c in table.columns],
            "dtypes": [str(t) for t in table.dtypes],
            "index_dtype": str(table.index.dtype),
            "index_name": str(table.index.name),
        }
    )
    return hashlib.sha256(
        header.encode() + pd.util.hash_pandas_object(table, index=True).values.tobytes()
    ).hexdigest()


def _completed_expert_history(
    history: pd.DataFrame,
    locked_payload: Mapping[str, Any],
    states: pd.Series,
    origin: pd.Timestamp,
) -> tuple[pd.DataFrame, bool]:
    required = {"model", "origin_date", "target_date", "current_state", "actual", "p_risk_on", "p_transition", "p_risk_off"}
    if not required <= set(history):
        raise OperationalPreparationError("expert history lacks state/probability contract")
    frame = history.loc[history.model.isin(EXPERTS)].copy()
    for field in ("origin_date", "target_date"):
        frame[field] = pd.to_datetime(frame[field], utc=True)
    from regime_lab.payload import normalized_probabilities
    positions = {pd.Timestamp(at).tz_convert("UTC"): i for i, at in enumerate(states.index)}
    for row in frame.to_dict("records"):
        position = positions.get(row["origin_date"])
        if position is None or position + 1 >= len(states) or row["target_date"] != states.index[position + 1]:
            raise OperationalPreparationError("expert history origin/target differs from official weekly index")
        if row["current_state"] != str(states.iloc[position]) or row["actual"] != str(states.iloc[position + 1]):
            raise OperationalPreparationError("expert history current/actual differs from official states")
        try:
            normalized_probabilities({s: row[f"p_{s}"] for s in STATE_ORDER})
        except (TypeError, ValueError) as exc:
            raise OperationalPreparationError("expert history probabilities are invalid") from exc
    if (frame.target_date > origin).any():
        raise OperationalPreparationError("expert history contains a future target")
    if frame.duplicated(["model", "origin_date"]).any():
        raise OperationalPreparationError(
            "expert history contains duplicate model origins"
        )
    appended = False
    if set(frame.loc[frame.target_date.eq(origin), "model"]) != set(EXPERTS):
        previous = locked_payload["weekly"][-1]
        previous_origin = pd.Timestamp(str(previous["date"])).date()
        if (origin.date() - previous_origin).days != 7:
            raise OperationalPreparationError(
                "expert history is stale; latest completed origin unavailable"
            )
        frozen = {
            r["model"]: r
            for r in previous.get("model_forecasts", [])
            if r["model"] in EXPERTS and r.get("date") == origin.date().isoformat()
        }
        if set(frozen) != set(EXPERTS):
            raise OperationalPreparationError(
                "frozen latest expert forecasts are incomplete"
            )
        if set(frame.loc[frame.target_date.eq(origin), "model"]):
            raise OperationalPreparationError(
                "partially completed expert history requires reconciliation"
            )
        previous_timestamp = states.index[-2]
        additions = []
        for model, row in frozen.items():
            additions.append(
                {
                    "origin_date": previous_timestamp,
                    "target_date": origin,
                    "model": model,
                    "evaluation_split": "operational_completed_frozen_forecast",
                    "current_state": str(states.iloc[-2]),
                    "actual": str(states.iloc[-1]),
                    "predicted": row["state"],
                    "train_size": len(states) - 3,
                    "gap": 1,
                    "fallback": bool(row.get("fallback", False)),
                    "fallback_reason": str(row.get("fallback_reason", "")),
                    **{f"p_{s}": float(row["probabilities"][s]) for s in STATE_ORDER},
                }
            )
        frame = pd.concat([frame, pd.DataFrame(additions)], ignore_index=True)
        appended = True
    expected_last = states.index[-2]
    budget = locked_payload["model"].get("candidate_manifest", {}).get("profile_budget", {})
    first = int(budget.get("minimum_train_weeks", 520)) + 1
    cutoff = pd.Timestamp(locked_payload["model"]["selection_end"], tz="UTC")
    eligible_positions = list(range(first, len(states) - 1))
    selection_positions = [i for i in eligible_positions if states.index[i + 1] < cutoff]
    diagnostic_positions = [i for i in eligible_positions if states.index[i] >= cutoff]
    maximum = budget.get("max_origins")
    if maximum is not None:
        diagnostic_positions = diagnostic_positions[-int(maximum):]
    expected_origins = {states.index[i] for i in selection_positions + diagnostic_positions}
    for expert in EXPERTS:
        eligible = frame.loc[frame.model.eq(expert) & frame.target_date.lt(origin)]
        if eligible.empty or eligible.target_date.max() != expected_last:
            raise OperationalPreparationError(
                "completed expert score history has a weekly gap"
            )
        actual_origins = set(frame.loc[frame.model.eq(expert), "origin_date"])
        if not expected_origins <= actual_origins:
            raise OperationalPreparationError("completed expert score history has an internal weekly gap")
        for split_positions in (selection_positions, [i for i in eligible_positions if states.index[i] >= cutoff]):
            present = [i for i in split_positions if states.index[i] in actual_origins]
            if present and any(states.index[i] not in actual_origins for i in range(min(present), max(present) + 1)):
                raise OperationalPreparationError("completed expert score history has an internal weekly gap")
    return frame, appended


def prepare_operational_forecast(
    features: pd.DataFrame,
    states: pd.Series,
    oos_predictions: pd.DataFrame,
    transition_predictions: pd.DataFrame,
    *,
    locked_payload: Mapping[str, Any],
    feature_manifest: Mapping[str, Any],
    expected_input_hashes: Mapping[str, str],
    decision_at: datetime | None = None,
    research_replay: bool = False,
) -> dict[str, Any]:
    """Refit only the latest two learned experts; preserve the locked selector.

    expected_input_hashes must be bound before calling this function. A cache
    with mismatched dates, schema, hashes, or stale ensemble scores is refused.
    An explicit past clock is accepted only for a labelled research replay.
    """
    started = perf_counter()
    if decision_at is not None and not research_replay:
        raise OperationalPreparationError(
            "a supplied decision clock is allowed only for research replay"
        )
    clock = pd.Timestamp(decision_at or datetime.now(timezone.utc))
    if clock.tzinfo is None:
        raise OperationalPreparationError("decision clock must be timezone-aware")
    clock = clock.tz_convert("UTC")
    if (
        not isinstance(features.index, pd.DatetimeIndex)
        or features.index.tz is None
        or features.index.has_duplicates
        or not features.index.is_monotonic_increasing
    ):
        raise OperationalPreparationError(
            "features require a unique sorted aware weekly index"
        )
    if not features.index.equals(states.index) or features.columns.has_duplicates:
        raise OperationalPreparationError(
            "feature/state index or feature schema differs"
        )
    if not states.isin(STATE_ORDER).all() or len(features) < 523:
        raise OperationalPreparationError("state history is invalid or too short")
    if any(
        (b.date() - a.date()).days != 7
        for a, b in zip(features.index, features.index[1:])
    ):
        raise OperationalPreparationError("feature/state history contains a weekly gap")
    origin = pd.Timestamp(features.index[-1]).tz_convert("UTC")
    if origin > clock:
        raise OperationalPreparationError("latest input cutoff is after decision clock")
    model = locked_payload.get("model", {})
    selection = locked_payload.get("selection", {})
    champion = str(selection.get("operating_champion", ""))
    if champion != model.get("champion") or champion not in {
        *EXPERTS,
        "causal_dynamic_ensemble",
        "causal_multiscale_ensemble",
    }:
        raise OperationalPreparationError(
            "locked champion is missing or unsupported by the latest-only path"
        )
    if model.get("selection_status") != "selected_by_gate":
        raise OperationalPreparationError(
            "operating champion is not locked by the selection gate"
        )
    manifest = model.get("candidate_manifest", {})
    if canonical_json_sha256_v1(manifest) != model.get("candidate_manifest_sha256"):
        raise OperationalPreparationError("locked model manifest hash differs")
    feature_body = {
        key: value for key, value in feature_manifest.items() if key != "sha256"
    }
    feature_hash = canonical_json_sha256_v1(feature_body)
    if feature_hash != feature_manifest.get("sha256") or feature_hash != model.get(
        "feature_manifest_sha256"
    ):
        raise OperationalPreparationError("locked feature manifest hash differs")
    expected_names = [
        name
        for group in feature_manifest.get("groups", [])
        for name in group.get("features", [])
    ]
    if set(features.columns) != set(expected_names) or len(features.columns) != int(
        feature_manifest.get("feature_count", 0)
    ):
        raise OperationalPreparationError(
            "prepared feature columns differ from the approved feature contract"
        )
    inputs = {
        "features": frame_sha256(features),
        "states": frame_sha256(states),
        "oos_predictions": frame_sha256(oos_predictions),
        "transition_predictions": frame_sha256(transition_predictions),
    }
    if dict(expected_input_hashes) != inputs:
        raise OperationalPreparationError(
            "prepared input hashes differ from the supplied bundle identity"
        )
    try:
        calibration_version = transition_calibration_version(transition_predictions)
    except ValueError as exc:
        raise OperationalPreparationError(str(exc)) from exc
    target = origin + timedelta(days=7)
    entry = _scheduled_nyse_entry_at(target.date().isoformat()).tz_convert("UTC")
    reasons = []
    if clock >= entry:
        reasons.append("scheduled_entry_missed")
    if (clock - origin).total_seconds() > 7 * 86400:
        reasons.append("stale_completed_week")
    key = {
        "origin_at": origin.isoformat(),
        "target_at": target.isoformat(),
        "champion": champion,
        "candidate_manifest_sha256": model["candidate_manifest_sha256"],
        "calibration_version": calibration_version,
        "inputs": inputs,
        "role": "research_replay" if research_replay else "operational_preparation",
    }
    envelope = {
        "schema_version": "regime-fast-operational-preparation/1",
        "key_sha256": canonical_json_sha256_v1(key),
        "key": key,
        "prepared_at": clock.isoformat(),
        "origin_at": origin.isoformat(),
        "target_at": target.isoformat(),
        "expires_at": entry.isoformat(),
        "status": "blocked" if reasons else "prepared",
        "blocked_reasons": reasons,
        "research_replay": research_replay,
        "issued": False,
        "operational_issue_eligible": False,
        "scope": "latest_origin_only_no_historical_retraining_no_automatic_issuance",
        "input_hashes": inputs,
        "champion": champion,
        "forecast": None,
    }
    if reasons and not research_replay:
        envelope["elapsed_seconds"] = perf_counter() - started
        return envelope
    history, completed_frozen = _completed_expert_history(
        oos_predictions, locked_payload, states, origin
    )
    cfg = BenchmarkProfile(name=str(manifest["profile"]), **manifest["profile_budget"])
    base = {
        name: forecast_next_regime(
            features,
            states,
            champion_name=name,
            as_of=origin,
            profile=cfg,
            gap=1,
            minimum_train_weeks=cfg.minimum_train_weeks,
            random_state=17,
        )
        for name in ("markov", "xgboost")
    }
    augmented = derive_causal_transition_features(features, states)
    numeric = _transition_numeric_frame(augmented)
    event, destination = _transition_targets(states, 1)
    raw, fallback, reason, _ = _fit_transition_candidate(
        "binary_xgboost",
        augmented,
        numeric,
        event,
        event,
        destination,
        horizon=1,
        train_stop=len(features) - 2,
        test_position=len(features) - 1,
        profile=cfg,
        random_state=17,
    )
    calibration = transition_predictions.loc[
        transition_predictions.model.eq("binary_xgboost")
        & transition_predictions.horizon.eq(1)
        & transition_predictions.evaluation_split.eq("selection")
    ].copy()
    calibration["target_end"] = pd.to_datetime(calibration.target_end, utc=True)
    selection_end = pd.Timestamp(model["selection_end"], tz="UTC")
    if (
        calibration.empty
        or (calibration.target_end > selection_end).any()
        or (calibration.target_end >= origin).any()
    ):
        raise OperationalPreparationError(
            "hazard calibration history crosses selection/origin boundary"
        )
    hazard, calibration_method, calibration_fallback, calibration_reason = (
        _calibrate_transition_probability(
            raw, calibration, minimum_rows=12, random_state=17,
            version=calibration_version, selection_end=selection_end, origin=origin,
        )
    )
    fallbacks = {
        "markov": bool(base["markov"].attrs.get("fallback", False)),
        "xgboost": bool(base["xgboost"].attrs.get("fallback", False)),
        "xgb_hazard_destination": bool(fallback or calibration_fallback),
    }
    structural = forecast_structural_probabilities(
        origin_date=origin,
        current_state=str(states.iloc[-1]),
        markov_probability=[float(base["markov"][s]) for s in STATE_ORDER],
        xgboost_probability=[float(base["xgboost"][s]) for s in STATE_ORDER],
        binary_xgboost_p_change=hazard,
        historical_oos_predictions=history,
        expert_fallbacks=fallbacks,
        current_duration_weeks=int(causal_state_durations(states).iloc[-1]),
        include_multiscale=True,
    )
    selected = structural.probabilities.loc[
        structural.probabilities.model.eq(champion)
    ].iloc[0]
    if bool(selected["fallback"]):
        reasons.append("selected_forecast_fallback")
    probabilities = {s: float(selected[f"p_{s}"]) for s in STATE_ORDER}
    reference = next(
        (
            r
            for r in locked_payload["weekly"][-1].get("model_forecasts", [])
            if r.get("model") == champion and r.get("date") == target.date().isoformat()
        ),
        None,
    )
    maximum_difference = (
        max(
            abs(probabilities[s] - float(reference["probabilities"][s]))
            for s in STATE_ORDER
        )
        if reference
        else None
    )
    envelope.update(
        {
            "status": (
                "research_replay"
                if research_replay
                else ("blocked" if reasons else "prepared_for_review")
            ),
            "blocked_reasons": reasons,
            "forecast": {
                "model": champion,
                "date": target.date().isoformat(),
                "state": str(selected["predicted"]),
                "probabilities": probabilities,
            },
            "calibration": {
                "version": calibration_version,
                "method": calibration_method,
                "raw_probability": raw,
                "probability": hazard,
                "fallback": bool(calibration_fallback),
                "reason": calibration_reason,
                "selection_rows": len(calibration),
            },
            "expert_fallbacks": fallbacks,
            "last_completed_scored_target": states.index[-2].isoformat(),
            "completed_prior_frozen_forecast": completed_frozen,
            "reference_parity": {
                "available": reference is not None,
                "maximum_absolute_probability_difference": maximum_difference,
                "tolerance": 1e-7,
                "passed": maximum_difference is not None and maximum_difference <= 1e-7,
            },
            "model_fits": {
                "xgboost": 1,
                "binary_xgboost": 1,
                "markov": 1,
                "historical_origins_retrained": 0,
            },
            "elapsed_seconds": perf_counter() - started,
        }
    )
    return envelope


def write_prepared_forecast(
    document: Mapping[str, Any], output_directory: str | Path
) -> Path:
    """Freeze a preparation once; exact forecast retry cannot overwrite it."""
    if document.get(
        "schema_version"
    ) != "regime-fast-operational-preparation/1" or canonical_json_sha256_v1(
        document.get("key")
    ) != document.get(
        "key_sha256"
    ):
        raise OperationalPreparationError("preparation identity hash is invalid")
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{document['key_sha256']}.json"
    encoded = (
        json.dumps(
            document, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2
        )
        + "\n"
    )
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=directory, prefix=".prepared.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
    try:
        # Same-filesystem hard link is an atomic create-only operation. A crash
        # during JSON serialisation cannot reserve the immutable final key.
        os.link(temporary, path)
    except FileExistsError:
        existing = json.loads(path.read_text())
        for field in ("key", "forecast", "research_replay"):
            if existing.get(field) != document.get(field):
                raise OperationalPreparationError(
                    "conflicting preparation already exists for this immutable key"
                )
    finally:
        temporary.unlink(missing_ok=True)
    return path
