"""Joint first-destination probabilities with a canonical one-week anchor.

The projection is parameter-free.  No observed target is consulted: in three
states there are two possible first destinations, so fixed departure totals
reduce the four long-horizon probabilities to a two-dimensional parallelogram.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import timedelta
from typing import Any, Mapping

import numpy as np
import pandas as pd

from regime_lab.schema import STATE_ORDER

COHERENCE_VERSION = "canonical-one-week-joint-first-destination/2"
HORIZONS = (1, 4, 13)


def _polygon_projection(
    point: np.ndarray, anchor: float, d4: float, d13: float
) -> np.ndarray:
    x, y = point
    if anchor <= x <= anchor + d4 and x <= y <= x + d13:
        return point.copy()
    vertices = np.asarray(
        [
            [anchor, anchor],
            [anchor + d4, anchor + d4],
            [anchor + d4, anchor + d4 + d13],
            [anchor, anchor + d13],
        ]
    )
    candidates = []
    for start, end in zip(vertices, np.roll(vertices, -1, axis=0), strict=True):
        edge = end - start
        length = float(edge @ edge)
        fraction = (
            0.0
            if length == 0
            else float(np.clip((point - start) @ edge / length, 0, 1))
        )
        candidates.append(start + fraction * edge)
    return min(
        candidates, key=lambda candidate: float(np.square(candidate - point).sum())
    )


def reconcile_directional_path(
    current_state: str,
    next_week_probabilities: Mapping[str, float],
    raw_directional_rows: Mapping[str, Mapping[str, Any]],
    *,
    canonical_model: str = "canonical_next_week_distribution",
) -> dict[str, dict[str, Any]]:
    """Return coherent rows, preserving supplied totals and provenance fields."""
    if current_state not in STATE_ORDER:
        raise ValueError("invalid current state")
    probabilities = np.asarray([next_week_probabilities[s] for s in STATE_ORDER], float)
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
        or not np.isclose(probabilities.sum(), 1, atol=1e-7, rtol=0)
    ):
        raise ValueError("next-week probabilities must form a distribution")
    alternatives = [s for s in STATE_ORDER if s != current_state]
    totals = np.asarray(
        [raw_directional_rows[f"{h}w"]["probability"] for h in HORIZONS], float
    )
    if (
        not np.isfinite(totals).all()
        or (totals < 0).any()
        or (totals > 1).any()
        or (np.diff(totals) < -1e-8).any()
    ):
        raise ValueError("departure totals must be finite, bounded and monotone")
    anchor_total = sum(float(next_week_probabilities[s]) for s in alternatives)
    if abs(totals[0] - anchor_total) > 2e-8:
        raise ValueError("one-week departure must match the canonical forecast")
    anchor = float(next_week_probabilities[alternatives[0]])
    raw = np.asarray(
        [
            [
                raw_directional_rows[f"{h}w"]["first_destination"][s]
                for s in alternatives
            ]
            for h in HORIZONS
        ],
        float,
    )
    if not np.isfinite(raw).all() or (raw < 0).any():
        raise ValueError(
            "raw first-destination probabilities must be finite and nonnegative"
        )
    # Fixed totals make the least-squares objective in each horizon proportional
    # to (x - (raw_first + total - raw_second)/2)^2, even with rounded inputs.
    target = (raw[1:, 0] + totals[1:] - raw[1:, 1]) / 2
    projected = _polygon_projection(
        target,
        anchor,
        max(0.0, totals[1] - anchor_total),
        max(0.0, totals[2] - totals[1]),
    )
    first = [anchor, *projected.tolist()]
    result = deepcopy(dict(raw_directional_rows))
    for i, h in enumerate(HORIZONS):
        row = result[f"{h}w"]
        if h == 1:
            destinations = {
                s: 0.0 if s == current_state else float(next_week_probabilities[s])
                for s in STATE_ORDER
            }
            row["model"] = canonical_model
        else:
            left = round(float(first[i]), 8)
            destinations = {
                current_state: 0.0,
                alternatives[0]: left,
                alternatives[1]: round(float(totals[i]) - left, 8),
            }
        row["first_destination"] = {s: destinations[s] for s in STATE_ORDER}
        row["no_departure"] = round(1.0 - float(totals[i]), 8)
    return result


def _score_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    if not records:
        return output
    frame = pd.DataFrame(records)
    for (split, horizon, variant), group in frame.groupby(
        ["evaluation_split", "horizon_weeks", "variant"], sort=True
    ):
        matrix = group[["p_no_departure", *[f"p_{s}" for s in STATE_ORDER]]].to_numpy(
            float
        )
        outcomes = ["no_departure", *STATE_ORDER]
        actual = np.asarray([outcomes.index(s) for s in group["actual_outcome"]])
        truth = np.eye(4)[actual]
        confidence = matrix.max(axis=1)
        correct = matrix.argmax(axis=1) == actual
        bins = np.minimum((confidence * 10).astype(int), 9)
        ece = sum(
            float(np.mean(bins == b))
            * abs(
                float(confidence[bins == b].mean()) - float(correct[bins == b].mean())
            )
            for b in range(10)
            if (bins == b).any()
        )
        event = actual != 0
        conditional = matrix[:, 1:] / np.maximum(
            matrix[:, 1:].sum(axis=1, keepdims=True), 1e-12
        )
        event_positions = np.flatnonzero(event)
        output.append(
            {
                "evaluation_split": split,
                "horizon_weeks": int(horizon),
                "variant": variant,
                "n_predictions": len(group),
                "event_count": int(event.sum()),
                "log_loss": float(
                    -np.log(
                        np.maximum(matrix[np.arange(len(matrix)), actual], 1e-12)
                    ).mean()
                ),
                "brier": float(np.square(matrix - truth).sum(axis=1).mean()),
                "top_label_ece": float(ece),
                "conditional_destination_log_loss": float(
                    -np.log(
                        np.maximum(
                            conditional[event_positions, actual[event] - 1], 1e-12
                        )
                    ).mean()
                )
                if event.any()
                else None,
                "mean_departure_probability": float((1 - matrix[:, 0]).mean()),
                "observed_departure_rate": float(event.mean()),
            }
        )
    return output


def upgrade_directional_payload(
    payload: Mapping[str, Any], states: pd.Series | None = None
) -> dict[str, Any]:
    """Copy and upgrade a payload; optionally score only fully observed targets.

    Existing reviewed payloads are never mutated.  The returned payload records
    raw rows so projected forecasts can be independently reproduced and scored.
    No model selection is changed by this semantic correction.
    """
    result = deepcopy(dict(payload))
    # Changed content cannot inherit an approval or a hash binding the original.
    if "meta" in result:
        result["meta"]["publication_status"] = "unpublished"
        result["meta"].pop("publication_review", None)
        result["meta"].pop("generation_manifest_sha256", None)
    lifecycle = result.get("model", {}).get("lifecycle")
    if isinstance(lifecycle, dict):
        lifecycle["publication"] = {"status": "unpublished"}
        lifecycle["deployment"] = {"status": "candidate"}
    if states is None:
        states = pd.Series(
            {pd.Timestamp(w["date"]): w["current"]["state"] for w in result["weekly"]}
        )
    labels = {pd.Timestamp(d).date(): str(v) for d, v in states.items()}
    cutoff = pd.Timestamp(result["model"].get("selection_end", "2023-01-01")).date()
    records = []
    changed = 0
    for week in result["weekly"]:
        raw = deepcopy(week.get("directional_risk_raw", week["directional_risk"]))
        projected = reconcile_directional_path(
            week["current"]["state"],
            week["next_week"]["probabilities"],
            raw,
            canonical_model=str(
                week["next_week"].get("model", "canonical_next_week_distribution")
            ),
        )
        changed += int(
            any(
                abs(
                    float(raw[f"{h}w"]["first_destination"][s])
                    - float(projected[f"{h}w"]["first_destination"][s])
                )
                > 1e-8
                for h in HORIZONS
                for s in STATE_ORDER
            )
        )
        week["directional_risk_raw"] = raw
        week["directional_risk"] = projected
        origin = pd.Timestamp(week["date"]).date()
        for h in HORIZONS:
            future = [origin + timedelta(weeks=k) for k in range(1, h + 1)]
            if not all(d in labels for d in future):
                continue
            outcome = next(
                (labels[d] for d in future if labels[d] != week["current"]["state"]),
                "no_departure",
            )
            split = "selection" if future[-1] < cutoff else "retrospective_diagnostic"
            for variant, rows in (("raw", raw), ("coherent", projected)):
                row = rows[f"{h}w"]
                records.append(
                    {
                        "origin_date": week["date"],
                        "target_end": future[-1].isoformat(),
                        "evaluation_split": split,
                        "horizon_weeks": h,
                        "variant": variant,
                        "actual_outcome": outcome,
                        "p_no_departure": row["no_departure"],
                        **{f"p_{s}": row["first_destination"][s] for s in STATE_ORDER},
                    }
                )
    directional = result["model"]["directional_transition"]
    directional["coherence_version"] = COHERENCE_VERSION
    directional["one_week_deployed_model"] = "canonical_next_week_champion"
    directional["coherence_evidence"] = {
        "schema_version": "regime-directional-coherence-evidence/2",
        "status": "evaluated" if records else "no_mature_targets",
        "method": "parameter_free_joint_l2_projection",
        "one_week_anchor": "canonical_next_week_distribution",
        "selection_effect": "none",
        "evidence_track": "reconstructed_oos",
        "origin_count": len(result["weekly"]),
        "adjusted_origin_count": changed,
        "metrics": _score_rows(records),
        "oos_predictions": records,
    }
    return result


def apply_full_directional_research(
    payload: Mapping[str, Any],
    states: pd.Series,
    benchmark: Any,
) -> dict[str, Any]:
    """Compose a completed full-origin benchmark into an unpublished copy.

    The canonical next-state champion and its selection are preserved.  Only
    direction research and its documented execution budget are replaced.
    """
    import hashlib
    import json
    from regime_lab.v5 import _directional_lookup, _directional_rows, _records
    from regime_lab.v5_artifacts import (
        V5_RESEARCH_ARTIFACTS,
        canonical_v5_artifact_csv_bytes,
    )

    result = deepcopy(dict(payload))
    lookup = _directional_lookup(benchmark)
    for week in result["weekly"]:
        week["directional_risk"] = _directional_rows(
            week, states=states, result=benchmark, lookup=lookup
        )
        week.pop("directional_risk_raw", None)
    metadata = result["model"]["directional_transition"]
    metadata.update(
        {
            "champions": {
                f"{h}w": str(benchmark.champions_by_horizon[h]) for h in HORIZONS
            },
            "leaderboard": _records(benchmark.leaderboard),
            "selection_diagnostics": _records(benchmark.selection_diagnostics),
            "selection_end": pd.Timestamp(benchmark.selection_end).date().isoformat(),
            "evaluation_coverage": "all_available_purged_origins",
        }
    )
    frames = {
        "directional_oos_predictions": benchmark.predictions,
        "directional_model_leaderboard": benchmark.leaderboard,
        "directional_walk_forward_splits": benchmark.split_audit,
        "directional_selection_diagnostics": benchmark.selection_diagnostics,
        "directional_forecasts": benchmark.latest_forecasts,
    }
    artifacts = result["model"].get("research_artifacts")
    if isinstance(artifacts, dict):
        for key, frame in frames.items():
            artifacts[key] = {
                "path": V5_RESEARCH_ARTIFACTS[key].path,
                "row_count": len(frame),
                "sha256": hashlib.sha256(
                    canonical_v5_artifact_csv_bytes(key, frame)
                ).hexdigest(),
            }
    parameters = result["model"]["execution_parameters"]
    parameters["directional_maximum_selection_origins"] = None
    parameters["directional_maximum_diagnostic_origins"] = None
    parameters.pop("sha256", None)
    parameters["sha256"] = hashlib.sha256(
        json.dumps(
            parameters, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()
    return upgrade_directional_payload(result, states)
