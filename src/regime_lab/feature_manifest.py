"""Canonical feature-family manifests for structural model artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from typing import Any


FEATURE_MANIFEST_SCHEMA_VERSION = "1.0.0"


def complete_feature_group_manifest(
    columns: Sequence[object],
    groups: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Return an exact-once manifest for the final model matrix.

    Dataset construction owns the structural-family assignments.  The regime
    pipeline subsequently adds causal boundary and duration columns; those
    protected v3 columns are folded into ``legacy_v3`` here so the published
    manifest describes the matrix that estimators actually receive.
    """

    ordered_columns = tuple(str(column) for column in columns)
    if len(set(ordered_columns)) != len(ordered_columns):
        raise ValueError("model feature columns must be unique")
    column_set = set(ordered_columns)
    seen_ids: set[str] = set()
    owner: dict[str, str] = {}
    normalized: list[dict[str, Any]] = []
    legacy_position: int | None = None
    for position, raw in enumerate(groups):
        if not isinstance(raw, Mapping):
            raise TypeError("feature manifest groups must be mappings")
        group_id = str(raw.get("id", "")).strip()
        if not group_id or group_id in seen_ids:
            raise ValueError("feature manifest group ids must be non-empty and unique")
        seen_ids.add(group_id)
        description = str(raw.get("description", "")).strip()
        raw_features = raw.get("features", ())
        if not isinstance(raw_features, Sequence) or isinstance(
            raw_features, (str, bytes)
        ):
            raise TypeError(f"feature manifest group {group_id!r} needs a feature list")
        features = tuple(str(feature) for feature in raw_features)
        if len(set(features)) != len(features):
            raise ValueError(f"feature manifest group {group_id!r} has duplicates")
        unknown = sorted(set(features).difference(column_set))
        if unknown:
            raise ValueError(
                f"feature manifest group {group_id!r} contains unknown columns: {unknown}"
            )
        for feature in features:
            previous = owner.get(feature)
            if previous is not None:
                raise ValueError(
                    f"feature {feature!r} belongs to both {previous!r} and {group_id!r}"
                )
            owner[feature] = group_id
        normalized.append(
            {
                "id": group_id,
                "description": description,
                "feature_count": len(features),
                "features": features,
            }
        )
        if group_id == "legacy_v3":
            legacy_position = position

    unassigned = tuple(column for column in ordered_columns if column not in owner)
    if unassigned:
        if legacy_position is None:
            normalized.append(
                {
                    "id": "legacy_v3",
                    "description": "Protected v3 features and causal regime-state transforms",
                    "feature_count": len(unassigned),
                    "features": unassigned,
                }
            )
        else:
            legacy = normalized[legacy_position]
            combined = tuple(
                column
                for column in ordered_columns
                if column in set((*legacy["features"], *unassigned))
            )
            legacy["features"] = combined
            legacy["feature_count"] = len(combined)

    assigned = [
        feature
        for group in normalized
        for feature in tuple(group["features"])
    ]
    if len(assigned) != len(ordered_columns) or set(assigned) != column_set:
        raise RuntimeError("feature manifest does not assign every model column exactly once")
    return tuple(normalized)


def feature_manifest_core(
    groups: Sequence[Mapping[str, Any]],
    *,
    feature_set_version: str,
) -> dict[str, Any]:
    """Build the JSON-safe, unhashed canonical manifest body."""

    if not isinstance(feature_set_version, str) or not feature_set_version:
        raise ValueError("feature_set_version must be non-empty")
    rows: list[dict[str, Any]] = []
    seen_features: set[str] = set()
    seen_ids: set[str] = set()
    for raw in groups:
        group_id = str(raw.get("id", "")).strip()
        if not group_id or group_id in seen_ids:
            raise ValueError("feature manifest group ids must be non-empty and unique")
        seen_ids.add(group_id)
        features = [str(value) for value in raw.get("features", ())]
        if len(features) != len(set(features)):
            raise ValueError(f"feature manifest group {group_id!r} has duplicates")
        overlap = seen_features.intersection(features)
        if overlap:
            raise ValueError(f"features assigned more than once: {sorted(overlap)}")
        seen_features.update(features)
        declared_count = int(raw.get("feature_count", len(features)))
        if declared_count != len(features):
            raise ValueError(f"feature count mismatch for group {group_id!r}")
        rows.append(
            {
                "id": group_id,
                "description": str(raw.get("description", "")),
                "feature_count": declared_count,
                "features": features,
            }
        )
    if not rows or not seen_features:
        raise ValueError("feature manifest must contain at least one feature")
    return {
        "schema_version": FEATURE_MANIFEST_SCHEMA_VERSION,
        "feature_set_version": feature_set_version,
        "feature_count": len(seen_features),
        "groups": rows,
    }


def feature_manifest_sha256(core: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        core,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def feature_manifest_document(
    groups: Sequence[Mapping[str, Any]],
    *,
    feature_set_version: str,
) -> dict[str, Any]:
    core = feature_manifest_core(groups, feature_set_version=feature_set_version)
    return {**core, "sha256": feature_manifest_sha256(core)}


__all__ = [
    "FEATURE_MANIFEST_SCHEMA_VERSION",
    "complete_feature_group_manifest",
    "feature_manifest_core",
    "feature_manifest_document",
    "feature_manifest_sha256",
]
