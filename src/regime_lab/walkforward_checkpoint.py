"""Private, identity-bound checkpoints for the V5 base walk-forward.

The checkpoint contains only run identity metadata, OOS prediction rows, and
split-audit rows.  Feature values and provider observations are hashed in
memory and are never written to the checkpoint directory.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
import os
from pathlib import Path
import platform
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from regime_lab.analysis.labels import STATE_ORDER
from regime_lab.analysis.models import (
    DIRECT_NEXT_STATE_MODEL_NAMES,
    MODEL_NAMES,
    MODEL_REGISTRY,
    SHADOW_NEXT_STATE_MODEL_NAMES,
    BenchmarkProfile,
)
from regime_lab.analysis.models import model_manifest_sha256
from regime_lab.io import write_json_atomic


MANIFEST_SCHEMA_VERSION = "regime-v5-base-walkforward-checkpoint-manifest/1"
RECORD_SCHEMA_VERSION = "regime-v5-base-walkforward-checkpoint-origin/1"
CHECKPOINT_KIND = "private_v5_base_run_benchmark"
CHECKPOINT_IMPLEMENTATION_VERSION = "1"

PREDICTION_COLUMNS: tuple[str, ...] = (
    "origin_date",
    "target_date",
    "model",
    "evaluation_split",
    "current_state",
    "actual",
    "predicted",
    *(f"p_{state}" for state in STATE_ORDER),
    "train_size",
    "gap",
    "fallback",
    "fallback_reason",
)
SPLIT_AUDIT_COLUMNS: tuple[str, ...] = (
    "origin_date",
    "target_date",
    "train_size",
    "train_start",
    "last_train_origin",
    "last_train_target",
    "purged_origin_count",
    "first_purged_origin",
    "gap",
    "evaluation_split",
)

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_INTEGER_PATTERN = re.compile(r"0|-?[1-9][0-9]*")
_MANIFEST_KEYS = frozenset(
    {"schema_version", "visibility", "identity", "run_signature"}
)
_IDENTITY_KEYS = frozenset(
    {
        "checkpoint_kind",
        "implementation_version",
        "feature_matrix",
        "state_vector",
        "benchmark_parameters",
        "source_fingerprint_sha256",
        "origins",
    }
)
_ORIGIN_MANIFEST_KEYS = frozenset(
    {
        "sequence",
        "origin_date",
        "target_date",
        "evaluation_split",
        "train_size",
        "train_start",
        "last_train_origin",
        "last_train_target",
        "purged_origin_count",
        "first_purged_origin",
        "gap",
        "signature",
        "record_file",
    }
)
_RECORD_KEYS = frozenset(
    {
        "schema_version",
        "run_signature",
        "origin_signature",
        "sequence",
        "prediction_rows",
        "split_audit",
        "record_sha256",
    }
)


def runtime_version_manifest() -> dict[str, str | None]:
    """Return the estimator runtime versions that can change fitted outputs."""

    packages = (
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "xgboost",
        "hmmlearn",
        "joblib",
        "threadpoolctl",
    )
    versions: dict[str, str | None] = {
        "python": platform.python_version(),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
    }
    for package in packages:
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = None
    return versions
_FORBIDDEN_PUBLIC_PATH_COMPONENTS = frozenset(
    {
        "docs",
        "gh-pages",
        "publication",
        "public",
        "public-site",
        "site",
        "sites",
        "web",
    }
)


class CheckpointError(RuntimeError):
    """Base error for a checkpoint that cannot be used safely."""


class CheckpointIdentityMismatch(CheckpointError):
    """The directory belongs to a different resolved benchmark run."""


class CheckpointCorruptionError(CheckpointError):
    """Stored checkpoint bytes fail their closed schema or integrity checks."""


class CheckpointPrivacyError(CheckpointError):
    """The requested checkpoint location is public or not private on disk."""


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_document(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise CheckpointCorruptionError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _reject_json_constant(value: str) -> None:
    raise CheckpointCorruptionError(f"non-finite JSON constant: {value}")


def _read_json_strict(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except CheckpointCorruptionError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CheckpointCorruptionError(
            f"checkpoint JSON cannot be read: {path.name}"
        ) from exc


def _timezone_name(value: pd.Timestamp) -> str | None:
    timezone = value.tz
    if timezone is None:
        return None
    return str(
        getattr(timezone, "key", None)
        or getattr(timezone, "zone", None)
        or timezone
    )


def _timestamp_document(value: pd.Timestamp | datetime) -> dict[str, Any]:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise TypeError("NaT is not a timestamp; use None for an absent value")
    try:
        nanoseconds = timestamp.value
    except OverflowError as exc:
        raise ValueError("timestamps must fit pandas nanosecond precision") from exc
    return {
        "nanoseconds": str(nanoseconds),
        "timezone": _timezone_name(timestamp),
    }


def _timestamp_from_document(value: Any) -> pd.Timestamp:
    if not isinstance(value, Mapping) or set(value) != {"nanoseconds", "timezone"}:
        raise CheckpointCorruptionError("invalid timestamp document")
    raw_nanoseconds = value["nanoseconds"]
    timezone = value["timezone"]
    if not isinstance(raw_nanoseconds, str) or not _INTEGER_PATTERN.fullmatch(
        raw_nanoseconds
    ):
        raise CheckpointCorruptionError("invalid timestamp nanoseconds")
    if timezone is not None and not isinstance(timezone, str):
        raise CheckpointCorruptionError("invalid timestamp timezone")
    try:
        if timezone is None:
            result = pd.Timestamp(int(raw_nanoseconds), unit="ns")
        else:
            result = pd.Timestamp(
                int(raw_nanoseconds), unit="ns", tz="UTC"
            ).tz_convert(timezone)
    except (OverflowError, TypeError, ValueError) as exc:
        raise CheckpointCorruptionError("invalid timestamp value") from exc
    if _timestamp_document(result) != dict(value):
        raise CheckpointCorruptionError("timestamp document is not canonical")
    return result


def _normalize_scalar(value: Any) -> Any:
    if value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, datetime):
        value = pd.Timestamp(value)
    return value


def encode_checkpoint_scalar(value: Any) -> dict[str, Any]:
    """Encode one supported scalar without losing its Python/Pandas type."""

    value = _normalize_scalar(value)
    if value is None:
        return {"type": "none"}
    if isinstance(value, pd.Timestamp):
        return {"type": "timestamp", **_timestamp_document(value)}
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    if isinstance(value, int):
        return {"type": "int", "value": str(value)}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("checkpoint floats must be finite")
        return {"type": "float", "value": value.hex()}
    if isinstance(value, str):
        return {"type": "str", "value": value}
    raise TypeError(
        "checkpoint scalars must be timestamps, bools, ints, strings, "
        "finite floats, or None"
    )


def decode_checkpoint_scalar(value: Any) -> Any:
    """Decode a scalar and reject non-canonical or type-confused JSON."""

    if not isinstance(value, Mapping) or not isinstance(value.get("type"), str):
        raise CheckpointCorruptionError("invalid typed checkpoint scalar")
    scalar_type = value["type"]
    if scalar_type == "none":
        if set(value) != {"type"}:
            raise CheckpointCorruptionError("invalid none scalar")
        return None
    if scalar_type == "timestamp":
        if set(value) != {"type", "nanoseconds", "timezone"}:
            raise CheckpointCorruptionError("invalid timestamp scalar")
        return _timestamp_from_document(
            {
                "nanoseconds": value["nanoseconds"],
                "timezone": value["timezone"],
            }
        )
    if set(value) != {"type", "value"}:
        raise CheckpointCorruptionError(f"invalid {scalar_type} scalar")
    raw = value["value"]
    if scalar_type == "bool":
        if not isinstance(raw, bool):
            raise CheckpointCorruptionError("invalid bool scalar")
        return raw
    if scalar_type == "int":
        if not isinstance(raw, str) or not _INTEGER_PATTERN.fullmatch(raw):
            raise CheckpointCorruptionError("invalid int scalar")
        return int(raw)
    if scalar_type == "float":
        if not isinstance(raw, str):
            raise CheckpointCorruptionError("invalid float scalar")
        try:
            result = float.fromhex(raw)
        except ValueError as exc:
            raise CheckpointCorruptionError("invalid float scalar") from exc
        if not math.isfinite(result) or result.hex() != raw:
            raise CheckpointCorruptionError("float scalar is not canonical and finite")
        return result
    if scalar_type == "str":
        if not isinstance(raw, str):
            raise CheckpointCorruptionError("invalid str scalar")
        return raw
    raise CheckpointCorruptionError(f"unsupported scalar type: {scalar_type}")


def _encode_row(row: Mapping[str, Any], columns: Sequence[str]) -> dict[str, Any]:
    expected = tuple(columns)
    if set(row) != set(expected) or len(row) != len(expected):
        raise ValueError(
            f"checkpoint row columns must exactly match {expected}; got {tuple(row)}"
        )
    return {column: encode_checkpoint_scalar(row[column]) for column in expected}


def _decode_row(value: Any, columns: Sequence[str]) -> dict[str, Any]:
    expected = tuple(columns)
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise CheckpointCorruptionError(
            f"checkpoint row columns must exactly match {expected}"
        )
    return {column: decode_checkpoint_scalar(value[column]) for column in expected}


def _hash_token(digest: Any, token: str) -> None:
    encoded = token.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
    digest.update(encoded)


def _hash_timestamp(digest: Any, value: pd.Timestamp) -> None:
    document = _timestamp_document(value)
    _hash_token(digest, str(document["nanoseconds"]))
    _hash_token(digest, "" if document["timezone"] is None else document["timezone"])


def _canonical_data_identity(
    features: pd.DataFrame,
    states: pd.Series,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any], dict[str, Any]]:
    if not isinstance(features, pd.DataFrame) or not isinstance(states, pd.Series):
        raise TypeError("features must be a DataFrame and states must be a Series")
    if not isinstance(features.index, pd.DatetimeIndex):
        raise TypeError("features must use a DatetimeIndex")
    if not features.index.equals(states.index):
        raise ValueError("features and states must use the identical index")
    if features.index.has_duplicates or not features.index.is_monotonic_increasing:
        raise ValueError("feature index must be unique and increasing")
    if features.empty or features.shape[1] == 0:
        raise ValueError("checkpoint identity needs non-empty feature content")
    columns = tuple(features.columns)
    if not all(isinstance(column, str) and column for column in columns):
        raise TypeError("checkpointed feature columns must be non-empty strings")
    if len(set(columns)) != len(columns):
        raise ValueError("checkpointed feature columns must be unique")
    non_numeric = [
        column
        for column in columns
        if not pd.api.types.is_numeric_dtype(features[column])
    ]
    if non_numeric:
        raise TypeError(f"checkpointed feature columns must be numeric: {non_numeric}")
    if states.isna().any():
        raise ValueError("checkpointed states must be complete")
    invalid_states = sorted(set(states.astype(str)).difference(STATE_ORDER))
    if invalid_states:
        raise ValueError(
            f"checkpointed states contain unsupported labels: {invalid_states}"
        )
    canonical_features = features.astype(float)
    canonical_states = states.astype(str)

    feature_digest = hashlib.sha256()
    _hash_token(feature_digest, "regime-v5-canonical-feature-matrix/1")
    _hash_token(feature_digest, str(len(canonical_features)))
    _hash_token(feature_digest, str(len(columns)))
    for column in columns:
        _hash_token(feature_digest, column)
    for at, values in canonical_features.iterrows():
        _hash_timestamp(feature_digest, pd.Timestamp(at))
        for value in values.to_numpy(dtype=float):
            number = float(value)
            if math.isnan(number):
                token = "nan"
            elif math.isinf(number):
                token = "+inf" if number > 0 else "-inf"
            else:
                token = number.hex()
            _hash_token(feature_digest, token)

    state_digest = hashlib.sha256()
    _hash_token(state_digest, "regime-v5-canonical-state-vector/1")
    _hash_token(state_digest, str(len(canonical_states)))
    for at, state in canonical_states.items():
        _hash_timestamp(state_digest, pd.Timestamp(at))
        _hash_token(state_digest, state)

    first = _timestamp_document(pd.Timestamp(canonical_features.index[0]))
    last = _timestamp_document(pd.Timestamp(canonical_features.index[-1]))
    feature_identity = {
        "sha256": feature_digest.hexdigest(),
        "row_count": len(canonical_features),
        "column_count": len(columns),
        "first_index": first,
        "last_index": last,
    }
    state_identity = {
        "sha256": state_digest.hexdigest(),
        "row_count": len(canonical_states),
        "first_index": first,
        "last_index": last,
    }
    return canonical_features, canonical_states, feature_identity, state_identity


def _normalize_selection_end(
    value: str | pd.Timestamp | None,
    index: pd.DatetimeIndex,
) -> pd.Timestamp | None:
    if value is None:
        return None
    cutoff = pd.Timestamp(value)
    if index.tz is None:
        if cutoff.tzinfo is not None:
            cutoff = cutoff.tz_localize(None)
    elif cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize(index.tz)
    else:
        cutoff = cutoff.tz_convert(index.tz)
    return cutoff


@dataclass(frozen=True)
class ResolvedBenchmarkParameters:
    """Every resolved argument that can affect the base benchmark result."""

    profile: BenchmarkProfile
    model_names: tuple[str, ...]
    include_hmm: bool
    gap: int
    random_state: int
    selection_end: str | pd.Timestamp | None
    selection_max_origins: int | None
    model_workers: int
    minimum_selection_predictions: int
    minimum_holdout_predictions: int
    minimum_log_loss_improvement: float

    @classmethod
    def from_arguments(
        cls,
        *,
        profile: BenchmarkProfile,
        models: Iterable[str] | None = None,
        include_hmm: bool = False,
        gap: int = 1,
        random_state: int = 17,
        selection_end: str | pd.Timestamp | None = None,
        selection_max_origins: int | None = None,
        model_workers: int = 1,
        minimum_selection_predictions: int = 12,
        minimum_holdout_predictions: int = 12,
        minimum_log_loss_improvement: float = 0.05,
    ) -> "ResolvedBenchmarkParameters":
        if not isinstance(profile, BenchmarkProfile):
            raise TypeError("profile must be an already resolved BenchmarkProfile")
        names = list(MODEL_NAMES if models is None else (str(name) for name in models))
        if include_hmm and "gaussian_hmm" not in names:
            names.append("gaussian_hmm")
        names = list(dict.fromkeys(names))
        # MODEL_NAMES is the intentionally small weekly operating roster.
        # Explicit callers may still request frozen/research models so old
        # checkpoints remain reproducible after a model is removed from the
        # weekly retraining set.
        supported = {
            name
            for name, spec in MODEL_REGISTRY.items()
            if spec.task == "multiclass_next_state" and spec.kind != "synthetic"
        }.union(
            DIRECT_NEXT_STATE_MODEL_NAMES,
            SHADOW_NEXT_STATE_MODEL_NAMES,
            {"gaussian_hmm"},
        )
        unknown = sorted(set(names).difference(supported))
        if unknown:
            raise ValueError(f"unknown benchmark models: {unknown}")
        resolved_selection_max = selection_max_origins
        if resolved_selection_max is None and profile.name == "quick" and selection_end is not None:
            resolved_selection_max = max(3, int(minimum_selection_predictions))
        result = cls(
            profile=profile,
            model_names=tuple(names),
            include_hmm=bool(include_hmm),
            gap=int(gap),
            random_state=int(random_state),
            selection_end=selection_end,
            selection_max_origins=(
                None if resolved_selection_max is None else int(resolved_selection_max)
            ),
            model_workers=int(model_workers),
            minimum_selection_predictions=int(minimum_selection_predictions),
            minimum_holdout_predictions=int(minimum_holdout_predictions),
            minimum_log_loss_improvement=float(
                minimum_log_loss_improvement
            ),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if not self.model_names or len(set(self.model_names)) != len(self.model_names):
            raise ValueError("resolved model_names must be non-empty and unique")
        if self.gap < 0:
            raise ValueError("gap must be non-negative")
        if self.model_workers < 1:
            raise ValueError("model_workers must be positive")
        if self.profile.minimum_train_weeks < 12:
            raise ValueError("profile minimum_train_weeks must be at least 12")
        if self.minimum_selection_predictions < 1 or self.minimum_holdout_predictions < 1:
            raise ValueError("minimum split prediction counts must be positive")
        if (
            not math.isfinite(self.minimum_log_loss_improvement)
            or self.minimum_log_loss_improvement < 0.0
        ):
            raise ValueError(
                "minimum_log_loss_improvement must be non-negative and finite"
            )
        if self.selection_max_origins is not None and self.selection_max_origins < 1:
            raise ValueError("selection_max_origins must be positive or None")

    def manifest(self, index: pd.DatetimeIndex) -> dict[str, Any]:
        self.validate()
        cutoff = _normalize_selection_end(self.selection_end, index)
        return {
            "profile": asdict(self.profile),
            "model_names": list(self.model_names),
            "include_hmm": self.include_hmm,
            "gap": self.gap,
            "random_state": self.random_state,
            "selection_end": None if cutoff is None else _timestamp_document(cutoff),
            "selection_max_origins": self.selection_max_origins,
            "model_workers": self.model_workers,
            "minimum_selection_predictions": self.minimum_selection_predictions,
            "minimum_holdout_predictions": self.minimum_holdout_predictions,
            "minimum_log_loss_improvement": self.minimum_log_loss_improvement,
            "model_manifest_sha256": model_manifest_sha256(
                self.profile,
                random_state=self.random_state,
                names=self.model_names,
            ),
            "runtime_versions": runtime_version_manifest(),
        }


@dataclass(frozen=True)
class CheckpointOrigin:
    sequence: int
    origin_date: pd.Timestamp
    target_date: pd.Timestamp
    evaluation_split: str
    train_size: int
    train_start: pd.Timestamp
    last_train_origin: pd.Timestamp
    last_train_target: pd.Timestamp
    purged_origin_count: int
    first_purged_origin: pd.Timestamp | None
    gap: int
    current_state: str
    actual: str
    signature: str
    record_file: str

    def public_manifest(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "origin_date": _timestamp_document(self.origin_date),
            "target_date": _timestamp_document(self.target_date),
            "evaluation_split": self.evaluation_split,
            "train_size": self.train_size,
            "train_start": _timestamp_document(self.train_start),
            "last_train_origin": _timestamp_document(self.last_train_origin),
            "last_train_target": _timestamp_document(self.last_train_target),
            "purged_origin_count": self.purged_origin_count,
            "first_purged_origin": (
                None
                if self.first_purged_origin is None
                else _timestamp_document(self.first_purged_origin)
            ),
            "gap": self.gap,
            "signature": self.signature,
            "record_file": self.record_file,
        }


def _resolve_test_positions(
    index: pd.DatetimeIndex,
    parameters: ResolvedBenchmarkParameters,
) -> tuple[list[int], set[int], set[int], pd.Timestamp | None]:
    supervised_count = len(index) - 1
    first_test_position = parameters.profile.minimum_train_weeks + parameters.gap
    if first_test_position >= supervised_count:
        raise ValueError(
            "not enough observations for checkpointed benchmark: "
            f"need > {first_test_position + 1}, got {len(index)}"
        )
    all_positions = list(range(first_test_position, supervised_count))
    cutoff = _normalize_selection_end(parameters.selection_end, index)
    if cutoff is None:
        positions = all_positions
        if parameters.profile.max_origins is not None:
            positions = positions[-parameters.profile.max_origins :]
        return positions, set(), set(), None

    selection_available = [
        position
        for position in all_positions
        if pd.Timestamp(index[position + 1]) < cutoff
    ]
    holdout_available = [
        position
        for position in all_positions
        if pd.Timestamp(index[position + 1]) >= cutoff
    ]
    if len(selection_available) < parameters.minimum_selection_predictions:
        raise ValueError("insufficient selection OOS predictions for checkpoint")
    if len(holdout_available) < parameters.minimum_holdout_predictions:
        raise ValueError("insufficient holdout OOS predictions for checkpoint")
    if parameters.selection_max_origins is None:
        selected = selection_available
    else:
        budget = max(
            parameters.minimum_selection_predictions,
            parameters.selection_max_origins,
        )
        selected = selection_available[-min(budget, len(selection_available)) :]
    if parameters.profile.max_origins is None:
        holdout = holdout_available
    else:
        if parameters.profile.max_origins < parameters.minimum_holdout_predictions:
            raise ValueError("profile max_origins cannot cover the holdout minimum")
        holdout = holdout_available[-min(parameters.profile.max_origins, len(holdout_available)) :]
    return sorted([*selected, *holdout]), set(selected), set(holdout), cutoff


def _origin_core(
    *,
    sequence: int,
    index: pd.DatetimeIndex,
    states: pd.Series,
    test_position: int,
    selection_positions: set[int],
    cutoff: pd.Timestamp | None,
    gap: int,
) -> dict[str, Any]:
    train_stop = test_position - gap
    purged = index[train_stop:test_position]
    evaluation_split = (
        "legacy"
        if cutoff is None
        else ("selection" if test_position in selection_positions else "holdout")
    )
    return {
        "sequence": sequence,
        "origin_date": _timestamp_document(pd.Timestamp(index[test_position])),
        "target_date": _timestamp_document(pd.Timestamp(index[test_position + 1])),
        "evaluation_split": evaluation_split,
        "train_size": train_stop,
        "train_start": _timestamp_document(pd.Timestamp(index[0])),
        "last_train_origin": _timestamp_document(pd.Timestamp(index[train_stop - 1])),
        "last_train_target": _timestamp_document(pd.Timestamp(index[train_stop])),
        "purged_origin_count": len(purged),
        "first_purged_origin": (
            None if len(purged) == 0 else _timestamp_document(pd.Timestamp(purged[0]))
        ),
        "gap": gap,
        "current_state": str(states.iloc[test_position]),
        "actual": str(states.iloc[test_position + 1]),
    }


@dataclass(frozen=True)
class BenchmarkCheckpointIdentity:
    feature_identity: Mapping[str, Any]
    state_identity: Mapping[str, Any]
    parameter_manifest: Mapping[str, Any]
    source_fingerprint_sha256: str | None
    origins: tuple[CheckpointOrigin, ...]
    model_names: tuple[str, ...]
    run_signature: str

    @classmethod
    def build(
        cls,
        features: pd.DataFrame,
        states: pd.Series,
        parameters: ResolvedBenchmarkParameters,
        *,
        source_fingerprint_sha256: str | None = None,
    ) -> "BenchmarkCheckpointIdentity":
        (
            canonical_features,
            canonical_states,
            feature_identity,
            state_identity,
        ) = _canonical_data_identity(features, states)
        fingerprint = None
        if source_fingerprint_sha256 is not None:
            fingerprint = str(source_fingerprint_sha256).lower()
            if not _SHA256_PATTERN.fullmatch(fingerprint):
                raise ValueError("source_fingerprint_sha256 must be a SHA-256 hex digest")
        positions, selection, _holdout, cutoff = _resolve_test_positions(
            canonical_features.index,
            parameters,
        )
        origins: list[CheckpointOrigin] = []
        origin_manifests: list[dict[str, Any]] = []
        for sequence, position in enumerate(positions, start=1):
            core = _origin_core(
                sequence=sequence,
                index=canonical_features.index,
                states=canonical_states,
                test_position=position,
                selection_positions=selection,
                cutoff=cutoff,
                gap=parameters.gap,
            )
            signature = _sha256_document(core)
            record_file = f"{sequence:06d}.json"
            origin = CheckpointOrigin(
                sequence=sequence,
                origin_date=_timestamp_from_document(core["origin_date"]),
                target_date=_timestamp_from_document(core["target_date"]),
                evaluation_split=str(core["evaluation_split"]),
                train_size=int(core["train_size"]),
                train_start=_timestamp_from_document(core["train_start"]),
                last_train_origin=_timestamp_from_document(core["last_train_origin"]),
                last_train_target=_timestamp_from_document(core["last_train_target"]),
                purged_origin_count=int(core["purged_origin_count"]),
                first_purged_origin=(
                    None
                    if core["first_purged_origin"] is None
                    else _timestamp_from_document(core["first_purged_origin"])
                ),
                gap=int(core["gap"]),
                current_state=str(core["current_state"]),
                actual=str(core["actual"]),
                signature=signature,
                record_file=record_file,
            )
            origins.append(origin)
            origin_manifests.append(origin.public_manifest())

        parameter_manifest = parameters.manifest(canonical_features.index)
        identity_core = {
            "checkpoint_kind": CHECKPOINT_KIND,
            "implementation_version": CHECKPOINT_IMPLEMENTATION_VERSION,
            "feature_matrix": feature_identity,
            "state_vector": state_identity,
            "benchmark_parameters": parameter_manifest,
            "source_fingerprint_sha256": fingerprint,
            "origins": origin_manifests,
        }
        run_signature = _sha256_document(identity_core)
        return cls(
            feature_identity=feature_identity,
            state_identity=state_identity,
            parameter_manifest=parameter_manifest,
            source_fingerprint_sha256=fingerprint,
            origins=tuple(origins),
            model_names=parameters.model_names,
            run_signature=run_signature,
        )

    def manifest_document(self) -> dict[str, Any]:
        identity_core = {
            "checkpoint_kind": CHECKPOINT_KIND,
            "implementation_version": CHECKPOINT_IMPLEMENTATION_VERSION,
            "feature_matrix": dict(self.feature_identity),
            "state_vector": dict(self.state_identity),
            "benchmark_parameters": dict(self.parameter_manifest),
            "source_fingerprint_sha256": self.source_fingerprint_sha256,
            "origins": [origin.public_manifest() for origin in self.origins],
        }
        if _sha256_document(identity_core) != self.run_signature:
            raise RuntimeError("in-memory checkpoint identity signature is inconsistent")
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "visibility": "private",
            "identity": identity_core,
            "run_signature": self.run_signature,
        }


@dataclass(frozen=True)
class CompletedCheckpointOrigin:
    origin: CheckpointOrigin
    prediction_rows: tuple[dict[str, Any], ...]
    split_audit: dict[str, Any]


def _coerce_row(row: Mapping[str, Any], columns: Sequence[str]) -> dict[str, Any]:
    return _decode_row(_encode_row(row, columns), columns)


def _timestamp_equal(left: Any, right: pd.Timestamp) -> bool:
    return isinstance(left, pd.Timestamp) and left == right and _timezone_name(left) == _timezone_name(right)


def _validate_completed_origin(
    identity: BenchmarkCheckpointIdentity,
    origin: CheckpointOrigin,
    prediction_rows: Sequence[Mapping[str, Any]],
    split_audit: Mapping[str, Any],
) -> None:
    if len(prediction_rows) != len(identity.model_names):
        raise CheckpointCorruptionError(
            "completed origin has missing or extra model rows"
        )
    row_models = tuple(row.get("model") for row in prediction_rows)
    if row_models != identity.model_names or len(set(row_models)) != len(row_models):
        raise CheckpointCorruptionError(
            "completed origin has duplicate, missing, or reordered model rows"
        )
    for row in prediction_rows:
        if tuple(row) != PREDICTION_COLUMNS:
            raise CheckpointCorruptionError("prediction row column order is invalid")
        if not _timestamp_equal(row["origin_date"], origin.origin_date):
            raise CheckpointCorruptionError("prediction row has the wrong origin")
        if not _timestamp_equal(row["target_date"], origin.target_date):
            raise CheckpointCorruptionError("prediction row has the wrong target")
        if row["evaluation_split"] != origin.evaluation_split:
            raise CheckpointCorruptionError("prediction row has the wrong split")
        if row["current_state"] != origin.current_state or row["actual"] != origin.actual:
            raise CheckpointCorruptionError("prediction row does not match bound states")
        if row["current_state"] not in STATE_ORDER or row["actual"] not in STATE_ORDER:
            raise CheckpointCorruptionError("prediction row contains an unsupported state")
        if row["predicted"] not in STATE_ORDER:
            raise CheckpointCorruptionError("prediction row contains an unsupported prediction")
        if type(row["train_size"]) is not int or row["train_size"] != origin.train_size:
            raise CheckpointCorruptionError("prediction row has the wrong train_size")
        if type(row["gap"]) is not int or row["gap"] != origin.gap:
            raise CheckpointCorruptionError("prediction row has the wrong gap")
        if type(row["fallback"]) is not bool or not isinstance(row["fallback_reason"], str):
            raise CheckpointCorruptionError("prediction fallback fields have invalid types")
        probabilities = [row[f"p_{state}"] for state in STATE_ORDER]
        if any(type(value) is not float or not math.isfinite(value) for value in probabilities):
            raise CheckpointCorruptionError("prediction probabilities must be finite floats")
        if any(value < 0.0 or value > 1.0 for value in probabilities):
            raise CheckpointCorruptionError("prediction probabilities must be in [0, 1]")
        if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-8):
            raise CheckpointCorruptionError("prediction probabilities must sum to one")
        expected_prediction = STATE_ORDER[int(np.argmax(probabilities))]
        if row["predicted"] != expected_prediction:
            raise CheckpointCorruptionError("predicted state does not match probabilities")

    if tuple(split_audit) != SPLIT_AUDIT_COLUMNS:
        raise CheckpointCorruptionError("split-audit row column order is invalid")
    expected_split: dict[str, Any] = {
        "origin_date": origin.origin_date,
        "target_date": origin.target_date,
        "train_size": origin.train_size,
        "train_start": origin.train_start,
        "last_train_origin": origin.last_train_origin,
        "last_train_target": origin.last_train_target,
        "purged_origin_count": origin.purged_origin_count,
        "first_purged_origin": origin.first_purged_origin,
        "gap": origin.gap,
        "evaluation_split": origin.evaluation_split,
    }
    for column, expected in expected_split.items():
        actual = split_audit[column]
        if isinstance(expected, pd.Timestamp):
            matches = _timestamp_equal(actual, expected)
        else:
            matches = type(actual) is type(expected) and actual == expected
        if not matches:
            raise CheckpointCorruptionError(
                f"split-audit row has the wrong {column}"
            )


def _validate_private_root(path: Path) -> Path:
    resolved = path.expanduser().resolve(strict=False)
    forbidden = _FORBIDDEN_PUBLIC_PATH_COMPONENTS.intersection(
        part.lower() for part in resolved.parts
    )
    if forbidden:
        raise CheckpointPrivacyError(
            "checkpoint root must not be inside a public output path: "
            + ", ".join(sorted(forbidden))
        )
    return resolved


def _require_private_mode(path: Path, *, directory: bool) -> None:
    try:
        mode = path.stat().st_mode
    except OSError as exc:
        raise CheckpointPrivacyError(f"cannot inspect checkpoint path: {path}") from exc
    if mode & 0o077:
        kind = "directory" if directory else "file"
        raise CheckpointPrivacyError(f"checkpoint {kind} is group/world accessible: {path}")


@dataclass(frozen=True)
class _StoredCheckpointInspection:
    run_signature: str
    model_names: tuple[str, ...]
    origins: tuple[CheckpointOrigin, ...]


def _require_real_private_directory(path: Path, *, label: str) -> None:
    if path.is_symlink():
        raise CheckpointPrivacyError(f"{label} must not be a symlink")
    if not path.is_dir():
        raise CheckpointPrivacyError(f"{label} must be a real directory")
    _require_private_mode(path, directory=True)


def _stored_origin(value: Any, *, expected_sequence: int) -> CheckpointOrigin:
    if not isinstance(value, Mapping) or set(value) != _ORIGIN_MANIFEST_KEYS:
        raise CheckpointCorruptionError("checkpoint origin manifest keys are invalid")
    sequence = value.get("sequence")
    if type(sequence) is not int or sequence != expected_sequence:
        raise CheckpointCorruptionError("checkpoint origin sequence is invalid")
    evaluation_split = value.get("evaluation_split")
    if evaluation_split not in {"legacy", "selection", "holdout"}:
        raise CheckpointCorruptionError("checkpoint origin split is invalid")
    train_size = value.get("train_size")
    gap = value.get("gap")
    purged_origin_count = value.get("purged_origin_count")
    if type(train_size) is not int or train_size < 1:
        raise CheckpointCorruptionError("checkpoint origin train_size is invalid")
    if type(gap) is not int or gap < 0:
        raise CheckpointCorruptionError("checkpoint origin gap is invalid")
    if (
        type(purged_origin_count) is not int
        or purged_origin_count < 0
        or purged_origin_count != gap
    ):
        raise CheckpointCorruptionError("checkpoint origin purge count is invalid")
    signature = value.get("signature")
    if not isinstance(signature, str) or not _SHA256_PATTERN.fullmatch(signature):
        raise CheckpointCorruptionError("checkpoint origin signature is invalid")
    record_file = value.get("record_file")
    if record_file != f"{sequence:06d}.json":
        raise CheckpointCorruptionError("checkpoint origin record file is invalid")

    origin_date = _timestamp_from_document(value.get("origin_date"))
    target_date = _timestamp_from_document(value.get("target_date"))
    train_start = _timestamp_from_document(value.get("train_start"))
    last_train_origin = _timestamp_from_document(value.get("last_train_origin"))
    last_train_target = _timestamp_from_document(value.get("last_train_target"))
    raw_first_purged = value.get("first_purged_origin")
    first_purged_origin = (
        None
        if raw_first_purged is None
        else _timestamp_from_document(raw_first_purged)
    )
    if (gap == 0) != (first_purged_origin is None):
        raise CheckpointCorruptionError("checkpoint origin first purge is invalid")
    if not (
        train_start <= last_train_origin <= last_train_target <= origin_date < target_date
    ):
        raise CheckpointCorruptionError("checkpoint origin dates are inconsistent")

    origin = CheckpointOrigin(
        sequence=sequence,
        origin_date=origin_date,
        target_date=target_date,
        evaluation_split=evaluation_split,
        train_size=train_size,
        train_start=train_start,
        last_train_origin=last_train_origin,
        last_train_target=last_train_target,
        purged_origin_count=purged_origin_count,
        first_purged_origin=first_purged_origin,
        gap=gap,
        # The public manifest deliberately omits labels.  A present record binds
        # them back to ``signature`` before it can be accepted below.
        current_state="transition",
        actual="transition",
        signature=signature,
        record_file=record_file,
    )
    if origin.public_manifest() != dict(value):
        raise CheckpointCorruptionError("checkpoint origin manifest is non-canonical")
    return origin


def _validate_stored_record(
    path: Path,
    *,
    run_signature: str,
    model_names: tuple[str, ...],
    stored_origin: CheckpointOrigin,
) -> None:
    _require_private_mode(path, directory=False)
    document = _read_json_strict(path)
    if not isinstance(document, Mapping) or set(document) != _RECORD_KEYS:
        raise CheckpointCorruptionError("checkpoint record keys are invalid")
    if document.get("schema_version") != RECORD_SCHEMA_VERSION:
        raise CheckpointCorruptionError("checkpoint record schema mismatch")
    body = {key: document[key] for key in _RECORD_KEYS if key != "record_sha256"}
    stored_sha256 = document.get("record_sha256")
    if (
        not isinstance(stored_sha256, str)
        or not _SHA256_PATTERN.fullmatch(stored_sha256)
        or _sha256_document(body) != stored_sha256
    ):
        raise CheckpointCorruptionError("checkpoint record digest mismatch")
    if document.get("run_signature") != run_signature:
        raise CheckpointCorruptionError("checkpoint record has the wrong run signature")
    if document.get("origin_signature") != stored_origin.signature:
        raise CheckpointCorruptionError("checkpoint record has the wrong origin signature")
    if document.get("sequence") != stored_origin.sequence:
        raise CheckpointCorruptionError("checkpoint record has the wrong origin sequence")

    raw_predictions = document.get("prediction_rows")
    if not isinstance(raw_predictions, list) or not raw_predictions:
        raise CheckpointCorruptionError("checkpoint prediction_rows must be a non-empty list")
    predictions = tuple(
        _decode_row(row, PREDICTION_COLUMNS) for row in raw_predictions
    )
    split = _decode_row(document.get("split_audit"), SPLIT_AUDIT_COLUMNS)
    current_state = predictions[0]["current_state"]
    actual = predictions[0]["actual"]
    bound_origin = CheckpointOrigin(
        sequence=stored_origin.sequence,
        origin_date=stored_origin.origin_date,
        target_date=stored_origin.target_date,
        evaluation_split=stored_origin.evaluation_split,
        train_size=stored_origin.train_size,
        train_start=stored_origin.train_start,
        last_train_origin=stored_origin.last_train_origin,
        last_train_target=stored_origin.last_train_target,
        purged_origin_count=stored_origin.purged_origin_count,
        first_purged_origin=stored_origin.first_purged_origin,
        gap=stored_origin.gap,
        current_state=current_state,
        actual=actual,
        signature=stored_origin.signature,
        record_file=stored_origin.record_file,
    )
    origin_core = {
        "sequence": bound_origin.sequence,
        "origin_date": _timestamp_document(bound_origin.origin_date),
        "target_date": _timestamp_document(bound_origin.target_date),
        "evaluation_split": bound_origin.evaluation_split,
        "train_size": bound_origin.train_size,
        "train_start": _timestamp_document(bound_origin.train_start),
        "last_train_origin": _timestamp_document(bound_origin.last_train_origin),
        "last_train_target": _timestamp_document(bound_origin.last_train_target),
        "purged_origin_count": bound_origin.purged_origin_count,
        "first_purged_origin": (
            None
            if bound_origin.first_purged_origin is None
            else _timestamp_document(bound_origin.first_purged_origin)
        ),
        "gap": bound_origin.gap,
        "current_state": bound_origin.current_state,
        "actual": bound_origin.actual,
    }
    if _sha256_document(origin_core) != bound_origin.signature:
        raise CheckpointCorruptionError("checkpoint origin signature mismatch")
    stored_identity = BenchmarkCheckpointIdentity(
        feature_identity={},
        state_identity={},
        parameter_manifest={},
        source_fingerprint_sha256=None,
        origins=(bound_origin,),
        model_names=model_names,
        run_signature=run_signature,
    )
    _validate_completed_origin(stored_identity, bound_origin, predictions, split)


def _inspect_stored_checkpoint(root: Path) -> _StoredCheckpointInspection:
    """Validate an existing checkpoint without assuming the requested identity."""

    _require_real_private_directory(root, label="checkpoint root")
    allowed_root_entries = {"manifest.json", "origins", "runs"}
    for entry in root.iterdir():
        if entry.is_symlink():
            raise CheckpointPrivacyError("checkpoint entries must not be symlinks")
        if entry.name not in allowed_root_entries:
            raise CheckpointCorruptionError(
                f"unexpected checkpoint root entry: {entry.name}"
            )

    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink():
        raise CheckpointPrivacyError("checkpoint manifest must not be a symlink")
    if not manifest_path.is_file():
        raise CheckpointCorruptionError("checkpoint manifest is missing")
    _require_private_mode(manifest_path, directory=False)
    manifest = _read_json_strict(manifest_path)
    if not isinstance(manifest, Mapping) or set(manifest) != _MANIFEST_KEYS:
        raise CheckpointCorruptionError("checkpoint manifest keys are invalid")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise CheckpointCorruptionError("checkpoint manifest schema mismatch")
    identity = manifest.get("identity")
    if manifest.get("visibility") != "private" or not isinstance(identity, Mapping):
        raise CheckpointCorruptionError("checkpoint manifest is not private and valid")
    if set(identity) != _IDENTITY_KEYS:
        raise CheckpointCorruptionError("checkpoint identity keys are invalid")
    if (
        identity.get("checkpoint_kind") != CHECKPOINT_KIND
        or identity.get("implementation_version")
        != CHECKPOINT_IMPLEMENTATION_VERSION
    ):
        raise CheckpointCorruptionError("checkpoint identity kind is invalid")
    run_signature = manifest.get("run_signature")
    if (
        not isinstance(run_signature, str)
        or not _SHA256_PATTERN.fullmatch(run_signature)
        or _sha256_document(identity) != run_signature
    ):
        raise CheckpointCorruptionError("checkpoint manifest signature mismatch")
    fingerprint = identity.get("source_fingerprint_sha256")
    if fingerprint is not None and (
        not isinstance(fingerprint, str)
        or not _SHA256_PATTERN.fullmatch(fingerprint)
    ):
        raise CheckpointCorruptionError("checkpoint source fingerprint is invalid")
    parameters = identity.get("benchmark_parameters")
    if not isinstance(parameters, Mapping):
        raise CheckpointCorruptionError("checkpoint benchmark parameters are invalid")
    raw_model_names = parameters.get("model_names")
    if (
        not isinstance(raw_model_names, list)
        or not raw_model_names
        or any(not isinstance(name, str) or not name for name in raw_model_names)
        or len(set(raw_model_names)) != len(raw_model_names)
    ):
        raise CheckpointCorruptionError("checkpoint model names are invalid")
    model_names = tuple(raw_model_names)
    raw_origins = identity.get("origins")
    if not isinstance(raw_origins, list) or not raw_origins:
        raise CheckpointCorruptionError("checkpoint origins are invalid")
    origins = tuple(
        _stored_origin(value, expected_sequence=sequence)
        for sequence, value in enumerate(raw_origins, start=1)
    )

    records_root = root / "origins"
    _require_real_private_directory(records_root, label="checkpoint records directory")
    expected_records = {origin.record_file: origin for origin in origins}
    for path in records_root.iterdir():
        if path.is_symlink():
            raise CheckpointPrivacyError("checkpoint records must not be symlinks")
        if path.name.startswith(".") and path.name.endswith(".tmp"):
            if not path.is_file():
                raise CheckpointCorruptionError(
                    f"unexpected checkpoint temporary record: {path.name}"
                )
            _require_private_mode(path, directory=False)
            continue
        stored_origin = expected_records.get(path.name)
        if stored_origin is None or not path.is_file():
            raise CheckpointCorruptionError(
                f"unexpected checkpoint record: {path.name}"
            )
        _validate_stored_record(
            path,
            run_signature=run_signature,
            model_names=model_names,
            stored_origin=stored_origin,
        )

    runs_root = root / "runs"
    if runs_root.exists() or runs_root.is_symlink():
        _require_real_private_directory(
            runs_root,
            label="versioned checkpoint runs directory",
        )
        for child in runs_root.iterdir():
            if child.is_symlink():
                raise CheckpointPrivacyError(
                    "versioned checkpoint run must not be a symlink"
                )
            if (
                not _SHA256_PATTERN.fullmatch(child.name)
                or not child.is_dir()
            ):
                raise CheckpointCorruptionError(
                    f"unexpected versioned checkpoint run: {child.name}"
                )
            _require_private_mode(child, directory=True)

    return _StoredCheckpointInspection(
        run_signature=run_signature,
        model_names=model_names,
        origins=origins,
    )


class WalkForwardCheckpoint:
    """Atomic reader/writer for fully validated V5 base origins."""

    def __init__(self, root: Path, identity: BenchmarkCheckpointIdentity) -> None:
        self.root = root
        self.records_root = root / "origins"
        self.identity = identity
        self._origins_by_sequence = {
            origin.sequence: origin for origin in identity.origins
        }

    @classmethod
    def open(
        cls,
        root: str | Path,
        identity: BenchmarkCheckpointIdentity,
    ) -> "WalkForwardCheckpoint":
        resolved = _validate_private_root(Path(root))
        resolved.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(resolved, 0o700)
        records_root = resolved / "origins"
        records_root.mkdir(mode=0o700, exist_ok=True)
        os.chmod(records_root, 0o700)
        _require_private_mode(resolved, directory=True)
        _require_private_mode(records_root, directory=True)
        checkpoint = cls(resolved, identity)
        checkpoint._open_manifest()
        checkpoint._reject_unexpected_records()
        return checkpoint

    @classmethod
    def open_versioned(
        cls,
        root: str | Path,
        identity: BenchmarkCheckpointIdentity,
    ) -> "WalkForwardCheckpoint":
        """Resume a matching legacy run or isolate a valid new run by signature.

        The first identity keeps the historical ``root`` layout.  A later valid
        identity uses ``root/runs/<run_signature>`` without moving or deleting
        the legacy run.  Only an identity mismatch permits rollover: malformed,
        non-private, or symlinked existing state always fails closed.
        """

        requested = Path(root).expanduser()
        if requested.is_symlink():
            raise CheckpointPrivacyError("checkpoint root must not be a symlink")
        resolved = _validate_private_root(requested)
        if not resolved.exists():
            return cls.open(resolved, identity)
        if not resolved.is_dir():
            raise CheckpointPrivacyError(
                "checkpoint root must be a real directory"
            )

        # ``open`` creates the root and ``origins`` before atomically writing
        # the manifest.  A hard interruption in that narrow window leaves no
        # trusted state to preserve, so an otherwise empty layout is safely
        # re-initializable.  Any record, extra entry, symlink, or manifest-like
        # file still fails closed through the normal inspector below.
        entries = list(resolved.iterdir())
        if not entries:
            os.chmod(resolved, 0o700)
            return cls.open(resolved, identity)
        if len(entries) == 1 and entries[0].name == "origins":
            origins = entries[0]
            if origins.is_symlink():
                raise CheckpointPrivacyError(
                    "checkpoint origins directory must not be a symlink"
                )
            if origins.is_dir() and not any(origins.iterdir()):
                os.chmod(resolved, 0o700)
                os.chmod(origins, 0o700)
                return cls.open(resolved, identity)

        stored = _inspect_stored_checkpoint(resolved)
        if stored.run_signature == identity.run_signature:
            return cls.open(resolved, identity)

        runs_root = resolved / "runs"
        if runs_root.is_symlink():
            raise CheckpointPrivacyError(
                "versioned checkpoint runs directory must not be a symlink"
            )
        if not runs_root.exists():
            runs_root.mkdir(mode=0o700)
        _require_real_private_directory(
            runs_root,
            label="versioned checkpoint runs directory",
        )

        child = runs_root / identity.run_signature
        if child.is_symlink():
            raise CheckpointPrivacyError("versioned checkpoint run must not be a symlink")
        if child.exists() and child.is_dir() and not any(child.iterdir()):
            os.chmod(child, 0o700)
            return cls.open(child, identity)
        if child.exists():
            child_stored = _inspect_stored_checkpoint(child)
            if child_stored.run_signature != identity.run_signature:
                raise CheckpointCorruptionError(
                    "versioned checkpoint manifest does not match its namespace"
                )
        return cls.open(child, identity)

    def _open_manifest(self) -> None:
        path = self.root / "manifest.json"
        expected = self.identity.manifest_document()
        if not path.exists():
            write_json_atomic(path, expected)
            os.chmod(path, 0o600)
            return
        if path.is_symlink():
            raise CheckpointPrivacyError("checkpoint manifest must not be a symlink")
        _require_private_mode(path, directory=False)
        actual = _read_json_strict(path)
        if not isinstance(actual, Mapping):
            raise CheckpointCorruptionError("checkpoint manifest must be an object")
        if actual.get("schema_version") != MANIFEST_SCHEMA_VERSION:
            raise CheckpointCorruptionError("checkpoint manifest schema mismatch")
        if set(actual) != {"schema_version", "visibility", "identity", "run_signature"}:
            raise CheckpointCorruptionError("checkpoint manifest keys are invalid")
        if actual.get("visibility") != "private" or not isinstance(actual.get("identity"), Mapping):
            raise CheckpointCorruptionError("checkpoint manifest is not private and valid")
        actual_signature = actual.get("run_signature")
        if not isinstance(actual_signature, str) or not _SHA256_PATTERN.fullmatch(actual_signature):
            raise CheckpointCorruptionError("checkpoint run signature is invalid")
        if _sha256_document(actual["identity"]) != actual_signature:
            raise CheckpointCorruptionError("checkpoint manifest signature mismatch")
        if actual_signature != self.identity.run_signature:
            raise CheckpointIdentityMismatch(
                "checkpoint identity differs from the requested V5 benchmark"
            )
        if actual != expected:
            raise CheckpointCorruptionError("checkpoint manifest is non-canonical")

    def _reject_unexpected_records(self) -> None:
        expected = {origin.record_file for origin in self.identity.origins}
        for path in self.records_root.iterdir():
            if path.name.startswith(".") and path.name.endswith(".tmp"):
                continue
            if path.is_symlink():
                raise CheckpointPrivacyError("checkpoint records must not be symlinks")
            if not path.is_file() or path.name not in expected:
                raise CheckpointCorruptionError(
                    f"unexpected checkpoint record: {path.name}"
                )

    def _origin(self, sequence: int) -> CheckpointOrigin:
        try:
            return self._origins_by_sequence[int(sequence)]
        except (KeyError, TypeError, ValueError) as exc:
            raise KeyError(f"unknown checkpoint origin sequence: {sequence}") from exc

    def save_origin(
        self,
        sequence: int,
        prediction_rows: Sequence[Mapping[str, Any]],
        split_audit: Mapping[str, Any],
    ) -> Path:
        origin = self._origin(sequence)
        try:
            normalized_predictions = tuple(
                _coerce_row(row, PREDICTION_COLUMNS) for row in prediction_rows
            )
            normalized_split = _coerce_row(split_audit, SPLIT_AUDIT_COLUMNS)
            _validate_completed_origin(
                self.identity,
                origin,
                normalized_predictions,
                normalized_split,
            )
        except CheckpointCorruptionError:
            raise
        except (TypeError, ValueError) as exc:
            raise CheckpointCorruptionError(
                f"origin {sequence} cannot be checkpointed"
            ) from exc
        body = {
            "schema_version": RECORD_SCHEMA_VERSION,
            "run_signature": self.identity.run_signature,
            "origin_signature": origin.signature,
            "sequence": origin.sequence,
            "prediction_rows": [
                _encode_row(row, PREDICTION_COLUMNS)
                for row in normalized_predictions
            ],
            "split_audit": _encode_row(normalized_split, SPLIT_AUDIT_COLUMNS),
        }
        document = {**body, "record_sha256": _sha256_document(body)}
        path = self.records_root / origin.record_file
        if path.exists():
            existing = self._load_origin(origin)
            if existing is None:
                raise CheckpointCorruptionError("existing record disappeared during load")
            actual = _read_json_strict(path)
            if actual != document:
                raise CheckpointCorruptionError(
                    "refusing to overwrite a different completed origin"
                )
            return path
        write_json_atomic(path, document)
        os.chmod(path, 0o600)
        # Read back the exact bytes and schema before reporting completion.
        self._load_origin(origin)
        return path

    def _load_origin(self, origin: CheckpointOrigin) -> CompletedCheckpointOrigin | None:
        path = self.records_root / origin.record_file
        if not path.exists():
            return None
        if path.is_symlink():
            raise CheckpointPrivacyError("checkpoint records must not be symlinks")
        _require_private_mode(path, directory=False)
        document = _read_json_strict(path)
        if not isinstance(document, Mapping):
            raise CheckpointCorruptionError("checkpoint record must be an object")
        expected_keys = {
            "schema_version",
            "run_signature",
            "origin_signature",
            "sequence",
            "prediction_rows",
            "split_audit",
            "record_sha256",
        }
        if set(document) != expected_keys:
            raise CheckpointCorruptionError("checkpoint record keys are invalid")
        if document.get("schema_version") != RECORD_SCHEMA_VERSION:
            raise CheckpointCorruptionError("checkpoint record schema mismatch")
        body = {key: document[key] for key in expected_keys if key != "record_sha256"}
        stored_sha256 = document.get("record_sha256")
        if not isinstance(stored_sha256, str) or _sha256_document(body) != stored_sha256:
            raise CheckpointCorruptionError("checkpoint record digest mismatch")
        if document.get("run_signature") != self.identity.run_signature:
            raise CheckpointCorruptionError("checkpoint record has the wrong run signature")
        if document.get("origin_signature") != origin.signature:
            raise CheckpointCorruptionError("checkpoint record has the wrong origin signature")
        if type(document.get("sequence")) is not int or document["sequence"] != origin.sequence:
            raise CheckpointCorruptionError("checkpoint record has the wrong origin sequence")
        raw_predictions = document.get("prediction_rows")
        if not isinstance(raw_predictions, list):
            raise CheckpointCorruptionError("checkpoint prediction_rows must be a list")
        predictions = tuple(
            _decode_row(row, PREDICTION_COLUMNS) for row in raw_predictions
        )
        split = _decode_row(document.get("split_audit"), SPLIT_AUDIT_COLUMNS)
        _validate_completed_origin(self.identity, origin, predictions, split)
        return CompletedCheckpointOrigin(
            origin=origin,
            prediction_rows=predictions,
            split_audit=split,
        )

    def load_origin(self, sequence: int) -> CompletedCheckpointOrigin | None:
        """Return one fully valid origin or ``None`` when it is incomplete."""

        self._reject_unexpected_records()
        return self._load_origin(self._origin(sequence))

    def load_completed_origins(self) -> tuple[CompletedCheckpointOrigin, ...]:
        """Return only complete records; any invalid present record aborts resume."""

        self._reject_unexpected_records()
        completed: list[CompletedCheckpointOrigin] = []
        for origin in self.identity.origins:
            record = self._load_origin(origin)
            if record is not None:
                completed.append(record)
        return tuple(completed)


__all__ = [
    "CHECKPOINT_KIND",
    "MANIFEST_SCHEMA_VERSION",
    "PREDICTION_COLUMNS",
    "RECORD_SCHEMA_VERSION",
    "SPLIT_AUDIT_COLUMNS",
    "BenchmarkCheckpointIdentity",
    "CheckpointCorruptionError",
    "CheckpointError",
    "CheckpointIdentityMismatch",
    "CheckpointOrigin",
    "CheckpointPrivacyError",
    "CompletedCheckpointOrigin",
    "ResolvedBenchmarkParameters",
    "WalkForwardCheckpoint",
    "decode_checkpoint_scalar",
    "encode_checkpoint_scalar",
    "runtime_version_manifest",
]
