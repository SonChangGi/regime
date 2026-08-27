from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from regime_lab.analysis import run_benchmark
import regime_lab.analysis.validation as validation_module
from regime_lab.analysis.models import BenchmarkProfile
from regime_lab.walkforward_checkpoint import PREDICTION_COLUMNS
from regime_lab.walkforward_checkpoint import RECORD_SCHEMA_VERSION
from regime_lab.walkforward_checkpoint import SPLIT_AUDIT_COLUMNS
from regime_lab.walkforward_checkpoint import BenchmarkCheckpointIdentity
from regime_lab.walkforward_checkpoint import CheckpointCorruptionError
from regime_lab.walkforward_checkpoint import CheckpointIdentityMismatch
from regime_lab.walkforward_checkpoint import CheckpointPrivacyError
from regime_lab.walkforward_checkpoint import ResolvedBenchmarkParameters
from regime_lab.walkforward_checkpoint import WalkForwardCheckpoint
from regime_lab.walkforward_checkpoint import _sha256_document
from regime_lab.walkforward_checkpoint import decode_checkpoint_scalar
from regime_lab.walkforward_checkpoint import encode_checkpoint_scalar


def _inputs(rows: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    index = pd.date_range("2020-01-03", periods=rows, freq="W-FRI")
    position = np.arange(rows, dtype=float)
    features = pd.DataFrame(
        {
            "trend": np.sin(position / 4.0),
            "stress": np.cos(position / 5.0),
        },
        index=index,
    )
    features.loc[index[7], "stress"] = np.nan
    states = pd.Series(
        np.resize(np.asarray(("risk_on", "transition", "risk_off")), rows),
        index=index,
        name="regime",
    )
    return features, states


def _profile() -> BenchmarkProfile:
    return BenchmarkProfile(
        name="standard",
        max_origins=5,
        minimum_train_weeks=12,
        random_forest_trees=5,
        extra_trees=5,
        hist_gradient_iterations=5,
        svm_calibration_splits=2,
        hmm_iterations=5,
        xgboost_trees=5,
        spline_pca_components=2,
    )


def _parameters(**overrides: object) -> ResolvedBenchmarkParameters:
    values: dict[str, object] = {
        "profile": _profile(),
        "models": ("majority", "persistence"),
        "gap": 1,
        "random_state": 17,
        "selection_end": "2020-07-03",
        "selection_max_origins": 4,
        "model_workers": 2,
        "minimum_selection_predictions": 3,
        "minimum_holdout_predictions": 3,
    }
    values.update(overrides)
    return ResolvedBenchmarkParameters.from_arguments(**values)  # type: ignore[arg-type]


def _identity(
    *,
    source: str | None = "a" * 64,
) -> BenchmarkCheckpointIdentity:
    features, states = _inputs()
    return BenchmarkCheckpointIdentity.build(
        features,
        states,
        _parameters(),
        source_fingerprint_sha256=source,
    )


def test_checkpoint_identity_accepts_opt_in_direct_candidates() -> None:
    parameters = _parameters(
        models=(
            "markov",
            "discounted_markov_208w",
            "pca_ridge_logistic",
            "recency_weighted_xgboost_208w",
            "recency_weighted_ridge_logistic_208w",
        )
    )

    assert parameters.model_names == (
        "markov",
        "discounted_markov_208w",
        "pca_ridge_logistic",
        "recency_weighted_xgboost_208w",
        "recency_weighted_ridge_logistic_208w",
    )


def _rows(identity: BenchmarkCheckpointIdentity, sequence: int = 1):
    origin = identity.origins[sequence - 1]
    rows: list[dict[str, object]] = []
    actual_position = tuple(("risk_on", "transition", "risk_off")).index(
        origin.actual
    )
    for model in identity.model_names:
        probability = [0.1, 0.1, 0.1]
        probability[actual_position] = 0.8
        rows.append(
            {
                "origin_date": origin.origin_date,
                "target_date": origin.target_date,
                "model": model,
                "evaluation_split": origin.evaluation_split,
                "current_state": origin.current_state,
                "actual": origin.actual,
                "predicted": origin.actual,
                "p_risk_on": float(probability[0]),
                "p_transition": float(probability[1]),
                "p_risk_off": float(probability[2]),
                "train_size": origin.train_size,
                "gap": origin.gap,
                "fallback": False,
                "fallback_reason": "",
            }
        )
    split = {
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
    assert tuple(rows[0]) == PREDICTION_COLUMNS
    assert tuple(split) == SPLIT_AUDIT_COLUMNS
    return rows, split


def _rewrite_record(path: Path, mutate) -> None:
    document = json.loads(path.read_text(encoding="utf-8"))
    mutate(document)
    body = {key: value for key, value in document.items() if key != "record_sha256"}
    document["record_sha256"] = _sha256_document(body)
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.chmod(path, 0o600)


def test_scalar_codec_exactly_round_trips_supported_types() -> None:
    timestamp = pd.Timestamp("2024-11-03 03:30:00.123456789", tz="America/New_York")
    values = (
        timestamp,
        True,
        False,
        2**80,
        -19,
        "001 and 1 are strings",
        -0.0,
        float.fromhex("0x1.123456789abcdp-7"),
        None,
        np.int64(7),
        np.bool_(True),
        np.float64(0.25),
    )
    decoded = [decode_checkpoint_scalar(encode_checkpoint_scalar(value)) for value in values]

    assert decoded[0].value == timestamp.value
    assert str(decoded[0].tz) == str(timestamp.tz)
    assert type(decoded[1]) is bool and type(decoded[3]) is int
    assert decoded[3] == 2**80
    assert math.copysign(1.0, decoded[6]) == -1.0
    assert decoded[7].hex() == values[7].hex()
    assert decoded[8] is None
    assert type(decoded[9]) is int
    assert type(decoded[10]) is bool
    assert type(decoded[11]) is float

    with pytest.raises(ValueError, match="finite"):
        encode_checkpoint_scalar(float("nan"))
    with pytest.raises(ValueError, match="finite"):
        encode_checkpoint_scalar(float("inf"))
    with pytest.raises(TypeError, match="checkpoint scalars"):
        encode_checkpoint_scalar([1, 2])


def test_identity_is_deterministic_and_binds_every_material_input() -> None:
    features, states = _inputs()
    parameters = _parameters()
    first = BenchmarkCheckpointIdentity.build(
        features,
        states,
        parameters,
        source_fingerprint_sha256="A" * 64,
    )
    second = BenchmarkCheckpointIdentity.build(
        features.copy(),
        states.copy(),
        parameters,
        source_fingerprint_sha256="a" * 64,
    )
    assert first.run_signature == second.run_signature
    assert first.manifest_document() == second.manifest_document()

    changed_feature = features.copy()
    changed_feature.iloc[2, 0] = np.nextafter(changed_feature.iloc[2, 0], np.inf)
    changed_state = states.copy()
    changed_state.iloc[2] = "risk_on"
    changed_parameter = _parameters(model_workers=1)
    changed_selection_policy = _parameters(
        minimum_log_loss_improvement=0.01
    )
    changed_source = "b" * 64

    signatures = {
        BenchmarkCheckpointIdentity.build(
            changed_feature, states, parameters, source_fingerprint_sha256="a" * 64
        ).run_signature,
        BenchmarkCheckpointIdentity.build(
            features, changed_state, parameters, source_fingerprint_sha256="a" * 64
        ).run_signature,
        BenchmarkCheckpointIdentity.build(
            features, states, changed_parameter, source_fingerprint_sha256="a" * 64
        ).run_signature,
        BenchmarkCheckpointIdentity.build(
            features,
            states,
            changed_selection_policy,
            source_fingerprint_sha256="a" * 64,
        ).run_signature,
        BenchmarkCheckpointIdentity.build(
            features, states, parameters, source_fingerprint_sha256=changed_source
        ).run_signature,
    }
    assert len(signatures) == 5
    assert first.run_signature not in signatures
    assert first.parameter_manifest["minimum_log_loss_improvement"] == 0.05


def test_origin_resolver_matches_run_benchmark_split_contract() -> None:
    features, states = _inputs()
    parameters = _parameters(model_workers=1)
    identity = BenchmarkCheckpointIdentity.build(features, states, parameters)
    result = run_benchmark(
        features,
        states,
        profile=parameters.profile,
        models=parameters.model_names,
        include_hmm=parameters.include_hmm,
        gap=parameters.gap,
        random_state=parameters.random_state,
        selection_end=parameters.selection_end,
        selection_max_origins=parameters.selection_max_origins,
        model_workers=parameters.model_workers,
        minimum_selection_predictions=parameters.minimum_selection_predictions,
        minimum_holdout_predictions=parameters.minimum_holdout_predictions,
        minimum_log_loss_improvement=parameters.minimum_log_loss_improvement,
    )

    expected = [
        (origin.origin_date, origin.target_date, origin.evaluation_split)
        for origin in identity.origins
    ]
    actual = [
        tuple(row)
        for row in result.split_audit[
            ["origin_date", "target_date", "evaluation_split"]
        ].itertuples(index=False, name=None)
    ]
    assert actual == expected


def test_store_resumes_only_fully_valid_atomic_origins(tmp_path: Path) -> None:
    identity = _identity()
    checkpoint = WalkForwardCheckpoint.open(tmp_path / "v5-checkpoint", identity)
    first_rows, first_split = _rows(identity, 1)
    second_rows, second_split = _rows(identity, 2)

    first_path = checkpoint.save_origin(1, first_rows, first_split)
    assert first_path.name == "000001.json"
    assert checkpoint.save_origin(1, first_rows, first_split) == first_path
    checkpoint.save_origin(2, second_rows, second_split)

    completed = checkpoint.load_completed_origins()
    assert [record.origin.sequence for record in completed] == [1, 2]
    assert len(completed) < len(identity.origins)
    assert completed[0].prediction_rows[0]["fallback"] is False
    assert type(completed[0].prediction_rows[0]["train_size"]) is int
    assert isinstance(completed[0].prediction_rows[0]["origin_date"], pd.Timestamp)
    assert checkpoint.load_origin(3) is None
    assert (checkpoint.root.stat().st_mode & 0o077) == 0
    assert (first_path.stat().st_mode & 0o077) == 0

    manifest_text = (checkpoint.root / "manifest.json").read_text(encoding="utf-8")
    assert "trend" not in manifest_text
    assert "stress" not in manifest_text
    assert "provider" not in manifest_text


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "target", "split"])
def test_save_rejects_incomplete_or_wrong_origin_rows(
    tmp_path: Path,
    mutation: str,
) -> None:
    identity = _identity()
    checkpoint = WalkForwardCheckpoint.open(tmp_path / "v5-checkpoint", identity)
    rows, split = _rows(identity)
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[1] = dict(rows[0])
    elif mutation == "target":
        rows[0]["target_date"] = pd.Timestamp(rows[0]["target_date"]) + pd.Timedelta(7, "D")
    elif mutation == "split":
        rows[0]["evaluation_split"] = "holdout"

    with pytest.raises(CheckpointCorruptionError):
        checkpoint.save_origin(1, rows, split)
    assert checkpoint.load_origin(1) is None


def test_save_rejects_nonfinite_probability_and_wrong_split_audit(tmp_path: Path) -> None:
    identity = _identity()
    checkpoint = WalkForwardCheckpoint.open(tmp_path / "v5-checkpoint", identity)
    rows, split = _rows(identity)
    rows[0]["p_risk_on"] = float("nan")
    with pytest.raises(CheckpointCorruptionError, match="cannot be checkpointed"):
        checkpoint.save_origin(1, rows, split)

    rows, split = _rows(identity)
    split["last_train_target"] = identity.origins[0].target_date
    with pytest.raises(CheckpointCorruptionError, match="last_train_target"):
        checkpoint.save_origin(1, rows, split)


def test_record_corruption_fails_closed_instead_of_being_skipped(tmp_path: Path) -> None:
    identity = _identity()
    checkpoint = WalkForwardCheckpoint.open(tmp_path / "v5-checkpoint", identity)
    rows, split = _rows(identity)
    record_path = checkpoint.save_origin(1, rows, split)
    record_path.write_text("{not-json", encoding="utf-8")
    os.chmod(record_path, 0o600)

    with pytest.raises(CheckpointCorruptionError, match="cannot be read"):
        checkpoint.load_completed_origins()


def test_validly_rehashed_wrong_target_and_signature_still_fail_closed(
    tmp_path: Path,
) -> None:
    identity = _identity()
    checkpoint = WalkForwardCheckpoint.open(tmp_path / "v5-checkpoint", identity)
    rows, split = _rows(identity)
    record_path = checkpoint.save_origin(1, rows, split)

    def wrong_target(document):
        document["prediction_rows"][0]["target_date"] = encode_checkpoint_scalar(
            identity.origins[0].target_date + pd.Timedelta(7, "D")
        )

    _rewrite_record(record_path, wrong_target)
    with pytest.raises(CheckpointCorruptionError, match="wrong target"):
        checkpoint.load_origin(1)

    record_path.unlink()
    checkpoint.save_origin(1, rows, split)

    def wrong_signature(document):
        document["run_signature"] = "b" * 64

    _rewrite_record(record_path, wrong_signature)
    with pytest.raises(CheckpointCorruptionError, match="wrong run signature"):
        checkpoint.load_origin(1)


def test_manifest_schema_corruption_and_identity_drift_fail_closed(tmp_path: Path) -> None:
    features, states = _inputs()
    identity = BenchmarkCheckpointIdentity.build(features, states, _parameters())
    root = tmp_path / "v5-checkpoint"
    checkpoint = WalkForwardCheckpoint.open(root, identity)

    changed = features.copy()
    changed.iloc[0, 0] += 0.5
    different_identity = BenchmarkCheckpointIdentity.build(changed, states, _parameters())
    with pytest.raises(CheckpointIdentityMismatch):
        WalkForwardCheckpoint.open(root, different_identity)

    manifest_path = checkpoint.root / "manifest.json"
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    document["schema_version"] = "future-schema"
    manifest_path.write_text(json.dumps(document) + "\n", encoding="utf-8")
    os.chmod(manifest_path, 0o600)
    with pytest.raises(CheckpointCorruptionError, match="schema mismatch"):
        WalkForwardCheckpoint.open(root, identity)


def test_versioned_open_reuses_matching_origin_and_preserves_prior_bytes(
    tmp_path: Path,
) -> None:
    features, states = _inputs()
    identity = BenchmarkCheckpointIdentity.build(
        features,
        states,
        _parameters(),
        source_fingerprint_sha256="a" * 64,
    )
    root = tmp_path / "v5-checkpoint"
    legacy = WalkForwardCheckpoint.open_versioned(root, identity)
    rows, split = _rows(identity)
    legacy_record = legacy.save_origin(1, rows, split)
    legacy_manifest_bytes = (root / "manifest.json").read_bytes()
    legacy_record_bytes = legacy_record.read_bytes()

    appended_features, appended_states = _inputs(43)
    appended_identity = BenchmarkCheckpointIdentity.build(
        appended_features,
        appended_states,
        _parameters(),
        source_fingerprint_sha256="a" * 64,
    )
    rolled = WalkForwardCheckpoint.open_versioned(root, appended_identity)

    assert legacy.root == root.resolve()
    assert rolled.root == (
        root / "runs" / appended_identity.run_signature
    ).resolve()
    assert [
        record.origin.training_slice_sha256
        for record in rolled.load_completed_origins()
    ] == [identity.origins[0].training_slice_sha256]
    assert (root / "manifest.json").read_bytes() == legacy_manifest_bytes
    assert legacy_record.read_bytes() == legacy_record_bytes
    resumed_legacy = WalkForwardCheckpoint.open_versioned(root, identity)
    assert resumed_legacy.root == root.resolve()
    assert resumed_legacy.load_origin(1) is not None
    assert (rolled.root.stat().st_mode & 0o077) == 0


def test_versioned_open_refits_origins_affected_by_historical_revision(
    tmp_path: Path,
) -> None:
    features, states = _inputs()
    parameters = _parameters(models=("majority", "ridge_logistic"))
    root = tmp_path / "v5-checkpoint"
    first_identity = BenchmarkCheckpointIdentity.build(
        features,
        states,
        parameters,
        source_fingerprint_sha256="a" * 64,
    )
    first = WalkForwardCheckpoint.open_versioned(root, first_identity)
    for sequence in range(1, len(first_identity.origins) + 1):
        rows, split = _rows(first_identity, sequence)
        first.save_origin(sequence, rows, split)

    revised = features.copy()
    revised.iloc[0, 0] += 0.25
    revised_identity = BenchmarkCheckpointIdentity.build(
        revised,
        states,
        parameters,
        source_fingerprint_sha256="a" * 64,
    )
    rolled = WalkForwardCheckpoint.open_versioned(root, revised_identity)

    assert rolled.load_completed_origins() == ()
    assert {
        origin.training_slice_sha256 for origin in first_identity.origins
    }.isdisjoint(
        {origin.training_slice_sha256 for origin in revised_identity.origins}
    )


@pytest.mark.parametrize("with_origins", [False, True])
def test_versioned_open_recovers_empty_interrupted_initialization(
    tmp_path: Path,
    with_origins: bool,
) -> None:
    features, states = _inputs()
    identity = BenchmarkCheckpointIdentity.build(
        features,
        states,
        _parameters(),
        source_fingerprint_sha256="a" * 64,
    )
    root = tmp_path / "v5-checkpoint"
    root.mkdir()
    if with_origins:
        (root / "origins").mkdir()

    checkpoint = WalkForwardCheckpoint.open_versioned(root, identity)

    assert checkpoint.root == root.resolve()
    assert (root / "manifest.json").is_file()
    assert checkpoint.load_completed_origins() == ()
    assert (root.stat().st_mode & 0o077) == 0
    assert ((root / "origins").stat().st_mode & 0o077) == 0


def test_checkpoint_identity_changes_with_estimator_runtime_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features, states = _inputs()
    first = BenchmarkCheckpointIdentity.build(
        features,
        states,
        _parameters(),
        source_fingerprint_sha256="a" * 64,
    )
    monkeypatch.setattr(
        "regime_lab.walkforward_checkpoint.runtime_version_manifest",
        lambda: {
            "python": "future",
            "platform_system": "future",
            "platform_release": "future",
            "platform_machine": "future",
            "numpy": "future",
            "pandas": "future",
            "scipy": "future",
            "scikit-learn": "future",
            "xgboost": "future",
            "hmmlearn": "future",
            "joblib": "future",
            "threadpoolctl": "future",
        },
    )
    second = BenchmarkCheckpointIdentity.build(
        features,
        states,
        _parameters(),
        source_fingerprint_sha256="a" * 64,
    )

    assert first.parameter_manifest["runtime_versions"] != second.parameter_manifest[
        "runtime_versions"
    ]
    assert first.run_signature != second.run_signature


def test_versioned_open_resumes_the_same_rolled_run(tmp_path: Path) -> None:
    features, states = _inputs()
    root = tmp_path / "v5-checkpoint"
    legacy_identity = BenchmarkCheckpointIdentity.build(
        features,
        states,
        _parameters(),
        source_fingerprint_sha256="a" * 64,
    )
    WalkForwardCheckpoint.open_versioned(root, legacy_identity)

    appended_features, appended_states = _inputs(43)
    appended_identity = BenchmarkCheckpointIdentity.build(
        appended_features,
        appended_states,
        _parameters(),
        source_fingerprint_sha256="a" * 64,
    )
    rolled = WalkForwardCheckpoint.open_versioned(root, appended_identity)
    rows, split = _rows(appended_identity)
    record_path = rolled.save_origin(1, rows, split)
    record_bytes = record_path.read_bytes()

    resumed = WalkForwardCheckpoint.open_versioned(root, appended_identity)

    assert resumed.root == rolled.root
    assert record_path.read_bytes() == record_bytes
    assert [record.origin.sequence for record in resumed.load_completed_origins()] == [1]


def test_versioned_open_uses_source_fingerprint_bound_namespaces(
    tmp_path: Path,
) -> None:
    features, states = _inputs()
    root = tmp_path / "v5-checkpoint"
    first = BenchmarkCheckpointIdentity.build(
        features,
        states,
        _parameters(),
        source_fingerprint_sha256="a" * 64,
    )
    second = BenchmarkCheckpointIdentity.build(
        features,
        states,
        _parameters(),
        source_fingerprint_sha256="b" * 64,
    )
    third = BenchmarkCheckpointIdentity.build(
        features,
        states,
        _parameters(),
        source_fingerprint_sha256="c" * 64,
    )

    assert WalkForwardCheckpoint.open_versioned(root, first).root == root.resolve()
    second_checkpoint = WalkForwardCheckpoint.open_versioned(root, second)
    third_checkpoint = WalkForwardCheckpoint.open_versioned(root, third)

    assert second_checkpoint.root == (root / "runs" / second.run_signature).resolve()
    assert third_checkpoint.root == (root / "runs" / third.run_signature).resolve()
    assert second_checkpoint.root != third_checkpoint.root
    assert WalkForwardCheckpoint.open_versioned(root, second).root == second_checkpoint.root


@pytest.mark.parametrize("corruption", ["manifest", "record"])
def test_versioned_open_does_not_roll_over_corrupt_legacy_state(
    tmp_path: Path,
    corruption: str,
) -> None:
    features, states = _inputs()
    root = tmp_path / "v5-checkpoint"
    identity = BenchmarkCheckpointIdentity.build(features, states, _parameters())
    legacy = WalkForwardCheckpoint.open_versioned(root, identity)
    rows, split = _rows(identity)
    record_path = legacy.save_origin(1, rows, split)
    if corruption == "manifest":
        manifest_path = root / "manifest.json"
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
        document["schema_version"] = "future-schema"
        manifest_path.write_text(json.dumps(document) + "\n", encoding="utf-8")
        os.chmod(manifest_path, 0o600)
    else:
        record_path.write_text("{broken", encoding="utf-8")
        os.chmod(record_path, 0o600)

    appended_features, appended_states = _inputs(43)
    appended_identity = BenchmarkCheckpointIdentity.build(
        appended_features,
        appended_states,
        _parameters(),
    )
    with pytest.raises(CheckpointCorruptionError):
        WalkForwardCheckpoint.open_versioned(root, appended_identity)
    assert not (root / "runs").exists()


@pytest.mark.parametrize("corruption", ["manifest", "record"])
def test_versioned_open_does_not_resume_corrupt_child_state(
    tmp_path: Path,
    corruption: str,
) -> None:
    features, states = _inputs()
    root = tmp_path / "v5-checkpoint"
    legacy_identity = BenchmarkCheckpointIdentity.build(features, states, _parameters())
    WalkForwardCheckpoint.open_versioned(root, legacy_identity)
    appended_features, appended_states = _inputs(43)
    appended_identity = BenchmarkCheckpointIdentity.build(
        appended_features,
        appended_states,
        _parameters(),
    )
    child = WalkForwardCheckpoint.open_versioned(root, appended_identity)
    rows, split = _rows(appended_identity)
    record_path = child.save_origin(1, rows, split)
    if corruption == "manifest":
        manifest_path = child.root / "manifest.json"
        manifest_path.write_text("{broken", encoding="utf-8")
        os.chmod(manifest_path, 0o600)
    else:
        record_path.write_text("{broken", encoding="utf-8")
        os.chmod(record_path, 0o600)

    with pytest.raises(CheckpointCorruptionError):
        WalkForwardCheckpoint.open_versioned(root, appended_identity)


def test_versioned_open_rejects_symlinked_or_non_private_rollover_state(
    tmp_path: Path,
) -> None:
    features, states = _inputs()
    identity = BenchmarkCheckpointIdentity.build(features, states, _parameters())

    legacy_root = tmp_path / "legacy-symlink"
    WalkForwardCheckpoint.open_versioned(legacy_root, identity)
    records_root = legacy_root / "origins"
    records_backing = tmp_path / "origins-backing"
    records_root.rename(records_backing)
    records_root.symlink_to(records_backing, target_is_directory=True)
    with pytest.raises(CheckpointPrivacyError, match="symlink"):
        WalkForwardCheckpoint.open_versioned(legacy_root, identity)

    root = tmp_path / "child-symlink"
    WalkForwardCheckpoint.open_versioned(root, identity)
    changed_identity = BenchmarkCheckpointIdentity.build(
        features,
        states,
        _parameters(),
        source_fingerprint_sha256="b" * 64,
    )
    runs_root = root / "runs"
    runs_root.mkdir(mode=0o700)
    child_backing = tmp_path / "child-backing"
    child_backing.mkdir(mode=0o700)
    (runs_root / changed_identity.run_signature).symlink_to(
        child_backing,
        target_is_directory=True,
    )
    with pytest.raises(CheckpointPrivacyError, match="symlink"):
        WalkForwardCheckpoint.open_versioned(root, changed_identity)

    (runs_root / changed_identity.run_signature).unlink()
    child = WalkForwardCheckpoint.open_versioned(root, changed_identity)
    os.chmod(child.root, 0o755)
    with pytest.raises(CheckpointPrivacyError, match="group/world accessible"):
        WalkForwardCheckpoint.open_versioned(root, changed_identity)


def test_record_schema_and_unexpected_files_fail_closed(tmp_path: Path) -> None:
    identity = _identity()
    checkpoint = WalkForwardCheckpoint.open(tmp_path / "v5-checkpoint", identity)
    rows, split = _rows(identity)
    record_path = checkpoint.save_origin(1, rows, split)

    def future_schema(document):
        document["schema_version"] = RECORD_SCHEMA_VERSION + ".future"

    _rewrite_record(record_path, future_schema)
    with pytest.raises(CheckpointCorruptionError, match="schema mismatch"):
        checkpoint.load_origin(1)

    record_path.unlink()
    unexpected = checkpoint.records_root / "copy.json"
    unexpected.write_text("{}\n", encoding="utf-8")
    os.chmod(unexpected, 0o600)
    with pytest.raises(CheckpointCorruptionError, match="unexpected"):
        checkpoint.load_completed_origins()


def test_public_checkpoint_path_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(CheckpointPrivacyError, match="public output"):
        WalkForwardCheckpoint.open(tmp_path / "web" / "checkpoint", _identity())


def test_private_file_mode_is_enforced_on_resume(tmp_path: Path) -> None:
    identity = _identity()
    checkpoint = WalkForwardCheckpoint.open(tmp_path / "v5-checkpoint", identity)
    rows, split = _rows(identity)
    record_path = checkpoint.save_origin(1, rows, split)
    os.chmod(record_path, 0o644)

    with pytest.raises(CheckpointPrivacyError, match="group/world accessible"):
        checkpoint.load_origin(1)


def _assert_benchmark_equal(left, right) -> None:
    assert left.champion == right.champion
    assert left.profile == right.profile
    assert left.selection_end == right.selection_end
    for attribute in (
        "leaderboard",
        "predictions",
        "split_audit",
        "selection_leaderboard",
        "holdout_leaderboard",
        "selection_diagnostics",
    ):
        pd.testing.assert_frame_equal(
            getattr(left, attribute),
            getattr(right, attribute),
            check_exact=True,
        )


def test_appended_week_versioned_benchmark_is_exact_and_resumes_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = {
        "profile": _profile(),
        "models": ("majority", "ridge_logistic"),
        "gap": 1,
        "random_state": 17,
        "selection_end": "2020-07-03",
        "selection_max_origins": 4,
        "model_workers": 1,
        "minimum_selection_predictions": 3,
        "minimum_holdout_predictions": 3,
    }
    parameters = _parameters(
        models=arguments["models"],
        model_workers=arguments["model_workers"],
    )
    source_fingerprint = "a" * 64
    root = tmp_path / "private-v5-checkpoint"

    features, states = _inputs()
    first_identity = BenchmarkCheckpointIdentity.build(
        features,
        states,
        parameters,
        source_fingerprint_sha256=source_fingerprint,
    )
    first_checkpoint = WalkForwardCheckpoint.open_versioned(root, first_identity)
    first = run_benchmark(
        features,
        states,
        checkpoint_directory=first_checkpoint.root,
        source_fingerprint_sha256=source_fingerprint,
        **arguments,
    )
    legacy_manifest_bytes = (root / "manifest.json").read_bytes()
    legacy_record_bytes = {
        path.name: path.read_bytes()
        for path in (root / "origins").glob("*.json")
    }

    appended_features, appended_states = _inputs(43)
    appended_identity = BenchmarkCheckpointIdentity.build(
        appended_features,
        appended_states,
        parameters,
        source_fingerprint_sha256=source_fingerprint,
    )
    appended_checkpoint = WalkForwardCheckpoint.open_versioned(
        root,
        appended_identity,
    )
    reused_hashes = {
        record.origin.training_slice_sha256
        for record in appended_checkpoint.load_completed_origins()
    }
    expected_reused_hashes = {
        origin.training_slice_sha256 for origin in first_identity.origins
    }.intersection(
        {origin.training_slice_sha256 for origin in appended_identity.origins}
    )
    assert reused_hashes == expected_reused_hashes
    assert reused_hashes
    expected = run_benchmark(appended_features, appended_states, **arguments)
    actual = run_benchmark(
        appended_features,
        appended_states,
        checkpoint_directory=appended_checkpoint.root,
        source_fingerprint_sha256=source_fingerprint,
        **arguments,
    )

    _assert_benchmark_equal(actual, expected)
    assert not first.predictions.empty
    assert appended_checkpoint.root == (
        root / "runs" / appended_identity.run_signature
    ).resolve()
    assert (root / "manifest.json").read_bytes() == legacy_manifest_bytes
    assert {
        path.name: path.read_bytes()
        for path in (root / "origins").glob("*.json")
    } == legacy_record_bytes

    child_records = sorted(appended_checkpoint.records_root.glob("*.json"))
    child_records[-1].unlink()
    learned_calls: list[pd.Timestamp] = []
    original_predict = validation_module._predict_learned_model

    def count_predict(*args, **kwargs):
        learned_calls.append(pd.Timestamp(args[3].index[0]))
        return original_predict(*args, **kwargs)

    monkeypatch.setattr(validation_module, "_predict_learned_model", count_predict)
    resumed = run_benchmark(
        appended_features,
        appended_states,
        checkpoint_directory=appended_checkpoint.root,
        source_fingerprint_sha256=source_fingerprint,
        **arguments,
    )

    _assert_benchmark_equal(resumed, expected)
    assert learned_calls == [resumed.predictions["origin_date"].max()]


def test_run_benchmark_resume_is_exact_and_refits_only_missing_origins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features, states = _inputs()
    arguments = {
        "profile": _profile(),
        "models": ("majority", "ridge_logistic"),
        "gap": 1,
        "random_state": 17,
        "selection_end": "2020-07-03",
        "selection_max_origins": 4,
        "model_workers": 1,
        "minimum_selection_predictions": 3,
        "minimum_holdout_predictions": 3,
    }
    expected = run_benchmark(features, states, **arguments)
    checkpoint_root = tmp_path / "private-v5-checkpoint"
    first = run_benchmark(
        features,
        states,
        checkpoint_directory=checkpoint_root,
        source_fingerprint_sha256="a" * 64,
        **arguments,
    )
    _assert_benchmark_equal(first, expected)

    records = sorted((checkpoint_root / "origins").glob("*.json"))
    assert len(records) == first.predictions["origin_date"].nunique()
    records[-1].unlink()

    learned_calls: list[pd.Timestamp] = []
    original_predict = validation_module._predict_learned_model

    def count_predict(*args, **kwargs):
        learned_calls.append(pd.Timestamp(args[3].index[0]))
        return original_predict(*args, **kwargs)

    monkeypatch.setattr(validation_module, "_predict_learned_model", count_predict)
    resumed = run_benchmark(
        features,
        states,
        checkpoint_directory=checkpoint_root,
        source_fingerprint_sha256="a" * 64,
        **arguments,
    )

    _assert_benchmark_equal(resumed, expected)
    assert learned_calls == [resumed.predictions["origin_date"].max()]
    assert len(list((checkpoint_root / "origins").glob("*.json"))) == len(records)


def test_run_benchmark_checkpoint_corruption_aborts_before_any_refit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features, states = _inputs()
    arguments = {
        "profile": _profile(),
        "models": ("majority", "ridge_logistic"),
        "selection_end": "2020-07-03",
        "selection_max_origins": 4,
        "minimum_selection_predictions": 3,
        "minimum_holdout_predictions": 3,
        "checkpoint_directory": tmp_path / "private-v5-checkpoint",
        "source_fingerprint_sha256": "a" * 64,
    }
    run_benchmark(features, states, **arguments)
    first_record = sorted(
        (tmp_path / "private-v5-checkpoint" / "origins").glob("*.json")
    )[0]
    first_record.write_text("{broken", encoding="utf-8")
    os.chmod(first_record, 0o600)
    monkeypatch.setattr(
        validation_module,
        "_predict_learned_model",
        lambda *_args, **_kwargs: pytest.fail("corruption must abort before refit"),
    )

    with pytest.raises(CheckpointCorruptionError, match="cannot be read"):
        run_benchmark(features, states, **arguments)


def test_run_benchmark_default_path_is_a_v4_compatible_checkpoint_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features, states = _inputs()
    monkeypatch.setattr(
        WalkForwardCheckpoint,
        "open",
        lambda *_args, **_kwargs: pytest.fail("default benchmark must not open V5 state"),
    )

    result = run_benchmark(
        features,
        states,
        profile=_profile(),
        models=("majority",),
        selection_end="2020-07-03",
        selection_max_origins=4,
        minimum_selection_predictions=3,
        minimum_holdout_predictions=3,
    )
    assert not result.predictions.empty

    with pytest.raises(ValueError, match="requires checkpoint_directory"):
        run_benchmark(
            features,
            states,
            profile=_profile(),
            models=("majority",),
            source_fingerprint_sha256="a" * 64,
        )
