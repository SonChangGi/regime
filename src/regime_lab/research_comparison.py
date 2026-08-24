"""Private matched-OOS evaluation for opt-in next-state challengers."""

from __future__ import annotations

from datetime import date, datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Mapping, Sequence
import uuid

import numpy as np
import pandas as pd

from regime_lab.analysis import (
    CausalRegimeLabeler,
    RegimeLabelConfig,
    derive_causal_transition_features,
    run_benchmark,
)
from regime_lab.analysis.models import DIRECT_NEXT_STATE_MODEL_NAMES
from regime_lab.collection import last_completed_week_cutoff, weekly_cutoffs
from regime_lab.config import project_root
from regime_lab.data import SQLiteSnapshotStore
from regime_lab.dataset import build_weekly_dataset
from regime_lab.feature_quality import (
    canonical_feature_quality_json_bytes,
    feature_quality_artifact_manifest,
    feature_quality_document,
)
from regime_lab.io import write_json_atomic
from regime_lab.pipeline import FEATURE_SET_VERSION, _profile
from regime_lab.provider_rights import (
    providers_for_live_config,
    verify_provider_rights,
)
from regime_lab.walkforward_checkpoint import runtime_version_manifest


UTC = timezone.utc
RESEARCH_MODEL_NAMES = (
    "majority",
    "persistence",
    "markov",
    "ridge_logistic",
    "xgboost",
    *DIRECT_NEXT_STATE_MODEL_NAMES,
)
CONTROL_MODEL = "markov"
RESEARCH_PAIRED_CHALLENGER_NAMES = tuple(
    name
    for name in RESEARCH_MODEL_NAMES
    if name not in {"majority", "persistence", CONTROL_MODEL}
)
STATE_ORDER = ("risk_on", "transition", "risk_off")
PROSPECTIVE_REGISTRY_START = "2026-08-21T20:00:00+00:00"
ARTIFACT_FRAMES = (
    ("leaderboard", "model-leaderboard.csv"),
    ("oos_predictions", "oos-predictions.csv"),
    ("walk_forward_splits", "walk-forward-splits.csv"),
    ("selection_diagnostics", "selection-diagnostics.csv"),
    ("fold_feature_availability", "fold-feature-availability.csv"),
)


def _update_fingerprint(
    digest: Any,
    *,
    label: str,
    payload: bytes,
) -> None:
    encoded_label = label.encode("utf-8")
    digest.update(len(encoded_label).to_bytes(4, "big"))
    digest.update(encoded_label)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def research_source_fingerprint(
    root: Path | None = None,
    *,
    config: Mapping[str, Any] | None = None,
) -> str:
    """Bind private comparisons to loaded code, config, and policy bytes."""

    selected_root = (root or project_root()).resolve()
    paths = sorted(
        [
            *(
                path
                for path in (selected_root / "src/regime_lab").rglob("*.py")
                if "__pycache__" not in path.parts
            ),
            selected_root / "scripts/evaluate_research_models.py",
            selected_root / "config/structural_v6_research.json",
            selected_root / "pyproject.toml",
            selected_root / "requirements-ci.lock",
            *(
                (
                    selected_root / "config/series.json",
                    selected_root / "config/provider_rights.json",
                )
                if config is None
                else ()
            ),
        ],
        key=lambda path: path.relative_to(selected_root).as_posix(),
    )
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"research source fingerprint input is missing: {path}")
        _update_fingerprint(
            digest,
            label=path.relative_to(selected_root).as_posix(),
            payload=path.read_bytes(),
        )
    if config is not None:
        config_payload = json.dumps(
            config,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        policy_value = config.get("provider_rights_policy")
        if not isinstance(policy_value, str) or not policy_value.strip():
            raise RuntimeError(
                "effective research config has no provider_rights_policy"
            )
        policy_path = Path(policy_value)
        if not policy_path.is_absolute():
            policy_path = selected_root / policy_path
        if not policy_path.is_file() or policy_path.is_symlink():
            raise RuntimeError(
                "effective provider-rights policy is missing or unsafe"
            )
        _update_fingerprint(
            digest,
            label="effective-config.json",
            payload=config_payload,
        )
        _update_fingerprint(
            digest,
            label="effective-provider-rights-policy.json",
            payload=policy_path.read_bytes(),
        )
    return digest.hexdigest()


def _require_unchanged_research_source(
    expected_sha256: str,
    *,
    config: Mapping[str, Any] | None = None,
) -> None:
    current = (
        research_source_fingerprint()
        if config is None
        else research_source_fingerprint(config=config)
    )
    if current != expected_sha256:
        raise RuntimeError(
            "research source changed during matched OOS evaluation; "
            "discarding the mixed-code result"
        )


def _require_preregistered_model_suite() -> None:
    path = project_root() / "config/structural_v6_research.json"
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("V6 research preregistration is unavailable") from exc
    declared = document.get("comparison_models")
    if declared != list(RESEARCH_MODEL_NAMES):
        raise RuntimeError(
            "research comparison model suite differs from V6 preregistration"
        )


def _frame_identity(frame: pd.DataFrame) -> str:
    values = pd.util.hash_pandas_object(frame, index=True).to_numpy(
        dtype="uint64",
        copy=False,
    )
    columns = json.dumps(
        [str(column) for column in frame.columns],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(columns + values.tobytes()).hexdigest()


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(
        frame.to_json(
            orient="records",
            date_format="iso",
            date_unit="us",
            double_precision=15,
        )
    )


def _paired_log_losses(
    predictions: pd.DataFrame,
    *,
    candidate: str,
    split: str,
    control: str = CONTROL_MODEL,
) -> np.ndarray:
    selected = predictions.loc[
        predictions["evaluation_split"].astype(str).eq(split)
        & predictions["model"].astype(str).isin((control, candidate))
    ].copy()
    keys = ["origin_date", "target_date", "actual"]
    probabilities: dict[str, pd.DataFrame] = {}
    for model in (control, candidate):
        rows = selected.loc[selected["model"].astype(str).eq(model)].sort_values(keys)
        if rows.duplicated(keys).any():
            raise RuntimeError(f"duplicate matched OOS rows for {model}/{split}")
        probabilities[model] = rows.set_index(keys)
    if not probabilities[control].index.equals(probabilities[candidate].index):
        raise RuntimeError(f"non-common OOS origins for {candidate}/{split}")
    actual = probabilities[control].index.get_level_values("actual").astype(str)
    positions = {state: index for index, state in enumerate(STATE_ORDER)}

    def loss(model: str) -> np.ndarray:
        matrix = probabilities[model][
            [f"p_{state}" for state in STATE_ORDER]
        ].to_numpy(dtype=float)
        observed = np.asarray(
            [matrix[index, positions[state]] for index, state in enumerate(actual)],
            dtype=float,
        )
        if not np.isfinite(observed).all() or np.any(observed <= 0):
            raise RuntimeError(f"invalid OOS probabilities for {model}/{split}")
        return -np.log(observed)

    return loss(candidate) - loss(control)


def _block_interval(
    values: np.ndarray,
    *,
    block_weeks: int = 13,
    resamples: int = 1_999,
    random_state: int = 20260824,
) -> tuple[float, float, int]:
    differential = np.asarray(values, dtype=float)
    if differential.ndim != 1 or not len(differential):
        raise ValueError("paired loss differential must be one-dimensional and non-empty")
    effective = min(block_weeks, max(1, len(differential) // 2))
    blocks = int(np.ceil(len(differential) / effective))
    generator = np.random.default_rng(random_state)
    starts = generator.integers(0, len(differential), size=(resamples, blocks))
    offsets = np.arange(effective)
    indices = (starts[..., np.newaxis] + offsets) % len(differential)
    means = differential[indices.reshape(resamples, -1)[:, : len(differential)]].mean(
        axis=1
    )
    lower, upper = np.quantile(means, (0.025, 0.975))
    return float(lower), float(upper), effective


def paired_control_comparison(predictions: pd.DataFrame) -> list[dict[str, Any]]:
    """Compare every opt-in challenger with Markov on exactly matched rows."""

    rows: list[dict[str, Any]] = []
    for internal_split, exported_split in (
        ("selection", "selection"),
        ("holdout", "retrospective_diagnostic"),
    ):
        for candidate in RESEARCH_PAIRED_CHALLENGER_NAMES:
            differential = _paired_log_losses(
                predictions,
                candidate=candidate,
                split=internal_split,
            )
            lower, upper, effective = _block_interval(differential)
            rows.append(
                {
                    "split": exported_split,
                    "candidate": candidate,
                    "control": CONTROL_MODEL,
                    "n_common_origins": int(len(differential)),
                    "mean_log_loss_delta_candidate_minus_control": float(
                        differential.mean()
                    ),
                    "median_log_loss_delta_candidate_minus_control": float(
                        np.median(differential)
                    ),
                    "candidate_week_win_rate": float(np.mean(differential < 0)),
                    "block_bootstrap_ci95_lower": lower,
                    "block_bootstrap_ci95_upper": upper,
                    "nominal_block_weeks": 13,
                    "effective_block_weeks": effective,
                    "bootstrap_resamples": 1_999,
                }
            )
    return rows


def fold_feature_availability(
    features: pd.DataFrame,
    split_audit: pd.DataFrame,
) -> pd.DataFrame:
    """Record which columns were available inside each causal training fold."""

    required = {"origin_date", "last_train_origin", "train_size", "evaluation_split"}
    missing = required.difference(split_audit.columns)
    if missing:
        raise ValueError(f"split audit lacks feature-availability fields: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    for raw in split_audit.sort_values("origin_date").to_dict(orient="records"):
        origin = pd.Timestamp(raw["origin_date"])
        last_train_origin = pd.Timestamp(raw["last_train_origin"])
        if last_train_origin >= origin:
            raise RuntimeError("feature availability audit is not causally ordered")
        training = features.loc[:last_train_origin]
        available = training.notna().any(axis=0)
        unavailable = sorted(str(name) for name in available.index[~available])
        unavailable_bytes = json.dumps(
            unavailable,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        rows.append(
            {
                "origin_date": origin,
                "last_train_origin": last_train_origin,
                "evaluation_split": str(raw["evaluation_split"]),
                "train_size": int(raw["train_size"]),
                "feature_count": int(features.shape[1]),
                "available_feature_count": int(available.sum()),
                "unavailable_feature_count": int((~available).sum()),
                "unavailable_features_sha256": hashlib.sha256(
                    unavailable_bytes
                ).hexdigest(),
            }
        )
    return pd.DataFrame(rows)


def _prepare_matrix(
    config: Mapping[str, Any],
    *,
    database: Path,
    as_of: datetime,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("as_of must include a timezone")
    cutoff = as_of.astimezone(UTC)
    if cutoff != last_completed_week_cutoff(cutoff):
        raise ValueError("as_of must be an exact completed Friday 16:00 ET cutoff")
    policy = config.get("provider_rights_policy")
    if not isinstance(policy, str) or not policy.strip():
        raise ValueError("provider_rights_policy is required")
    policy_path = Path(policy)
    if not policy_path.is_absolute():
        policy_path = project_root() / policy_path
    verify_provider_rights(
        providers_for_live_config(config),
        policy_path=policy_path,
        capabilities=("local_storage", "model_training"),
    )
    with SQLiteSnapshotStore(database, read_only=True) as store:
        observations = store.read_last_good_observations(available_as_of=cutoff)
    if not observations:
        raise RuntimeError("no last-good observations are available at as_of")
    cutoffs = weekly_cutoffs(date(2006, 1, 1), cutoff)
    dataset = build_weekly_dataset(config, cutoffs, observations)
    canonical = dataset.canonical.loc[dataset.canonical["spy_close"].notna()].copy()
    if len(canonical) < 650:
        raise RuntimeError("research comparison requires at least 650 weekly rows")
    features = dataset.features.reindex(canonical.index).copy()
    labeler = CausalRegimeLabeler(
        RegimeLabelConfig(price_column="spy_close", minimum_fit_observations=260)
    )
    labeler.fit(canonical.iloc[:520])
    states = labeler.transform(canonical)
    scores = labeler.score_frame(canonical)
    for column in scores.columns:
        features[f"regime_boundary__{column}"] = scores[column]
    features = derive_causal_transition_features(
        features,
        states,
        risk_score_col="regime_boundary__risk_score",
        lower_threshold=labeler.lower_threshold_,
        upper_threshold=labeler.upper_threshold_,
    ).drop(columns=["current_state"])
    metadata = {
        "observation_count": len(observations),
        "weekly_row_count": len(features),
        "feature_count": features.shape[1],
        "feature_matrix_sha256": _frame_identity(features),
        "state_vector_sha256": _frame_identity(states.to_frame("state")),
    }
    alpha_rows = [
        record for record in observations if record.source == "alpha_vantage"
    ]
    if not alpha_rows:
        raise RuntimeError(
            "research comparison requires the market-label source provenance"
        )
    initial_market_retrieval = min(record.retrieved_at for record in alpha_rows)
    pre_initialization_rows = sum(
        record.observed_period_end < initial_market_retrieval.date()
        for record in alpha_rows
    )
    metadata["market_vintage_evidence"] = {
        "classification": "retrospective_current_adjusted_market_backfill",
        "initial_retrieved_at": initial_market_retrieval.isoformat(),
        "pre_initialization_observation_count": int(pre_initialization_rows),
        "historical_market_vintage_certified": False,
    }
    return features, states, metadata


def run_research_comparison(
    config: Mapping[str, Any],
    *,
    database: Path,
    as_of: datetime,
    profile_name: str = "standard",
    checkpoint_directory: Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame], dict[str, Any]]:
    source_fingerprint = research_source_fingerprint(config=config)
    _require_preregistered_model_suite()
    features, states, input_metadata = _prepare_matrix(
        config,
        database=database,
        as_of=as_of,
    )
    _require_unchanged_research_source(source_fingerprint, config=config)
    profile = _profile(profile_name, len(features))
    split_minimum = 3 if profile_name == "quick" else 12
    selection_end = str(config["model"]["final_holdout_start"])
    benchmark = run_benchmark(
        features,
        states,
        profile=profile,
        models=RESEARCH_MODEL_NAMES,
        gap=1,
        minimum_train_weeks=profile.minimum_train_weeks,
        random_state=17,
        selection_end=selection_end,
        selection_max_origins=3 if profile_name == "quick" else None,
        model_workers=1 if profile_name == "quick" else 4,
        minimum_selection_predictions=split_minimum,
        minimum_holdout_predictions=split_minimum,
        progress=progress,
        checkpoint_directory=checkpoint_directory,
        source_fingerprint_sha256=(
            source_fingerprint if checkpoint_directory is not None else None
        ),
    )
    _require_unchanged_research_source(source_fingerprint, config=config)
    diagnostics = (
        benchmark.selection_diagnostics
        if benchmark.selection_diagnostics is not None
        else pd.DataFrame()
    )
    exported_leaderboard = benchmark.leaderboard.copy()
    if "evaluation_split" in exported_leaderboard.columns:
        exported_leaderboard["evaluation_split"] = exported_leaderboard[
            "evaluation_split"
        ].replace({"holdout": "retrospective_diagnostic"})
    exported_predictions = benchmark.predictions.copy()
    exported_predictions["evaluation_split"] = exported_predictions[
        "evaluation_split"
    ].replace({"holdout": "retrospective_diagnostic"})
    exported_split_audit = benchmark.split_audit.copy()
    exported_split_audit["evaluation_split"] = exported_split_audit[
        "evaluation_split"
    ].replace({"holdout": "retrospective_diagnostic"})
    availability = fold_feature_availability(features, exported_split_audit)
    frames = {
        "leaderboard": exported_leaderboard,
        "oos_predictions": exported_predictions,
        "walk_forward_splits": exported_split_audit,
        "selection_diagnostics": diagnostics,
        "fold_feature_availability": availability,
    }
    quality = feature_quality_document(features)
    report = {
        "schema_version": "regime-private-model-comparison/1",
        "generated_at": datetime.now(UTC).isoformat(),
        "data_as_of": as_of.astimezone(UTC).isoformat(),
        "mode": "private_real_data_retrospective_research",
        "public_release_eligible": False,
        "promotion_status": "comparison_only_noncertifying",
        "profile": profile.name,
        "selection_end": selection_end,
        "feature_set_version": FEATURE_SET_VERSION,
        "models": list(RESEARCH_MODEL_NAMES),
        "control_model": CONTROL_MODEL,
        "restricted_suite_champion": benchmark.champion,
        "input": {
            **input_metadata,
            "analysis_source_fingerprint_sha256": source_fingerprint,
            "runtime_versions": runtime_version_manifest(),
        },
        "leaderboard": _json_records(exported_leaderboard),
        "selection_diagnostics": _json_records(diagnostics),
        "paired_control_comparison": paired_control_comparison(
            benchmark.predictions
        ),
        "evidence": {
            "evaluation_design": "matched_expanding_walk_forward",
            "market_history_status": input_metadata[
                "market_vintage_evidence"
            ]["classification"],
            "historical_market_vintage_certified": False,
            "post_2023_evidence_status": "retrospective_external_period_diagnostic",
            "prospective_registry_start": PROSPECTIVE_REGISTRY_START,
            "predictive_improvement_claim_eligible": False,
            "automatic_promotion_eligible": False,
        },
        "feature_quality": feature_quality_artifact_manifest(quality),
        "fold_feature_availability": {
            "origin_count": int(len(availability)),
            "maximum_unavailable_feature_count": int(
                availability["unavailable_feature_count"].max()
            ),
            "latest_unavailable_feature_count": int(
                availability.iloc[-1]["unavailable_feature_count"]
            ),
        },
        "interpretation": {
            "selection_rows_choose_models": True,
            "retrospective_diagnostic_rows_do_not_choose_models": True,
            "negative_log_loss_delta_favors_candidate": True,
            "automatic_champion_promotion": False,
        },
    }
    _require_unchanged_research_source(source_fingerprint, config=config)
    return report, frames, quality


def write_research_generation(
    output_root: Path,
    report: dict[str, Any],
    frames: Mapping[str, pd.DataFrame],
    quality: dict[str, Any],
    *,
    expected_source_fingerprint_sha256: str | None = None,
    source_config: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically expose one immutable, derived-only research generation."""

    if expected_source_fingerprint_sha256 is not None:
        _require_unchanged_research_source(
            expected_source_fingerprint_sha256,
            config=source_config,
        )

    root = output_root.resolve()
    runs = root / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    generation_id = (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        + "-"
        + uuid.uuid4().hex[:8]
    )
    staging = Path(tempfile.mkdtemp(prefix=".research-generation-", dir=runs))
    final = runs / generation_id
    try:
        manifest: dict[str, dict[str, Any]] = {}
        for key, filename in ARTIFACT_FRAMES:
            frame = frames[key]
            path = staging / filename
            frame.to_csv(path, index=False, lineterminator="\n")
            payload = path.read_bytes()
            manifest[key] = {
                "path": filename,
                "row_count": int(len(frame)),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        quality_path = staging / "feature-quality.json"
        quality_path.write_bytes(canonical_feature_quality_json_bytes(quality))
        report_document = {**report, "artifact_manifest": manifest}
        write_json_atomic(staging / "research-model-comparison.json", report_document)
        report_bytes = (staging / "research-model-comparison.json").read_bytes()
        if expected_source_fingerprint_sha256 is not None:
            _require_unchanged_research_source(
                expected_source_fingerprint_sha256,
                config=source_config,
            )
        os.replace(staging, final)
        write_json_atomic(
            root / "latest.json",
            {
                "schema_version": 1,
                "generation": f"runs/{generation_id}",
                "report": "research-model-comparison.json",
                "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
            },
        )
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return final


__all__ = [
    "CONTROL_MODEL",
    "RESEARCH_MODEL_NAMES",
    "RESEARCH_PAIRED_CHALLENGER_NAMES",
    "paired_control_comparison",
    "fold_feature_availability",
    "research_source_fingerprint",
    "run_research_comparison",
    "write_research_generation",
]
