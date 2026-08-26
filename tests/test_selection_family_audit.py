from __future__ import annotations

import copy
from datetime import timedelta
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import pandas as pd
import pytest

from regime_lab.analysis.labels import STATE_ORDER
from regime_lab.analysis.models import (
    BenchmarkProfile,
    model_manifest,
    model_manifest_sha256,
)
from regime_lab.analysis.selection_evaluation import build_selection_evaluation
from regime_lab.selection_family_audit import (
    build_selection_family_audit,
    build_selection_family_audit_from_artifacts,
    validate_selection_family_audit,
    validate_selection_family_payload_binding,
)
from regime_lab.integrity import canonical_json_sha256_v1
from regime_lab import integrity
from regime_lab.artifact_inventory import write_artifact_inventory
from regime_lab.artifact_inventory import verify_artifact_inventory
from regime_lab import cli
from types import SimpleNamespace


MODELS = ("markov", "causal_dynamic_ensemble", "causal_multiscale_ensemble")


def _diagnostics() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": "markov",
                "is_reference": True,
                "selected": False,
                "gate_passed": True,
                "gate_reason": "passed",
                "log_loss": -math.log(0.7),
                "brier": 0.135,
                "calibration_error": 0.10,
                "fallback_count": 0,
                "n_predictions": 4,
                "raw_p_value": None,
                "holm_adjusted_p_value": None,
            },
            {
                "model": "causal_dynamic_ensemble",
                "is_reference": False,
                "selected": True,
                "gate_passed": True,
                "gate_reason": "passed",
                "log_loss": -math.log(0.7),
                "brier": 0.135,
                "calibration_error": 0.06,
                "fallback_count": 0,
                "n_predictions": 4,
                "raw_p_value": 0.01,
                "holm_adjusted_p_value": 0.02,
            },
            {
                "model": "causal_multiscale_ensemble",
                "is_reference": False,
                "selected": False,
                "gate_passed": True,
                "gate_reason": "passed",
                "log_loss": -math.log(0.7),
                "brier": 0.135,
                "calibration_error": 0.07,
                "fallback_count": 0,
                "n_predictions": 4,
                "raw_p_value": 0.01,
                "holm_adjusted_p_value": 0.02,
            },
        ]
    )


def _predictions() -> pd.DataFrame:
    origins = pd.date_range("2022-01-07", periods=4, freq="W-FRI")
    rows = []
    for model in MODELS:
        for position, origin in enumerate(origins):
            actual = STATE_ORDER[position % len(STATE_ORDER)]
            probability = {state: 0.15 for state in STATE_ORDER}
            probability[actual] = 0.7
            rows.append(
                {
                    "model": model,
                    "origin_date": origin,
                    "target_date": origin + pd.to_timedelta(7, unit="D"),
                    "evaluation_split": "selection",
                    "current_state": STATE_ORDER[(position - 1) % len(STATE_ORDER)],
                    "actual": actual,
                    "predicted": actual,
                    "p_risk_on": probability["risk_on"],
                    "p_transition": probability["transition"],
                    "p_risk_off": probability["risk_off"],
                    "train_size": 520 + position,
                    "gap": 1,
                    "fallback": False,
                }
            )
    return pd.DataFrame(rows)


def _document() -> dict:
    predictions = _predictions()
    diagnostics = _diagnostics()
    supplemental = build_selection_evaluation(
        predictions,
        diagnostics,
        evidence_status="historical_reconstructed_oos",
        mcs_resamples=99,
    )
    return build_selection_family_audit(
        diagnostics,
        predictions,
        champion="causal_dynamic_ensemble",
        selection_reason="simplicity_within_tolerance",
        policy_sha256="a" * 64,
        complexity_registry={
            "markov": 2,
            "causal_dynamic_ensemble": 15,
            "causal_multiscale_ensemble": 16,
        },
        evidence_track="reconstructed_oos",
        generation_id="20260826T000000.000000Z",
        candidate_manifest_sha256="b" * 64,
        declared_selection_period="2022-01-14–2022-02-04",
        selection_end_at="2022-02-05T00:00:00+00:00",
        source_artifacts={
            "selection_diagnostics": {
                "path": "selection-diagnostics.csv",
                "sha256": "c" * 64,
                "row_count": 3,
            },
            "oos_predictions": {
                "path": "oos-predictions.csv",
                "sha256": "d" * 64,
                "row_count": 12,
            },
        },
        expected_candidate_set=MODELS,
        runner_up="causal_multiscale_ensemble",
        fallback={
            "model": "markov",
            "trigger": "no_challenger_passes_gate",
            "reason": "conservative probability baseline",
        },
        supplemental_evaluation=supplemental,
    )


def _required_kwargs() -> dict:
    return {
        "generation_id": "20260826T000000.000000Z",
        "candidate_manifest_sha256": "b" * 64,
        "declared_selection_period": "2022-01-14–2022-02-04",
        "selection_end_at": "2022-02-05T00:00:00+00:00",
        "source_artifacts": {
            "selection_diagnostics": {
                "path": "selection-diagnostics.csv",
                "sha256": "c" * 64,
                "row_count": 3,
            },
            "oos_predictions": {
                "path": "oos-predictions.csv",
                "sha256": "d" * 64,
                "row_count": 12,
            },
        },
        "expected_candidate_set": MODELS,
        "supplemental_evaluation": build_selection_evaluation(
            _predictions(),
            _diagnostics(),
            evidence_status="historical_reconstructed_oos",
            mcs_resamples=99,
        ),
    }


def _artifact_fixture(directory: Path) -> dict:
    directory.mkdir(parents=True)
    diagnostics = _diagnostics()
    selection = _predictions()
    holdout = selection.copy()
    holdout["evaluation_split"] = "holdout"
    holdout["origin_date"] = holdout["origin_date"] + timedelta(days=365)
    holdout["target_date"] = holdout["target_date"] + timedelta(days=365)
    predictions = pd.concat([selection, holdout], ignore_index=True)
    diagnostics.to_csv(directory / "selection-diagnostics.csv", index=False)
    predictions.to_csv(directory / "oos-predictions.csv", index=False)
    ranks = dict(zip(MODELS, (2, 15, 16), strict=True))
    manifest_body = {
        "models": [
            {"name": model, "complexity_rank": ranks[model]} for model in MODELS
        ]
    }
    manifest_sha256 = canonical_json_sha256_v1(manifest_body)
    (directory / "candidate-manifest.json").write_text(
        json.dumps({**manifest_body, "sha256": manifest_sha256}),
        encoding="utf-8",
    )
    return {
        "meta": {
            "mode": "demo",
            "generation_id": "20260826T000000.000000Z",
        },
        "model": {
            "champion": "causal_dynamic_ensemble",
            "candidate_manifest_sha256": manifest_sha256,
            "candidate_manifest": manifest_body,
            "selection_period": "2022-01-14–2022-02-04",
            "selection_end": "2022-02-05",
        },
        "selection": {
            "candidate_set": list(MODELS),
            "runner_up": "causal_multiscale_ensemble",
            "selection_reason": "simplicity_within_tolerance",
            "policy_sha256": "a" * 64,
        },
        "forecast": {"evidence_track": "reconstructed_oos"},
    }


def test_builder_contains_all_candidates_gates_policy_and_fallback() -> None:
    document = _document()
    validate_selection_family_audit(document)
    assert document["schema_version"] == "selection-family-audit/v2"
    assert document["candidate_set"] == list(MODELS)
    assert document["champion"] == "causal_dynamic_ensemble"
    assert document["runner_up"] == "causal_multiscale_ensemble"
    assert document["common_origin_contract"]["status"] == "matched"
    assert document["common_origin_contract"]["origin_count"] == 4
    assert all("gate" in row for row in document["candidates"])
    assert document["fallback"]["model"] == "markov"


def test_candidate_origin_mismatch_fails_closed() -> None:
    predictions = _predictions()
    mask = predictions["model"].eq("causal_multiscale_ensemble")
    predictions.loc[mask, "target_date"] += pd.to_timedelta(7, unit="D")
    with pytest.raises(ValueError, match="does not share exact"):
        build_selection_family_audit(
            _diagnostics(),
            predictions,
            champion="causal_dynamic_ensemble",
            selection_reason="simplicity_within_tolerance",
            policy_sha256="a" * 64,
            complexity_registry=dict(zip(MODELS, (2, 15, 16), strict=True)),
            evidence_track="reconstructed_oos",
            **_required_kwargs(),
        )


def test_validation_rejects_semantic_or_hash_tampering() -> None:
    document = _document()
    tampered = copy.deepcopy(document)
    tampered["champion"] = "markov"
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_selection_family_audit(tampered)


def test_builder_rejects_unpassed_runner_up_and_incomplete_registry() -> None:
    diagnostics = _diagnostics()
    diagnostics.loc[
        diagnostics["model"].eq("causal_multiscale_ensemble"), "gate_passed"
    ] = False
    with pytest.raises(ValueError, match="runner_up must pass"):
        build_selection_family_audit(
            diagnostics,
            _predictions(),
            champion="causal_dynamic_ensemble",
            runner_up="causal_multiscale_ensemble",
            selection_reason="manual",
            policy_sha256="a" * 64,
            complexity_registry=dict(zip(MODELS, (2, 15, 16), strict=True)),
            evidence_track="operational_oos",
            **{
                **_required_kwargs(),
                "supplemental_evaluation": build_selection_evaluation(
                    _predictions(),
                    diagnostics,
                    evidence_status="operational_oos",
                    mcs_resamples=99,
                ),
            },
        )
    with pytest.raises(ValueError, match="every candidate exactly"):
        build_selection_family_audit(
            _diagnostics(),
            _predictions(),
            champion="causal_dynamic_ensemble",
            selection_reason="manual",
            policy_sha256="a" * 64,
            complexity_registry={"markov": 2},
            evidence_track="operational_oos",
            **{
                **_required_kwargs(),
                "supplemental_evaluation": build_selection_evaluation(
                    _predictions(),
                    _diagnostics(),
                    evidence_status="operational_oos",
                    mcs_resamples=99,
                ),
            },
        )


def test_builder_rejects_post_selection_targets() -> None:
    predictions = _predictions()
    with pytest.raises(ValueError, match="post-selection"):
        build_selection_family_audit(
            _diagnostics(),
            predictions,
            champion="causal_dynamic_ensemble",
            selection_reason="manual",
            policy_sha256="a" * 64,
            complexity_registry=dict(zip(MODELS, (2, 15, 16), strict=True)),
            evidence_track="reconstructed_oos",
            **{
                **_required_kwargs(),
                "selection_end_at": "2022-01-20T00:00:00+00:00",
            },
        )


def test_validator_rejects_mismatched_generation() -> None:
    with pytest.raises(ValueError, match="generation_id mismatch"):
        validate_selection_family_audit(
            _document(),
            expected_generation_id="different-generation",
        )


def test_artifact_builder_uses_selection_only_and_marks_demo_synthetic(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "artifacts"
    payload = _artifact_fixture(artifacts)

    document = build_selection_family_audit_from_artifacts(payload, artifacts)

    assert document["evidence_track"] == "reconstructed_oos"
    assert document["evidence_status"] == "synthetic_fixture"
    assert document["supplemental_evaluation"]["evidence_status"] == (
        "synthetic_fixture"
    )
    assert document["common_origin_contract"]["origin_count"] == 4
    assert document["source_artifacts"]["oos_predictions"]["row_count"] == 24
    assert document["source_artifacts"]["oos_predictions"]["sha256"] == (
        hashlib.sha256((artifacts / "oos-predictions.csv").read_bytes()).hexdigest()
    )
    validate_selection_family_payload_binding(document, payload)
    payload["selection"]["policy_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="policy_sha256 differs from payload"):
        validate_selection_family_payload_binding(document, payload)


def test_artifact_builder_rejects_unsupported_split_and_manifest_drift(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "artifacts"
    payload = _artifact_fixture(artifacts)
    predictions = pd.read_csv(artifacts / "oos-predictions.csv")
    predictions.loc[predictions.index[-1], "evaluation_split"] = "validation"
    predictions.to_csv(artifacts / "oos-predictions.csv", index=False)
    with pytest.raises(ValueError, match="unsupported evaluation split"):
        build_selection_family_audit_from_artifacts(payload, artifacts)

    payload = _artifact_fixture(tmp_path / "second-artifacts")
    second = tmp_path / "second-artifacts"
    manifest = json.loads((second / "candidate-manifest.json").read_text())
    manifest["models"][0]["complexity_rank"] = 99
    (second / "candidate-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="candidate manifest differs"):
        build_selection_family_audit_from_artifacts(payload, second)


def test_replay_cli_writes_only_a_new_confined_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = tmp_path / "artifacts"
    payload = _artifact_fixture(artifacts)
    payload_path = tmp_path / "payload.json"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts" / (
        "build_selection_family_audit.py"
    )
    spec = importlib.util.spec_from_file_location("selection_family_replay", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "project_root", lambda: tmp_path)
    output = tmp_path / "build" / "replay" / "selection-family-audit.json"

    result = module.rebuild_selection_family_audit(
        payload_path=payload_path,
        artifact_directory=artifacts,
        output_path=output,
    )

    assert result["ok"] is True
    assert result["evidence_status"] == "synthetic_fixture"
    assert output.is_file()
    with pytest.raises(module.ReplayError, match="refusing overwrite"):
        module.rebuild_selection_family_audit(
            payload_path=payload_path,
            artifact_directory=artifacts,
            output_path=output,
        )


def test_generation_manifest_binds_selection_family_and_rejects_foreign_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    artifacts = project / "private-artifacts"
    payload = _artifact_fixture(artifacts)
    generation_id = payload["meta"]["generation_id"]
    payload["meta"].update(
        {
            "data_as_of": "2022-02-04T21:00:00+00:00",
            "publication_status": "unpublished",
        }
    )
    payload["model"].update(
        {
            "selection_status": "selected_by_gate",
            "label_version": "market-causal-3state-v1",
            "execution_parameters": {"sha256": "e" * 64},
            "lifecycle": {
                "selection": {"status": "selected_by_gate"},
                "deployment": {"status": "reviewed"},
                "publication": {"status": "unpublished"},
            },
        }
    )
    payload["label"] = {
        "spec_id": "v1_spy_hysteresis",
        "spec_version": "market-causal-3state-v1",
        "spec_sha256": "b" * 64,
    }
    (artifacts / "build-generation.json").write_text(
        json.dumps({"generation_id": generation_id}), encoding="utf-8"
    )
    write_artifact_inventory(artifacts)
    run = project / "run"
    run.mkdir(parents=True)
    payload_path = run / "regime-results.json"
    sidecar_path = run / "selection-family-audit.json"
    manifest_path = run / "generation-manifest.json"
    label_path = project / "config" / "label-spec.json"
    label_path.parent.mkdir(parents=True)
    label_path.write_text('{"version":"market-causal-3state-v1"}\n')
    sidecar = build_selection_family_audit_from_artifacts(payload, artifacts)
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    monkeypatch.setattr(integrity, "project_root", lambda: project)

    manifest = integrity.build_generation_manifest(
        payload=payload,
        payload_path=payload_path,
        artifact_directory=artifacts,
        input_snapshot={
            "data_as_of": payload["meta"]["data_as_of"],
            "sha256": "a" * 64,
        },
        label_spec_path=label_path,
        selection_family=sidecar,
        selection_family_path=sidecar_path,
    )
    bound = integrity.bind_payload_to_generation_manifest(payload, manifest)
    payload_path.write_text(json.dumps(bound), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = integrity.validate_generation_manifest(
        manifest_path,
        require_comparison=False,
        require_selection_family=True,
        require_artifacts=False,
    )
    assert result["selection_family"] == sidecar
    foreign = copy.deepcopy(sidecar)
    foreign["generation_id"] = "foreign-generation"
    foreign_body = dict(foreign)
    foreign_body.pop("sha256")
    foreign["sha256"] = canonical_json_sha256_v1(foreign_body)
    with pytest.raises(integrity.IntegrityError, match="selection-family.*invalid"):
        integrity.build_generation_manifest(
            payload=payload,
            payload_path=payload_path,
            artifact_directory=artifacts,
            input_snapshot={
                "data_as_of": payload["meta"]["data_as_of"],
                "sha256": "a" * 64,
            },
            label_spec_path=label_path,
            selection_family=foreign,
            selection_family_path=sidecar_path,
        )


def test_v5_artifact_generation_emits_selection_family_before_inventory(
    tmp_path: Path,
) -> None:
    diagnostics = _diagnostics()
    selection_predictions = _predictions()
    holdout = selection_predictions.copy()
    holdout["evaluation_split"] = "holdout"
    holdout["origin_date"] += timedelta(days=365)
    holdout["target_date"] += timedelta(days=365)
    predictions = pd.concat([selection_predictions, holdout], ignore_index=True)
    profile = BenchmarkProfile.quick()
    manifest_body = model_manifest(profile, random_state=17, names=MODELS)
    manifest_sha256 = model_manifest_sha256(
        profile,
        random_state=17,
        names=MODELS,
    )
    payload = {
        "meta": {
            "mode": "demo",
            "generation_id": "20260826T000000.000000Z",
        },
        "model": {
            "champion": "causal_dynamic_ensemble",
            "candidate_manifest_sha256": manifest_sha256,
            "candidate_manifest": manifest_body,
            "selection_period": "2022-01-14–2022-02-04",
            "selection_end": "2022-02-05",
        },
        "selection": {
            "candidate_set": list(MODELS),
            "runner_up": "causal_multiscale_ensemble",
            "selection_reason": "simplicity_within_tolerance",
            "policy_sha256": "a" * 64,
        },
        "forecast": {"evidence_track": "reconstructed_oos"},
    }
    benchmark = SimpleNamespace(
        leaderboard=pd.DataFrame({"model": list(MODELS)}),
        predictions=predictions,
        split_audit=pd.DataFrame({"origin_date": []}),
        selection_diagnostics=diagnostics,
        profile=profile,
    )
    output = tmp_path / "artifacts"

    cli._write_supporting_results(
        benchmark,
        output,
        generation_id=payload["meta"]["generation_id"],
        write_inventory=True,
        selection_context=payload,
    )

    inventory = verify_artifact_inventory(output)
    document = json.loads(
        (output / "selection-family-audit.json").read_text(encoding="utf-8")
    )
    validate_selection_family_audit(
        document,
        expected_generation_id=payload["meta"]["generation_id"],
    )
    assert inventory["file_count"] == 7
    assert "selection-family-audit.json" in (
        output / "SHA256SUMS"
    ).read_text(encoding="ascii")
