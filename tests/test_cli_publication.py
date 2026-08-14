from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from regime_lab import cli


def _payload(generation_id: str) -> dict[str, object]:
    return {"meta": {"generation_id": generation_id}}


def _install_stage_fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "validate_dashboard_payload", lambda _payload: None)

    def write_artifacts(
        _benchmark: object,
        directory: Path,
        *,
        generation_id: str | None = None,
    ) -> None:
        directory.mkdir(parents=True)
        (directory / "build-generation.json").write_text(
            json.dumps({"generation_id": generation_id}),
            encoding="utf-8",
        )
        (directory / "marker.txt").write_text("new artifacts", encoding="utf-8")

    def write_payload(payload: dict[str, object], path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    monkeypatch.setattr(cli, "_write_supporting_results", write_artifacts)
    monkeypatch.setattr(cli, "write_dashboard_payload", write_payload)


def _old_generation(root: Path) -> tuple[Path, Path]:
    artifacts = root / "artifacts" / "latest"
    output = root / "web" / "data" / "regime-results.json"
    artifacts.mkdir(parents=True)
    output.parent.mkdir(parents=True)
    (artifacts / "marker.txt").write_text("old artifacts", encoding="utf-8")
    output.write_text(
        json.dumps({"meta": {"generation_id": "old"}}),
        encoding="utf-8",
    )
    return artifacts, output


def _assert_no_transaction_directories(artifacts: Path, output: Path) -> None:
    assert not list(artifacts.parent.glob(f".{artifacts.name}-publish-*"))
    assert not list(output.parent.glob(f".{output.name}-publish-*"))


def test_publish_active_generation_replaces_both_outputs_together(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_stage_fakes(monkeypatch)
    artifacts, output = _old_generation(tmp_path)

    cli._publish_active_generation(
        _payload("new"),
        object(),
        output=output,
        artifacts=artifacts,
    )

    artifact_generation = json.loads(
        (artifacts / "build-generation.json").read_text(encoding="utf-8")
    )["generation_id"]
    payload_generation = json.loads(output.read_text(encoding="utf-8"))["meta"][
        "generation_id"
    ]
    assert artifact_generation == payload_generation == "new"
    assert (artifacts / "marker.txt").read_text(encoding="utf-8") == "new artifacts"
    _assert_no_transaction_directories(artifacts, output)


def test_staging_failure_leaves_old_generation_untouched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_stage_fakes(monkeypatch)
    artifacts, output = _old_generation(tmp_path)

    def fail_staging(*_args, **_kwargs) -> None:
        raise OSError("artifact staging failed")

    monkeypatch.setattr(cli, "_write_supporting_results", fail_staging)
    with pytest.raises(OSError, match="artifact staging failed"):
        cli._publish_active_generation(
            _payload("new"),
            object(),
            output=output,
            artifacts=artifacts,
        )

    assert (artifacts / "marker.txt").read_text(encoding="utf-8") == "old artifacts"
    assert json.loads(output.read_text(encoding="utf-8"))["meta"][
        "generation_id"
    ] == "old"
    _assert_no_transaction_directories(artifacts, output)


def test_payload_cutover_failure_rolls_back_both_active_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_stage_fakes(monkeypatch)
    artifacts, output = _old_generation(tmp_path)
    real_replace = cli.os.replace
    failed = False

    def fail_new_payload(source: str | Path, target: str | Path) -> None:
        nonlocal failed
        source_path = Path(source)
        target_path = Path(target)
        if not failed and source_path.name == "next.json" and target_path == output:
            failed = True
            raise OSError("payload cutover failed")
        real_replace(source, target)

    monkeypatch.setattr(cli.os, "replace", fail_new_payload)
    with pytest.raises(OSError, match="payload cutover failed"):
        cli._publish_active_generation(
            _payload("new"),
            object(),
            output=output,
            artifacts=artifacts,
        )

    assert failed is True
    assert (artifacts / "marker.txt").read_text(encoding="utf-8") == "old artifacts"
    assert json.loads(output.read_text(encoding="utf-8"))["meta"][
        "generation_id"
    ] == "old"
    _assert_no_transaction_directories(artifacts, output)


def test_failed_first_publication_leaves_no_partial_active_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_stage_fakes(monkeypatch)
    artifacts = tmp_path / "artifacts" / "latest"
    output = tmp_path / "web" / "data" / "regime-results.json"
    real_replace = cli.os.replace

    def fail_new_payload(source: str | Path, target: str | Path) -> None:
        if Path(source).name == "next.json" and Path(target) == output:
            raise OSError("payload cutover failed")
        real_replace(source, target)

    monkeypatch.setattr(cli.os, "replace", fail_new_payload)
    with pytest.raises(OSError, match="payload cutover failed"):
        cli._publish_active_generation(
            _payload("new"),
            object(),
            output=output,
            artifacts=artifacts,
        )

    assert not artifacts.exists()
    assert not output.exists()
    _assert_no_transaction_directories(artifacts, output)


def test_rollback_failure_preserves_recovery_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_stage_fakes(monkeypatch)
    artifacts, output = _old_generation(tmp_path)
    real_replace = cli.os.replace
    cutover_failed = False

    def fail_cutover_and_rollback(source: str | Path, target: str | Path) -> None:
        nonlocal cutover_failed
        source_path = Path(source)
        target_path = Path(target)
        if (
            not cutover_failed
            and source_path.name == "next.json"
            and target_path == output
        ):
            cutover_failed = True
            raise OSError("payload cutover failed")
        if cutover_failed and source_path.name == "previous" and target_path == artifacts:
            raise OSError("artifact rollback failed")
        real_replace(source, target)

    monkeypatch.setattr(cli.os, "replace", fail_cutover_and_rollback)
    with pytest.raises(RuntimeError, match="recovery paths retained"):
        cli._publish_active_generation(
            _payload("new"),
            object(),
            output=output,
            artifacts=artifacts,
        )

    artifact_recovery = list(
        artifacts.parent.glob(f".{artifacts.name}-publish-*")
    )
    payload_recovery = list(output.parent.glob(f".{output.name}-publish-*"))
    assert len(artifact_recovery) == len(payload_recovery) == 1
    assert (
        artifact_recovery[0] / "previous" / "marker.txt"
    ).read_text(encoding="utf-8") == "old artifacts"
    assert json.loads(
        (payload_recovery[0] / "previous.json").read_text(encoding="utf-8")
    )["meta"]["generation_id"] == "old"


def test_payload_path_inside_artifacts_is_rejected_before_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_stage_fakes(monkeypatch)
    artifacts = tmp_path / "artifacts" / "latest"
    with pytest.raises(ValueError, match="must not be stored inside artifacts"):
        cli._publish_active_generation(
            _payload("new"),
            object(),
            output=artifacts / "regime-results.json",
            artifacts=artifacts,
        )
    assert not artifacts.exists()


def test_mutable_path_allows_project_local_and_system_temp_targets(
    tmp_path: Path,
) -> None:
    project_target = cli.project_root() / "build" / "safe-target.json"
    assert cli._mutable_path(
        "build/safe-target.json", label="test output"
    ) == project_target
    temp_target = tmp_path / "safe-target.json"
    assert cli._mutable_path(temp_target, label="test output") == temp_target


def test_mutable_path_rejects_absolute_other_project_before_write() -> None:
    other_project = cli.project_root().parent / "do-not-touch" / "result.json"
    with pytest.raises(ValueError, match="must stay below"):
        cli._mutable_path(other_project, label="test output")


def test_publish_rejects_symlink_parent_even_within_system_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_stage_fakes(monkeypatch)
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(ValueError, match="symbolic-link"):
        cli._publish_active_generation(
            _payload("new"),
            object(),
            output=linked_parent / "regime-results.json",
            artifacts=tmp_path / "artifacts" / "latest",
        )
    assert not (real_parent / "regime-results.json").exists()


def test_v4_staged_evidence_linkage_requires_both_exact_raw_hashes(
    tmp_path: Path,
) -> None:
    label_bytes = b"date,state\n2026-08-07,risk_on\n"
    forecast_bytes = b"origin_date,model\n2026-08-07,markov\n"
    (tmp_path / "state-label-history.csv").write_bytes(label_bytes)
    (tmp_path / "weekly-state-forecasts.csv").write_bytes(forecast_bytes)
    payload = {
        "meta": {"result_version": "weekly-regime-result-v4"},
        "model": {
            "evidence_artifacts": {
                "state_label_history": {
                    "path": "state-label-history.csv",
                    "sha256": hashlib.sha256(label_bytes).hexdigest(),
                },
                "weekly_state_forecasts": {
                    "path": "weekly-state-forecasts.csv",
                    "sha256": hashlib.sha256(forecast_bytes).hexdigest(),
                },
            }
        },
    }

    cli._verify_staged_evidence_artifacts(payload, tmp_path)
    payload["model"]["evidence_artifacts"]["weekly_state_forecasts"][
        "sha256"
    ] = "0" * 64
    with pytest.raises(RuntimeError, match="hash mismatch"):
        cli._verify_staged_evidence_artifacts(payload, tmp_path)


def test_non_v4_staged_publication_preserves_legacy_optional_evidence(
    tmp_path: Path,
) -> None:
    cli._verify_staged_evidence_artifacts(
        {"meta": {"result_version": "weekly-regime-result-v3"}},
        tmp_path,
    )


def test_missing_v4_evidence_fails_before_active_generation_cutover(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_stage_fakes(monkeypatch)
    artifacts, output = _old_generation(tmp_path)
    payload = _payload("new")
    payload["meta"]["result_version"] = "weekly-regime-result-v4"
    payload["model"] = {
        "evidence_artifacts": {
            "state_label_history": {
                "path": "state-label-history.csv",
                "sha256": "0" * 64,
            },
            "weekly_state_forecasts": {
                "path": "weekly-state-forecasts.csv",
                "sha256": "0" * 64,
            },
        }
    }

    with pytest.raises(RuntimeError, match="staged evidence artifact is missing"):
        cli._publish_active_generation(
            payload,
            object(),
            output=output,
            artifacts=artifacts,
        )

    assert (artifacts / "marker.txt").read_text(encoding="utf-8") == "old artifacts"
    assert json.loads(output.read_text(encoding="utf-8"))["meta"][
        "generation_id"
    ] == "old"
    _assert_no_transaction_directories(artifacts, output)
