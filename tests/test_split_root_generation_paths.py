from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from dataclasses import replace

import pytest

from regime_lab import integrity
from regime_lab.path_safety import (
    UnsafeMutablePath,
    confined_generation_path,
    generation_contract_path,
)
from test_generation_integrity import _generation, _json_bytes
from test_weekly_automation import _settings, CANDIDATE_CONTEXT, LIVE_TARGET
from regime_lab import automation


def test_split_generation_uses_explicit_state_files_and_source_label_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, paths = _generation(tmp_path, monkeypatch)
    source = paths["project"]
    state = tmp_path / "state"
    state.mkdir()
    shutil.move(str(source / "run"), state / "run")
    payload_path = state / "run/regime-results.json"
    comparison_path = state / "run/comparison.json"
    artifacts = state / "run/artifacts"
    manifest_path = state / "run/generation-manifest.json"
    payload = json.loads(payload_path.read_text())
    comparison = json.loads(comparison_path.read_text())
    original_manifest = json.loads(manifest_path.read_text())
    sentinels = {}
    for name in ("regime-results.json", "comparison.json"):
        target = source / "run" / name
        target.parent.mkdir(exist_ok=True)
        sentinel = json.loads(_json_bytes(payload if name == "regime-results.json" else comparison))
        if name == "regime-results.json":
            sentinel["meta"]["generation_id"] = "previous-source-generation"
        else:
            sentinel["result"]["log_loss"] = 0.75
        target.write_bytes(_json_bytes(sentinel))
        sentinels[target] = target.read_bytes()

    manifest = integrity.build_generation_manifest(
        payload=payload, payload_path=payload_path, artifact_directory=artifacts,
        input_snapshot=original_manifest["input_snapshot"],
        label_spec_path=source / "config/label-spec.json",
        comparison=comparison, comparison_path=comparison_path,
        payload_contract_path=generation_contract_path(
            payload_path, source_root=source, state_root=state,
        ),
        comparison_contract_path=generation_contract_path(
            comparison_path, source_root=source, state_root=state,
        ),
    )
    payload = integrity.bind_payload_to_generation_manifest(payload, manifest)
    payload_path.write_bytes(_json_bytes(payload))
    comparison["inputs"]["v5"]["regime_results"]["sha256"] = hashlib.sha256(
        payload_path.read_bytes()
    ).hexdigest()
    comparison_path.write_bytes(_json_bytes(comparison))
    manifest_path.write_bytes(_json_bytes(manifest))

    result = integrity.validate_generation_manifest(
        manifest_path, require_comparison=True, artifact_directory=artifacts,
        payload_path_override=payload_path, comparison_path_override=comparison_path,
    )
    assert result["payload_path"] == payload_path
    assert result["comparison_path"] == comparison_path
    assert result["artifact_directory"] == artifacts
    assert result["declared_payload_path"] == source / "run/regime-results.json"
    assert result["label_spec"]["registry_sha256"] == hashlib.sha256(
        (source / "config/label-spec.json").read_bytes()
    ).hexdigest()
    assert manifest["runtime_fingerprint"] == integrity.build_runtime_fingerprint(source)
    with pytest.raises(integrity.IntegrityError, match="payload contract hash"):
        integrity.validate_generation_manifest(manifest_path, artifact_directory=artifacts)
    assert {path: path.read_bytes() for path in sentinels} == sentinels


@pytest.mark.parametrize("outside", ["source", "other-state", "temporary"])
def test_generation_members_have_no_global_temp_or_other_root_fallback(
    tmp_path: Path, outside: str,
) -> None:
    state = tmp_path / "state"
    state.mkdir()
    with pytest.raises(UnsafeMutablePath):
        confined_generation_path(tmp_path / outside / "payload.json", root=state)
    with pytest.raises(UnsafeMutablePath):
        confined_generation_path("../escape.json", root=state)


@pytest.mark.parametrize("logical", [False, True])
@pytest.mark.parametrize("kind", ["parent", "leaf", "dangling"])
def test_generation_mapping_rejects_symlink_components_in_both_roots(
    tmp_path: Path, logical: bool, kind: str,
) -> None:
    source, state = tmp_path / "source", tmp_path / "state"
    source.mkdir()
    state.mkdir()
    selected = source if logical else state
    real = selected / "real"
    real.mkdir()
    if kind == "parent":
        (selected / "generation").symlink_to(real, target_is_directory=True)
        member = state / "generation/payload.json"
    else:
        (selected / "generation").mkdir()
        destination = real / "payload.json"
        if kind == "leaf":
            destination.write_text("sentinel")
        (selected / "generation/payload.json").symlink_to(destination)
        member = state / "generation/payload.json"
    with pytest.raises(UnsafeMutablePath, match="symbolic-link"):
        generation_contract_path(member, source_root=source, state_root=state)
    if kind == "leaf":
        assert destination.read_text() == "sentinel"


def test_generation_mapping_rejects_source_as_physical_state_root(tmp_path: Path) -> None:
    source, state = tmp_path / "source", tmp_path / "state"
    source.mkdir()
    state.mkdir()
    with pytest.raises(UnsafeMutablePath):
        generation_contract_path(source / "run/result.json", source_root=source, state_root=state)


def _split_settings(tmp_path):
    source, state = tmp_path / "source", tmp_path / "state"
    source.mkdir()
    state.mkdir()
    return replace(_settings(state, root=source), state_root=state, contract="v5")


def test_automation_hands_actual_state_members_to_first_audit_and_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _split_settings(tmp_path)
    commands = []

    def capture(command, **_kwargs):
        command = tuple(map(str, command))
        commands.append(command)
        if "scripts/promote_v5_publication.py" in command:
            raise automation.AutomationError("promotion handoff captured")
        return b""

    monkeypatch.setattr(automation, "_run", capture)
    monkeypatch.setattr(automation, "_candidate_context", lambda _settings: CANDIDATE_CONTEXT)
    monkeypatch.setattr(automation, "_sqlite_quick_check", lambda _path: None)
    with pytest.raises(automation.AutomationError, match="handoff captured"):
        automation._build_candidate(settings, target=LIVE_TARGET, context=CANDIDATE_CONTEXT)

    first_audit = next(c for c in commands if "scripts/audit_outputs.py" in c)
    promotion = next(c for c in commands if "scripts/promote_v5_publication.py" in c)
    for flag, expected in {
        "--payload": settings.payload, "--artifacts": settings.artifacts,
        "--selection-family": settings.artifacts / "selection-family-audit.json",
        "--state-root": settings.storage_root,
    }.items():
        assert first_audit[first_audit.index(flag) + 1] == str(expected)
    assert promotion[promotion.index("--state-root") + 1] == str(settings.storage_root)
    assert promotion[promotion.index("--candidate") + 1] == str(settings.payload)
    assert promotion[promotion.index("--publication-contract-directory") + 1] == "publication/live"


def test_automation_package_handoff_keeps_private_members_and_temp_output_in_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _split_settings(tmp_path)
    commands = []

    def capture(command, **_kwargs):
        command = tuple(map(str, command))
        commands.append(command)
        if "scripts/verify_public_package.py" in command:
            raise automation.AutomationError("package handoff captured")
        return b""

    monkeypatch.setattr(automation, "_run", capture)
    with pytest.raises(automation.AutomationError, match="handoff captured"):
        automation._verify_candidate_package(
            settings, settings.reviewed_payload_path, settings.comparison_path,
            settings.reviewed_generation_manifest_path, settings.reviewed_selection_family_path,
        )
    package = next(c for c in commands if "scripts/package_public_demo.py" in c)
    assert package[package.index("--state-root") + 1] == str(settings.storage_root)
    assert package[package.index("--artifacts") + 1] == str(settings.artifacts)
    assert Path(package[package.index("--output") + 1]).is_relative_to(settings.storage_root)
    assert package[package.index("--staged-generation-contract-directory") + 1] == "publication/live"
    assert not (settings.root / "build").exists()
