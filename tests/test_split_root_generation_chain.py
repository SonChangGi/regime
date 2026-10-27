"""Split-root contracts plus an opt-in, real last-good bundle acceptance chain.

The external acceptance fixture is never copied into the repository. Set
REGIME_SPLIT_ROOT_FIXTURE_ROOT to a local checkout containing a verified V5
generation and the pinned V4 reference to run the complete no-fitting chain.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

import pytest

from regime_lab import cli, integrity
from regime_lab.artifact_inventory import write_artifact_inventory
from regime_lab.analysis.label_spec import default_label_spec_path
from regime_lab.path_safety import (
    UnsafeMutablePath,
    confined_generation_path,
    generation_contract_path,
)
from regime_lab.selection_family_audit import build_selection_family_audit_from_artifacts


ROOT = Path(__file__).resolve().parents[1]


def _module(relative: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


@pytest.fixture
def split_generation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A compact real manifest graph, including reconstructed selection evidence."""
    fixture = _module("tests/test_selection_family_audit.py", "_split_selection_fixture")
    source, state = tmp_path / "source", tmp_path / "state"
    source.mkdir()
    state.mkdir()
    artifacts = state / "build/generation/artifacts"
    payload = fixture._artifact_fixture(artifacts)
    payload["meta"].update({"data_as_of": "2022-02-04T21:00:00+00:00", "publication_status": "unpublished"})
    payload["model"].update({
        "selection_status": "selected_by_gate", "label_version": "market-causal-3state-v1",
        "execution_parameters": {"sha256": "e" * 64},
        "lifecycle": {"selection": {"status": "selected_by_gate"},
                      "deployment": {"status": "reviewed"},
                      "publication": {"status": "unpublished"}},
    })
    payload["label"] = {"spec_id": "v1_spy_hysteresis", "spec_version": "market-causal-3state-v1", "spec_sha256": "b" * 64}
    label = source / "config/label-spec.json"
    label.parent.mkdir()
    label.write_text('{"version":"market-causal-3state-v1"}\n')
    (source / "requirements-ci.lock").write_text("source-runtime-lock\n")
    (state / "requirements-ci.lock").write_text("different-state-file\n")
    sidecar = build_selection_family_audit_from_artifacts(payload, artifacts)
    sidecar_path = artifacts / "selection-family-audit.json"
    _write_json(sidecar_path, sidecar)
    _write_json(artifacts / "build-generation.json", {"generation_id": payload["meta"]["generation_id"]})
    write_artifact_inventory(artifacts)
    output = state / "build/generation/regime-results.json"
    manifest_path = output.with_name("generation-manifest.json")
    sentinel = generation_contract_path(output, source_root=source, state_root=state)
    sentinel.parent.mkdir(parents=True)
    sentinel.write_bytes(b"old source sentinel\n")
    selection_sentinel = generation_contract_path(sidecar_path, source_root=source, state_root=state)
    selection_sentinel.parent.mkdir(parents=True)
    selection_sentinel.write_bytes(b"old source selection sentinel\n")
    monkeypatch.setattr(integrity, "project_root", lambda: source)
    manifest = integrity.build_generation_manifest(
        payload=payload, payload_path=output, artifact_directory=artifacts,
        input_snapshot={"data_as_of": payload["meta"]["data_as_of"], "sha256": "a" * 64},
        label_spec_path=label, selection_family=sidecar, selection_family_path=sidecar_path,
        payload_contract_path=sentinel, selection_family_contract_path=selection_sentinel,
    )
    bound = integrity.bind_payload_to_generation_manifest(payload, manifest)
    _write_json(output, bound)
    _write_json(manifest_path, manifest)
    return {"source": source, "state": state, "artifacts": artifacts, "payload": output,
            "manifest": manifest_path, "selection": sidecar_path, "document": bound,
            "manifest_document": manifest, "label": label, "sentinel": sentinel,
            "selection_sentinel": selection_sentinel}


def _validate_graph(paths):
    return integrity.validate_generation_manifest(
        paths["manifest"], require_comparison=False, require_selection_family=True,
        artifact_directory=paths["artifacts"], payload_path_override=paths["payload"],
        selection_family_path_override=paths["selection"],
    )


def test_split_manifest_reads_state_and_keeps_source_label_runtime_authority(split_generation):
    p = split_generation
    result = _validate_graph(p)
    assert result["payload_path"] == p["payload"]
    assert result["declared_payload_path"] == p["sentinel"]
    assert result["selection_family_path"] == p["selection"]
    assert result["label_spec"]["registry_verified"] is True
    assert result["runtime_fingerprint"] == integrity.build_runtime_fingerprint(p["source"])
    assert p["sentinel"].read_bytes() == b"old source sentinel\n"
    assert p["selection_sentinel"].read_bytes() == b"old source selection sentinel\n"


@pytest.mark.parametrize("kind", ["foreign_root", "escape", "leaf_link", "parent_link", "dangling_link"])
def test_generation_physical_paths_reject_wrong_roots_and_links(split_generation, kind):
    p = split_generation
    if kind == "foreign_root":
        candidate = p["sentinel"]
    elif kind == "escape":
        candidate = p["state"] / ".." / "escape.json"
    elif kind == "leaf_link":
        candidate = p["state"] / "payload-alias.json"
        candidate.symlink_to(p["payload"])
    elif kind == "parent_link":
        alias = p["state"] / "parent-alias"
        alias.symlink_to(p["payload"].parent, target_is_directory=True)
        candidate = alias / p["payload"].name
    else:
        candidate = p["state"] / "dangling.json"
        candidate.symlink_to(p["state"] / "missing.json")
    with pytest.raises(UnsafeMutablePath):
        confined_generation_path(candidate, root=p["state"])


def test_manifest_contract_rejects_source_parent_alias(split_generation):
    p = split_generation
    (p["source"] / "alias").symlink_to(p["source"] / "build", target_is_directory=True)
    with pytest.raises(UnsafeMutablePath):
        confined_generation_path("alias/generation/regime-results.json", root=p["source"])


@pytest.mark.parametrize("member", ["payload", "selection"])
def test_split_generation_rejects_hash_drift(split_generation, member):
    p = split_generation
    path = p[member]
    body = json.loads(path.read_text())
    if member == "payload":
        body["model"]["execution_parameters"]["sha256"] = "f" * 64
    else:
        body["generation_id"] = "foreign-generation"
    _write_json(path, body)
    with pytest.raises(integrity.IntegrityError, match="hash mismatch"):
        _validate_graph(p)


def test_split_generation_rejects_cutoff_drift(split_generation):
    p = split_generation
    with pytest.raises(integrity.IntegrityError, match="data_as_of differs"):
        integrity.build_generation_manifest(
            payload=p["document"], payload_path=p["payload"], artifact_directory=p["artifacts"],
            input_snapshot={"data_as_of": "2022-02-11T21:00:00+00:00", "sha256": "a" * 64},
            label_spec_path=p["label"], payload_contract_path=p["sentinel"],
        )


@pytest.mark.parametrize("stage", ["audit", "promotion", "packaging"])
def test_split_stage_rejects_wrong_physical_root(split_generation, monkeypatch, stage):
    p = split_generation
    if stage == "audit":
        script = _module("scripts/audit_outputs.py", "_split_wrong_root_audit")
        monkeypatch.setattr(script, "PROJECT_ROOT", p["source"])
        with pytest.raises(UnsafeMutablePath):
            script._audit_local_generation(
                p["manifest"], payload_path=p["payload"],
                selection_family_path=p["selection"], artifact_directory=p["artifacts"],
                state_root=p["source"],
            )
    elif stage == "promotion":
        script = _module("scripts/promote_v5_publication.py", "_split_wrong_root_promote")
        monkeypatch.setattr(script, "project_root", lambda: p["source"])
        with pytest.raises(UnsafeMutablePath):
            script.promote(
                candidate_path=p["payload"], v5_artifacts=p["artifacts"],
                comparison_path=p["state"] / "comparison.json", generation_manifest_path=p["manifest"],
                output_path=p["state"] / "reviewed.json",
                reviewed_at=datetime.fromisoformat(p["document"]["meta"]["data_as_of"]),
                publication_contract_directory=Path("publication/live"), state_root=p["source"],
            )
    else:
        script = _module("scripts/package_public_demo.py", "_split_wrong_root_package")
        monkeypatch.setattr(script, "project_root", lambda: p["source"])
        with pytest.raises(UnsafeMutablePath):
            script.package_public_dashboard(
                web_root=p["source"] / "web", payload_path=p["payload"],
                output_directory=p["state"] / "package", state_root=p["source"],
            )
    assert p["sentinel"].read_bytes() == b"old source sentinel\n"


@pytest.mark.parametrize("member", ["manifest_path", "payload_path", "selection_family_path", "artifact_directory"])
def test_split_audit_rejects_source_member_substitution(split_generation, monkeypatch, member):
    p = split_generation
    script = _module("scripts/audit_outputs.py", "_split_source_substitution_audit")
    monkeypatch.setattr(script, "PROJECT_ROOT", p["source"])
    arguments = {"manifest_path": p["manifest"], "payload_path": p["payload"],
                 "selection_family_path": p["selection"], "artifact_directory": p["artifacts"]}
    arguments[member] = generation_contract_path(
        arguments[member], source_root=p["source"], state_root=p["state"],
    )
    with pytest.raises(UnsafeMutablePath):
        script._audit_local_generation(**arguments, state_root=p["state"])


def test_split_audit_rejects_override_parent_symlink(split_generation, monkeypatch):
    p = split_generation
    script = _module("scripts/audit_outputs.py", "_split_symlink_audit")
    monkeypatch.setattr(script, "PROJECT_ROOT", p["source"])
    alias = p["state"] / "generation-alias"
    alias.symlink_to(p["payload"].parent, target_is_directory=True)
    with pytest.raises(UnsafeMutablePath, match="symbolic"):
        script._audit_local_generation(
            p["manifest"], payload_path=alias / p["payload"].name,
            selection_family_path=p["selection"], artifact_directory=p["artifacts"],
            state_root=p["state"],
        )


def test_real_split_root_cutover_audit_promotion_package_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    fixture_value = os.environ.get("REGIME_SPLIT_ROOT_FIXTURE_ROOT")
    if not fixture_value:
        pytest.skip("set REGIME_SPLIT_ROOT_FIXTURE_ROOT for the verified local no-fitting acceptance chain")
    fixture = Path(fixture_value).resolve(strict=True)
    original = fixture / "build/weekly-automation/generation-v5"
    original_artifacts = original / "artifacts"
    payload = json.loads((original / "regime-results.json").read_text())
    original_manifest = json.loads((original / "generation-manifest.json").read_text())
    assert payload["meta"]["mode"] == "live"
    assert payload["model"]["profile"] == "standard"
    source, state = tmp_path / "source", tmp_path / "state"
    source.mkdir()
    state.mkdir()
    for directory in ("config", "web"):
        shutil.copytree(ROOT / directory, source / directory)
    shutil.copy2(ROOT / "requirements-ci.lock", source / "requirements-ci.lock")
    preregistration = Path(payload["model"]["structural_preregistration"]["path"])
    (source / preregistration).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / preregistration, source / preregistration)
    baseline_relative = Path("artifacts/baselines/v4-20260821")
    shutil.copytree(fixture / baseline_relative, source / baseline_relative)
    baseline_payload = Path("publication/baselines/v4-20260821/regime-results.json")
    (source / baseline_payload).parent.mkdir(parents=True)
    shutil.copy2(ROOT / baseline_payload, source / baseline_payload)
    label_relative = default_label_spec_path().relative_to(ROOT)
    audit = _module("scripts/audit_outputs.py", "_split_real_audit")
    promote = _module("scripts/promote_v5_publication.py", "_split_real_promote")
    package = _module("scripts/package_public_demo.py", "_split_real_package")
    verify = _module("scripts/verify_public_package.py", "_split_real_verify")
    comparison = _module("scripts/compare_v5_to_frozen_v4.py", "_split_real_comparison")
    monkeypatch.setattr(cli, "project_root", lambda: source)
    monkeypatch.setattr(cli, "default_label_spec_path", lambda: source / label_relative)
    monkeypatch.setattr(integrity, "project_root", lambda: source)
    monkeypatch.setattr(audit, "PROJECT_ROOT", source)
    monkeypatch.setattr(promote, "project_root", lambda: source)
    monkeypatch.setattr(package, "project_root", lambda: source)
    monkeypatch.setattr(comparison, "PROJECT_ROOT", source)
    monkeypatch.setattr(comparison.build_comparison, "__defaults__", (source / baseline_relative,))
    monkeypatch.setattr(promote, "_comparison_builder_module", lambda: comparison)

    def copy_completed_support(_benchmark, destination, *, generation_id, **_kwargs):
        assert generation_id == payload["meta"]["generation_id"]
        shutil.copytree(original_artifacts, destination, dirs_exist_ok=True)

    # Only artifact materialization uses the precomputed fixture. All payload,
    # manifest, reconstruction, audit, promotion and packaging checks stay real.
    monkeypatch.setattr(cli, "_write_supporting_results", copy_completed_support)
    run_relative = Path("build/weekly-automation/generation-v5")
    output = state / run_relative / "regime-results.json"
    artifacts = state / run_relative / "artifacts"
    sentinels = {}
    for relative in (run_relative / "regime-results.json", run_relative / "artifacts/selection-family-audit.json",
                     Path("publication/live/regime-results.json"), Path("publication/live/v5-vs-v4-comparison.json"),
                     Path("publication/live/selection-family-audit.json")):
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        sentinels[path] = b"stale source sentinel\n"
        path.write_bytes(sentinels[path])
    payload = deepcopy(payload)
    payload["meta"].pop("generation_manifest_sha256", None)
    replay = json.loads((original_artifacts / "research-replay-input.json").read_text())
    token = cli._CLI_STATE_ROOT.set(state)
    try:
        published = cli._publish_active_generation(
            payload, object(), output=output, artifacts=artifacts,
            input_snapshot_sha256=original_manifest["input_snapshot"]["sha256"],
            research_replay_input=replay,
        )
    finally:
        cli._CLI_STATE_ROOT.reset(token)
    manifest_path = output.with_name("generation-manifest.json")
    result = audit._audit_local_generation(
        manifest_path, payload_path=output,
        selection_family_path=artifacts / "selection-family-audit.json",
        artifact_directory=artifacts, state_root=state,
    )
    assert result["ok"] is True
    assert result["generation_id"] == published["meta"]["generation_id"]
    report = comparison.build_comparison(artifacts, v5_payload=output)
    comparison_path = state / "build/unreviewed-comparison.json"
    _write_json(comparison_path, report)
    reviewed = state / "build/reviewed/regime-results.json"
    review = promote.promote(
        candidate_path=output, v5_artifacts=artifacts,
        comparison_path=comparison_path, generation_manifest_path=manifest_path,
        output_path=reviewed, reviewed_at=datetime.fromisoformat(published["meta"]["generated_at"]),
        publication_contract_directory=Path("publication/live"), state_root=state,
    )
    assert review["ok"] is True
    reviewed_manifest = reviewed.with_name("generation-manifest.json")
    reviewed_comparison = reviewed.with_name("v5-vs-v4-comparison.json")
    reviewed_selection = reviewed.with_name("selection-family-audit.json")
    # A stale default artifact sibling must never shadow the explicit artifacts.
    (reviewed.parent / "artifacts").mkdir()
    (reviewed.parent / "artifacts/stale").write_text("unrelated sibling\n")
    reviewed_audit = audit._audit_local_generation(
        reviewed_manifest, payload_path=reviewed, comparison_path=reviewed_comparison,
        selection_family_path=reviewed_selection, artifact_directory=artifacts,
        state_root=state,
    )
    assert reviewed_audit["ok"] is True
    package_dir = state / "build/public-package"
    package.package_public_dashboard(
        web_root=source / "web", payload_path=reviewed, output_directory=package_dir,
        comparison_path=reviewed_comparison, generation_manifest_path=reviewed_manifest,
        selection_family_path=reviewed_selection, artifact_directory=artifacts,
        staged_generation_contract_directory=Path("publication/live"),
        publication_mode=package.PUBLICATION_MODE_LIVE_DERIVED,
        rights_acknowledged=True, state_root=state,
        forecast_enhancements_mode="required",
    )
    assert verify.verify_public_package(package_dir)["ok"] is True
    for path, before in sentinels.items():
        assert path.read_bytes() == before
    final_manifest = json.loads(reviewed_manifest.read_text())
    assert final_manifest["label_spec"]["registry_sha256"] == hashlib.sha256((source / label_relative).read_bytes()).hexdigest()
    assert final_manifest["runtime_fingerprint"] == integrity.build_runtime_fingerprint(source)
