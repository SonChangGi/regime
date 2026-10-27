"""Release entry points reject physical members outside their explicit state root."""

from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

from regime_lab.path_safety import UnsafeMutablePath


ROOT = Path(__file__).resolve().parents[1]


def _module(name):
    spec = importlib.util.spec_from_file_location(f"split_root_boundary_{name}", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PROMOTE = _module("promote_v5_publication")
AUDIT = _module("audit_outputs")
PACKAGE = _module("package_public_demo")


@pytest.fixture
def roots(tmp_path, monkeypatch):
    source, state, outside = (tmp_path / name for name in ("source", "state", "outside"))
    for directory in (source, state, outside):
        directory.mkdir()
    (source / "web").mkdir()
    monkeypatch.setattr(PROMOTE, "project_root", lambda: source)
    monkeypatch.setattr(PACKAGE, "project_root", lambda: source)
    monkeypatch.setattr(AUDIT, "PROJECT_ROOT", source)
    return source, state, outside


def _arguments(kind, source, state):
    if kind == "promote":
        return {
            "candidate_path": state / "run/payload.json", "v5_artifacts": state / "run/artifacts",
            "comparison_path": state / "run/comparison.json", "generation_manifest_path": state / "run/manifest.json",
            "output_path": state / "reviewed/payload.json", "output_comparison_path": state / "reviewed/comparison.json",
            "output_manifest_path": state / "reviewed/manifest.json", "output_selection_family_path": state / "reviewed/selection.json",
            "reviewed_at": datetime.now(timezone.utc), "state_root": state,
        }
    if kind == "audit":
        return {
            "manifest_path": state / "run/manifest.json", "payload_path": state / "run/payload.json",
            "artifact_directory": state / "run/artifacts", "comparison_path": state / "run/comparison.json",
            "selection_family_path": state / "run/selection.json", "state_root": state,
        }
    return {
        "web_root": source / "web", "payload_path": state / "run/payload.json",
        "output_directory": state / "package", "comparison_path": state / "run/comparison.json",
        "generation_manifest_path": state / "run/manifest.json", "selection_family_path": state / "run/selection.json",
        "forecast_enhancements_path": state / "run/enhancement.json", "artifact_directory": state / "run/artifacts",
        "state_root": state,
    }


def _invoke(kind, arguments):
    function = {"promote": PROMOTE.promote, "audit": AUDIT._audit_local_generation,
                "package": PACKAGE.package_public_dashboard}[kind]
    return function(**arguments)


@pytest.mark.parametrize("kind,field", [
    *(('promote', name) for name in ("candidate_path", "v5_artifacts", "comparison_path", "generation_manifest_path", "output_path", "output_comparison_path", "output_manifest_path", "output_selection_family_path")),
    *(('audit', name) for name in ("manifest_path", "payload_path", "artifact_directory", "comparison_path", "selection_family_path")),
    *(('package', name) for name in ("payload_path", "output_directory", "comparison_path", "generation_manifest_path", "selection_family_path", "forecast_enhancements_path", "artifact_directory")),
])
def test_release_members_cannot_use_an_unapproved_physical_root(roots, kind, field):
    source, state, outside = roots
    sentinel = outside / "member.json"
    sentinel.write_text("private sentinel", encoding="utf-8")
    arguments = _arguments(kind, source, state)
    arguments[field] = sentinel
    with pytest.raises(UnsafeMutablePath):
        _invoke(kind, arguments)
    assert sentinel.read_text(encoding="utf-8") == "private sentinel"
    assert list(state.iterdir()) == []


@pytest.mark.parametrize("kind,field", [("promote", "candidate_path"), ("audit", "manifest_path"), ("package", "payload_path")])
@pytest.mark.parametrize("alias_kind", ["inside_leaf", "outside_leaf", "dangling_leaf", "inside_parent"])
def test_release_members_reject_symlink_aliases_even_inside_state(roots, kind, field, alias_kind):
    source, state, outside = roots
    real = state / "real"
    real.mkdir()
    (real / "member.json").write_text("{}", encoding="utf-8")
    (outside / "member.json").write_text("{}", encoding="utf-8")
    alias = state / "alias"
    if alias_kind == "inside_parent":
        alias.symlink_to(real, target_is_directory=True)
        supplied = alias / "member.json"
    else:
        target = {"inside_leaf": real / "member.json", "outside_leaf": outside / "member.json",
                  "dangling_leaf": real / "absent.json"}[alias_kind]
        alias.symlink_to(target)
        supplied = alias
    arguments = _arguments(kind, source, state)
    arguments[field] = supplied
    with pytest.raises(UnsafeMutablePath):
        _invoke(kind, arguments)
    assert not (state / "package").exists()
    assert not (state / "reviewed").exists()


@pytest.mark.parametrize("missing", ["payload_path", "artifact_directory", "selection_family_path", "comparison_path"])
def test_split_root_audit_requires_explicit_bound_physical_members(roots, missing):
    source, state, _ = roots
    arguments = _arguments("audit", source, state)
    arguments["manifest_path"].parent.mkdir()
    arguments["manifest_path"].write_text(json.dumps({"comparison_sidecar": {"path": "run/comparison.json"},
                                                   "selection_family_sidecar": {"path": "run/selection.json"}}))
    arguments[missing] = None
    with pytest.raises(AUDIT.AuditFailure, match="requires explicit"):
        _invoke("audit", arguments)


def test_split_root_audit_rejects_source_contract_parent_alias(roots):
    source, state, _ = roots
    real = source / "real"
    real.mkdir()
    (source / "alias").symlink_to(real, target_is_directory=True)
    arguments = _arguments("audit", source, state)
    arguments["manifest_path"].parent.mkdir()
    arguments["manifest_path"].write_text(json.dumps({"payload": {"path": "alias/payload.json"}}))
    with pytest.raises(UnsafeMutablePath, match="symbolic-link"):
        _invoke("audit", arguments)


@pytest.mark.parametrize("output", ["run/artifacts/package", "run"])
def test_private_package_cannot_write_into_bound_generation_artifacts(roots, output):
    source, state, _ = roots
    arguments = _arguments("package", source, state)
    arguments["output_directory"] = state / output
    with pytest.raises(PACKAGE.PackagingError, match="must not overlap"):
        _invoke("package", arguments)
    assert list(state.iterdir()) == []


def test_private_package_requires_available_bound_artifacts(tmp_path):
    # Reuse the tracked public fixture; private source/cache material is not
    # copied into a portable regression fixture.
    state = tmp_path / "state"
    generation = state / "reviewed"
    generation.mkdir(parents=True)
    for name in ("regime-results.json", "v5-vs-v4-comparison.json",
                 "generation-manifest.json", "selection-family-audit.json"):
        shutil.copyfile(ROOT / "publication/live" / name, generation / name)
    with pytest.raises(PACKAGE.PackagingError, match="artifact directory is unavailable"):
        PACKAGE.package_public_dashboard(
            web_root=ROOT / "web", payload_path=generation / "regime-results.json",
            output_directory=state / "package", state_root=state,
            artifact_directory=state / "missing-artifacts",
            publication_mode=PACKAGE.PUBLICATION_MODE_LIVE_DERIVED,
            rights_acknowledged=True,
            staged_generation_contract_directory="publication/live",
        )
    assert not (state / "package").exists()
