"""Keep scheduled readback compatible with the actual Pages packager."""

from dataclasses import replace
from datetime import datetime, timedelta
import hashlib
import importlib.util
import json
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from regime_lab import automation


ROOT = Path(__file__).resolve().parents[1]
EXPORTS = (
    "data/forecast_research.json",
    "data/calibration_audit.json",
    "data/forecast_information.json",
)


@pytest.fixture(scope="module")
def released_package(tmp_path_factory):
    spec = importlib.util.spec_from_file_location(
        "automation_export_packager", ROOT / "scripts/package_public_demo.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path_factory.mktemp("automation-exports") / "package"
    live = ROOT / "publication/live"
    with pytest.MonkeyPatch.context() as patch:
        # Provider permission policy is independent of this package/readback contract.
        patch.setattr(module, "verify_provider_rights", lambda *args, **kwargs: None)
        module.package_public_dashboard(
            web_root=ROOT / "web",
            payload_path=live / "regime-results.json",
            comparison_path=live / "v5-vs-v4-comparison.json",
            generation_manifest_path=live / "generation-manifest.json",
            selection_family_path=live / "selection-family-audit.json",
            publication_mode=module.PUBLICATION_MODE_LIVE_DERIVED,
            rights_acknowledged=True,
            output_directory=output,
        )
    files = {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in output.rglob("*") if path.is_file()
    }
    assert all(name in files for name in EXPORTS)
    return files


def _readback(files):
    settings = automation.AutomationSettings.load()
    kwargs = {
        "expected_payload": files[automation.PUBLIC_PAYLOAD_PATH],
        "expected_comparison": files[automation.PUBLIC_COMPARISON_PATH],
        "expected_generation_manifest": files[automation.PUBLIC_GENERATION_MANIFEST_PATH],
        "expected_selection_family": files[automation.PUBLIC_SELECTION_FAMILY_PATH],
    }

    def fetch(url):
        prefix = urlsplit(settings.public_root).path
        path = urlsplit(url).path
        assert path.startswith(prefix)
        return files[path[len(prefix):]]

    return settings, kwargs, fetch


def test_scheduled_readback_accepts_actual_pages_package(released_package):
    settings, kwargs, fetch = _readback(released_package)
    automation.verify_public_readback(settings, **kwargs, fetch=fetch)


@pytest.mark.parametrize("name", EXPORTS)
@pytest.mark.parametrize("fault", ["missing", "changed", "manifest_hash", "manifest_size"])
def test_scheduled_readback_rejects_download_damage(released_package, name, fault):
    files = dict(released_package)
    manifest = json.loads(files[automation.PUBLIC_MANIFEST_PATH])
    if fault == "missing":
        del files[name]
        del manifest["files"][name]
        message = "file inventory is not exact"
    elif fault == "changed":
        block = json.loads(files[name])
        block["tampered"] = True
        files[name] = json.dumps(block).encode()
        # A matching public manifest cannot legitimize a different reviewed result.
        manifest["files"][name] = {
            "bytes": len(files[name]), "sha256": hashlib.sha256(files[name]).hexdigest()
        }
        message = "does not match the expected generation"
    elif fault == "manifest_hash":
        manifest["files"][name]["sha256"] = "0" * 64
        message = "SHA-256 is incorrect"
    else:
        manifest["files"][name]["bytes"] += 1
        message = "byte count is incorrect"
    files[automation.PUBLIC_MANIFEST_PATH] = json.dumps(manifest).encode()
    settings, kwargs, fetch = _readback(files)
    with pytest.raises(automation.AutomationError, match=message):
        automation.verify_public_readback(settings, **kwargs, fetch=fetch)


def test_scheduled_readback_still_rejects_unexpected_files(released_package):
    files = dict(released_package)
    manifest = json.loads(files[automation.PUBLIC_MANIFEST_PATH])
    manifest["files"]["data/unreviewed.json"] = {"bytes": 2, "sha256": hashlib.sha256(b"{}").hexdigest()}
    files[automation.PUBLIC_MANIFEST_PATH] = json.dumps(manifest).encode()
    settings, kwargs, fetch = _readback(files)
    with pytest.raises(automation.AutomationError, match="file inventory is not exact"):
        automation.verify_public_readback(settings, **kwargs, fetch=fetch)


def test_current_research_release_does_not_request_redeployment(
    released_package, tmp_path, monkeypatch
):
    settings, kwargs, fetch = _readback(released_package)
    settings = replace(settings, state_directory=tmp_path)
    target = datetime.fromisoformat(json.loads(kwargs["expected_payload"])["meta"]["data_as_of"])
    remote = automation.RemotePublication(
        head_sha="a" * 40,
        payload_bytes=kwargs["expected_payload"],
        data_as_of=target,
        comparison_bytes=kwargs["expected_comparison"],
        generation_manifest_bytes=kwargs["expected_generation_manifest"],
        selection_family_bytes=kwargs["expected_selection_family"],
    )
    monkeypatch.setattr(automation, "_git_preflight", lambda *args: remote)
    real_readback = automation.verify_public_readback
    monkeypatch.setattr(
        automation, "verify_public_readback",
        lambda settings, **kwargs: real_readback(settings, **kwargs, fetch=fetch),
    )

    def unnecessary_action(*args, **kwargs):
        raise AssertionError("a verified current release must not build or republish")

    monkeypatch.setattr(automation, "_validate_local_authorization", unnecessary_action)
    monkeypatch.setattr(automation, "_build_candidate", unnecessary_action)
    monkeypatch.setattr(automation, "_publish_candidate", unnecessary_action)
    monkeypatch.setattr(automation, "_notify_status_best_effort", lambda *args, **kwargs: None)
    result = automation.run_weekly_release(settings, now=target + timedelta(days=2))
    assert result["status"] == "succeeded"
    assert result["stage"] == "already_current"
