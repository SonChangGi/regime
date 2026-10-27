"""Split-root builds must retain the V4 and operational write boundaries."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from regime_lab import cli, path_safety


V4_TARGETS = (
    "web/data/regime-results.json", "artifacts/latest", "artifacts/demo",
    "publication/live/regime-results.json",
    "build/weekly-automation/generation/regime-results.json",
    "build/weekly-automation/generation/artifacts",
)
H10_TARGETS = V4_TARGETS + (
    "build/weekly-automation", "build/v5-live/regime-results.json",
    "build/v5-live/artifacts", "build/v5-demo/regime-results.json",
    "build/v5-demo/artifacts",
)
OFR_TARGETS = (
    "data/regime.sqlite3", "build/weekly-automation", "build/v5-live",
    "build/v5-demo", "publication/live", "web", "artifacts/latest", "artifacts/demo",
)


class ConfigReached(RuntimeError):
    """Stop a valid command before any config, provider, model or DB work."""


def _inventory(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


@pytest.fixture
def roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source, state = tmp_path / "source", tmp_path / "state"
    for root in (source, state):
        root.mkdir()
        for relative in V4_TARGETS:
            target = root / relative
            if target.suffix == ".json":
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(f"preserved V4 payload at {root.name}/{relative}\n".encode())
            else:
                target.mkdir(parents=True, exist_ok=True)
                (target / "preserved.csv").write_bytes(b"existing V4 evidence\n")
        (root / "data").mkdir()
        (root / "data/regime.sqlite3").write_bytes(b"existing operational SQLite bytes")
    monkeypatch.setattr(cli, "project_root", lambda: source)
    # Permit absolute paths in either temporary root so the reserved-path guard
    # itself must reject source paths, independently of ordinary confinement.
    monkeypatch.setattr(path_safety.tempfile, "gettempdir", lambda: str(tmp_path))
    calls = []

    def load_config(path):
        calls.append(path)
        raise ConfigReached("stopped before config, providers, models or SQLite")

    monkeypatch.setattr(cli, "load_config", load_config)
    token = cli._CLI_STATE_ROOT.set(state)
    before = _inventory(source), _inventory(state)
    try:
        yield source, state, calls
    finally:
        cli._CLI_STATE_ROOT.reset(token)
        assert (_inventory(source), _inventory(state)) == before


def _overlap(target: Path, relationship: str) -> Path:
    if relationship == "ancestor":
        return target.parent
    if relationship == "descendant":
        return target / "nested-result.json"
    return target


def test_v5_cli_rejects_state_v4_payload_before_loading_config(roots) -> None:
    _, state, calls = roots
    with pytest.raises(ValueError, match="v4-owned target"):
        cli.main(["build", "--state-root", str(state), "--contract", "v5",
                  "--output", "web/data/regime-results.json"])
    assert calls == []


@pytest.mark.parametrize("root_name", ["source", "state"])
@pytest.mark.parametrize("relative", V4_TARGETS)
@pytest.mark.parametrize("relationship", ["equal", "ancestor", "descendant"])
@pytest.mark.parametrize("option", ["--output", "--artifacts", "--checkpoint-directory"])
def test_v5_cli_rejects_all_v4_overlaps_in_both_roots_before_config(
    roots, root_name: str, relative: str, relationship: str, option: str,
) -> None:
    source, state, calls = roots
    root = source if root_name == "source" else state
    candidate = _overlap(root / relative, relationship)
    with pytest.raises(ValueError, match="[vV]4-owned target"):
        cli.main(["build", "--state-root", str(state), "--contract", "v5", option, str(candidate)])
    assert calls == []


def test_isolated_v5_defaults_and_explicit_targets_reach_config_without_mutation(roots) -> None:
    _, state, calls = roots
    for extra in ([], ["--output", "build/reviewed-v5/result.json",
                       "--artifacts", "build/reviewed-v5/artifacts",
                       "--checkpoint-directory", "build/reviewed-v5/checkpoints"]):
        with pytest.raises(ConfigReached, match="stopped before config"):
            cli.main(["build", "--state-root", str(state), "--contract", "v5", *extra])
    assert len(calls) == 2
    output, artifacts = cli._resolve_contract_write_targets(
        command="build", contract_version="v5", output=None, artifacts=None)
    assert output == state / "build/v5-live/regime-results.json"
    assert artifacts == state / "build/v5-live/artifacts"


@pytest.mark.parametrize("split", [False, True])
def test_v4_legacy_default_targets_remain_allowed(roots, split: bool) -> None:
    source, state, calls = roots
    arguments = ["build", "--contract", "v4"] + (["--state-root", str(state)] if split else [])
    with pytest.raises(ConfigReached, match="stopped before config"):
        cli.main(arguments)
    assert len(calls) == 1
    token = cli._CLI_STATE_ROOT.set(state if split else None)
    try:
        output, artifacts = cli._resolve_contract_write_targets(
            command="build", contract_version="v4", output=None, artifacts=None)
    finally:
        cli._CLI_STATE_ROOT.reset(token)
    selected = state if split else source
    assert output == selected / "web/data/regime-results.json"
    assert artifacts == selected / "artifacts/latest"


def test_v4_explicit_checkpoint_still_fails_before_config(roots) -> None:
    _, state, calls = roots
    with pytest.raises(ValueError, match="only for V5"):
        cli.main(["build", "--state-root", str(state), "--contract", "v4",
                  "--checkpoint-directory", "build/private-checkpoint"])
    assert calls == []


@pytest.mark.parametrize("root_name", ["source", "state"])
@pytest.mark.parametrize("relative", H10_TARGETS)
@pytest.mark.parametrize("relationship", ["equal", "ancestor", "descendant"])
def test_h10_receipt_guard_retains_both_roots_and_all_overlaps(
    roots, root_name: str, relative: str, relationship: str,
) -> None:
    source, state, calls = roots
    root = source if root_name == "source" else state
    with pytest.raises(ValueError, match="overlaps"):
        cli._resolve_h10_collection_receipt(
            _overlap(root / relative, relationship), database=state / "data/regime.sqlite3")
    assert calls == []


@pytest.mark.parametrize("root_name", ["source", "state"])
@pytest.mark.parametrize("relative", OFR_TARGETS)
@pytest.mark.parametrize("relationship", ["equal", "ancestor", "descendant"])
@pytest.mark.parametrize("field", ["database", "receipt"])
def test_ofr_targets_guard_retains_both_roots_and_all_overlaps(
    roots, root_name: str, relative: str, relationship: str, field: str,
) -> None:
    source, state, calls = roots
    root = source if root_name == "source" else state
    candidate = _overlap(root / relative, relationship)
    with pytest.raises(ValueError, match="overlaps an operating/public target"):
        cli._resolve_ofr_fsi_collection_targets(
            database_value=candidate if field == "database" else None,
            receipt_value=candidate if field == "receipt" else None,
        )
    assert calls == []


def test_private_h10_and_ofr_defaults_are_allowed_in_state_root(roots) -> None:
    _, state, calls = roots
    receipt = cli._resolve_h10_collection_receipt(None, database=state / "data/regime.sqlite3")
    assert receipt == state / "build/v5-h10/collection-receipt.json"
    database, receipt, lock = cli._resolve_ofr_fsi_collection_targets(database_value=None, receipt_value=None)
    assert database == state / "data/ofr-fsi-shadow.sqlite3"
    assert receipt == state / "build/v6-ofr-fsi/collection-receipt.json"
    assert lock == state / "data/ofr-fsi-shadow.sqlite3.ofr-fsi-collect.lock"
    assert calls == []
