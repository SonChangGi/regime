from __future__ import annotations

import hashlib
from pathlib import Path
import shutil

import pytest

from regime_lab import frozen_v4, runtime_inputs
from regime_lab.path_safety import UnsafeMutablePath
from test_frozen_v4 import _baseline_fixture


def inventory(root: Path) -> dict[str, tuple[str, int, int]]:
    return {str(path.relative_to(root)): (hashlib.sha256(path.read_bytes()).hexdigest(),
                path.stat().st_mtime_ns, path.stat().st_ino)
            for path in root.rglob("*") if path.is_file()}


@pytest.fixture
def runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    state, original, contract = _baseline_fixture(tmp_path / "existing", monkeypatch)
    (state / "editor.py").write_bytes(b"uncommitted user edit\n")
    (state / "data/automation").mkdir(parents=True)
    (state / "data/regime.sqlite3").write_bytes(b"unchanged database")
    (state / "data/automation/authorization.json").write_bytes(b"existing authorization")
    source = tmp_path / "runtime"
    payload = source / str(contract["payload_path"])
    payload.parent.mkdir(parents=True)
    shutil.copy2(original / "regime-results.json", payload)
    target = source / str(contract["artifacts_path"])
    return source, state, target


def test_preparation_copies_exact_verified_baseline_and_preserves_all_originals(runtime: tuple[Path, Path, Path]) -> None:
    source, state, target = runtime
    before = inventory(state)
    public = source / str(frozen_v4.FROZEN_V4_BASELINE["payload_path"])
    public_before = public.read_bytes(), public.stat().st_mtime_ns, public.stat().st_ino

    result = runtime_inputs.prepare_runtime_inputs(source, state)

    assert result["ok"] is True and result["status"] == "prepared"
    assert result["payload_sha256"] == frozen_v4.FROZEN_V4_BASELINE["payload_sha256"]
    assert frozen_v4.verify_frozen_v4_baseline(project_directory=source) == dict(frozen_v4.FROZEN_V4_BASELINE)
    original = state / str(frozen_v4.FROZEN_V4_BASELINE["artifacts_path"])
    assert {path.name: path.read_bytes() for path in target.iterdir()} == {
        path.name: path.read_bytes() for path in original.iterdir()}
    assert inventory(state) == before
    assert (public.read_bytes(), public.stat().st_mtime_ns, public.stat().st_ino) == public_before
    assert not list((source / "build").glob(".runtime-inputs-*"))


def test_existing_verified_target_is_idempotent_without_reading_or_copying_state(runtime: tuple[Path, Path, Path]) -> None:
    source, state, _ = runtime
    runtime_inputs.prepare_runtime_inputs(source, state)
    before = inventory(source)
    shutil.rmtree(state)

    result = runtime_inputs.prepare_runtime_inputs(source, state)

    assert result["status"] == "already_present"
    assert inventory(source) == before


@pytest.mark.parametrize("contract,same_root", [("v4", False), ("v5", True)])
def test_v4_and_single_checkout_are_noops_without_private_artifacts(
    tmp_path: Path, contract: str, same_root: bool,
) -> None:
    source, state = tmp_path / "source", tmp_path / "state"
    source.mkdir()
    state.mkdir()
    result = runtime_inputs.prepare_runtime_inputs(source, source if same_root else state, contract=contract)
    assert result["status"] == "not_required"
    assert list(source.iterdir()) == [] and list(state.iterdir()) == []


def test_corrupt_original_is_rejected_before_target_creation(runtime: tuple[Path, Path, Path]) -> None:
    source, state, target = runtime
    original = state / str(frozen_v4.FROZEN_V4_BASELINE["artifacts_path"])
    (original / "oos-predictions.csv").write_text("mutated historical data\n")
    before = inventory(state)
    with pytest.raises(frozen_v4.FrozenV4BaselineError, match="artifact hash"):
        runtime_inputs.prepare_runtime_inputs(source, state)
    assert not target.exists()
    assert not (source / "build").exists()
    assert inventory(state) == before


def test_corrupt_existing_target_is_never_repaired_or_overwritten(runtime: tuple[Path, Path, Path]) -> None:
    source, state, target = runtime
    target.mkdir(parents=True)
    (target / "preserve-me.txt").write_text("invalid but user-owned target")
    before = inventory(source), inventory(state)
    with pytest.raises(frozen_v4.FrozenV4BaselineError):
        runtime_inputs.prepare_runtime_inputs(source, state)
    assert (inventory(source), inventory(state)) == before


def test_corrupt_source_public_payload_is_rejected_without_installing(runtime: tuple[Path, Path, Path]) -> None:
    source, state, target = runtime
    public = source / str(frozen_v4.FROZEN_V4_BASELINE["payload_path"])
    public.write_text("wrong published baseline")
    before = inventory(source), inventory(state)
    with pytest.raises(frozen_v4.FrozenV4BaselineError, match="public payload hash"):
        runtime_inputs.prepare_runtime_inputs(source, state)
    assert not target.exists()
    assert (inventory(source), inventory(state)) == before


def test_partial_staging_failure_never_exposes_target(
    runtime: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, state, target = runtime
    before = inventory(state)

    def incomplete(original: Path, destination: Path, **kwargs: object) -> None:
        destination.mkdir()
        (destination / "partial.csv").write_text("partial copy")
        assert not target.exists()
        raise OSError("copy interrupted")

    monkeypatch.setattr(runtime_inputs.shutil, "copytree", incomplete)
    with pytest.raises(OSError, match="copy interrupted"):
        runtime_inputs.prepare_runtime_inputs(source, state)
    assert not target.exists()
    assert not list((source / "build").glob(".runtime-inputs-*"))
    assert inventory(state) == before


def test_staging_is_reverified_before_atomic_visibility(
    runtime: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, state, target = runtime
    copy = shutil.copytree

    def corrupt(original: Path, destination: Path, **kwargs: object) -> Path:
        result = copy(original, destination, **kwargs)
        (destination / "oos-predictions.csv").write_text("corrupt staged bytes")
        assert not target.exists()
        return result

    monkeypatch.setattr(runtime_inputs.shutil, "copytree", corrupt)
    with pytest.raises(frozen_v4.FrozenV4BaselineError, match="artifact hash"):
        runtime_inputs.prepare_runtime_inputs(source, state)
    assert not target.exists()
    assert not list((source / "build").glob(".runtime-inputs-*"))


@pytest.mark.parametrize("alias", ["artifacts", "build", "publication"])
def test_symlink_paths_are_rejected_without_writing_through_them(
    runtime: tuple[Path, Path, Path], tmp_path: Path, alias: str,
) -> None:
    source, state, target = runtime
    destination = tmp_path / "do-not-modify"
    destination.mkdir()
    selected = source / alias
    if selected.exists():
        shutil.rmtree(selected)
    selected.symlink_to(destination, target_is_directory=True)
    before = inventory(state)
    with pytest.raises(UnsafeMutablePath):
        runtime_inputs.prepare_runtime_inputs(source, state)
    assert list(destination.iterdir()) == []
    assert inventory(state) == before


def test_target_that_appears_during_staging_is_not_overwritten(
    runtime: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, state, target = runtime
    copy = shutil.copytree

    def concurrent(original: Path, destination: Path, **kwargs: object) -> Path:
        result = copy(original, destination, **kwargs)
        target.mkdir(parents=True)
        (target / "preserve.txt").write_text("new user target")
        return result

    monkeypatch.setattr(runtime_inputs.shutil, "copytree", concurrent)
    with pytest.raises(FileExistsError, match="appeared during preparation"):
        runtime_inputs.prepare_runtime_inputs(source, state)
    assert (target / "preserve.txt").read_text() == "new user target"
    assert not list((source / "build").glob(".runtime-inputs-*"))


def test_empty_target_created_after_last_check_is_not_replaced(
    runtime: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, state, target = runtime
    rename = runtime_inputs._rename_no_replace
    target_inode = None

    def concurrent(staged: Path, destination: Path) -> None:
        nonlocal target_inode
        destination.mkdir()
        target_inode = destination.stat().st_ino
        rename(staged, destination)

    monkeypatch.setattr(runtime_inputs, "_rename_no_replace", concurrent)
    with pytest.raises(FileExistsError):
        runtime_inputs.prepare_runtime_inputs(source, state)
    assert target.stat().st_ino == target_inode
    assert list(target.iterdir()) == []
    assert not list((source / "build").glob(".runtime-inputs-*"))
