"""Provision verified immutable inputs into a dedicated automation checkout."""

from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

from regime_lab import frozen_v4
from regime_lab.path_safety import UnsafeMutablePath, confined_mutable_path


def _source_path(root: Path, value: str, *, label: str) -> Path:
    path = confined_mutable_path(value, project_directory=root, label=label)
    # The shared path guard also permits system temp. Provisioning specifically
    # owns only this source checkout, including in tests whose roots are temp.
    if path == root or not path.is_relative_to(root):
        raise UnsafeMutablePath(f"{label} must stay below the runtime source root")
    return path


def _verify_public_payload(path: Path, expected_sha256: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise frozen_v4.FrozenV4BaselineError("runtime frozen v4 public payload is missing or unsafe")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise frozen_v4.FrozenV4BaselineError("runtime frozen v4 public payload hash does not match")


def _rename_no_replace(source: Path, target: Path) -> None:
    """Atomically publish a directory without replacing even an empty target."""
    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        rename = library.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(os.fsencode(source), os.fsencode(target), 0x00000004)  # RENAME_EXCL
    elif sys.platform.startswith("linux") and hasattr(library, "renameat2"):
        rename = library.renameat2
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(-100, os.fsencode(source), -100, os.fsencode(target), 1)  # AT_FDCWD, RENAME_NOREPLACE
    else:
        raise RuntimeError("atomic non-replacing runtime input installation is unavailable")
    if result:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(target))


def prepare_runtime_inputs(
    source_root: str | Path,
    state_root: str | Path,
    contract: str = "v5",
) -> dict[str, Any]:
    """Copy a missing frozen baseline only after its full staged verification.

    Callers hold the normal automation locks. Existing baseline directories are
    verified in place and are never repaired, merged or overwritten here.
    """
    source, state = Path(source_root).resolve(), Path(state_root).resolve()
    receipt: dict[str, Any] = {
        "ok": True, "status": "not_required", "contract": contract,
        "source_root": str(source), "state_root": str(state),
    }
    if contract not in {"v4", "v5"}:
        raise ValueError("runtime input contract must be v4 or v5")
    if source == state or contract == "v4":
        return receipt

    pinned = frozen_v4.FROZEN_V4_BASELINE
    target = _source_path(source, str(pinned["artifacts_path"]), label="runtime frozen v4 artifacts")
    if target.exists():
        verified = frozen_v4.verify_frozen_v4_baseline(project_directory=source)
        return {**receipt, "status": "already_present",
                "artifacts_inventory_sha256": verified["artifacts_inventory_sha256"],
                "payload_sha256": verified["payload_sha256"]}

    verified = frozen_v4.verify_frozen_v4_baseline(project_directory=state)
    original = _source_path(state, str(verified["artifacts_path"]), label="existing frozen v4 artifacts")
    public_payload = _source_path(source, str(verified["payload_path"]), label="runtime frozen v4 public payload")
    _verify_public_payload(public_payload, str(verified["payload_sha256"]))
    build = _source_path(source, "build", label="runtime input staging directory")
    build.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".runtime-inputs-", dir=build) as temporary:
        staged = Path(temporary)
        staged_artifacts = staged / str(verified["artifacts_path"])
        staged_artifacts.parent.mkdir(parents=True)
        shutil.copytree(original, staged_artifacts, symlinks=True)
        staged_payload = staged / str(verified["payload_path"])
        staged_payload.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(public_payload, staged_payload, follow_symlinks=False)
        frozen_v4.verify_frozen_v4_baseline(project_directory=staged)
        _verify_public_payload(staged_payload, str(verified["payload_sha256"]))
        target = _source_path(source, str(verified["artifacts_path"]), label="runtime frozen v4 artifacts")
        if target.exists():
            raise FileExistsError(f"runtime frozen v4 target appeared during preparation: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        _rename_no_replace(staged_artifacts, target)
        frozen_v4.verify_frozen_v4_baseline(project_directory=source)
    return {**receipt, "status": "prepared",
            "artifacts_inventory_sha256": verified["artifacts_inventory_sha256"],
            "payload_sha256": verified["payload_sha256"]}


__all__ = ["prepare_runtime_inputs"]
