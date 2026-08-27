from __future__ import annotations

from pathlib import Path

import pytest

from regime_lab.runtime_fingerprint import (
    RuntimeFingerprintError,
    build_runtime_fingerprint,
    validate_runtime_fingerprint,
)


def test_runtime_fingerprint_binds_lock_runtime_and_packages(tmp_path: Path) -> None:
    (tmp_path / "requirements-ci.lock").write_text(
        "numpy==2.3.0 --hash=sha256:" + "a" * 64 + "\n",
        encoding="utf-8",
    )

    first = build_runtime_fingerprint(tmp_path)
    second = build_runtime_fingerprint(tmp_path)

    assert first == second
    assert validate_runtime_fingerprint(first) == first["sha256"]
    assert first["lockfile"]["path"] == "requirements-ci.lock"
    assert "scikit-learn" in first["packages"]


def test_runtime_fingerprint_changes_with_lockfile(tmp_path: Path) -> None:
    lockfile = tmp_path / "requirements-ci.lock"
    lockfile.write_text("first\n", encoding="utf-8")
    first = build_runtime_fingerprint(tmp_path)
    lockfile.write_text("second\n", encoding="utf-8")
    second = build_runtime_fingerprint(tmp_path)

    assert first["sha256"] != second["sha256"]


def test_runtime_fingerprint_fails_closed_without_lock(tmp_path: Path) -> None:
    with pytest.raises(RuntimeFingerprintError, match="locked runtime file"):
        build_runtime_fingerprint(tmp_path)


def test_runtime_fingerprint_rejects_tampering(tmp_path: Path) -> None:
    (tmp_path / "requirements-ci.lock").write_text("locked\n", encoding="utf-8")
    fingerprint = build_runtime_fingerprint(tmp_path)
    fingerprint["python"] = "0.0.0"

    with pytest.raises(RuntimeFingerprintError, match="checksum mismatch"):
        validate_runtime_fingerprint(fingerprint)
