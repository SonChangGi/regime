from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from regime_lab import cli
from regime_lab.analysis import BenchmarkProfile
from regime_lab.artifact_inventory import (
    ArtifactInventoryError,
    verify_artifact_inventory,
    write_artifact_inventory,
)


ROOT = Path(__file__).resolve().parents[1]
AUDIT_SPEC = importlib.util.spec_from_file_location(
    "artifact_inventory_audit_outputs",
    ROOT / "scripts" / "audit_outputs.py",
)
assert AUDIT_SPEC is not None and AUDIT_SPEC.loader is not None
audit_outputs = importlib.util.module_from_spec(AUDIT_SPEC)
AUDIT_SPEC.loader.exec_module(audit_outputs)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _benchmark() -> SimpleNamespace:
    return SimpleNamespace(
        leaderboard=pd.DataFrame({"model": ["markov"]}),
        predictions=pd.DataFrame({"model": ["markov"]}),
        split_audit=pd.DataFrame({"origin_date": []}),
        profile=BenchmarkProfile.quick(),
    )


def test_inventory_is_canonical_sorted_and_binds_the_exact_file_set(
    tmp_path: Path,
) -> None:
    b_raw = b"b\n"
    a_raw = b"a\n"
    (tmp_path / "b.csv").write_bytes(b_raw)
    (tmp_path / "a.json").write_bytes(a_raw)

    inventory = write_artifact_inventory(tmp_path)

    assert inventory.read_bytes() == (
        f"{_sha256(a_raw)}  a.json\n"
        f"{_sha256(b_raw)}  b.csv\n"
    ).encode("ascii")
    assert verify_artifact_inventory(tmp_path)["file_count"] == 2

    (tmp_path / "extra.csv").write_text("extra\n", encoding="utf-8")
    with pytest.raises(ArtifactInventoryError, match="file set differs"):
        verify_artifact_inventory(tmp_path)


def test_inventory_verification_rejects_hash_and_canonical_order_drift(
    tmp_path: Path,
) -> None:
    (tmp_path / "a.csv").write_text("a\n", encoding="utf-8")
    (tmp_path / "b.csv").write_text("b\n", encoding="utf-8")
    inventory = write_artifact_inventory(tmp_path)

    (tmp_path / "a.csv").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ArtifactInventoryError, match="hash does not match"):
        verify_artifact_inventory(tmp_path)

    (tmp_path / "a.csv").write_text("a\n", encoding="utf-8")
    lines = inventory.read_bytes().splitlines(keepends=True)
    inventory.write_bytes(b"".join(reversed(lines)))
    with pytest.raises(ArtifactInventoryError, match="not canonical"):
        verify_artifact_inventory(tmp_path)


def test_v5_staging_writer_adds_inventory_before_atomic_cutover(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "artifacts"
    output.mkdir()
    (output / "old.txt").write_text("old\n", encoding="utf-8")

    monkeypatch.setattr(
        cli,
        "write_artifact_inventory",
        lambda _directory: (_ for _ in ()).throw(RuntimeError("inventory failed")),
    )
    with pytest.raises(RuntimeError, match="inventory failed"):
        cli._write_supporting_results(
            _benchmark(),
            output,
            generation_id="new",
            write_inventory=True,
        )
    assert {path.name for path in output.iterdir()} == {"old.txt"}

    monkeypatch.undo()
    cli._write_supporting_results(
        _benchmark(),
        output,
        generation_id="new",
        write_inventory=True,
    )
    summary = verify_artifact_inventory(output)
    assert summary["file_count"] == 5
    assert "SHA256SUMS" not in (output / "SHA256SUMS").read_text(
        encoding="ascii"
    )


def test_v5_auditor_requires_a_valid_whole_generation_inventory(
    tmp_path: Path,
) -> None:
    (tmp_path / "artifact.csv").write_text("value\n1\n", encoding="utf-8")
    with pytest.raises(audit_outputs.AuditFailure, match="inventory failed"):
        audit_outputs.audit_v5_artifact_inventory(tmp_path)

    write_artifact_inventory(tmp_path)
    assert audit_outputs.audit_v5_artifact_inventory(tmp_path)["file_count"] == 1
