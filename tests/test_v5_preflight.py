from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3

import pytest

from regime_lab import v5_preflight


def _project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    (root / "config").mkdir(parents=True)
    (root / "src" / "regime_lab").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    (root / "config" / "series.json").write_text("{}\n")
    (root / "config" / "structural_v4.json").write_text("{}\n")
    (root / "config" / "structural_v5.json").write_text("{\"v\":5}\n")
    (root / "src" / "regime_lab" / "model.py").write_text("VALUE = 1\n")
    return root


def test_source_fingerprint_changes_with_analysis_source(tmp_path: Path) -> None:
    root = _project(tmp_path)
    before, count = v5_preflight.v5_analysis_source_fingerprint(
        project_directory=root,
    )

    (root / "src" / "regime_lab" / "model.py").write_text("VALUE = 2\n")
    after, after_count = v5_preflight.v5_analysis_source_fingerprint(
        project_directory=root,
    )

    assert count == after_count == 5
    assert before != after


def test_source_fingerprint_ignores_build_outputs(tmp_path: Path) -> None:
    root = _project(tmp_path)
    before, _ = v5_preflight.v5_analysis_source_fingerprint(
        project_directory=root,
    )
    (root / "build").mkdir()
    (root / "build" / "candidate.json").write_text("changed")

    after, _ = v5_preflight.v5_analysis_source_fingerprint(
        project_directory=root,
    )

    assert after == before


def test_source_fingerprint_binds_effective_config_and_rights_policy(
    tmp_path: Path,
) -> None:
    root = _project(tmp_path)
    policy = root / "config/custom-rights.json"
    policy.write_text('{"schema_version":1}', encoding="utf-8")
    config = {
        "provider_rights_policy": "config/custom-rights.json",
        "model": {"final_holdout_start": "2023-01-01"},
    }
    before, count = v5_preflight.v5_analysis_source_fingerprint(
        project_directory=root,
        config=config,
    )
    policy.write_text('{"schema_version":1,"changed":true}', encoding="utf-8")
    after, after_count = v5_preflight.v5_analysis_source_fingerprint(
        project_directory=root,
        config=config,
    )

    assert count == after_count == 6
    assert before != after


def test_source_stability_guard_fails_closed(tmp_path: Path) -> None:
    root = _project(tmp_path)
    expected, _ = v5_preflight.v5_analysis_source_fingerprint(
        project_directory=root,
    )
    (root / "config" / "series.json").write_text('{"changed":true}\n')

    with pytest.raises(v5_preflight.V5PreflightError, match="changed during"):
        v5_preflight.require_v5_analysis_source_unchanged(
            expected,
            project_directory=root,
        )


def test_structural_preregistration_hash_is_checked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _project(tmp_path)
    expected = hashlib.sha256(
        (root / "config" / "structural_v5.json").read_bytes()
    ).hexdigest()
    monkeypatch.setattr(
        v5_preflight,
        "STRUCTURAL_V5_PREREGISTRATION_SHA256",
        expected,
    )
    assert v5_preflight.verify_structural_v5_preregistration(
        project_directory=root,
    ) == expected

    (root / "config" / "structural_v5.json").write_text("{}\n")
    with pytest.raises(v5_preflight.V5PreflightError, match="SHA-256"):
        v5_preflight.verify_structural_v5_preregistration(
            project_directory=root,
        )


def test_database_quick_check_rejects_corrupt_file(tmp_path: Path) -> None:
    database = tmp_path / "corrupt.sqlite3"
    database.write_bytes(b"not sqlite")

    with pytest.raises(v5_preflight.V5PreflightError, match="quick_check"):
        v5_preflight._database_quick_check(database)


def test_database_quick_check_accepts_database_and_missing_path(tmp_path: Path) -> None:
    database = tmp_path / "ok.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE sample (value INTEGER)")

    assert v5_preflight._database_quick_check(database) == "ok"
    assert v5_preflight._database_quick_check(tmp_path / "missing.sqlite3") == "not_present"


def test_full_profile_requires_hmm_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        v5_preflight.importlib.util,
        "find_spec",
        lambda name: None if name == "hmmlearn" else object(),
    )

    with pytest.raises(v5_preflight.V5PreflightError, match="hmmlearn"):
        v5_preflight._require_profile_dependencies("full")


def test_preflight_receipt_is_secret_free_and_hash_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _project(tmp_path)
    preregistration_sha = hashlib.sha256(
        (root / "config" / "structural_v5.json").read_bytes()
    ).hexdigest()
    monkeypatch.setattr(
        v5_preflight,
        "STRUCTURAL_V5_PREREGISTRATION_SHA256",
        preregistration_sha,
    )
    monkeypatch.setattr(
        v5_preflight,
        "verify_frozen_v4_baseline",
        lambda **_kwargs: {"payload_sha256": "a" * 64},
    )
    monkeypatch.setattr(
        v5_preflight.importlib.util,
        "find_spec",
        lambda _name: object(),
    )

    receipt = v5_preflight.verify_v5_preflight(
        profile="standard",
        database_path=tmp_path / "missing.sqlite3",
        project_directory=root,
    )

    assert receipt["profile"] == "standard"
    assert receipt["structural_v5_sha256"] == preregistration_sha
    assert receipt["frozen_v4_payload_sha256"] == "a" * 64
    assert receipt["database_quick_check"] == "not_present"
    assert len(receipt["source_fingerprint_sha256"]) == 64
    assert "secret" not in json.dumps(receipt).lower()
