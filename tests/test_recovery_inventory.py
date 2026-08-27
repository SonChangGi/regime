from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3

import pytest

from regime_lab.database_backup import create_database_backup
from regime_lab.recovery_inventory import (
    RecoveryInventoryError,
    build_recovery_inventory,
    run_restore_drill,
)
from regime_lab.recovery_policy import (
    RecoveryPolicy,
    RecoveryPolicyError,
    load_recovery_policy,
)


def _database(path: Path) -> Path:
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE snapshots (
                snapshot_id TEXT PRIMARY KEY,
                retrieved_at TEXT NOT NULL
            );
            CREATE TABLE observations (
                snapshot_id TEXT NOT NULL,
                series_id TEXT NOT NULL,
                value REAL,
                FOREIGN KEY (snapshot_id) REFERENCES snapshots(snapshot_id)
            );
            INSERT INTO snapshots VALUES ('one', '2026-08-27T00:00:00Z');
            INSERT INTO observations VALUES ('one', 'SPY', 1.0);
            """
        )
    return path


def _policy(**overrides: int | bool) -> RecoveryPolicy:
    values: dict[str, int | bool] = {
        "automatic_deletion": False,
        "retention_valid_generations": 1,
        "max_source_bytes": 1024 * 1024,
        "max_total_backup_bytes": 16 * 1024 * 1024,
        "minimum_free_bytes_after_backup": 1,
        "max_backup_seconds": 30,
        "minimum_assumed_bytes_per_second": 1024,
        "restore_drill_max_seconds": 30,
        "restore_drill_max_age_hours": 24,
    }
    values.update(overrides)
    return RecoveryPolicy(**values)  # type: ignore[arg-type]


def test_repository_recovery_policy_forbids_automatic_deletion() -> None:
    policy = load_recovery_policy()

    assert policy.automatic_deletion is False
    assert policy.retention_valid_generations == 4


def test_backup_preflight_refuses_byte_ceiling_before_generation_write(
    tmp_path: Path,
) -> None:
    source = _database(tmp_path / "source.sqlite3")
    backup_root = tmp_path / "backups"

    with pytest.raises(RecoveryPolicyError, match="max_source_bytes"):
        create_database_backup(
            source,
            backup_root,
            recovery_policy=_policy(max_source_bytes=1),
        )

    assert source.exists()
    assert not list(backup_root.glob("regime-snapshot-*"))


def test_inventory_classifies_preserved_recovery_material_and_restore_drills(
    tmp_path: Path,
) -> None:
    source = _database(tmp_path / "source.sqlite3")
    backup_root = tmp_path / "backups"
    first = create_database_backup(
        source,
        backup_root,
        retain=1,
        recovery_policy=_policy(),
        created_at=datetime(2026, 8, 25, tzinfo=timezone.utc),
    )
    second = create_database_backup(
        source,
        backup_root,
        retain=1,
        recovery_policy=_policy(),
        created_at=datetime(2026, 8, 26, tzinfo=timezone.utc),
    )
    legacy = create_database_backup(
        source,
        backup_root,
        retain=1,
        recovery_policy=_policy(),
        created_at=datetime(2026, 8, 23, tzinfo=timezone.utc),
    )
    legacy_manifest = json.loads(legacy.manifest_path.read_text(encoding="utf-8"))
    legacy_manifest["schema_version"] = "regime.database-backup.v1"
    legacy.manifest_path.write_text(json.dumps(legacy_manifest), encoding="utf-8")
    os.chmod(legacy.manifest_path, 0o600)
    corrupt = create_database_backup(
        source,
        backup_root,
        retain=1,
        recovery_policy=_policy(),
        created_at=datetime(2026, 8, 24, tzinfo=timezone.utc),
    )
    corrupt.manifest_path.write_text("{broken", encoding="utf-8")
    os.chmod(corrupt.manifest_path, 0o600)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "manifest.json").write_text("{}", encoding="utf-8")
    preview = tmp_path / "preview.json"
    preview.write_text("{}", encoding="utf-8")

    inventory = build_recovery_inventory(
        backup_root,
        checkpoint_paths=[checkpoint],
        preview_paths=[preview],
        policy=_policy(),
        generated_at=datetime(2026, 8, 27, tzinfo=timezone.utc),
    )

    assert inventory["summary"]["counts"] == {
        "valid-current": 2,
        "legacy": 1,
        "corrupt": 1,
        "checkpoint": 1,
        "preview": 1,
    }
    assert inventory["retention"] == {
        "automatic_deletion": False,
        "target_valid_generations": 1,
        "valid_generations": 2,
        "over_target_generations": 1,
        "action": "review_only_no_automatic_deletion",
    }
    assert all(path.exists() for path in (first.path, second.path, legacy.path, corrupt.path))

    drill = run_restore_drill(
        backup_root,
        policy=_policy(),
        performed_at=datetime(2026, 8, 27, tzinfo=timezone.utc),
    )
    assert drill["generation_id"] == second.generation_id
    assert drill["within_time_ceiling"] is True
    assert drill["capacity_preflight"]["projected_bytes"] > 0
    assert drill["verification"]["quick_check"] == "ok"
    assert drill["verification"]["integrity_check"] == "ok"
    assert drill["verification"]["foreign_key_check"] == "ok"

    with pytest.raises(RecoveryInventoryError, match="estimated time ceiling"):
        run_restore_drill(
            backup_root,
            generation_path=second.path,
            policy=_policy(
                minimum_assumed_bytes_per_second=1,
                restore_drill_max_seconds=1,
            ),
        )
