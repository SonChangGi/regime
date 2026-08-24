from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sqlite3
import stat

import pytest

from regime_lab import database_backup
from regime_lab.database_backup import (
    DatabaseBackupIntegrityError,
    DatabaseBackupPathError,
    create_database_backup,
    list_database_backups,
    verify_database_restore,
)


def _snapshot_database(path: Path, *, observation_count: int = 3) -> Path:
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            PRAGMA user_version = 5;
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
            CREATE INDEX observations_series_idx ON observations(series_id);
            INSERT INTO snapshots VALUES ('snapshot-1', '2026-08-24T00:00:00Z');
            """
        )
        connection.executemany(
            "INSERT INTO observations VALUES ('snapshot-1', ?, ?)",
            [(f"series-{index}", float(index)) for index in range(observation_count)],
        )
    return path


def _count(path: Path, table: str) -> int:
    with sqlite3.connect(path) as connection:
        return int(connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat(follow_symlinks=False).st_mode)


def test_online_backup_is_private_hash_bound_and_restore_verified(tmp_path: Path) -> None:
    source = _snapshot_database(tmp_path / "source.sqlite3")
    # Keep a WAL connection open to exercise SQLite's online backup path rather
    # than relying on a closed-file byte copy.
    writer = sqlite3.connect(source)
    writer.execute("PRAGMA journal_mode = WAL")
    writer.execute(
        "INSERT INTO observations VALUES ('snapshot-1', 'committed', 10.0)"
    )
    writer.commit()
    try:
        generation = create_database_backup(
            source,
            tmp_path / "backups",
            source_code_fingerprint_sha256="a" * 64,
            created_at=datetime(2026, 8, 24, 1, 2, 3, tzinfo=timezone.utc),
        )
    finally:
        writer.close()

    assert generation.path.exists()
    assert _mode(generation.path.parent) == 0o700
    assert _mode(generation.path) == 0o700
    assert _mode(generation.database_path) == 0o600
    assert _mode(generation.manifest_path) == 0o600
    assert _count(generation.database_path, "observations") == 4

    manifest = json.loads(generation.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "regime.database-backup.v2"
    assert len(manifest["backup"]["sha256"]) == 64
    assert manifest["backup"]["bytes"] == generation.database_path.stat().st_size
    assert len(manifest["source"]["fingerprint_sha256"]) == 64
    assert manifest["source"]["code_fingerprint_sha256"] == "a" * 64
    assert manifest["source"]["quick_check"] == "ok"
    assert manifest["source"]["integrity_check"] == "ok"
    assert manifest["source"]["foreign_key_check"] == "ok"
    assert manifest["sqlite"]["quick_check"] == "ok"
    assert manifest["sqlite"]["integrity_check"] == "ok"
    assert manifest["sqlite"]["foreign_key_check"] == "ok"
    assert manifest["sqlite"]["core_table_counts"] == {
        "observations": 4,
        "snapshots": 1,
    }
    # The local source path is fingerprinted rather than disclosed.
    assert str(source) not in generation.manifest_path.read_text(encoding="utf-8")

    restored = verify_database_restore(generation.path)
    assert restored.generation_id == generation.generation_id
    assert restored.quick_check == restored.integrity_check == "ok"
    assert restored.foreign_key_check == "ok"
    assert restored.core_table_counts == {"snapshots": 1, "observations": 4}


def test_corrupt_source_is_rejected_before_generation_commit(tmp_path: Path) -> None:
    source = tmp_path / "corrupt.sqlite3"
    source.write_bytes(b"not-a-sqlite-database")
    backup_root = tmp_path / "backups"

    with pytest.raises(DatabaseBackupIntegrityError, match="online backup"):
        create_database_backup(source, backup_root)

    assert backup_root.exists()


def test_orphan_foreign_key_is_rejected_before_generation_commit(
    tmp_path: Path,
) -> None:
    source = _snapshot_database(tmp_path / "orphan.sqlite3")
    with sqlite3.connect(source) as connection:
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute(
            "INSERT INTO observations VALUES ('missing-snapshot', 'orphan', 1.0)"
        )
        connection.commit()
    backup_root = tmp_path / "backups"

    with pytest.raises(DatabaseBackupIntegrityError, match="foreign_key_check"):
        create_database_backup(source, backup_root)

    assert list(backup_root.glob("regime-snapshot-*")) == []
    assert not [path for path in backup_root.iterdir() if path.name.startswith("regime-snapshot-")]
    assert not [path for path in backup_root.iterdir() if path.name.endswith(".tmp")]


def test_corrupt_backup_is_not_listed_or_restorable(tmp_path: Path) -> None:
    source = _snapshot_database(tmp_path / "source.sqlite3")
    generation = create_database_backup(source, tmp_path / "backups")
    with generation.database_path.open("r+b") as handle:
        handle.seek(0)
        handle.write(b"broken!!")

    with pytest.raises(DatabaseBackupIntegrityError, match="hash or byte count"):
        verify_database_restore(generation.path)
    assert list_database_backups(tmp_path / "backups") == ()
    # Corrupt evidence is preserved for diagnosis, never rotated as if valid.
    assert generation.path.exists()


def test_rotation_keeps_latest_three_valid_generations_and_source(tmp_path: Path) -> None:
    source = _snapshot_database(tmp_path / "source.sqlite3", observation_count=1)
    backup_root = tmp_path / "backups"
    start = datetime(2026, 8, 20, tzinfo=timezone.utc)
    created = []
    for offset in range(5):
        with sqlite3.connect(source) as connection:
            connection.execute(
                "INSERT INTO observations VALUES ('snapshot-1', ?, ?)",
                (f"new-{offset}", float(offset)),
            )
        created.append(
            create_database_backup(
                source,
                backup_root,
                retain=3,
                created_at=start + timedelta(seconds=offset),
            )
        )

    listed = list_database_backups(backup_root)
    assert [item.generation_id for item in listed] == [
        item.generation_id for item in reversed(created[-3:])
    ]
    assert not created[0].path.exists()
    assert not created[1].path.exists()
    assert all(item.path.exists() for item in created[-3:])
    assert source.exists()
    assert _count(source, "observations") == 6


def test_rotation_preserves_corrupt_old_generation(tmp_path: Path) -> None:
    source = _snapshot_database(tmp_path / "source.sqlite3")
    backup_root = tmp_path / "backups"
    start = datetime(2026, 8, 20, tzinfo=timezone.utc)
    first = create_database_backup(source, backup_root, retain=2, created_at=start)
    second = create_database_backup(
        source,
        backup_root,
        retain=2,
        created_at=start + timedelta(seconds=1),
    )
    first.manifest_path.write_text("{broken", encoding="utf-8")
    os.chmod(first.manifest_path, 0o600)
    third = create_database_backup(
        source,
        backup_root,
        retain=2,
        created_at=start + timedelta(seconds=2),
    )

    assert first.path.exists()
    assert second.path.exists()
    assert third.path.exists()
    assert [item.generation_id for item in list_database_backups(backup_root)] == [
        third.generation_id,
        second.generation_id,
    ]


def test_corrupt_newest_generation_does_not_consume_retention_slot(
    tmp_path: Path,
) -> None:
    source = _snapshot_database(tmp_path / "source.sqlite3")
    backup_root = tmp_path / "backups"
    start = datetime(2026, 8, 20, tzinfo=timezone.utc)
    first = create_database_backup(source, backup_root, retain=2, created_at=start)
    corrupt = create_database_backup(
        source,
        backup_root,
        retain=2,
        created_at=start + timedelta(days=1),
    )
    corrupt.manifest_path.write_text("{broken", encoding="utf-8")
    os.chmod(corrupt.manifest_path, 0o600)
    latest = create_database_backup(
        source,
        backup_root,
        retain=2,
        created_at=start + timedelta(days=2),
    )

    assert corrupt.path.exists()
    assert first.path.exists()
    assert latest.path.exists()
    assert [item.generation_id for item in list_database_backups(backup_root)] == [
        latest.generation_id,
        first.generation_id,
    ]


def test_permission_drift_is_preserved_without_consuming_retention_slot(
    tmp_path: Path,
) -> None:
    source = _snapshot_database(tmp_path / "source.sqlite3")
    backup_root = tmp_path / "backups"
    start = datetime(2026, 8, 20, tzinfo=timezone.utc)
    first = create_database_backup(source, backup_root, retain=2, created_at=start)
    drifted = create_database_backup(
        source,
        backup_root,
        retain=2,
        created_at=start + timedelta(days=1),
    )
    os.chmod(drifted.manifest_path, 0o644)
    latest = create_database_backup(
        source,
        backup_root,
        retain=2,
        created_at=start + timedelta(days=2),
    )

    assert drifted.path.exists()
    assert first.path.exists()
    assert latest.path.exists()
    assert [item.generation_id for item in list_database_backups(backup_root)] == [
        latest.generation_id,
        first.generation_id,
    ]


def test_rotation_prefers_distinct_days_over_same_day_retries(
    tmp_path: Path,
) -> None:
    source = _snapshot_database(tmp_path / "source.sqlite3")
    backup_root = tmp_path / "backups"
    start = datetime(2026, 8, 20, tzinfo=timezone.utc)
    older = [
        create_database_backup(
            source,
            backup_root,
            retain=3,
            created_at=start + timedelta(days=offset),
        )
        for offset in range(2)
    ]
    same_day = [
        create_database_backup(
            source,
            backup_root,
            retain=3,
            created_at=start + timedelta(days=2, seconds=offset),
        )
        for offset in range(3)
    ]

    listed_ids = {item.generation_id for item in list_database_backups(backup_root)}
    assert listed_ids == {
        older[0].generation_id,
        older[1].generation_id,
        same_day[-1].generation_id,
    }


def test_atomic_commit_failure_preserves_existing_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _snapshot_database(tmp_path / "source.sqlite3")
    backup_root = tmp_path / "backups"
    existing = create_database_backup(source, backup_root)

    def fail_replace(_source: object, _target: object) -> None:
        raise OSError("injected atomic rename failure")

    monkeypatch.setattr(database_backup.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected"):
        create_database_backup(source, backup_root)

    assert existing.path.exists()
    assert verify_database_restore(existing).quick_check == "ok"
    assert {entry.name for entry in backup_root.iterdir()} == {existing.generation_id}


def test_symlink_paths_fail_closed(tmp_path: Path) -> None:
    source = _snapshot_database(tmp_path / "source.sqlite3")
    source_link = tmp_path / "source-link.sqlite3"
    source_link.symlink_to(source)
    with pytest.raises(DatabaseBackupPathError, match="symlink"):
        create_database_backup(source_link, tmp_path / "backups-a")

    real_root = tmp_path / "real-backups"
    real_root.mkdir()
    linked_root = tmp_path / "linked-backups"
    linked_root.symlink_to(real_root, target_is_directory=True)
    with pytest.raises(DatabaseBackupPathError, match="symlink"):
        create_database_backup(source, linked_root)

    safe_root = tmp_path / "backups-b"
    safe_root.mkdir()
    malicious_name = "regime-snapshot-20260824T010203000000Z-aaaaaaaaaaaa"
    (safe_root / malicious_name).symlink_to(real_root, target_is_directory=True)
    with pytest.raises(DatabaseBackupPathError, match="symlink"):
        create_database_backup(source, safe_root)


def test_trial_restore_detects_manifest_count_tampering(tmp_path: Path) -> None:
    source = _snapshot_database(tmp_path / "source.sqlite3")
    generation = create_database_backup(source, tmp_path / "backups")
    document = json.loads(generation.manifest_path.read_text(encoding="utf-8"))
    document["sqlite"]["core_table_counts"]["observations"] += 1
    generation.manifest_path.write_text(json.dumps(document) + "\n", encoding="utf-8")
    os.chmod(generation.manifest_path, 0o600)

    with pytest.raises(DatabaseBackupIntegrityError, match="manifest evidence"):
        verify_database_restore(generation.path)


def test_unsafe_public_and_broad_backup_paths_are_rejected(tmp_path: Path) -> None:
    source = _snapshot_database(tmp_path / "source.sqlite3")
    with pytest.raises(DatabaseBackupPathError, match="public output"):
        create_database_backup(source, tmp_path / "web" / "backups")
    with pytest.raises(DatabaseBackupPathError, match="too broad"):
        create_database_backup(source, Path("/"))
