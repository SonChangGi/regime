"""Private, integrity-checked backups for the local SQLite snapshot store."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import stat
import tempfile
import time
from typing import Any, Mapping, Sequence
from uuid import uuid4

from regime_lab.recovery_policy import (
    RecoveryPolicy,
    RecoveryPolicyError,
    backup_capacity_preflight,
    load_recovery_policy,
)


BACKUP_SCHEMA_VERSION = "regime.database-backup.v2"
DEFAULT_CORE_TABLES = ("snapshots", "observations")
DEFAULT_RETENTION = 4

_DATABASE_FILENAME = "snapshot.sqlite3"
_MANIFEST_FILENAME = "manifest.json"
_GENERATION_RE = re.compile(
    r"^regime-snapshot-(?P<stamp>\d{8}T\d{12}Z)-(?P<nonce>[0-9a-f]{12})$"
)
_TABLE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PUBLIC_DIRECTORY_NAMES = frozenset({"docs", "gh-pages", "public", "site", "web"})


class DatabaseBackupError(RuntimeError):
    """Base class for backup, verification, and restore failures."""


class DatabaseBackupPathError(DatabaseBackupError):
    """Raised when a mutable or database path is not safely confined."""


class DatabaseBackupIntegrityError(DatabaseBackupError):
    """Raised when SQLite or manifest integrity verification fails."""


@dataclass(frozen=True)
class BackupGeneration:
    """One committed backup generation and its audited manifest."""

    generation_id: str
    path: Path
    database_path: Path
    manifest_path: Path
    manifest: Mapping[str, Any]


@dataclass(frozen=True)
class RestoreVerification:
    """Evidence that a generation can be copied and opened as a database."""

    generation_id: str
    sha256: str
    bytes: int
    quick_check: str
    integrity_check: str
    foreign_key_check: str
    core_table_counts: Mapping[str, int]


def create_database_backup(
    source_database: str | Path,
    backup_directory: str | Path,
    *,
    retain: int = DEFAULT_RETENTION,
    core_tables: Sequence[str] = DEFAULT_CORE_TABLES,
    source_code_fingerprint_sha256: str | None = None,
    created_at: datetime | None = None,
    recovery_policy: RecoveryPolicy | None = None,
) -> BackupGeneration:
    """Create and atomically commit a consistent online SQLite backup.

    The source is opened read-only and copied with SQLite's backup API, so WAL
    activity does not produce a torn filesystem-level copy.  A generation is
    visible only after its database, manifest, integrity checks, and trial
    restore all succeed.
    """

    if not isinstance(retain, int) or isinstance(retain, bool) or not 1 <= retain <= 32:
        raise ValueError("retain must be an integer between 1 and 32")
    tables = _validated_core_tables(core_tables)
    external_fingerprint = _validated_optional_sha256(
        source_code_fingerprint_sha256,
        field_name="source_code_fingerprint_sha256",
    )
    source = _safe_source_database(source_database)
    backup_root = _safe_backup_root(backup_directory, source=source, create=True)
    # A matching symlink is never ignored because inventory and backup writes
    # must not be usable as path traversal primitives.
    _reject_generation_symlinks(backup_root)
    policy = recovery_policy or load_recovery_policy()
    preflight = backup_capacity_preflight(source, backup_root, policy=policy)
    started_monotonic = time.monotonic()
    deadline_monotonic = started_monotonic + policy.max_backup_seconds
    moment = created_at or datetime.now(timezone.utc)
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError("created_at must be timezone-aware")
    moment = moment.astimezone(timezone.utc)
    generation_id = (
        f"regime-snapshot-{moment.strftime('%Y%m%dT%H%M%S%fZ')}-"
        f"{uuid4().hex[:12]}"
    )
    final_path = backup_root / generation_id
    if final_path.exists() or final_path.is_symlink():
        raise DatabaseBackupPathError("backup generation destination already exists")

    temporary_path = Path(
        tempfile.mkdtemp(prefix=".regime-snapshot-", suffix=".tmp", dir=backup_root)
    )
    os.chmod(temporary_path, 0o700)
    temporary_database = temporary_path / _DATABASE_FILENAME
    temporary_manifest = temporary_path / _MANIFEST_FILENAME
    try:
        before = _source_file_state(source)
        source_checks, source_summary = _online_backup(
            source,
            temporary_database,
            core_tables=tables,
            deadline_monotonic=deadline_monotonic,
        )
        os.chmod(temporary_database, 0o600)
        _fsync_file(temporary_database)
        after = _source_file_state(source)

        backup_checks, backup_summary = _inspect_database(
            temporary_database,
            core_tables=tables,
        )
        _require_matching_logical_snapshot(source_summary, backup_summary)
        database_sha256, database_bytes = _hash_file(temporary_database)
        source_path_sha256 = hashlib.sha256(os.fsencode(source)).hexdigest()
        source_fingerprint_sha256 = _canonical_sha256(
            {
                "path_sha256": source_path_sha256,
                "file_state_before": before,
                "file_state_after": after,
                "logical_snapshot": backup_summary,
                "backup_sha256": database_sha256,
            }
        )
        manifest: dict[str, Any] = {
            "schema_version": BACKUP_SCHEMA_VERSION,
            "generation_id": generation_id,
            "created_at": moment.isoformat().replace("+00:00", "Z"),
            "source": {
                "path_sha256": source_path_sha256,
                "fingerprint_sha256": source_fingerprint_sha256,
                "code_fingerprint_sha256": external_fingerprint,
                "file_state_before": before,
                "file_state_after": after,
                "quick_check": source_checks["quick_check"],
                "integrity_check": source_checks["integrity_check"],
                "foreign_key_check": source_checks["foreign_key_check"],
            },
            "backup": {
                "filename": _DATABASE_FILENAME,
                "sha256": database_sha256,
                "bytes": database_bytes,
            },
            "sqlite": {
                **backup_checks,
                **backup_summary,
            },
            "recovery_policy": {
                **policy.as_document(),
                "requested_retention_valid_generations": retain,
            },
            "capacity_preflight": preflight,
        }
        _write_private_json(temporary_manifest, manifest)
        _fsync_directory(temporary_path)

        temporary_generation = BackupGeneration(
            generation_id=generation_id,
            path=temporary_path,
            database_path=temporary_database,
            manifest_path=temporary_manifest,
            manifest=manifest,
        )
        _validate_generation(
            temporary_generation,
            core_tables=tables,
            allow_staging_path=True,
        )
        _verify_restore_copy(
            temporary_generation,
            core_tables=tables,
            deadline_monotonic=deadline_monotonic,
        )

        os.replace(temporary_path, final_path)
        _fsync_directory(backup_root)
        committed = BackupGeneration(
            generation_id=generation_id,
            path=final_path,
            database_path=final_path / _DATABASE_FILENAME,
            manifest_path=final_path / _MANIFEST_FILENAME,
            manifest=manifest,
        )
        # Retention is deliberately advisory.  Recovery points are inventoried
        # and reported as over target, never deleted by routine backup work.
        return committed
    except Exception:
        # Never touch an older generation on a failed create.  The random
        # temporary directory is private and is the only eligible cleanup.
        if temporary_path.exists() and not temporary_path.is_symlink():
            shutil.rmtree(temporary_path)
        raise


def list_database_backups(
    backup_directory: str | Path,
    *,
    core_tables: Sequence[str] = DEFAULT_CORE_TABLES,
) -> tuple[BackupGeneration, ...]:
    """Return committed, fully validated generations, newest first."""

    tables = _validated_core_tables(core_tables)
    root = _safe_backup_root(backup_directory, source=None, create=False)
    _reject_generation_symlinks(root)
    generations: list[BackupGeneration] = []
    for path in _structural_generation_paths(root):
        try:
            generation = _load_generation(path)
            _validate_generation(generation, core_tables=tables)
        except DatabaseBackupIntegrityError:
            # Corrupt ordinary directories are preserved for diagnosis and are
            # not advertised as usable backups.
            continue
        generations.append(generation)
    return tuple(sorted(generations, key=lambda item: item.generation_id, reverse=True))


def verify_database_restore(
    generation: BackupGeneration | str | Path,
    *,
    core_tables: Sequence[str] = DEFAULT_CORE_TABLES,
    max_duration_seconds: int | None = None,
) -> RestoreVerification:
    """Copy a backup to an isolated temporary DB and verify it can restore."""

    tables = _validated_core_tables(core_tables)
    loaded = _load_generation(
        generation.path if isinstance(generation, BackupGeneration) else Path(generation)
    )
    _validate_generation(loaded, core_tables=tables)
    if max_duration_seconds is not None and (
        not isinstance(max_duration_seconds, int)
        or isinstance(max_duration_seconds, bool)
        or max_duration_seconds <= 0
    ):
        raise ValueError("max_duration_seconds must be a positive integer")
    deadline = (
        None
        if max_duration_seconds is None
        else time.monotonic() + max_duration_seconds
    )
    return _verify_restore_copy(
        loaded,
        core_tables=tables,
        deadline_monotonic=deadline,
    )


def classify_database_backup(
    generation_path: str | Path,
    *,
    core_tables: Sequence[str] = DEFAULT_CORE_TABLES,
) -> tuple[str, str | None]:
    """Classify one preserved generation without hiding unusable evidence."""

    path = Path(generation_path)
    manifest_path = path / _MANIFEST_FILENAME
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return "corrupt", f"manifest_unreadable:{type(exc).__name__}"
    if not isinstance(document, dict):
        return "corrupt", "manifest_root_invalid"
    schema = document.get("schema_version")
    if schema != BACKUP_SCHEMA_VERSION:
        if isinstance(schema, str) and schema.startswith("regime.database-backup."):
            return "legacy", schema
        return "corrupt", "manifest_schema_invalid"
    try:
        generation = _load_generation(path)
        _validate_generation(
            generation,
            core_tables=_validated_core_tables(core_tables),
        )
    except (DatabaseBackupError, OSError, ValueError) as exc:
        return "corrupt", type(exc).__name__
    return "valid-current", None


def _safe_source_database(value: str | Path) -> Path:
    path = _lexical_absolute(value)
    _reject_symlink_components(path, label="source database")
    if not path.exists() or not path.is_file():
        raise DatabaseBackupPathError("source database must be an existing regular file")
    try:
        if not stat.S_ISREG(path.stat(follow_symlinks=False).st_mode):
            raise DatabaseBackupPathError("source database must be a regular file")
    except OSError as exc:
        raise DatabaseBackupPathError("source database could not be inspected") from exc
    return path


def _safe_backup_root(
    value: str | Path,
    *,
    source: Path | None,
    create: bool,
) -> Path:
    path = _lexical_absolute(value)
    if path == Path(path.anchor) or path == Path.home().resolve():
        raise DatabaseBackupPathError("backup directory is too broad")
    if any(part.casefold() in _PUBLIC_DIRECTORY_NAMES for part in path.parts):
        raise DatabaseBackupPathError("backup directory must not be a public output path")
    _reject_symlink_components(path, label="backup directory")
    if source is not None and (source == path or source.is_relative_to(path)):
        raise DatabaseBackupPathError("source database must be outside the backup directory")
    if path.exists():
        if not path.is_dir():
            raise DatabaseBackupPathError("backup directory must be a directory")
    elif create:
        path.mkdir(mode=0o700, parents=True)
    else:
        raise DatabaseBackupPathError("backup directory does not exist")
    os.chmod(path, 0o700)
    if stat.S_IMODE(path.stat(follow_symlinks=False).st_mode) != 0o700:
        raise DatabaseBackupPathError("backup directory is not private")
    return path


def _lexical_absolute(value: str | Path) -> Path:
    raw = Path(value).expanduser()
    return Path(os.path.abspath(raw))


def _reject_symlink_components(path: Path, *, label: str) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise DatabaseBackupPathError(f"{label} must not traverse a symlink: {current}")
        if not current.exists():
            break


def _reject_generation_symlinks(root: Path) -> None:
    for entry in root.iterdir():
        if _GENERATION_RE.fullmatch(entry.name) and entry.is_symlink():
            raise DatabaseBackupPathError("backup generation must not be a symlink")


def _validated_core_tables(values: Sequence[str]) -> tuple[str, ...]:
    tables = tuple(dict.fromkeys(values))
    if not tables:
        raise ValueError("core_tables must not be empty")
    if any(not isinstance(name, str) or not _TABLE_NAME_RE.fullmatch(name) for name in tables):
        raise ValueError("core_tables contains an invalid SQLite identifier")
    return tables


def _validated_optional_sha256(value: str | None, *, field_name: str) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", normalized):
        raise ValueError(f"{field_name} must be a 64-character SHA-256")
    return normalized


def _source_file_state(source: Path) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for label, path in (
        ("database", source),
        ("wal", Path(f"{source}-wal")),
        ("shm", Path(f"{source}-shm")),
    ):
        if path.is_symlink():
            raise DatabaseBackupPathError(f"source SQLite {label} file must not be a symlink")
        if not path.exists():
            state[label] = {"exists": False}
            continue
        info = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode):
            raise DatabaseBackupPathError(f"source SQLite {label} file must be regular")
        state[label] = {
            "exists": True,
            "device": info.st_dev,
            "inode": info.st_ino,
            "bytes": info.st_size,
            "mtime_ns": info.st_mtime_ns,
        }
    return state


def _readonly_connection(path: Path) -> sqlite3.Connection:
    uri = f"{path.as_uri()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=30)
    connection.execute("PRAGMA query_only = ON")
    connection.execute("PRAGMA trusted_schema = OFF")
    connection.execute("PRAGMA busy_timeout = 30000")
    return connection


def _online_backup(
    source: Path,
    destination: Path,
    *,
    core_tables: tuple[str, ...],
    deadline_monotonic: float | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    try:
        with _readonly_connection(source) as source_connection:
            # Pin checks, counts, and backup to one source snapshot.  In WAL
            # mode this remains online for writers; in rollback-journal mode it
            # deliberately holds the ordinary SQLite read lock.
            source_connection.execute("BEGIN")
            checks = _sqlite_checks(source_connection, label="source")
            summary = _sqlite_summary(source_connection, core_tables=core_tables)
            with sqlite3.connect(destination, timeout=30) as destination_connection:
                def progress(_status: int, _remaining: int, _total: int) -> None:
                    _require_before_deadline(
                        deadline_monotonic,
                        context="SQLite online backup",
                    )

                source_connection.backup(
                    destination_connection,
                    pages=256,
                    progress=progress,
                    sleep=0.01,
                )
                destination_connection.commit()
                # A source in WAL mode transfers that persistent journal-mode
                # setting into the copy.  Backups are immutable generations;
                # normalize them to a single-file database before hashing so a
                # read-only verification cannot create WAL/SHM sidecars.
                mode = str(
                    destination_connection.execute("PRAGMA journal_mode = DELETE").fetchone()[0]
                ).lower()
                if mode != "delete":
                    raise DatabaseBackupIntegrityError(
                        "backup database could not be normalized to DELETE journal mode"
                    )
    except sqlite3.Error as exc:
        raise DatabaseBackupIntegrityError("SQLite online backup failed") from exc
    return checks, summary


def _inspect_database(
    path: Path,
    *,
    core_tables: tuple[str, ...],
) -> tuple[dict[str, str], dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise DatabaseBackupPathError("backup database must be a regular file")
    try:
        with _readonly_connection(path) as connection:
            return (
                _sqlite_checks(connection, label="backup"),
                _sqlite_summary(connection, core_tables=core_tables),
            )
    except sqlite3.Error as exc:
        raise DatabaseBackupIntegrityError("backup SQLite inspection failed") from exc


def _sqlite_checks(connection: sqlite3.Connection, *, label: str) -> dict[str, str]:
    output: dict[str, str] = {}
    for pragma in ("quick_check", "integrity_check"):
        rows = tuple(str(row[0]) for row in connection.execute(f"PRAGMA {pragma}"))
        if rows != ("ok",):
            raise DatabaseBackupIntegrityError(f"{label} database {pragma} failed")
        output[pragma] = "ok"
    foreign_key_rows = tuple(connection.execute("PRAGMA foreign_key_check"))
    if foreign_key_rows:
        raise DatabaseBackupIntegrityError(
            f"{label} database foreign_key_check failed"
        )
    output["foreign_key_check"] = "ok"
    return output


def _sqlite_summary(
    connection: sqlite3.Connection,
    *,
    core_tables: tuple[str, ...],
) -> dict[str, Any]:
    schema_rows = [
        [str(kind), str(name), str(table), sql]
        for kind, name, table, sql in connection.execute(
            """
            SELECT type, name, tbl_name, sql
            FROM sqlite_schema
            WHERE name NOT LIKE 'sqlite_%'
            ORDER BY type, name, tbl_name
            """
        )
    ]
    table_names = {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_schema WHERE type = 'table'"
        )
    }
    missing = sorted(set(core_tables) - table_names)
    if missing:
        raise DatabaseBackupIntegrityError(
            "database is missing core tables: " + ", ".join(missing)
        )
    counts = {
        table: int(
            connection.execute(
                f'SELECT COUNT(*) FROM "{table}"'
            ).fetchone()[0]
        )
        for table in core_tables
    }
    return {
        "schema_sha256": _canonical_sha256(schema_rows),
        "core_table_counts": counts,
        "page_count": int(connection.execute("PRAGMA page_count").fetchone()[0]),
        "page_size": int(connection.execute("PRAGMA page_size").fetchone()[0]),
        "user_version": int(connection.execute("PRAGMA user_version").fetchone()[0]),
    }


def _require_matching_logical_snapshot(
    source_summary: Mapping[str, Any],
    backup_summary: Mapping[str, Any],
) -> None:
    for field in ("schema_sha256", "core_table_counts", "user_version"):
        if source_summary.get(field) != backup_summary.get(field):
            raise DatabaseBackupIntegrityError(
                f"online backup does not match source logical snapshot: {field}"
            )


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _write_private_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    ).encode("utf-8") + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        if path.exists() and not path.is_symlink():
            path.unlink()
        raise
    os.chmod(path, 0o600)


def _load_generation(path: Path) -> BackupGeneration:
    path = _lexical_absolute(path)
    _reject_symlink_components(path, label="backup generation")
    if path.is_symlink() or not path.is_dir():
        raise DatabaseBackupPathError("backup generation must be a regular directory")
    match = _GENERATION_RE.fullmatch(path.name)
    if match is None:
        raise DatabaseBackupPathError("backup generation name is invalid")
    manifest_path = path / _MANIFEST_FILENAME
    database_path = path / _DATABASE_FILENAME
    if manifest_path.is_symlink() or database_path.is_symlink():
        raise DatabaseBackupPathError("backup generation files must not be symlinks")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DatabaseBackupIntegrityError("backup manifest is unreadable") from exc
    if not isinstance(manifest, dict):
        raise DatabaseBackupIntegrityError("backup manifest must be an object")
    return BackupGeneration(
        generation_id=path.name,
        path=path,
        database_path=database_path,
        manifest_path=manifest_path,
        manifest=manifest,
    )


def _validate_generation(
    generation: BackupGeneration,
    *,
    core_tables: tuple[str, ...],
    allow_staging_path: bool = False,
) -> None:
    if generation.path.is_symlink() or not generation.path.is_dir():
        raise DatabaseBackupPathError("backup generation must be a regular directory")
    if _GENERATION_RE.fullmatch(generation.generation_id) is None:
        raise DatabaseBackupIntegrityError("backup generation name is invalid")
    if not allow_staging_path and generation.path.name != generation.generation_id:
        raise DatabaseBackupIntegrityError("backup generation path does not match its id")
    entries = {entry.name for entry in generation.path.iterdir()}
    if entries != {_DATABASE_FILENAME, _MANIFEST_FILENAME}:
        raise DatabaseBackupIntegrityError("backup generation has unexpected entries")
    _require_private_mode(generation.path, expected=0o700, label="backup generation")
    for path, label in (
        (generation.database_path, "backup database"),
        (generation.manifest_path, "backup manifest"),
    ):
        if path.is_symlink() or not path.is_file():
            raise DatabaseBackupPathError(f"{label} must be a regular file")
        _require_private_mode(path, expected=0o600, label=label)

    manifest = generation.manifest
    if manifest.get("schema_version") != BACKUP_SCHEMA_VERSION:
        raise DatabaseBackupIntegrityError("backup manifest schema mismatch")
    if manifest.get("generation_id") != generation.generation_id:
        raise DatabaseBackupIntegrityError("backup manifest generation id mismatch")
    backup = manifest.get("backup")
    sqlite_document = manifest.get("sqlite")
    source = manifest.get("source")
    if (
        not isinstance(backup, dict)
        or not isinstance(sqlite_document, dict)
        or not isinstance(source, dict)
    ):
        raise DatabaseBackupIntegrityError("backup manifest sections are invalid")
    if backup.get("filename") != _DATABASE_FILENAME:
        raise DatabaseBackupIntegrityError("backup manifest filename is invalid")
    expected_sha = _validated_manifest_sha256(backup.get("sha256"), "backup sha256")
    expected_bytes = backup.get("bytes")
    if (
        not isinstance(expected_bytes, int)
        or isinstance(expected_bytes, bool)
        or expected_bytes <= 0
    ):
        raise DatabaseBackupIntegrityError("backup manifest byte count is invalid")
    _validated_manifest_sha256(source.get("path_sha256"), "source path sha256")
    _validated_manifest_sha256(source.get("fingerprint_sha256"), "source fingerprint")
    external = source.get("code_fingerprint_sha256")
    if external is not None:
        _validated_manifest_sha256(external, "source code fingerprint")
    actual_sha, actual_bytes = _hash_file(generation.database_path)
    if (actual_sha, actual_bytes) != (expected_sha, expected_bytes):
        raise DatabaseBackupIntegrityError("backup database hash or byte count mismatch")
    checks, summary = _inspect_database(generation.database_path, core_tables=core_tables)
    expected_sqlite = {**checks, **summary}
    if sqlite_document != expected_sqlite:
        raise DatabaseBackupIntegrityError("backup SQLite manifest evidence mismatch")


def _validated_manifest_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise DatabaseBackupIntegrityError(f"{label} is invalid")
    return value


def _require_private_mode(path: Path, *, expected: int, label: str) -> None:
    actual = stat.S_IMODE(path.stat(follow_symlinks=False).st_mode)
    if actual != expected:
        raise DatabaseBackupIntegrityError(
            f"{label} must have mode {expected:04o}, found {actual:04o}"
        )


def _verify_restore_copy(
    generation: BackupGeneration,
    *,
    core_tables: tuple[str, ...],
    deadline_monotonic: float | None = None,
) -> RestoreVerification:
    with tempfile.TemporaryDirectory(
        prefix=".restore-verification-",
        dir=generation.path.parent,
    ) as temporary_name:
        temporary_root = Path(temporary_name)
        os.chmod(temporary_root, 0o700)
        restored = temporary_root / "restored.sqlite3"
        with generation.database_path.open("rb") as source, restored.open("xb") as destination:
            os.chmod(restored, 0o600)
            while True:
                _require_before_deadline(
                    deadline_monotonic,
                    context="backup restore drill",
                )
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                destination.write(chunk)
            destination.flush()
            os.fsync(destination.fileno())
        restored_sha, restored_bytes = _hash_file(restored)
        backup_document = generation.manifest["backup"]
        if (
            restored_sha != backup_document["sha256"]
            or restored_bytes != backup_document["bytes"]
        ):
            raise DatabaseBackupIntegrityError("trial restore copy hash mismatch")
        checks, summary = _inspect_database(restored, core_tables=core_tables)
        _require_before_deadline(
            deadline_monotonic,
            context="backup restore drill",
        )
        sqlite_document = generation.manifest["sqlite"]
        for field in (
            "schema_sha256",
            "core_table_counts",
            "page_count",
            "page_size",
            "user_version",
        ):
            if summary[field] != sqlite_document[field]:
                raise DatabaseBackupIntegrityError(
                    f"trial restore SQLite evidence mismatch: {field}"
                )
        return RestoreVerification(
            generation_id=generation.generation_id,
            sha256=restored_sha,
            bytes=restored_bytes,
            quick_check=checks["quick_check"],
            integrity_check=checks["integrity_check"],
            foreign_key_check=checks["foreign_key_check"],
            core_table_counts=dict(summary["core_table_counts"]),
        )


def _require_before_deadline(deadline_monotonic: float | None, *, context: str) -> None:
    if deadline_monotonic is not None and time.monotonic() > deadline_monotonic:
        raise RecoveryPolicyError(f"{context} exceeded time ceiling")


def _structural_generation_paths(root: Path) -> tuple[Path, ...]:
    paths = []
    for entry in root.iterdir():
        if _GENERATION_RE.fullmatch(entry.name):
            if entry.is_symlink():
                raise DatabaseBackupPathError("backup generation must not be a symlink")
            if entry.is_dir():
                paths.append(entry)
    return tuple(sorted(paths, key=lambda path: path.name, reverse=True))


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "BACKUP_SCHEMA_VERSION",
    "BackupGeneration",
    "DatabaseBackupError",
    "DatabaseBackupIntegrityError",
    "DatabaseBackupPathError",
    "RestoreVerification",
    "classify_database_backup",
    "create_database_backup",
    "list_database_backups",
    "verify_database_restore",
]
