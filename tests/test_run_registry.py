from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from regime_lab.run_registry import (
    RunRegistryError,
    append_run_event,
    completed_run_for_generation,
    current_run_status,
)


UTC = timezone.utc


def test_run_registry_is_append_only_and_enforces_lifecycle(tmp_path: Path) -> None:
    path = tmp_path / "run-registry.jsonl"
    run_id = "20260826T120000Z-test"
    for status in ("started", "collecting", "analyzing", "completed", "publication_reviewed", "published"):
        append_run_event(
            path,
            run_id=run_id,
            status=status,
            occurred_at=datetime(2026, 8, 26, 12, tzinfo=UTC),
        )
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["status"] for row in rows] == [
        "started", "collecting", "analyzing", "completed", "publication_reviewed", "published"
    ]
    assert current_run_status(path, run_id) == "published"
    assert all(len(row["line_checksum_sha256"]) == 64 for row in rows)


def test_run_registry_reads_legacy_rows_and_repairs_partial_tail(
    tmp_path: Path,
) -> None:
    path = tmp_path / "run-registry.jsonl"
    legacy = {
        "schema_version": "regime-run-registry-event/1",
        "run_id": "20260826T120000Z-test",
        "status": "started",
        "occurred_at": "2026-08-26T12:00:00+00:00",
        "generation_id": None,
        "detail": None,
    }
    path.write_bytes(json.dumps(legacy, sort_keys=True).encode() + b"\n{\"status\":")

    event = append_run_event(
        path,
        run_id=legacy["run_id"],
        status="collecting",
    )

    assert event.line_checksum_sha256 is not None
    assert current_run_status(path, legacy["run_id"]) == "collecting"
    quarantined = list(tmp_path.glob("run-registry.jsonl.partial-*.corrupt"))
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == b'{"status":'


def test_run_registry_rejects_checksum_tampering(tmp_path: Path) -> None:
    path = tmp_path / "run-registry.jsonl"
    append_run_event(path, run_id="20260826T120000Z-test", status="started")
    row = json.loads(path.read_text(encoding="utf-8"))
    row["status"] = "collecting"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(RunRegistryError, match="checksum mismatch"):
        current_run_status(path, "20260826T120000Z-test")


def test_completed_run_for_generation_requires_unique_lifecycle(
    tmp_path: Path,
) -> None:
    path = tmp_path / "run-registry.jsonl"
    run_id = "20260826T120000Z-test"
    for status in ("started", "collecting", "analyzing"):
        append_run_event(path, run_id=run_id, status=status)
    append_run_event(
        path,
        run_id=run_id,
        status="completed",
        generation_id="20260826T120000.000000Z",
    )

    assert completed_run_for_generation(
        path,
        "20260826T120000.000000Z",
    ) == run_id


def test_run_registry_rejects_skipped_or_terminal_transition(tmp_path: Path) -> None:
    path = tmp_path / "run-registry.jsonl"
    run_id = "20260826T120000Z-test"
    append_run_event(path, run_id=run_id, status="started")
    with pytest.raises(RunRegistryError, match="invalid run transition"):
        append_run_event(path, run_id=run_id, status="completed")
    append_run_event(path, run_id=run_id, status="interrupted")
    with pytest.raises(RunRegistryError, match="invalid run transition"):
        append_run_event(path, run_id=run_id, status="collecting")


def test_run_registry_requires_review_before_published(tmp_path: Path) -> None:
    path = tmp_path / "run-registry.jsonl"
    run_id = "20260826T120000Z-test"
    for status in ("started", "collecting", "analyzing", "completed"):
        append_run_event(path, run_id=run_id, status=status)

    with pytest.raises(RunRegistryError, match="invalid run transition"):
        append_run_event(path, run_id=run_id, status="published")
