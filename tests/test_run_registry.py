from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from regime_lab.run_registry import (
    RunRegistryError,
    append_run_event,
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


def test_run_registry_rejects_skipped_or_terminal_transition(tmp_path: Path) -> None:
    path = tmp_path / "run-registry.jsonl"
    run_id = "20260826T120000Z-test"
    append_run_event(path, run_id=run_id, status="started")
    with pytest.raises(RunRegistryError, match="invalid run transition"):
        append_run_event(path, run_id=run_id, status="completed")
    append_run_event(path, run_id=run_id, status="interrupted")
    with pytest.raises(RunRegistryError, match="invalid run transition"):
        append_run_event(path, run_id=run_id, status="collecting")
