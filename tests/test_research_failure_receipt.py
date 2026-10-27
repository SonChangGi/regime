"""Only a validated research transport receipt may change a late retry policy."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys

import pytest

from regime_lab import automation, cli
from regime_lab.research.additional_sources import AdditionalSourceUnavailable, SOURCES
from test_automation_operations import _settings


UTC = timezone.utc
TARGET = datetime(2026, 10, 23, 20, tzinfo=UTC)
STARTED = datetime(2026, 10, 27, 4, 46, 9, tzinfo=UTC)
FAILED = datetime(2026, 10, 27, 6, 15, 44, tzinfo=UTC)


def primary_receipt() -> dict[str, object]:
    return {
        "schema_version": 1, "expected_cutoff": TARGET.isoformat(),
        "model_cutoff": TARGET.isoformat(), "ready_for_training": True,
        "overall_health": "ok", "sources": [{"id": "frb_h10", "status": "ok"}],
        "issues": [], "gate_error": None,
    }


def failure_receipt(source: str = "vix") -> dict[str, object]:
    return {
        **primary_receipt(),
        "post_collection_failure": {
            "schema_version": 1, "error_code": "additional_source_unavailable",
            "source_id": source, "expected_cutoff": TARGET.isoformat(),
        },
    }


def write_receipt(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")


def test_cli_records_only_safe_transport_fields_without_changing_primary_gate(tmp_path: Path) -> None:
    path = tmp_path / "collection-report.json"
    original = primary_receipt()
    write_receipt(path, original)
    error = AdditionalSourceUnavailable("vix")
    error.__cause__ = RuntimeError("DO_NOT_SERIALIZE_DOWNLOADER_TRACEBACK")

    cli._record_additional_source_failure(path, TARGET, error)

    document = json.loads(path.read_text())
    assert document == failure_receipt()
    assert document["ready_for_training"] is True
    assert document["sources"] == original["sources"]
    assert "DO_NOT_SERIALIZE" not in path.read_text()
    assert "retry_class" not in document["post_collection_failure"]


@pytest.mark.parametrize("damage", [
    "missing", "json", "list", "schema_bool", "schema_wrong", "not_ready",
    "truthy_ready", "wrong_cutoff", "missing_model_cutoff", "stale_model_cutoff", "gate_error",
])
def test_cli_does_not_create_or_rewrite_unvalidated_primary_receipt(tmp_path: Path, damage: str) -> None:
    path = tmp_path / "collection-report.json"
    document = primary_receipt()
    if damage == "missing":
        before = None
    elif damage == "json":
        path.write_bytes(b"not JSON; keep exact bytes")
        before = path.read_bytes()
    else:
        if damage == "list":
            document = []
        elif damage == "schema_bool":
            document["schema_version"] = True
        elif damage == "schema_wrong":
            document["schema_version"] = 2
        elif damage == "not_ready":
            document["ready_for_training"] = False
        elif damage == "truthy_ready":
            document["ready_for_training"] = 1
        elif damage == "missing_model_cutoff":
            del document["model_cutoff"]
        elif damage == "stale_model_cutoff":
            document["model_cutoff"] = (TARGET - timedelta(days=7)).isoformat()
        elif damage == "gate_error":
            document["gate_error"] = "integrity validation failed"
        else:
            document["expected_cutoff"] = (TARGET - timedelta(days=7)).isoformat()
        write_receipt(path, document)
        before = path.read_bytes()

    cli._record_additional_source_failure(path, TARGET, AdditionalSourceUnavailable("vix"))

    assert (path.read_bytes() if path.exists() else None) == before


def test_cli_missing_optional_receipt_does_not_create_files(tmp_path: Path) -> None:
    cli._record_additional_source_failure(None, TARGET, AdditionalSourceUnavailable("vix"))
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("source", tuple(SOURCES))
def test_parent_accepts_each_known_transport_receipt_with_bounded_retry(tmp_path: Path, source: str) -> None:
    settings = replace(_settings(tmp_path), contract="v5")
    failure = automation._post_collection_failure(settings, failure_receipt(source), TARGET, FAILED)

    assert isinstance(failure, automation.ScheduledRetry)
    assert failure.error_code == "additional_source_unavailable"
    assert failure.retry_class == "transient"
    assert failure.next_retry_at == FAILED + settings.transient_retry_delay
    assert automation._failure_policy(
        failure, stage="train_models", attempt_started_at=STARTED, settings=settings,
    ) == ("additional_source_unavailable", "transient", failure.next_retry_at)


@pytest.mark.parametrize("damage", [
    "absent", "list", "string", "schema_bool", "schema_wrong", "unknown_code",
    "unknown_source", "source_list", "extra_field", "missing_field", "stale_cutoff",
    "noncanonical_cutoff", "top_schema_bool", "top_schema_wrong", "not_ready",
    "truthy_ready", "top_stale_cutoff", "missing_model_cutoff", "stale_model_cutoff", "gate_error",
])
def test_parent_rejects_malformed_or_unbound_transport_evidence(tmp_path: Path, damage: str) -> None:
    settings = replace(_settings(tmp_path), contract="v5")
    document = failure_receipt()
    failure = document["post_collection_failure"]
    if damage == "absent":
        del document["post_collection_failure"]
    elif damage == "list":
        document["post_collection_failure"] = []
    elif damage == "string":
        document["post_collection_failure"] = "additional_source_unavailable"
    elif damage == "schema_bool":
        failure["schema_version"] = True
    elif damage == "schema_wrong":
        failure["schema_version"] = 2
    elif damage == "unknown_code":
        failure["error_code"] = "cache_checksum_failed"
    elif damage == "unknown_source":
        failure["source_id"] = "unreviewed-source"
    elif damage == "source_list":
        failure["source_id"] = ["vix"]
    elif damage == "extra_field":
        failure["retry_class"] = "transient"
    elif damage == "missing_field":
        del failure["source_id"]
    elif damage == "stale_cutoff":
        failure["expected_cutoff"] = (TARGET - timedelta(days=7)).isoformat()
    elif damage == "noncanonical_cutoff":
        failure["expected_cutoff"] = TARGET.isoformat().replace("+00:00", "Z")
    elif damage == "top_schema_bool":
        document["schema_version"] = True
    elif damage == "top_schema_wrong":
        document["schema_version"] = 2
    elif damage == "not_ready":
        document["ready_for_training"] = False
    elif damage == "truthy_ready":
        document["ready_for_training"] = 1
    elif damage == "missing_model_cutoff":
        del document["model_cutoff"]
    elif damage == "stale_model_cutoff":
        document["model_cutoff"] = (TARGET - timedelta(days=7)).isoformat()
    elif damage == "gate_error":
        document["gate_error"] = "integrity validation failed"
    else:
        document["expected_cutoff"] = (TARGET - timedelta(days=7)).isoformat()
    before = deepcopy(document)

    assert automation._post_collection_failure(settings, document, TARGET, FAILED) is None
    assert document == before


def test_real_child_exit_uses_valid_receipt_instead_of_traceback_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = replace(_settings(tmp_path), contract="v5")
    real_run = automation._run
    sentinel = settings.root / "preserved-generation.json"
    sentinel.write_bytes(b"previous complete generation")

    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return FAILED.astimezone(tz) if tz is not None else FAILED.replace(tzinfo=None)

    monkeypatch.setattr(automation, "datetime", FrozenDateTime)
    monkeypatch.setattr(automation, "_candidate_context", lambda _settings: {})

    def child(_command, **_kwargs):
        program = (
            "import json,pathlib,sys; p=pathlib.Path(sys.argv[1]); "
            "p.parent.mkdir(parents=True,exist_ok=True); "
            "p.write_text(sys.argv[2]); raise SystemExit(1)"
        )
        return real_run(
            [sys.executable, "-c", program, str(settings.collection_report_path),
             json.dumps(failure_receipt())], cwd=settings.root, capture=True,
        )

    monkeypatch.setattr(automation, "_run", child)
    with pytest.raises(automation.ScheduledRetry) as caught:
        automation._build_candidate_generation(settings, target=TARGET, context={}, started_at=STARTED)

    assert caught.value.error_code == "additional_source_unavailable"
    assert caught.value.retry_class == "transient"
    assert caught.value.next_retry_at == FAILED + settings.transient_retry_delay
    assert isinstance(caught.value.__cause__, automation.AutomationError)
    assert "failed with exit code 1" in str(caught.value.__cause__)
    assert json.loads(settings.collection_report_path.read_text()) == failure_receipt()
    assert sentinel.read_bytes() == b"previous complete generation"


@pytest.mark.parametrize("receipt", [primary_receipt(), {
    **failure_receipt(), "post_collection_failure": {
        **failure_receipt()["post_collection_failure"], "source_id": "unreviewed-source",
    },
}])
def test_generic_late_child_failure_remains_blocked_without_exact_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, receipt: dict[str, object],
) -> None:
    settings = replace(_settings(tmp_path), contract="v5")

    def fail(_command, **_kwargs):
        write_receipt(settings.collection_report_path, receipt)
        raise automation.AutomationError("python failed with exit code 1")

    monkeypatch.setattr(automation, "_run", fail)
    with pytest.raises(automation.AutomationError) as caught:
        automation._build_candidate_generation(settings, target=TARGET, context={}, started_at=STARTED)

    assert not isinstance(caught.value, automation.ScheduledRetry)
    assert automation._failure_policy(
        caught.value, stage="train_models", attempt_started_at=STARTED, settings=settings,
    ) == ("analysis_build_failed", "blocked", None)


def test_prior_attempt_transport_receipt_is_removed_before_new_child_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = replace(_settings(tmp_path), contract="v5")
    write_receipt(settings.collection_report_path, failure_receipt())

    def fail_before_receipt(_command, **_kwargs):
        assert not settings.collection_report_path.exists()
        raise automation.AutomationError("python failed with exit code 1")

    monkeypatch.setattr(automation, "_run", fail_before_receipt)
    with pytest.raises(automation.ScheduledRetry) as caught:
        automation._build_candidate_generation(settings, target=TARGET, context={}, started_at=STARTED)
    assert caught.value.error_code == "child_precollection_failed"
    assert not settings.collection_report_path.exists()


def test_changed_source_remains_blocked_even_with_valid_transport_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = replace(_settings(tmp_path), contract="v5")
    marker = settings.root / "source-marker.py"
    marker.write_bytes(b"unchanged source bytes")
    previous = settings.root / "preserved-generation.json"
    previous.write_bytes(b"previous complete generation")

    def failed_child(_command, **_kwargs):
        write_receipt(settings.collection_report_path, failure_receipt())
        marker.write_bytes(b"changed source bytes")
        raise automation.AutomationError("python failed with exit code 1")

    monkeypatch.setattr(automation, "_run", failed_child)
    monkeypatch.setattr(
        automation, "_candidate_context",
        lambda _settings: {"changed": "x"} if marker.read_bytes() != b"unchanged source bytes" else {},
    )
    with pytest.raises(automation.AutomationError, match="tracked source or config changed") as caught:
        automation._build_candidate_generation(settings, target=TARGET, context={}, started_at=STARTED)

    assert not isinstance(caught.value, automation.ScheduledRetry)
    assert automation._failure_policy(
        caught.value, stage="train_models", attempt_started_at=STARTED, settings=settings,
    ) == ("train_models_blocked", "blocked", None)
    assert previous.read_bytes() == b"previous complete generation"
