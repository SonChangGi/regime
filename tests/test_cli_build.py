from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd
import pytest

from regime_lab import cli
from regime_lab.automation import automation_lock
from regime_lab.collection import LiveCollection
from regime_lab.data import HealthStatus
from regime_lab.v5_preflight import V5PreflightError


TARGET = datetime(2026, 8, 14, 20, tzinfo=timezone.utc)


def test_build_checks_keychain_then_backs_up_inside_lock_before_collection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(cli, "load_config", lambda _path: {})
    monkeypatch.setattr(
        cli,
        "_mutable_path",
        lambda value, **_kwargs: Path(value),
    )
    monkeypatch.setattr(
        cli,
        "_backup_database_before_mutation",
        lambda *_args, **_kwargs: events.append("backup"),
    )
    monkeypatch.setattr(
        cli, "verify_v5_preflight", lambda **_kwargs: (
            events.append("v5_preflight")
            or {"source_fingerprint_sha256": "a" * 64}
        )
    )
    secrets = {"FRED_API_KEY": "secret-a", "ALPHA_VANTAGE_API_KEY": "secret-b"}

    @contextmanager
    def loaded_secrets(*_args, **_kwargs):
        events.append("keychain_load")
        try:
            yield secrets
        finally:
            for name in secrets:
                secrets[name] = ""
            events.append("keychain_clear")

    @contextmanager
    def credentials(values, *_args, **_kwargs):
        assert values is secrets
        assert all(values.values())
        events.append("credentials")
        yield

    monkeypatch.setattr(cli, "provider_secrets_from_keychain", loaded_secrets)
    monkeypatch.setattr(cli, "provider_environment_from_secrets", credentials)

    def collect(*_args, **_kwargs):
        events.append("collect")
        raise RuntimeError("stop after ordering assertion")

    monkeypatch.setattr(cli, "collect_live_data", collect)
    args = argparse.Namespace(
        profile="standard",
        contract="v5",
        config=tmp_path / "series.json",
        database=tmp_path / "regime.sqlite3",
        output=tmp_path / "result.json",
        artifacts=tmp_path / "artifacts",
        checkpoint_directory=None,
        collection_report=None,
        expected_cutoff=TARGET,
        backup_directory=tmp_path / "backups",
        backup_source_code_fingerprint_sha256="a" * 64,
        from_env=False,
        require_ac_power=False,
    )

    with pytest.raises(RuntimeError, match="ordering assertion"):
        cli.command_build(args)

    assert events == [
        "v5_preflight",
        "keychain_load",
        "backup",
        "credentials",
        "collect",
        "keychain_clear",
    ]
    assert all(value == "" for value in secrets.values())


def _degraded_collection(database: Path) -> LiveCollection:
    return LiveCollection(
        records=(),
        cutoffs=(TARGET,),
        sources=(
            {
                "id": "alpha_vantage",
                "status": "degraded",
                "available_at": "2026-08-07T20:00:00+00:00",
                "coverage": "2006-01-06–2026-08-07",
                "issues": ["Alpha Vantage SPY: request timed out"],
            },
            {
                "id": "alfred",
                "status": "ok",
                "available_at": TARGET.isoformat(),
                "coverage": "2006-01-01–2026-08-14",
                "issues": [],
            },
        ),
        overall_health=HealthStatus.DEGRADED,
        issues=("Alpha Vantage SPY: request timed out",),
        model_cutoff=TARGET,
        database_path=database,
    )


def _ready_collection(database: Path) -> LiveCollection:
    return LiveCollection(
        records=(),
        cutoffs=(TARGET,),
        sources=(
            {
                "id": "alpha_vantage",
                "status": "ok",
                "available_at": TARGET.isoformat(),
                "coverage": "2006-01-06–2026-08-14",
                "issues": [],
            },
            {
                "id": "alfred",
                "status": "ok",
                "available_at": TARGET.isoformat(),
                "coverage": "2006-01-01–2026-08-14",
                "issues": [],
            },
        ),
        overall_health=HealthStatus.OK,
        issues=(),
        model_cutoff=TARGET,
        database_path=database,
    )


def test_build_writes_degraded_report_and_stops_before_analysis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "regime.sqlite3"
    output = tmp_path / "regime-results.json"
    artifacts = tmp_path / "artifacts"
    report = tmp_path / "collection-report.json"
    output.write_text("old payload\n", encoding="utf-8")
    artifacts.mkdir()
    (artifacts / "old.txt").write_text("old artifacts\n", encoding="utf-8")

    monkeypatch.setattr(cli, "load_config", lambda _path: {})
    monkeypatch.setattr(
        cli,
        "_mutable_path",
        lambda value, **_kwargs: Path(value),
    )
    monkeypatch.setattr(
        cli,
        "collect_live_data",
        lambda *_args, **_kwargs: _degraded_collection(database),
    )
    monkeypatch.setattr(
        cli,
        "build_weekly_dataset",
        lambda *_args, **_kwargs: pytest.fail("analysis must not start"),
    )
    monkeypatch.setattr(
        cli,
        "build_dashboard_result",
        lambda *_args, **_kwargs: pytest.fail("model training must not start"),
    )
    args = argparse.Namespace(
        alfred_rights_confirmed=True,
        profile="standard",
        contract="v4",
        config=tmp_path / "series.json",
        database=database,
        output=output,
        artifacts=artifacts,
        from_env=True,
        expected_cutoff=TARGET,
        collection_report=report,
        require_ac_power=False,
    )

    with pytest.raises(SystemExit, match="stopped before analysis"):
        cli.command_build(args)

    receipt = json.loads(report.read_text(encoding="utf-8"))
    assert receipt["expected_cutoff"] == TARGET.isoformat()
    assert receipt["ready_for_training"] is False
    assert receipt["overall_health"] == "degraded"
    assert "health is degraded" in receipt["gate_error"]
    assert output.read_text(encoding="utf-8") == "old payload\n"
    assert (artifacts / "old.txt").read_text(encoding="utf-8") == "old artifacts\n"


def test_build_parser_requires_timezone_for_expected_cutoff() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "build",
            "--expected-cutoff",
            "2026-08-14T20:00:00+00:00",
            "--collection-report",
            "build/collection-report.json",
            "--require-ac-power",
            "--alfred-rights-confirmed",
        ]
    )
    assert args.expected_cutoff == TARGET
    assert args.require_ac_power is True

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "build",
                "--expected-cutoff",
                "2026-08-14T20:00:00",
                "--alfred-rights-confirmed",
            ]
        )


def test_ac_power_failure_updates_report_and_stops_before_analysis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "regime.sqlite3"
    report = tmp_path / "collection-report.json"
    monkeypatch.setattr(cli, "load_config", lambda _path: {})
    monkeypatch.setattr(cli, "_mutable_path", lambda value, **_kwargs: Path(value))
    monkeypatch.setattr(
        cli,
        "collect_live_data",
        lambda *_args, **_kwargs: _ready_collection(database),
    )
    monkeypatch.setattr(
        cli,
        "build_weekly_dataset",
        lambda *_args, **_kwargs: pytest.fail("analysis must not start"),
    )
    monkeypatch.setattr(cli.sys, "platform", "darwin")
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *_args, **_kwargs: argparse.Namespace(
            returncode=0,
            stdout=b"Now drawing from 'Battery Power'\n",
            stderr=b"",
        ),
    )
    args = argparse.Namespace(
        alfred_rights_confirmed=True,
        profile="standard",
        contract="v4",
        config=tmp_path / "series.json",
        database=database,
        output=tmp_path / "result.json",
        artifacts=tmp_path / "artifacts",
        from_env=True,
        expected_cutoff=TARGET,
        collection_report=report,
        require_ac_power=True,
    )

    with pytest.raises(SystemExit, match="AC power is required"):
        cli.command_build(args)

    receipt = json.loads(report.read_text(encoding="utf-8"))
    assert receipt["ready_for_training"] is False
    assert receipt["gate_error"] == "AC power is required before analysis"


def test_ac_power_requirement_is_noop_off_macos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli.sys, "platform", "linux")
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("pmset must not run off macOS"),
    )
    cli._require_ac_power_before_analysis(enabled=True)


def test_database_lock_collision_writes_transient_collection_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "regime.sqlite3"
    report = tmp_path / "collection-report.json"
    lock = database.with_name(f"{database.name}.live-build.lock")
    monkeypatch.setattr(cli, "load_config", lambda _path: {})
    monkeypatch.setattr(cli, "_mutable_path", lambda value, **_kwargs: Path(value))
    monkeypatch.setattr(
        cli,
        "collect_live_data",
        lambda *_args, **_kwargs: pytest.fail("lock collision must stop before providers"),
    )
    args = argparse.Namespace(
        alfred_rights_confirmed=True,
        profile="standard",
        contract="v4",
        config=tmp_path / "series.json",
        database=database,
        output=tmp_path / "result.json",
        artifacts=tmp_path / "artifacts",
        from_env=True,
        expected_cutoff=TARGET,
        collection_report=report,
        require_ac_power=False,
    )

    with automation_lock(lock):
        with pytest.raises(SystemExit, match="another build owns"):
            cli.command_build(args)

    receipt = json.loads(report.read_text(encoding="utf-8"))
    assert receipt["ready_for_training"] is False
    assert receipt["error_code"] == "database_build_lock_busy"
    assert receipt["expected_cutoff"] == TARGET.isoformat()


def test_v5_preflight_stops_before_provider_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "regime.sqlite3"
    monkeypatch.setattr(cli, "load_config", lambda _path: {})
    monkeypatch.setattr(cli, "_mutable_path", lambda value, **_kwargs: Path(value))
    monkeypatch.setattr(
        cli,
        "verify_v5_preflight",
        lambda **_kwargs: (_ for _ in ()).throw(
            V5PreflightError("frozen baseline mismatch")
        ),
    )
    monkeypatch.setattr(
        cli,
        "collect_live_data",
        lambda *_args, **_kwargs: pytest.fail(
            "v5 preflight must fail before provider collection"
        ),
    )
    args = argparse.Namespace(
        alfred_rights_confirmed=True,
        profile="standard",
        contract="v5",
        config=tmp_path / "series.json",
        database=database,
        output=tmp_path / "result.json",
        artifacts=tmp_path / "artifacts",
        from_env=True,
        expected_cutoff=TARGET,
        collection_report=None,
        require_ac_power=False,
    )

    with pytest.raises(V5PreflightError, match="frozen baseline mismatch"):
        cli.command_build(args)


def test_v5_live_build_forwards_preflight_fingerprint_and_private_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import regime_lab.data as data_module
    import regime_lab.h10_store as h10_store_module

    class BuildIntercept(RuntimeError):
        pass

    class DummyStore:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

    database = tmp_path / "regime.sqlite3"
    output = tmp_path / "v5-live" / "regime-results.json"
    artifacts = tmp_path / "v5-live" / "artifacts"
    report = tmp_path / "collection-report.json"
    captured: dict[str, object] = {}
    source_fingerprint = "c" * 64
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda _path: {
            "model": {
                "final_holdout_start": "2023-01-01",
                "minimum_log_loss_improvement": 0.01,
            }
        },
    )
    monkeypatch.setattr(cli, "_mutable_path", lambda value, **_kwargs: Path(value))
    monkeypatch.setattr(
        cli,
        "verify_v5_preflight",
        lambda **_kwargs: {"source_fingerprint_sha256": source_fingerprint},
    )
    monkeypatch.setattr(
        cli,
        "collect_live_data",
        lambda *_args, **_kwargs: _ready_collection(database),
    )
    monkeypatch.setattr(
        cli,
        "build_weekly_dataset",
        lambda *_args, **_kwargs: argparse.Namespace(
            features=pd.DataFrame(
                [[0.0, 1.0]] * 700,
                columns=["one", "two"],
            )
        ),
    )
    monkeypatch.setattr(data_module, "SQLiteSnapshotStore", DummyStore)
    archive_client = object()
    h10_client = object()
    monkeypatch.setattr(data_module, "H10ArchiveClient", lambda: archive_client)
    monkeypatch.setattr(data_module, "H10Client", lambda: h10_client)
    h10_calls: list[str] = []

    def refresh_h10_archive(_store, actual_client, **kwargs):
        assert actual_client is archive_client
        assert kwargs["as_of"] == TARGET
        h10_calls.append("archive")
        return object()

    monkeypatch.setattr(
        h10_store_module,
        "refresh_h10_archive_store",
        refresh_h10_archive,
    )

    def refresh_h10(_store, actual_client, **kwargs):
        assert actual_client is h10_client
        assert kwargs["as_of"] == TARGET
        assert h10_calls == ["archive"]
        h10_calls.append("current")
        pending = json.loads(report.read_text(encoding="utf-8"))
        assert pending["ready_for_training"] is False
        assert pending["overall_health"] == "pending"
        assert pending["sources"][-1] == {
            "id": "frb_h10",
            "name": "Federal Reserve H.10 foreign exchange rates",
            "status": "pending",
            "issues": [],
        }
        return argparse.Namespace(
            fx_features=None,
            fx_context=None,
            source_row={"id": "frb_h10", "status": "ok", "issues": []},
        )

    monkeypatch.setattr(
        h10_store_module,
        "refresh_h10_store",
        refresh_h10,
    )

    def intercept(*_args, **kwargs):
        captured.update(kwargs)
        raise BuildIntercept

    monkeypatch.setattr(cli, "build_dashboard_result", intercept)
    args = argparse.Namespace(
        alfred_rights_confirmed=True,
        profile="standard",
        contract="v5",
        config=tmp_path / "series.json",
        database=database,
        output=output,
        artifacts=artifacts,
        checkpoint_directory=None,
        from_env=True,
        expected_cutoff=TARGET,
        collection_report=report,
        require_ac_power=False,
    )

    with pytest.raises(BuildIntercept):
        cli.command_build(args)

    assert captured["checkpoint_directory"] == (
        output.parent / ".private-checkpoints" / "base-walk-forward"
    )
    assert captured["source_fingerprint_sha256"] == source_fingerprint
    assert captured["minimum_log_loss_improvement"] == 0.01
    receipt = json.loads(report.read_text(encoding="utf-8"))
    assert receipt["ready_for_training"] is True
    assert receipt["gate_error"] is None
    assert [source["id"] for source in receipt["sources"]] == [
        "alpha_vantage",
        "alfred",
        "frb_h10",
    ]
    assert receipt["sources"][-1]["status"] == "ok"
    assert receipt["sources"][-1]["issues"] == []
    assert h10_calls == ["archive", "current"]


def test_v5_h10_degradation_updates_report_and_stops_before_analysis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import regime_lab.data as data_module
    import regime_lab.h10_store as h10_store_module

    class DummyStore:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

    database = tmp_path / "regime.sqlite3"
    report = tmp_path / "collection-report.json"
    monkeypatch.setattr(cli, "load_config", lambda _path: {})
    monkeypatch.setattr(cli, "_mutable_path", lambda value, **_kwargs: Path(value))
    monkeypatch.setattr(
        cli,
        "verify_v5_preflight",
        lambda **_kwargs: {"source_fingerprint_sha256": "d" * 64},
    )
    monkeypatch.setattr(
        cli,
        "collect_live_data",
        lambda *_args, **_kwargs: _ready_collection(database),
    )
    monkeypatch.setattr(data_module, "SQLiteSnapshotStore", DummyStore)
    monkeypatch.setattr(data_module, "H10ArchiveClient", object)
    monkeypatch.setattr(
        h10_store_module,
        "refresh_h10_archive_store",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        h10_store_module,
        "refresh_h10_store",
        lambda *_args, **_kwargs: argparse.Namespace(
            fx_features=None,
            fx_context=None,
            source_row={
                "id": "frb_h10",
                "status": "degraded",
                "issues": ["h10_collection_failed_last_good_retained"],
            },
        ),
    )
    monkeypatch.setattr(
        cli,
        "build_weekly_dataset",
        lambda *_args, **_kwargs: pytest.fail("analysis must not start"),
    )
    monkeypatch.setattr(
        cli,
        "build_dashboard_result",
        lambda *_args, **_kwargs: pytest.fail("model training must not start"),
    )
    args = argparse.Namespace(
        alfred_rights_confirmed=True,
        profile="standard",
        contract="v5",
        config=tmp_path / "series.json",
        database=database,
        output=tmp_path / "result.json",
        artifacts=tmp_path / "artifacts",
        checkpoint_directory=None,
        from_env=True,
        expected_cutoff=TARGET,
        collection_report=report,
        require_ac_power=False,
    )

    with pytest.raises(SystemExit, match="frb_h10 source health is not ok"):
        cli.command_build(args)

    receipt = json.loads(report.read_text(encoding="utf-8"))
    assert receipt["ready_for_training"] is False
    assert receipt["overall_health"] == "degraded"
    assert receipt["gate_error"] == "frb_h10 source health is not ok"
    assert receipt["sources"][-1]["status"] == "degraded"
    assert receipt["issues"] == ["h10_collection_failed_last_good_retained"]


def test_v5_h10_exception_records_unavailable_before_propagating(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import regime_lab.data as data_module
    import regime_lab.h10_store as h10_store_module

    class DummyStore:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

    database = tmp_path / "regime.sqlite3"
    report = tmp_path / "collection-report.json"
    monkeypatch.setattr(cli, "load_config", lambda _path: {})
    monkeypatch.setattr(cli, "_mutable_path", lambda value, **_kwargs: Path(value))
    monkeypatch.setattr(
        cli,
        "verify_v5_preflight",
        lambda **_kwargs: {"source_fingerprint_sha256": "e" * 64},
    )
    monkeypatch.setattr(
        cli,
        "collect_live_data",
        lambda *_args, **_kwargs: _ready_collection(database),
    )
    monkeypatch.setattr(data_module, "SQLiteSnapshotStore", DummyStore)
    monkeypatch.setattr(data_module, "H10ArchiveClient", object)
    monkeypatch.setattr(
        h10_store_module,
        "refresh_h10_archive_store",
        lambda *_args, **_kwargs: object(),
    )

    def fail_refresh(*_args, **_kwargs):
        raise RuntimeError("H.10 endpoint unavailable")

    monkeypatch.setattr(h10_store_module, "refresh_h10_store", fail_refresh)
    monkeypatch.setattr(
        cli,
        "build_weekly_dataset",
        lambda *_args, **_kwargs: pytest.fail("analysis must not start"),
    )
    args = argparse.Namespace(
        alfred_rights_confirmed=True,
        profile="standard",
        contract="v5",
        config=tmp_path / "series.json",
        database=database,
        output=tmp_path / "result.json",
        artifacts=tmp_path / "artifacts",
        checkpoint_directory=None,
        from_env=True,
        expected_cutoff=TARGET,
        collection_report=report,
        require_ac_power=False,
    )

    with pytest.raises(RuntimeError, match="H.10 endpoint unavailable"):
        cli.command_build(args)

    receipt = json.loads(report.read_text(encoding="utf-8"))
    assert receipt["ready_for_training"] is False
    assert receipt["overall_health"] == "degraded"
    assert receipt["gate_error"] == "frb_h10 refresh failed before training"
    assert receipt["sources"][-1]["status"] == "unavailable"
    assert receipt["sources"][-1]["issues"] == [
        "h10_refresh_failed_before_training"
    ]
