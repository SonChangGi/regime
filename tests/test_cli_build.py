from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from regime_lab import cli
from regime_lab.automation import automation_lock
from regime_lab.collection import LiveCollection
from regime_lab.data import HealthStatus


TARGET = datetime(2026, 8, 14, 20, tzinfo=timezone.utc)


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
