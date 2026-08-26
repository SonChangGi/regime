from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd
import pytest

from regime_lab import cli
from regime_lab.analysis.fx import FXFeatureResult
from regime_lab.automation import AlreadyRunning, automation_lock
from regime_lab.data import CollectionResult, PreparedSnapshot, SnapshotMode
import regime_lab.data as data_module
import regime_lab.h10_store as h10_store
from regime_lab.h10_store import (
    H10StoreRefresh,
    h10_collection_receipt_document,
)
from regime_lab.provider_rights import ProviderRightsError


UTC = timezone.utc
AS_OF = datetime(2026, 8, 21, 20, tzinfo=UTC)


def _refresh(*, fx_result: FXFeatureResult | None = None) -> H10StoreRefresh:
    prepared = PreparedSnapshot(
        snapshot_result=CollectionResult(records=()),
        effective_records=(),
        snapshot_mode=SnapshotMode.DELTA,
        added_count=3,
        changed_count=2,
        unchanged_count=7,
        removed_count=1,
    )
    return H10StoreRefresh(
        snapshot_id="private-snapshot-id",
        prepared=prepared,
        effective_records=(),
        fx_features=fx_result,
        source_row={
            "status": "degraded",
            "issues": ["h10_collection_failed_last_good_retained"],
            "official_release_archive_ingest": False,
            "availability_basis": "collection_first_seen_at",
            "archive_revision_policy": (
                "later_official_release_preserved_as_new_vintage"
            ),
            "archive_correction_availability_basis": (
                "date_only_conservative_next_day"
            ),
        },
        fx_context={"status": "degraded"},
        used_last_good=True,
    )


def _fx_result() -> FXFeatureResult:
    weeks = pd.date_range("2026-08-07", periods=2, freq="W-FRI")
    features = pd.DataFrame(
        {
            "fx__brd__usd_log_return_1w": [0.1, 0.2],
            "fx__eur__usd_log_return_1w": [0.3, 0.4],
        },
        index=weeks,
    )
    coverage = pd.DataFrame(
        {
            "feature_available_at": pd.to_datetime(
                ["2026-08-10T20:15:00Z", "2026-08-17T20:15:00Z"],
                utc=True,
            )
        },
        index=weeks,
    )
    empty = pd.DataFrame(index=weeks)
    return FXFeatureResult(
        features=features,
        weekly_usd_log_levels=empty,
        weekly_availability=empty,
        coverage=coverage,
        status=empty,
    )


def test_collect_h10_parser_requires_explicit_v5_opt_in() -> None:
    parser = cli.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["collect-h10"])
    with pytest.raises(SystemExit):
        parser.parse_args(["collect-h10", "--contract", "v4"])

    args = parser.parse_args(["collect-h10", "--contract", "v5"])
    assert args.contract == "v5"
    assert args.receipt is None
    assert args.func is cli.command_collect_h10

    archive = parser.parse_args(
        [
            "collect-h10",
            "--contract",
            "v5",
            "--official-release-archive-ingest",
        ]
    )
    assert archive.official_release_archive_ingest is True
    assert archive.archive_start is None
    assert archive.archive_through is None


def test_collect_h10_checks_provider_rights_before_any_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = cli.build_parser().parse_args(["collect-h10", "--contract", "v5"])

    def blocked(*_args: object, **_kwargs: object) -> None:
        raise ProviderRightsError("frb_h10 rights blocked")

    monkeypatch.setattr(cli, "verify_provider_rights", blocked)
    monkeypatch.setattr(
        cli,
        "_mutable_path",
        lambda *_args, **_kwargs: pytest.fail("write target must not be resolved"),
    )

    with pytest.raises(SystemExit, match="frb_h10 rights blocked"):
        cli.command_collect_h10(args)


def test_receipt_is_derived_only_and_reports_prospective_readiness() -> None:
    document = h10_collection_receipt_document(
        _refresh(fx_result=_fx_result()),
        requested_at=datetime(2026, 8, 21, 21, tzinfo=UTC),
        as_of=AS_OF,
    )

    assert document["contract"] == "v5"
    assert document["operation"] == "collect_h10"
    assert document["snapshot_mode"] == "delta"
    assert document["collection_status"] == "ok"
    assert document["source_status"] == "degraded"
    assert document["fx_status"] == "degraded"
    assert document["last_good_used"] is True
    assert document["added_records"] == 3
    assert document["changed_records"] == 2
    assert document["removed_records"] == 1
    assert document["effective_record_count"] == 0
    assert document["eligible_common_weeks"] == 2
    assert document["readiness"] == "insufficient_history"
    assert document["minimum_common_weeks"] == 156
    assert document["first_eligible_cutoff"] == "2026-08-14"
    assert document["last_eligible_cutoff"] == "2026-08-21"
    assert document["historical_availability_backfill"] is False

    encoded = json.dumps(document, sort_keys=True)
    for forbidden in (
        "private-snapshot-id",
        "snapshot_id",
        "source_url",
        "raw_sha256",
        "raw_value_token",
        "request_params",
    ):
        assert forbidden not in encoded


def test_collect_h10_shares_live_build_lock_and_writes_only_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = tmp_path / "regime.sqlite3"
    receipt = tmp_path / "h10-receipt.json"
    lock = database.with_name(f"{database.name}.live-build.lock")
    client = object()
    refresh = _refresh()
    calls: list[tuple[datetime, datetime]] = []

    monkeypatch.setattr(data_module, "H10Client", lambda: client)

    def fake_refresh(store, actual_client, *, requested_at, as_of):
        del store
        assert actual_client is client
        with pytest.raises(AlreadyRunning):
            with automation_lock(lock):
                pass
        calls.append((requested_at, as_of))
        return refresh

    monkeypatch.setattr(h10_store, "refresh_h10_store", fake_refresh)

    def forbidden(*_args, **_kwargs):
        pytest.fail("standalone H.10 collection must not train or publish")

    monkeypatch.setattr(cli, "collect_live_data", forbidden)
    monkeypatch.setattr(cli, "build_weekly_dataset", forbidden)
    monkeypatch.setattr(cli, "build_dashboard_result", forbidden)
    monkeypatch.setattr(cli, "_publish_active_generation", forbidden)

    args = cli.build_parser().parse_args(
        [
            "collect-h10",
            "--contract",
            "v5",
            "--database",
            str(database),
            "--receipt",
            str(receipt),
            "--as-of",
            AS_OF.isoformat(),
        ]
    )
    assert args.func(args) == 0

    assert len(calls) == 1
    assert calls[0][0].tzinfo is not None
    assert calls[0][1] == AS_OF
    document = json.loads(receipt.read_text(encoding="utf-8"))
    assert document["source_status"] == "degraded"
    assert document["readiness"] == "unavailable"
    assert document["eligible_common_weeks"] == 0
    output = json.loads(capsys.readouterr().out)
    assert output["receipt"] == str(receipt)


def test_collect_h10_lock_collision_stops_before_client_or_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "regime.sqlite3"
    receipt = tmp_path / "h10-receipt.json"
    lock = database.with_name(f"{database.name}.live-build.lock")
    monkeypatch.setattr(
        data_module,
        "H10Client",
        lambda: pytest.fail("lock collision must stop before provider setup"),
    )
    args = cli.build_parser().parse_args(
        [
            "collect-h10",
            "--contract",
            "v5",
            "--database",
            str(database),
            "--receipt",
            str(receipt),
        ]
    )

    with automation_lock(lock):
        with pytest.raises(SystemExit, match="another build owns"):
            args.func(args)

    assert not receipt.exists()
    assert not database.exists()


def test_collect_h10_archive_routes_only_to_isolated_archive_refresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "archive.sqlite3"
    receipt = tmp_path / "archive-receipt.json"
    client = object()
    calls: list[tuple[datetime, datetime, object, object]] = []
    monkeypatch.setattr(data_module, "H10ArchiveClient", lambda: client)

    def fake_archive_refresh(
        store,
        actual_client,
        *,
        requested_at,
        as_of,
        start_date,
        end_date,
    ):
        del store
        assert actual_client is client
        calls.append((requested_at, as_of, start_date, end_date))
        return _refresh()

    monkeypatch.setattr(
        h10_store,
        "refresh_h10_archive_store",
        fake_archive_refresh,
    )
    monkeypatch.setattr(
        h10_store,
        "refresh_h10_store",
        lambda *_args, **_kwargs: pytest.fail("archive opt-in must not use XML refresh"),
    )
    args = cli.build_parser().parse_args(
        [
            "collect-h10",
            "--contract",
            "v5",
            "--official-release-archive-ingest",
            "--database",
            str(database),
            "--receipt",
            str(receipt),
            "--as-of",
            AS_OF.isoformat(),
            "--archive-start",
            "2022-01-01",
            "--archive-through",
            "2026-08-21",
        ]
    )

    assert args.func(args) == 0
    assert len(calls) == 1
    assert calls[0][1] == AS_OF
    assert str(calls[0][2]) == "2022-01-01"
    assert str(calls[0][3]) == "2026-08-21"
    assert receipt.exists()


def test_archive_bounds_and_non_friday_cutoff_fail_before_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        data_module,
        "H10Client",
        lambda: pytest.fail("invalid CLI contract must stop before provider setup"),
    )
    monkeypatch.setattr(
        cli,
        "_backup_database_before_mutation",
        lambda *_args, **_kwargs: pytest.fail(
            "invalid CLI contract must stop before database backup"
        ),
    )
    bounded_without_opt_in = cli.build_parser().parse_args(
        [
            "collect-h10",
            "--contract",
            "v5",
            "--archive-start",
            "2022-01-01",
        ]
    )
    with pytest.raises(SystemExit, match="require --official"):
        bounded_without_opt_in.func(bounded_without_opt_in)

    invalid_cutoff = cli.build_parser().parse_args(
        [
            "collect-h10",
            "--contract",
            "v5",
            "--as-of",
            "2026-08-20T20:00:00+00:00",
        ]
    )
    with pytest.raises(SystemExit, match="exact completed Friday"):
        invalid_cutoff.func(invalid_cutoff)


def test_collect_h10_rejects_v4_receipt_target_before_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        data_module,
        "H10Client",
        lambda: pytest.fail("unsafe receipt must fail before provider setup"),
    )
    args = cli.build_parser().parse_args(
        [
            "collect-h10",
            "--contract",
            "v5",
            "--receipt",
            "web/data/regime-results.json",
        ]
    )

    with pytest.raises(ValueError, match="overlaps a v4-owned target"):
        args.func(args)


@pytest.mark.parametrize(
    "receipt",
    (
        "build/weekly-automation/automation-health.json",
        "build/v5-live/artifacts/h10-receipt.json",
        "build/v5-demo/regime-results.json",
    ),
)
def test_collect_h10_rejects_automation_and_model_result_targets(
    receipt: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        data_module,
        "H10Client",
        lambda: pytest.fail("unsafe receipt must fail before provider setup"),
    )
    args = cli.build_parser().parse_args(
        ["collect-h10", "--contract", "v5", "--receipt", receipt]
    )

    with pytest.raises(ValueError, match="automation or model-result target"):
        args.func(args)


def test_direct_command_call_rejects_missing_contract() -> None:
    args = argparse.Namespace(
        contract=None,
        database="data/regime.sqlite3",
        receipt=None,
        as_of=None,
    )

    with pytest.raises(SystemExit, match="explicit --contract v5"):
        cli.command_collect_h10(args)


def test_collect_h10_backs_up_inside_lock_before_provider_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(cli, "verify_provider_rights", lambda *_args, **_kwargs: None)
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

    def client():
        events.append("provider")
        raise RuntimeError("stop after ordering assertion")

    monkeypatch.setattr(data_module, "H10Client", client)
    args = cli.build_parser().parse_args(
        [
            "collect-h10",
            "--contract",
            "v5",
            "--database",
            str(tmp_path / "regime.sqlite3"),
            "--receipt",
            str(tmp_path / "receipt.json"),
            "--as-of",
            AS_OF.isoformat(),
        ]
    )

    with pytest.raises(RuntimeError, match="ordering assertion"):
        cli.command_collect_h10(args)

    assert events == ["backup", "provider"]
