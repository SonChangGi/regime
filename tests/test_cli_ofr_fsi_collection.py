from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from regime_lab import cli
from regime_lab.automation import AlreadyRunning, automation_lock
import regime_lab.data as data_module
from regime_lab.data.ofr_fsi import load_ofr_fsi_contract, parse_ofr_fsi_csv
from regime_lab.provider_rights import ProviderRightsError


UTC = timezone.utc
ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = (ROOT / "tests" / "fixtures" / "ofr_fsi.csv").read_bytes()


class FixtureClient:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def collect(self):
        at = datetime.now(UTC)
        return parse_ofr_fsi_csv(
            PAYLOAD,
            load_ofr_fsi_contract(),
            first_seen_at=at,
            retrieved_at=at,
        )


def test_collect_ofr_fsi_parser_requires_explicit_v6_opt_in() -> None:
    parser = cli.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["collect-ofr-fsi"])
    with pytest.raises(SystemExit):
        parser.parse_args(["collect-ofr-fsi", "--contract", "v5"])

    args = parser.parse_args(["collect-ofr-fsi", "--contract", "v6"])
    assert args.contract == "v6"
    assert args.database is None
    assert args.receipt is None
    assert args.func is cli.command_collect_ofr_fsi


def test_collect_ofr_fsi_checks_rights_before_resolving_write_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = cli.build_parser().parse_args(
        ["collect-ofr-fsi", "--contract", "v6"]
    )

    def blocked(*_args: object, **_kwargs: object) -> None:
        raise ProviderRightsError("ofr_fsi rights blocked")

    monkeypatch.setattr(cli, "verify_provider_rights", blocked)
    monkeypatch.setattr(
        cli,
        "_resolve_ofr_fsi_collection_targets",
        lambda **_kwargs: pytest.fail("rights failure must precede target resolution"),
    )

    with pytest.raises(SystemExit, match="ofr_fsi rights blocked"):
        args.func(args)


def test_collect_ofr_fsi_isolated_command_writes_private_db_and_value_free_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = tmp_path / "ofr-shadow.sqlite3"
    receipt = tmp_path / "ofr-receipt.json"
    monkeypatch.setattr(data_module, "OFRFSIClient", FixtureClient)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("OFR shadow collection must not train, build, or publish")

    monkeypatch.setattr(cli, "collect_live_data", forbidden)
    monkeypatch.setattr(cli, "build_weekly_dataset", forbidden)
    monkeypatch.setattr(cli, "build_dashboard_result", forbidden)
    monkeypatch.setattr(cli, "_publish_active_generation", forbidden)
    args = cli.build_parser().parse_args(
        [
            "collect-ofr-fsi",
            "--contract",
            "v6",
            "--database",
            str(database),
            "--receipt",
            str(receipt),
        ]
    )

    assert args.func(args) == 0
    assert database.is_file()
    assert receipt.is_file()
    document = json.loads(receipt.read_text(encoding="utf-8"))
    assert document["operation"] == "collect_ofr_fsi"
    assert document["collection_status"] == "ok"
    assert document["public_package_inclusion"] is False
    assert document["raw_payload_publication"] is False
    assert document["effective_record_count"] == 27
    assert '"values"' not in json.dumps(document)
    stdout = json.loads(capsys.readouterr().out)
    assert stdout["receipt"] == str(receipt)


@pytest.mark.parametrize(
    ("database", "receipt"),
    [
        ("data/regime.sqlite3", "build/v6-ofr-fsi/receipt.json"),
        ("data/ofr-private.sqlite3", "web/data/ofr.json"),
        ("data/ofr-private.sqlite3", "publication/live/ofr.json"),
        ("build/weekly-automation/ofr.sqlite3", "build/v6-ofr-fsi/receipt.json"),
    ],
)
def test_collect_ofr_fsi_rejects_core_public_and_automation_targets(
    database: str,
    receipt: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        data_module,
        "OFRFSIClient",
        lambda *_args, **_kwargs: pytest.fail("unsafe target must precede client"),
    )
    args = cli.build_parser().parse_args(
        [
            "collect-ofr-fsi",
            "--contract",
            "v6",
            "--database",
            database,
            "--receipt",
            receipt,
        ]
    )

    with pytest.raises(ValueError, match="operating/public target"):
        args.func(args)


def test_collect_ofr_fsi_uses_its_own_lock_not_the_live_build_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "ofr.sqlite3"
    receipt = tmp_path / "receipt.json"
    own_lock = database.with_name(f"{database.name}.ofr-fsi-collect.lock")
    live_lock = database.with_name(f"{database.name}.live-build.lock")
    monkeypatch.setattr(
        data_module,
        "OFRFSIClient",
        lambda *_args, **_kwargs: pytest.fail("lock collision must precede client"),
    )
    args = cli.build_parser().parse_args(
        [
            "collect-ofr-fsi",
            "--contract",
            "v6",
            "--database",
            str(database),
            "--receipt",
            str(receipt),
        ]
    )

    with automation_lock(own_lock):
        with pytest.raises(SystemExit, match="another shadow collection owns"):
            args.func(args)

    assert not receipt.exists()
    assert not database.exists()
    assert not live_lock.exists()


def test_direct_command_call_rejects_missing_contract() -> None:
    with pytest.raises(SystemExit, match="explicit --contract v6"):
        cli.command_collect_ofr_fsi(
            type(
                "Args",
                (),
                {"contract": None, "database": None, "receipt": None, "as_of": None},
            )()
        )
