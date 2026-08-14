from __future__ import annotations

import pytest

from regime_lab.cli import build_parser
from regime_lab.smoke import main


def test_alfred_smoke_requires_explicit_rights_confirmation() -> None:
    with pytest.raises(SystemExit, match="alfred-rights-confirmed"):
        main(["alfred"])


def test_cli_smoke_uses_shared_database_and_rights_flag() -> None:
    args = build_parser().parse_args(
        [
            "smoke",
            "all",
            "--database",
            "data/test.sqlite3",
            "--alfred-rights-confirmed",
        ]
    )
    assert args.database == "data/test.sqlite3"
    assert args.alfred_rights_confirmed is True
