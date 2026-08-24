from __future__ import annotations

from contextlib import contextmanager

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


@pytest.mark.parametrize(
    "argv",
    (
        ["alfred", "--alfred-rights-confirmed"],
        ["alpha_vantage"],
    ),
)
def test_reviewed_policy_allows_provider_smoke_to_reach_keychain(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
) -> None:
    calls: list[bool] = []

    @contextmanager
    def fake_provider_environment_from_keychain(*, rights_acknowledged: bool):
        calls.append(rights_acknowledged)
        raise RuntimeError("keychain boundary reached")
        yield  # pragma: no cover - makes this a context manager generator

    monkeypatch.setattr(
        "regime_lab.smoke.provider_environment_from_keychain",
        fake_provider_environment_from_keychain,
    )

    with pytest.raises(RuntimeError, match="keychain boundary reached"):
        main(argv)
    assert calls == ["--alfred-rights-confirmed" in argv]
