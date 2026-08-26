from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from regime_lab import cli
from regime_lab.config import project_root
from regime_lab.schema import ContractError


@pytest.mark.parametrize("command", ("build", "demo"))
def test_active_v5_is_the_default_cli_contract(command: str) -> None:
    args = cli.build_parser().parse_args([command])

    assert args.contract == "v5"
    assert args.output is None
    assert args.artifacts is None


def test_validate_and_serve_default_to_reviewed_live_v5() -> None:
    parser = cli.build_parser()

    validate = parser.parse_args(["validate"])
    serve = parser.parse_args(["serve"])

    assert validate.path == "publication/live/regime-results.json"
    assert serve.payload == "publication/live/regime-results.json"
    assert serve.comparison is None


def test_serve_auto_selects_payload_sibling_comparison(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = tmp_path / "regime-results.json"
    comparison = tmp_path / "v5-vs-v4-comparison.json"
    payload.write_text(
        '{"meta":{"result_version":"weekly-regime-result-v5"}}',
        encoding="utf-8",
    )
    comparison.write_text("{}", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_serve(
        web_root: Path,
        host: str,
        port: int,
        *,
        payload_bytes: bytes,
        comparison_bytes: bytes | None,
    ) -> None:
        captured.update(
            web_root=web_root,
            host=host,
            port=port,
            payload_bytes=payload_bytes,
            comparison_bytes=comparison_bytes,
        )

    monkeypatch.setattr(cli, "serve_dashboard", fake_serve)
    monkeypatch.setattr(cli, "validate_dashboard_payload", lambda _payload: None)
    monkeypatch.setattr(
        cli,
        "validate_v5_comparison_sidecar",
        lambda *_args, **_kwargs: None,
    )
    args = SimpleNamespace(
        web_root="web",
        host="127.0.0.1",
        port=9988,
        payload=str(payload),
        comparison=None,
    )

    assert cli.command_serve(args) == 0
    assert captured == {
        "web_root": project_root() / "web",
        "host": "127.0.0.1",
        "port": 9988,
        "payload_bytes": payload.read_bytes(),
        "comparison_bytes": comparison.read_bytes(),
    }


def test_serve_rejects_invalid_payload_before_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = tmp_path / "regime-results.json"
    payload.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        cli,
        "serve_dashboard",
        lambda *_args, **_kwargs: pytest.fail("invalid payload must not be served"),
    )
    args = SimpleNamespace(
        web_root="web",
        host="127.0.0.1",
        port=9988,
        payload=str(payload),
        comparison=None,
    )

    with pytest.raises(ContractError):
        cli.command_serve(args)


@pytest.mark.parametrize(
    ("command", "contract", "expected_output", "expected_artifacts"),
    (
        (
            "build",
            "v4",
            "web/data/regime-results.json",
            "artifacts/latest",
        ),
        (
            "demo",
            "v4",
            "web/data/regime-results.json",
            "artifacts/demo",
        ),
        (
            "build",
            "v5",
            "build/v5-live/regime-results.json",
            "build/v5-live/artifacts",
        ),
        (
            "demo",
            "v5",
            "build/v5-demo/regime-results.json",
            "build/v5-demo/artifacts",
        ),
    ),
)
def test_contract_defaults_resolve_to_separate_targets(
    command: str,
    contract: str,
    expected_output: str,
    expected_artifacts: str,
) -> None:
    output, artifacts = cli._resolve_contract_write_targets(
        command=command,
        contract_version=contract,
        output=None,
        artifacts=None,
    )

    root = project_root()
    assert output == root / expected_output
    assert artifacts == root / expected_artifacts


@pytest.mark.parametrize("command", ("build", "demo"))
@pytest.mark.parametrize("contract", ("v4", "v5"))
def test_parser_leaves_contract_dependent_targets_for_resolver(
    command: str,
    contract: str,
) -> None:
    args = cli.build_parser().parse_args(
        [command, "--contract", contract]
    )

    assert args.output is None
    assert args.artifacts is None
    assert args.contract == contract
    if command == "build":
        assert args.checkpoint_directory is None


def test_live_checkpoint_defaults_to_private_v5_output_sibling(
    tmp_path: Path,
) -> None:
    output = tmp_path / "v5-live" / "regime-results.json"
    artifacts = tmp_path / "v5-live" / "artifacts"

    checkpoint = cli._resolve_live_checkpoint_directory(
        contract_version="v5",
        output=output,
        artifacts=artifacts,
        value=None,
    )

    assert checkpoint == (
        output.parent / ".private-checkpoints" / "base-walk-forward"
    )
    assert cli._resolve_live_checkpoint_directory(
        contract_version="v4",
        output=output,
        artifacts=artifacts,
        value=None,
    ) is None


def test_explicit_checkpoint_is_v5_only_and_must_be_isolated(
    tmp_path: Path,
) -> None:
    output = tmp_path / "v5-live" / "regime-results.json"
    artifacts = tmp_path / "v5-live" / "artifacts"
    with pytest.raises(ValueError, match="only for V5"):
        cli._resolve_live_checkpoint_directory(
            contract_version="v4",
            output=output,
            artifacts=artifacts,
            value=tmp_path / "checkpoint",
        )
    with pytest.raises(ValueError, match="must not overlap"):
        cli._resolve_live_checkpoint_directory(
            contract_version="v5",
            output=output,
            artifacts=artifacts,
            value=artifacts / "checkpoint",
        )

    args = cli.build_parser().parse_args(
        [
            "build",
            "--contract",
            "v5",
            "--checkpoint-directory",
            str(tmp_path / "checkpoint"),
            "--alfred-rights-confirmed",
        ]
    )
    assert args.checkpoint_directory == str(tmp_path / "checkpoint")


def test_v4_explicit_checkpoint_is_rejected_before_config_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = cli.build_parser().parse_args(
        [
            "build",
            "--contract",
            "v4",
            "--checkpoint-directory",
            str(tmp_path / "checkpoint"),
            "--alfred-rights-confirmed",
        ]
    )
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda *_args: pytest.fail("V4 checkpoint must fail before config load"),
    )
    with pytest.raises(ValueError, match="only for V5"):
        args.func(args)


def test_v5_accepts_explicit_isolated_targets(tmp_path: Path) -> None:
    output, artifacts = cli._resolve_contract_write_targets(
        command="build",
        contract_version="v5",
        output=tmp_path / "result.json",
        artifacts=tmp_path / "artifacts",
    )

    assert output == tmp_path / "result.json"
    assert artifacts == tmp_path / "artifacts"


@pytest.mark.parametrize(
    ("output", "artifacts"),
    (
        ("build/v5-custom/result.json", "build/v5-custom/result.json"),
        ("build/v5-custom", "build/v5-custom/artifacts"),
        ("build/v5-custom/artifacts/result.json", "build/v5-custom/artifacts"),
    ),
)
def test_contract_targets_reject_payload_artifact_overlap(
    output: str,
    artifacts: str,
) -> None:
    with pytest.raises(ValueError, match="must not overlap"):
        cli._resolve_contract_write_targets(
            command="build",
            contract_version="v5",
            output=output,
            artifacts=artifacts,
        )


@pytest.mark.parametrize(
    ("output", "artifacts"),
    (
        ("web/data/regime-results.json", "build/v5-custom/artifacts"),
        ("publication/live/regime-results.json", "build/v5-custom/artifacts"),
        ("build/v5-custom/result.json", "artifacts/latest"),
        ("build/v5-custom/result.json", "artifacts/demo"),
        ("build/v5-custom/result.json", "artifacts"),
        (
            "build/weekly-automation/generation/regime-results.json",
            "build/v5-custom/artifacts",
        ),
        (
            "build/v5-custom/result.json",
            "build/weekly-automation/generation/artifacts",
        ),
    ),
)
def test_v5_rejects_explicit_v4_owned_or_overlapping_targets(
    output: str,
    artifacts: str,
) -> None:
    with pytest.raises(ValueError, match="overlaps a v4-owned target"):
        cli._resolve_contract_write_targets(
            command="build",
            contract_version="v5",
            output=output,
            artifacts=artifacts,
        )


def test_v5_rejects_absolute_spelling_of_v4_target() -> None:
    with pytest.raises(ValueError, match="overlaps a v4-owned target"):
        cli._resolve_contract_write_targets(
            command="demo",
            contract_version="v5",
            output=project_root() / "web/data/regime-results.json",
            artifacts=None,
        )


@pytest.mark.parametrize(
    "argv",
    (
        (
            "build",
            "--contract",
            "v5",
            "--artifacts",
            "artifacts/latest",
            "--alfred-rights-confirmed",
        ),
        (
            "demo",
            "--contract",
            "v5",
            "--output",
            "web/data/regime-results.json",
        ),
    ),
)
def test_v5_commands_reject_v4_targets_before_loading_config(
    argv: tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = cli.build_parser().parse_args(argv)
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda *_args: pytest.fail("unsafe target must fail before config load"),
    )

    with pytest.raises(ValueError, match="overlaps a v4-owned target"):
        args.func(args)


def test_contract_target_resolver_rejects_unknown_modes() -> None:
    with pytest.raises(ValueError, match="require build/demo and v4/v5"):
        cli._resolve_contract_write_targets(
            command="serve",
            contract_version="v5",
            output=None,
            artifacts=None,
        )
