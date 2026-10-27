from pathlib import Path

import pytest

from regime_lab import cli, path_safety


def test_explicit_state_root_routes_build_writes_and_does_not_change_read_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, state, temporary = [tmp_path / name for name in ("source", "state", "temporary")]
    for directory in (source, state, temporary):
        directory.mkdir()
    monkeypatch.setattr(cli, "project_root", lambda: source)
    monkeypatch.setattr(path_safety.tempfile, "gettempdir", lambda: str(temporary))
    original = source / "editor.txt"
    original.write_text("uncommitted editor content")

    def build(args):
        assert cli._root_path("config/series.json") == source / "config/series.json"
        for name in ("data/regime.sqlite3", "build/result.json", "build/artifacts/data.csv",
                     "build/backups/receipt.json", "build/checkpoints/receipt.json"):
            output = cli._mutable_path(name, label="build output")
            assert output == state / name
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("state output")
        with pytest.raises(path_safety.UnsafeMutablePath):
            cli._mutable_path(original, label="editor overwrite")
        return 0

    monkeypatch.setattr(cli, "command_build", build)
    assert cli.main(["build", "--state-root", str(state)]) == 0
    assert original.read_text() == "uncommitted editor content"
    assert cli._mutable_path("build/default.json", label="default output") == source / "build/default.json"


def test_explicit_state_root_preserves_escape_and_symlink_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, state, temporary = [tmp_path / name for name in ("source", "state", "temporary")]
    for directory in (source, state, temporary):
        directory.mkdir()
    (state / "alias").symlink_to(state, target_is_directory=True)
    monkeypatch.setattr(cli, "project_root", lambda: source)
    monkeypatch.setattr(path_safety.tempfile, "gettempdir", lambda: str(temporary))

    def build(args):
        for name in ("../escape.json", "alias/escaped.json"):
            with pytest.raises(path_safety.UnsafeMutablePath):
                cli._mutable_path(name, label="write target")
        raise RuntimeError("stop before collection")

    monkeypatch.setattr(cli, "command_build", build)
    with pytest.raises(RuntimeError, match="stop before collection"):
        cli.main(["build", "--state-root", str(state)])
    assert cli._mutable_path("after.json", label="write target") == source / "after.json"

