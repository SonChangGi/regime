from pathlib import Path
import pytest
from scripts.build_comprehensive_preview import publish_preview_generation


def test_failed_preview_stage_preserves_previous_generation(tmp_path, monkeypatch):
    output = tmp_path / "data"
    publish_preview_generation(
        output, {"regime-results.json": b"old", "regime-core.json": b"oldcore"}
    )
    previous = output.resolve()
    write = Path.write_bytes

    def failed(self, data):
        if self.name == "regime-core.json":
            raise OSError("injected interrupted staging")
        return write(self, data)

    monkeypatch.setattr(Path, "write_bytes", failed)
    with pytest.raises(OSError):
        publish_preview_generation(
            output, {"regime-results.json": b"new", "regime-core.json": b"newcore"}
        )
    assert output.resolve() == previous
    assert (output / "regime-results.json").read_bytes() == b"old"


def test_preview_cutover_preserves_old_complete_directory(tmp_path):
    output = tmp_path / "data"
    output.mkdir()
    (output / "regime-results.json").write_bytes(b"first")
    publish_preview_generation(
        output, {"regime-results.json": b"second", "regime-core.json": b"core"}
    )
    assert output.is_symlink()
    assert (output / "regime-results.json").read_bytes() == b"second"
    assert (
        next(
            (tmp_path / "generations").glob("previous-*/regime-results.json")
        ).read_bytes()
        == b"first"
    )
