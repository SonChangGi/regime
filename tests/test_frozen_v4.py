from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import pytest

import regime_lab.frozen_v4 as frozen_v4
import regime_lab.pipeline as pipeline


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _baseline_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    artifact_generation_id: str = "frozen-generation",
) -> tuple[Path, Path, dict[str, object]]:
    root = tmp_path / "project"
    directory = root / "artifacts/baselines/v4-20260821"
    directory.mkdir(parents=True)
    payload = {
        "meta": {
            "result_version": "weekly-regime-result-v4",
            "generation_id": "frozen-generation",
            "data_as_of": "2026-08-07T20:00:00+00:00",
            "mode": "live",
        },
        "model": {
            "version": "weekly-nondl-structural-v4",
            "label_version": "market-causal-3state-v1",
            "feature_set_version": "weekly-pit-structural-v4",
            "champion": "markov",
            "profile": "standard",
        },
    }
    (directory / "regime-results.json").write_text(
        json.dumps(payload, sort_keys=True),
        encoding="utf-8",
    )
    (directory / "build-generation.json").write_text(
        json.dumps({"generation_id": artifact_generation_id}, sort_keys=True),
        encoding="utf-8",
    )
    (directory / "oos-predictions.csv").write_text(
        "origin_date,target_date,model\n"
        "2026-07-31,2026-08-07,markov\n"
        "2026-08-07,2026-08-14,markov\n",
        encoding="utf-8",
    )
    files = {
        path.name: path
        for path in directory.iterdir()
        if path.is_file()
    }
    inventory = "".join(
        f"{_sha256(files[name])}  {name}\n" for name in sorted(files)
    ).encode("ascii")
    (directory / "SHA256SUMS").write_bytes(inventory)
    contract: dict[str, object] = {
        **dict(frozen_v4.FROZEN_V4_BASELINE),
        "payload_sha256": _sha256(directory / "regime-results.json"),
        "artifacts_inventory_sha256": hashlib.sha256(inventory).hexdigest(),
        "generation_id": "frozen-generation",
    }
    monkeypatch.setattr(
        frozen_v4,
        "FROZEN_V4_BASELINE",
        MappingProxyType(contract),
    )
    monkeypatch.setattr(
        frozen_v4,
        "FROZEN_V4_OOS_PREDICTIONS",
        MappingProxyType(
            {
                "path": "oos-predictions.csv",
                "row_count": 2,
                "sha256": _sha256(directory / "oos-predictions.csv"),
            }
        ),
    )
    monkeypatch.setattr(frozen_v4, "FROZEN_V4_INVENTORY_FILE_COUNT", 3)
    return root, directory, contract


def test_frozen_v4_verifier_accepts_exact_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _directory, contract = _baseline_fixture(tmp_path, monkeypatch)

    assert frozen_v4.verify_frozen_v4_baseline(
        project_directory=root
    ) == contract


def test_frozen_v4_verifier_rejects_mutated_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, directory, _contract = _baseline_fixture(tmp_path, monkeypatch)
    (directory / "regime-results.json").write_bytes(b"mutated")

    with pytest.raises(
        frozen_v4.FrozenV4BaselineError,
        match="artifact hash does not match: regime-results.json",
    ):
        frozen_v4.verify_frozen_v4_baseline(project_directory=root)


def test_frozen_v4_verifier_rejects_extra_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, directory, _contract = _baseline_fixture(tmp_path, monkeypatch)
    (directory / "unreviewed.csv").write_text("value\n1\n", encoding="utf-8")

    with pytest.raises(
        frozen_v4.FrozenV4BaselineError,
        match="file set does not match",
    ):
        frozen_v4.verify_frozen_v4_baseline(project_directory=root)


def test_frozen_v4_verifier_rejects_generation_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _directory, _contract = _baseline_fixture(
        tmp_path,
        monkeypatch,
        artifact_generation_id="other-generation",
    )

    with pytest.raises(
        frozen_v4.FrozenV4BaselineError,
        match="build generation does not match",
    ):
        frozen_v4.verify_frozen_v4_baseline(project_directory=root)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    (
        ("sha256", "0" * 64, "OOS predictions hash"),
        ("row_count", 1, "OOS predictions row count"),
    ),
)
def test_frozen_v4_verifier_binds_immutable_oos_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    error: str,
) -> None:
    root, directory, _contract = _baseline_fixture(tmp_path, monkeypatch)
    record = {
        "path": "oos-predictions.csv",
        "row_count": 2,
        "sha256": _sha256(directory / "oos-predictions.csv"),
    }
    record[field] = value
    monkeypatch.setattr(
        frozen_v4,
        "FROZEN_V4_OOS_PREDICTIONS",
        MappingProxyType(record),
    )

    with pytest.raises(frozen_v4.FrozenV4BaselineError, match=error):
        frozen_v4.verify_frozen_v4_baseline(project_directory=root)


def test_v5_pipeline_verifies_baseline_before_dataset_analysis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class VerificationReached(RuntimeError):
        pass

    monkeypatch.setattr(
        pipeline,
        "verify_frozen_v4_baseline",
        lambda: (_ for _ in ()).throw(VerificationReached()),
    )

    with pytest.raises(VerificationReached):
        pipeline.build_dashboard_result(
            object(),
            None,
            contract_version="v5",
        )


def test_v4_pipeline_does_not_require_frozen_v5_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def verify() -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(pipeline, "verify_frozen_v4_baseline", verify)

    with pytest.raises(AttributeError):
        pipeline.build_dashboard_result(
            object(),
            None,
            contract_version="v4",
        )
    assert called is False
