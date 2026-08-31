from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "recompose_local_preview.py"
SPEC = importlib.util.spec_from_file_location("recompose_local_preview", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _legacy_payload(*, shadow_count: int = 2) -> dict[str, object]:
    return {
        "forecast": {
            "prospective_ledger": {
                "schema_version": "regime-prospective-ledger-summary/1",
                "status": "recorded",
                "entry_count": 2,
                "key_manifest_sha256": "a" * 64,
                "hash_scope": "ordered_ledger_primary_keys_only",
            }
        },
        "research": {
            "prospective_decision_shadow": {
                "prospective_ledger": {
                    "ledger_entry_count": shadow_count,
                    "realized_evaluation_count": 0,
                }
            }
        },
    }


def test_legacy_pending_ledger_is_upgraded_without_inventing_outcomes() -> None:
    summary = MODULE._prospective_ledger_summary(_legacy_payload())

    assert summary["schema_version"] == "regime-prospective-ledger-summary/2"
    assert summary["status"] == "pending"
    assert summary["entry_count"] == 2
    assert summary["pending_evaluation_count"] == 2
    assert summary["realized_evaluation_count"] == 0
    assert summary["performance"]["weeks"] == 0
    assert summary["performance"]["net_cumulative_return"] is None


def test_legacy_ledger_count_mismatch_is_refused() -> None:
    with pytest.raises(
        MODULE.PreviewRecomposeError,
        match="ledger counts disagree",
    ):
        MODULE._prospective_ledger_summary(_legacy_payload(shadow_count=1))


def test_recompose_preserves_forecast_identity_and_unrelated_research(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matched = pd.DatetimeIndex([pd.Timestamp("2023-01-06", tz="UTC")])
    monkeypatch.setattr(
        MODULE,
        "_conditional_research",
        lambda *args, **kwargs: (
            {"conditional_asset_stats": {"rows": ["actual"]}},
            SimpleNamespace(
                outcomes=pd.DataFrame(),
                statistics=pd.DataFrame(),
            ),
            matched,
        ),
    )
    monkeypatch.setattr(
        MODULE,
        "_model_conditioned_research",
        lambda *args, **kwargs: (
            {"model_conditioned_asset_stats": {"rows": ["forecast"]}},
            pd.DataFrame(),
            pd.DataFrame(),
        ),
    )
    monkeypatch.setattr(
        MODULE,
        "build_decision_shadow",
        lambda *args, **kwargs: {
            "schema_version": "shadow-v2",
            "current_signal": {"action": "no_trade"},
        },
    )
    monkeypatch.setattr(
        MODULE,
        "build_allocation_shadow_candidate",
        lambda *args, **kwargs: {"schema_version": "allocation-v1"},
    )
    monkeypatch.setattr(
        MODULE,
        "allocation_calibration_evidence",
        lambda *args, **kwargs: {"status": "passed"},
    )
    monkeypatch.setattr(MODULE, "reviewed_candidate_payload", lambda value: value)
    monkeypatch.setattr(MODULE, "validate_v5_payload", lambda value: None)
    monkeypatch.setattr(MODULE, "_prospective_ledger_summary", lambda value: None)
    payload = {
        "meta": {
            "generated_at": "2026-08-27T15:09:39+00:00",
            "data_as_of": "2026-08-21T20:00:00+00:00",
        },
        "forecast": {"decision_at": "2026-08-27T15:09:39+00:00"},
        "weekly": [{"date": "2023-01-06"}],
        "selection": {"operating_champion": "champion"},
        "model": {
            "execution_parameters": {
                "conditional_outcome_bootstrap_resamples": 1_999,
            },
            "forecast_comparison": {"models": ["champion"]},
            "selection_end": "2023-01-01",
        },
        "research": {"unchanged_evidence": {"status": "keep"}},
    }

    candidate = MODULE.recompose_payload(
        payload,
        canonical=pd.DataFrame(),
        states=pd.Series(dtype="object"),
    )

    assert candidate["meta"] == payload["meta"]
    assert candidate["forecast"] == payload["forecast"]
    assert candidate["research"]["unchanged_evidence"] == {"status": "keep"}
    assert candidate["research"]["conditional_asset_stats"]["rows"] == [
        "actual"
    ]
    assert candidate["research"]["model_conditioned_asset_stats"]["rows"] == [
        "forecast"
    ]
    assert candidate["research"]["prospective_decision_shadow"] == {
        "schema_version": "shadow-v2",
        "current_signal": {"action": "no_trade"},
        "allocation_candidate": {"schema_version": "allocation-v1"},
    }


def test_canonical_recompose_separates_research_and_operational_input_snapshots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeStore:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def __enter__(self) -> "FakeStore":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read_last_good_observations(self, **kwargs: object) -> list[object]:
            return [] if kwargs else [object()]

    index = pd.date_range(
        end=pd.Timestamp("2026-08-21", tz="UTC"),
        periods=260,
        freq="W-FRI",
    )
    canonical = pd.DataFrame({"spy_close": range(260)}, index=index)
    monkeypatch.setattr(MODULE, "SQLiteSnapshotStore", FakeStore)
    monkeypatch.setattr(
        MODULE,
        "build_weekly_dataset",
        lambda *args, **kwargs: SimpleNamespace(
            canonical=canonical,
            availability_basis="reconstructed_market",
            input_vintages=(object(),),
        ),
    )
    monkeypatch.setattr(
        MODULE,
        "build_research_replay_input_document",
        lambda **kwargs: {
            "input_vintages": {"count": 1, "sha256": "a" * 64},
            "operational_generation_input_snapshot_sha256": kwargs[
                "operational_input_snapshot_sha256"
            ],
        },
    )
    monkeypatch.setattr(
        MODULE,
        "CausalRegimeLabeler",
        lambda *args, **kwargs: SimpleNamespace(
            fit=lambda value: None,
            transform=lambda value: pd.Series("risk_on", index=value.index),
        ),
    )
    payload = {
        "meta": {"data_as_of": "2026-08-21T20:00:00+00:00"},
        "forecast": {
            "origin_at": "2026-08-21T20:00:00+00:00",
            "decision_at": "2026-08-27T15:09:39+00:00",
        },
        "label": {
            "fit_period": {
                "weeks": 260,
                "start": index[0].date().isoformat(),
                "end": index[-1].date().isoformat(),
            }
        },
    }

    _, _, _, snapshot = MODULE._canonical_and_states(
        payload,
        config={},
        database=Path("unused.sqlite3"),
        operational_input_snapshot_sha256="b" * 64,
    )

    assert snapshot["input_vintages"] == {"count": 1, "sha256": "a" * 64}
    assert snapshot["operational_generation_input_snapshot_sha256"] == "b" * 64


def _mock_release_candidate_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    fail_validation_call: int,
) -> tuple[Path, list[Path]]:
    root = tmp_path
    (root / "build").mkdir()
    source_directory = root / "source"
    source_artifacts = source_directory / "artifacts"
    source_artifacts.mkdir(parents=True)
    (source_artifacts / "source-evidence.txt").write_text(
        "frozen",
        encoding="utf-8",
    )
    for name in (
        "regime-results.json",
        "generation-manifest.json",
        "v5-vs-v4-comparison.json",
        "selection-family-audit.json",
    ):
        (source_directory / name).write_text("{}", encoding="utf-8")
    (root / "config.json").write_text("{}", encoding="utf-8")
    (root / "snapshot.sqlite3").write_bytes(b"")

    payload = {
        "meta": {
            "mode": "live",
            "data_as_of": "2026-08-21T20:00:00+00:00",
            "publication_status": "unpublished",
        },
        "weekly": [{}],
        "model": {"research_artifacts": {}},
        "research": {
            "conditional_asset_stats": {"rows": []},
            "model_conditioned_asset_stats": {"rows": []},
            "prospective_decision_shadow": {
                "schema_version": "shadow-v2",
                "historical_reconstructed_shadow": {
                    "strategies": {"probability_shadow": {"weeks": 1}}
                },
            },
        },
    }
    source_generation = {
        "payload": payload,
        "input_snapshot": {
            "data_as_of": "2026-08-21T20:00:00+00:00",
            "sha256": "a" * 64,
        },
        "label_spec": {"path": "labels.json"},
    }
    validation_artifact_directories: list[Path] = []

    def fake_validate_generation_manifest(
        manifest_path: Path,
        **kwargs: object,
    ) -> dict[str, object]:
        artifact_directory = Path(str(kwargs["artifact_directory"]))
        validation_artifact_directories.append(artifact_directory)
        call_number = len(validation_artifact_directories)
        if call_number == 1:
            assert artifact_directory != source_artifacts
            assert (artifact_directory / "source-evidence.txt").read_text(
                encoding="utf-8"
            ) == "frozen"
            return source_generation
        if call_number == fail_validation_call:
            raise MODULE.IntegrityError("forced validation failure")
        return {
            "generation_id": "generation",
            "manifest_sha256": "b" * 64,
        }

    monkeypatch.setattr(MODULE, "project_root", lambda: root)
    monkeypatch.setattr(
        MODULE,
        "validate_generation_manifest",
        fake_validate_generation_manifest,
    )
    monkeypatch.setattr(MODULE, "validate_v5_payload", lambda value: None)
    monkeypatch.setattr(MODULE, "load_config", lambda value: {})
    monkeypatch.setattr(
        MODULE,
        "_canonical_and_states",
        lambda *args, **kwargs: (
            pd.DataFrame(),
            pd.Series(dtype="object"),
            1,
            {
                "input_vintages": {"count": 1},
                "operational_generation_input_snapshot_sha256": "a" * 64,
            },
        ),
    )
    monkeypatch.setattr(
        MODULE,
        "_recompose_payload_with_frames",
        lambda *args, **kwargs: (payload, {}),
    )
    monkeypatch.setattr(
        MODULE,
        "build_selection_family_audit_from_artifacts",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(MODULE, "write_artifact_inventory", lambda value: None)
    monkeypatch.setattr(
        MODULE,
        "verify_staged_v5_research_artifacts",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        MODULE,
        "build_generation_manifest",
        lambda **kwargs: {"schema_version": "test"},
    )
    monkeypatch.setattr(
        MODULE,
        "bind_payload_to_generation_manifest",
        lambda candidate, manifest: candidate,
    )

    release_root = root / "build" / "candidate"
    with pytest.raises(
        MODULE.PreviewRecomposeError,
        match="release candidate is invalid",
    ):
        MODULE.build_release_candidate(
            payload_path=source_directory / "regime-results.json",
            database_path=root / "snapshot.sqlite3",
            config_path=root / "config.json",
            source_manifest_path=source_directory / "generation-manifest.json",
            source_artifacts_path=source_artifacts,
            release_root_path=release_root,
        )
    return release_root, validation_artifact_directories


def test_release_candidate_validates_frozen_artifacts_before_recompose_and_cleans_stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    release_root, calls = _mock_release_candidate_dependencies(
        monkeypatch,
        tmp_path,
        fail_validation_call=2,
    )

    assert len(calls) == 2
    assert not release_root.exists()
    assert not list(release_root.parent.glob(".candidate-*"))


def test_release_candidate_rolls_back_after_post_install_validation_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    release_root, calls = _mock_release_candidate_dependencies(
        monkeypatch,
        tmp_path,
        fail_validation_call=3,
    )

    assert len(calls) == 3
    assert not release_root.exists()
    assert not list(release_root.parent.glob(".candidate-*"))


def test_release_candidate_rejects_nested_source_artifact_target_before_copy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    build = tmp_path / "build"
    source_artifacts = build / "source" / "artifacts"
    source_artifacts.mkdir(parents=True)
    monkeypatch.setattr(MODULE, "project_root", lambda: tmp_path)
    release_root = source_artifacts / "candidate"

    with pytest.raises(
        MODULE.PreviewRecomposeError,
        match="overlaps a read-only input",
    ):
        MODULE.build_release_candidate(
            payload_path=tmp_path / "payload.json",
            database_path=tmp_path / "snapshot.sqlite3",
            config_path=tmp_path / "config.json",
            source_manifest_path=tmp_path / "generation-manifest.json",
            source_artifacts_path=source_artifacts,
            release_root_path=release_root,
        )

    assert not release_root.exists()
    assert not list(source_artifacts.glob(".candidate-*"))
