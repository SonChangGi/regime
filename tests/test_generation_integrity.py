from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from regime_lab.artifact_inventory import write_artifact_inventory
from regime_lab import integrity
from regime_lab.web_contract import render_browser_contract_javascript


ROOT = Path(__file__).resolve().parents[1]


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, dict[str, Path]]:
    project = tmp_path / "project"
    run = project / "run"
    artifacts = run / "artifacts"
    label_path = project / "config" / "label-spec.json"
    artifacts.mkdir(parents=True)
    label_path.parent.mkdir(parents=True)
    label_raw = b'{"version":"market-causal-3state-v1"}\n'
    label_path.write_bytes(label_raw)
    (project / "requirements-ci.lock").write_text("locked-test-runtime\n", encoding="utf-8")
    monkeypatch.setattr(integrity, "project_root", lambda: project)

    generation_id = "20260826T000000.000000Z"
    (artifacts / "build-generation.json").write_bytes(
        _json_bytes({"generation_id": generation_id})
    )
    (artifacts / "evidence.csv").write_text("value\n1\n", encoding="utf-8")
    inventory_path = write_artifact_inventory(artifacts)

    payload_path = run / "regime-results.json"
    comparison_path = run / "comparison.json"
    manifest_path = run / integrity.GENERATION_MANIFEST_FILENAME
    payload: dict[str, object] = {
        "meta": {
            "generation_id": generation_id,
            "data_as_of": "2026-08-21T20:00:00+00:00",
            "publication_status": "unpublished",
            "generation_manifest_sha256": "0" * 64,
        },
        "model": {
            "selection_status": "selected_by_gate",
            "label_version": "market-causal-3state-v1",
            "execution_parameters": {"profile": "standard", "sha256": "e" * 64},
            "lifecycle": {
                "selection": {"status": "selected_by_gate"},
                "deployment": {"status": "reviewed"},
                "publication": {"status": "unpublished"},
            },
        },
        "label": {
            "spec_id": "v1_spy_hysteresis",
            "spec_version": "market-causal-3state-v1",
            "spec_sha256": "b" * 64,
        },
    }
    comparison: dict[str, object] = {
        "schema_version": "test-comparison/1",
        "inputs": {
            "v5": {
                "regime_results": {
                    "path": payload_path.name,
                    "sha256": "0" * 64,
                }
            }
        },
        "result": {"log_loss": 0.25},
    }
    manifest: dict[str, object] = {
        "schema_version": integrity.GENERATION_MANIFEST_SCHEMA_VERSION,
        "generation_id": generation_id,
        "payload": {
            "path": "run/regime-results.json",
            "payload_contract_sha256": (
                integrity.canonical_json_sha256_v1_without_generation_binding(payload)
            ),
        },
        "comparison_sidecar": {
            "path": "run/comparison.json",
            "comparison_contract_sha256": (
                integrity.canonical_comparison_contract_sha256_v1(comparison)
            ),
        },
        "selection_family_sidecar": None,
        "artifact_inventory": {
            "sha256": hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
            "file_count": 2,
        },
        "input_snapshot": {
            "data_as_of": "2026-08-21T20:00:00+00:00",
            "sha256": "a" * 64,
        },
        "label_spec": {
            "path": "config/label-spec.json",
            "registry_sha256": hashlib.sha256(label_raw).hexdigest(),
            "spec_id": "v1_spy_hysteresis",
            "version": "market-causal-3state-v1",
            "spec_sha256": "b" * 64,
        },
        "execution_spec": {"sha256": "e" * 64},
        "runtime_fingerprint": integrity.build_runtime_fingerprint(project),
    }
    payload["meta"]["generation_manifest_sha256"] = (
        integrity.canonical_json_sha256_v1(manifest)
    )
    payload_path.write_bytes(_json_bytes(payload))
    comparison["inputs"]["v5"]["regime_results"]["sha256"] = hashlib.sha256(
        payload_path.read_bytes()
    ).hexdigest()
    comparison_path.write_bytes(_json_bytes(comparison))
    manifest_path.write_bytes(_json_bytes(manifest))
    return manifest_path, {
        "project": project,
        "payload": payload_path,
        "comparison": comparison_path,
        "artifacts": artifacts,
        "inventory": inventory_path,
    }


def test_canonical_json_sha256_v1_ignores_formatting_and_key_order() -> None:
    left = json.loads('{"b": 2, "a": [1, 3]}')
    right = json.loads('{\n  "a": [1,3],\n  "b": 2\n}')

    assert integrity.canonical_json_sha256_v1(left) == (
        integrity.canonical_json_sha256_v1(right)
    )


def test_generation_manifest_binds_every_generation_edge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, paths = _generation(tmp_path, monkeypatch)

    result = integrity.validate_generation_manifest(
        manifest_path,
        require_comparison=True,
    )

    assert result["generation_id"] == "20260826T000000.000000Z"
    assert result["payload_path"] == paths["payload"]
    assert result["comparison_path"] == paths["comparison"]
    assert result["artifact_directory"] == paths["artifacts"]


def test_current_manifest_can_require_selection_family_while_legacy_remains_readable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, paths = _generation(tmp_path, monkeypatch)
    with pytest.raises(
        integrity.IntegrityError,
        match="selection-family sidecar is required",
    ):
        integrity.validate_generation_manifest(
            manifest_path,
            require_selection_family=True,
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = integrity.LEGACY_GENERATION_MANIFEST_SCHEMA_VERSION
    manifest.pop("selection_family_sidecar")
    manifest.pop("runtime_fingerprint")
    payload = json.loads(paths["payload"].read_text(encoding="utf-8"))
    payload["meta"]["generation_manifest_sha256"] = (
        integrity.canonical_json_sha256_v1(manifest)
    )
    paths["payload"].write_bytes(_json_bytes(payload))
    comparison = json.loads(paths["comparison"].read_text(encoding="utf-8"))
    comparison["inputs"]["v5"]["regime_results"]["sha256"] = hashlib.sha256(
        paths["payload"].read_bytes()
    ).hexdigest()
    paths["comparison"].write_bytes(_json_bytes(comparison))
    manifest_path.write_bytes(_json_bytes(manifest))

    result = integrity.validate_generation_manifest(
        manifest_path,
        require_comparison=True,
    )
    assert result["schema_version"] == (
        integrity.LEGACY_GENERATION_MANIFEST_SCHEMA_VERSION
    )
    assert result["selection_family"] is None


def test_generation_manifest_builder_matches_the_validated_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, paths = _generation(tmp_path, monkeypatch)
    payload = json.loads(paths["payload"].read_text(encoding="utf-8"))
    comparison = json.loads(paths["comparison"].read_text(encoding="utf-8"))
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))

    built = integrity.build_generation_manifest(
        payload=payload,
        payload_path=paths["payload"],
        artifact_directory=paths["artifacts"],
        input_snapshot=expected["input_snapshot"],
        label_spec_path=paths["project"] / "config/label-spec.json",
        comparison=comparison,
        comparison_path=paths["comparison"],
    )

    assert built == expected
    bound = integrity.bind_payload_to_generation_manifest(payload, built)
    assert bound["meta"]["generation_manifest_sha256"] == (
        integrity.canonical_json_sha256_v1(built)
    )


def test_generation_manifest_builder_accepts_an_unwritten_comparison_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, paths = _generation(tmp_path, monkeypatch)
    payload = json.loads(paths["payload"].read_text(encoding="utf-8"))
    comparison = json.loads(paths["comparison"].read_text(encoding="utf-8"))
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths["comparison"].unlink()

    built = integrity.build_generation_manifest(
        payload=payload,
        payload_path=paths["payload"],
        artifact_directory=paths["artifacts"],
        input_snapshot=expected["input_snapshot"],
        label_spec_path=paths["project"] / "config/label-spec.json",
        comparison=comparison,
        comparison_path=paths["comparison"],
    )

    assert built["comparison_sidecar"] == expected["comparison_sidecar"]


def test_staged_generation_can_bind_final_publication_member_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, paths = _generation(tmp_path, monkeypatch)
    payload = json.loads(paths["payload"].read_text(encoding="utf-8"))
    comparison = json.loads(paths["comparison"].read_text(encoding="utf-8"))
    existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    final = paths["project"] / "publication" / "live"

    manifest = integrity.build_generation_manifest(
        payload=payload,
        payload_path=paths["payload"],
        artifact_directory=paths["artifacts"],
        input_snapshot=existing["input_snapshot"],
        label_spec_path=paths["project"] / "config/label-spec.json",
        comparison=comparison,
        comparison_path=paths["comparison"],
        payload_contract_path=final / "regime-results.json",
        comparison_contract_path=final / "v5-vs-v4-comparison.json",
    )
    payload = integrity.bind_payload_to_generation_manifest(payload, manifest)
    paths["payload"].write_bytes(_json_bytes(payload))
    comparison["inputs"]["v5"]["regime_results"]["sha256"] = hashlib.sha256(
        paths["payload"].read_bytes()
    ).hexdigest()
    paths["comparison"].write_bytes(_json_bytes(comparison))
    manifest_path.write_bytes(_json_bytes(manifest))

    with pytest.raises(integrity.IntegrityError, match="manifest.payload"):
        integrity.validate_generation_manifest(manifest_path)

    result = integrity.validate_generation_manifest(
        manifest_path,
        payload_path_override=paths["payload"],
        comparison_path_override=paths["comparison"],
    )
    assert result["payload_path"] == paths["payload"]
    assert result["declared_payload_path"] == final / "regime-results.json"
    assert result["declared_comparison_path"] == (
        final / "v5-vs-v4-comparison.json"
    )


def test_publication_validation_keeps_artifact_paths_and_raw_snapshot_private(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, paths = _generation(tmp_path, monkeypatch)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    serialized = integrity.canonical_json_bytes_v1(manifest)
    assert b"artifact" not in json.dumps(
        manifest["artifact_inventory"], sort_keys=True
    ).encode("utf-8")
    assert b"snapshot_id" not in serialized

    unavailable = paths["artifacts"].with_name("private-artifacts-not-in-checkout")
    paths["artifacts"].rename(unavailable)
    result = integrity.validate_generation_manifest(
        manifest_path,
        require_comparison=True,
        require_artifacts=False,
    )
    assert result["artifact_inventory"]["verified"] is False
    with pytest.raises(integrity.IntegrityError, match="artifact directory is unavailable"):
        integrity.validate_generation_manifest(manifest_path)


def test_generation_manifest_rejects_payload_and_sidecar_generation_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, paths = _generation(tmp_path, monkeypatch)
    payload = json.loads(paths["payload"].read_text(encoding="utf-8"))
    payload["model"]["execution_parameters"]["profile"] = "quick"
    paths["payload"].write_bytes(_json_bytes(payload))

    with pytest.raises(integrity.IntegrityError, match="payload contract hash mismatch"):
        integrity.validate_generation_manifest(manifest_path)

    manifest_path, paths = _generation(tmp_path / "second", monkeypatch)
    comparison = json.loads(paths["comparison"].read_text(encoding="utf-8"))
    comparison["result"]["log_loss"] = 0.4
    paths["comparison"].write_bytes(_json_bytes(comparison))
    with pytest.raises(integrity.IntegrityError, match="sidecar contract hash mismatch"):
        integrity.validate_generation_manifest(manifest_path)


def test_generation_manifest_rejects_payload_manifest_back_reference_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, paths = _generation(tmp_path, monkeypatch)
    payload = json.loads(paths["payload"].read_text(encoding="utf-8"))
    payload["meta"]["generation_manifest_sha256"] = "f" * 64
    paths["payload"].write_bytes(_json_bytes(payload))

    with pytest.raises(integrity.IntegrityError, match="back-reference mismatch"):
        integrity.validate_generation_manifest(manifest_path)


@pytest.mark.parametrize(
    ("deployment", "publication"),
    (
        ("candidate", "reviewed_publication"),
        ("reviewed", "reviewed_publication"),
        ("operating", "unpublished"),
        ("provisional_predeployment", "reviewed_publication"),
    ),
)
def test_lifecycle_rejects_contradictory_combinations(
    deployment: str,
    publication: str,
) -> None:
    payload = {
        "meta": {"publication_status": publication},
        "model": {
            "selection_status": "selected_by_gate",
            "lifecycle": {
                "selection": {"status": "selected_by_gate"},
                "deployment": {"status": deployment},
                "publication": {"status": publication},
            },
        },
    }

    with pytest.raises(integrity.IntegrityError):
        integrity.validate_lifecycle_consistency(payload)


def test_reviewed_candidate_hash_is_semantic_and_ignores_manifest_back_reference() -> None:
    candidate = {
        "meta": {
            "publication_status": "unpublished",
            "generation_manifest_sha256": "a" * 64,
        },
        "model": {
            "selection_status": "selected_by_gate",
            "lifecycle": {
                "selection": {"status": "selected_by_gate"},
                "deployment": {"status": "candidate"},
                "publication": {"status": "unpublished"},
            },
        },
        "weekly": [{"date": "2026-08-21"}],
    }
    expected = integrity.reviewed_candidate_sha256_v1(candidate)
    publication = deepcopy(candidate)
    publication["meta"]["publication_status"] = "reviewed_publication"
    publication["meta"]["generation_manifest_sha256"] = "b" * 64
    publication["meta"]["publication_review"] = {
        "reviewed_candidate_sha256": expected
    }
    publication["model"]["lifecycle"]["deployment"]["status"] = "operating"
    publication["model"]["lifecycle"]["publication"]["status"] = (
        "reviewed_publication"
    )

    assert integrity.validate_reviewed_candidate_hash(publication) == expected

    publication["weekly"][0]["date"] = "2026-08-28"
    with pytest.raises(integrity.IntegrityError, match="canonical JSON hash"):
        integrity.validate_reviewed_candidate_hash(publication)


def test_audit_cli_requires_an_explicit_target_and_local_manifest() -> None:
    script = ROOT / "scripts" / "audit_outputs.py"
    spec = importlib.util.spec_from_file_location("audit_outputs_targets", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    with pytest.raises(SystemExit):
        module.parse_args([])
    with pytest.raises(SystemExit):
        module.parse_args(["--target", "local-generation"])
    args = module.parse_args(
        ["--target", "local-generation", "--manifest", "run/generation-manifest.json"]
    )
    assert args.manifest == Path("run/generation-manifest.json")


def test_publication_live_target_cannot_accept_a_stale_v4_payload(
    tmp_path: Path,
) -> None:
    script = ROOT / "scripts" / "audit_outputs.py"
    spec = importlib.util.spec_from_file_location("audit_outputs_no_stale_v4", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    live = tmp_path / "publication" / "live"
    live.mkdir(parents=True)
    (live / "regime-results.json").write_bytes(
        _json_bytes(
            {
                "meta": {
                    "result_version": "weekly-regime-result-v4",
                    "mode": "live",
                    "data_as_of": "2026-08-07T20:00:00+00:00",
                }
            }
        )
    )
    module.PROJECT_ROOT = tmp_path

    with pytest.raises(module.AuditFailure, match="V5.*V4 cannot pass"):
        module._audit_publication_live()


def test_publication_live_freshness_is_recomputed_at_audit_time() -> None:
    script = ROOT / "scripts" / "audit_outputs.py"
    spec = importlib.util.spec_from_file_location("audit_outputs_wall_clock", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    meta = {
        "data_as_of": "2026-08-01T20:00:00+00:00",
        "freshness": {"maximum_age_days": 10},
    }

    with pytest.raises(module.AuditFailure, match="stale at audit time"):
        module._require_wall_clock_freshness(
            meta,
            now=datetime(2026, 8, 12, 20, 0, tzinfo=timezone.utc),
        )


def test_packaging_cache_keys_are_derived_from_asset_content() -> None:
    script = ROOT / "scripts" / "package_public_demo.py"
    spec = importlib.util.spec_from_file_location("package_cache_keys", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    styles = b"body { color: black; }\n"
    app = b"console.log('content');\n"
    source = (
        b'<link rel="stylesheet" href="./styles.css?v=20260826-v5-12">\n'
        b'<script src="./app.js?v=20260826-v5-12" defer></script>\n'
    )

    rewritten = module.rewrite_index_asset_versions(
        source,
        styles_raw=styles,
        app_raw=app,
    )

    assert b"20260826-v5-12" not in rewritten
    assert hashlib.sha256(styles).hexdigest().encode("ascii") in rewritten
    assert hashlib.sha256(app).hexdigest().encode("ascii") in rewritten
    module.validate_index_asset_versions(
        rewritten,
        styles_raw=styles,
        app_raw=app,
    )


def test_packaging_requires_manifest_when_payload_carries_generation_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = ROOT / "scripts" / "package_public_demo.py"
    spec = importlib.util.spec_from_file_location("package_manifest_required", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    web = tmp_path / "web"
    web.mkdir()
    (web / "index.html").write_text("<main>test</main>\n", encoding="utf-8")
    (web / "styles.css").write_text("main {}\n", encoding="utf-8")
    (web / "operating-contract.generated.js").write_bytes(
        render_browser_contract_javascript()
    )
    (web / "app.js").write_text("'use strict';\n", encoding="utf-8")
    payload_path = tmp_path / "regime-results.json"
    payload_path.write_bytes(
        _json_bytes(
            {
                "meta": {
                    "result_version": module.V5_RESULT_VERSION,
                    "generation_manifest_sha256": "0" * 64,
                }
            }
        )
    )
    comparison = tmp_path / module.V5_COMPARISON_FILENAME
    comparison.write_bytes(_json_bytes({"schema_version": "test"}))
    monkeypatch.setattr(module, "validate_public_payload", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        module,
        "validate_v5_comparison_sidecar",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(module.PackagingError, match="generation manifest is invalid"):
        module.package_public_dashboard(
            web_root=web,
            payload_path=payload_path,
            comparison_path=comparison,
            output_directory=tmp_path / "output",
        )
