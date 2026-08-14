from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "package_public_demo.py"
PAGES_WORKFLOW = ROOT / ".github" / "workflows" / "pages.yml"
LIVE_PUBLICATION = ROOT / "publication" / "live" / "regime-results.json"
SPEC = importlib.util.spec_from_file_location("package_public_demo", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
package_public_demo = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(package_public_demo)


def _web_root(tmp_path: Path) -> Path:
    root = tmp_path / "web"
    root.mkdir()
    (root / "index.html").write_text("<main>demo</main>\n", encoding="utf-8")
    (root / "styles.css").write_text("main { color: black; }\n", encoding="utf-8")
    (root / "app.js").write_text("console.log('demo');\n", encoding="utf-8")
    return root


def _payload(tmp_path: Path, *, mode: str = "demo", synthetic: bool = True) -> Path:
    source_id = "synthetic_market" if synthetic else "alpha_vantage"
    license_class = "synthetic_fixture" if synthetic else "private_noncommercial"
    estimate = {
        "state": "transition",
        "probabilities": {"risk_on": 0.25, "transition": 0.5, "risk_off": 0.25},
        "confidence": 0.5,
        "entropy": 0.95,
    }
    payload = {
        "meta": {
            "schema_version": "1.0.0",
            "generated_at": "2026-08-11T00:00:00Z",
            "mode": mode,
            "data_as_of": "2026-08-07T20:00:00+00:00",
            "timezone": "America/New_York",
        },
        "states": [
            {"id": "risk_on", "label": "Risk-on"},
            {"id": "transition", "label": "Transition"},
            {"id": "risk_off", "label": "Risk-off"},
        ],
        "model": {
            "champion": "markov",
            "selection_status": "provisional_predeployment",
            "leaderboard": [],
        },
        "sources": [
            {
                "id": source_id,
                "license_class": license_class,
                "status": "degraded",
            }
        ],
        "weekly": [
            {
                "date": "2026-08-07",
                "current": estimate,
                "next_week": estimate,
                "transition_probability": 0.2,
                "scores": {
                    "trend": 0.1,
                    "stress": -0.1,
                    "macro": 0.0,
                    "financial_conditions": 0.0,
                },
            }
        ],
        "feature_catalog": [
            {
                "id": "synthetic_feature",
                "category": "test",
                "frequency": "weekly",
                "source": "synthetic_market",
            }
        ],
    }
    path = tmp_path / f"{mode}-payload.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_package_copies_only_allowlisted_assets_and_synthetic_payload(tmp_path: Path) -> None:
    web_root = _web_root(tmp_path)
    (web_root / ".env").write_text("SECRET=must-not-copy\n", encoding="utf-8")
    (web_root / "provider.sqlite3").write_bytes(b"private database")
    nested = web_root / "data"
    nested.mkdir()
    (nested / "live.json").write_text('{"mode":"live"}', encoding="utf-8")

    payload = _payload(tmp_path)
    output = tmp_path / "public-demo"
    manifest = package_public_demo.package_public_demo(
        web_root=web_root,
        payload_path=payload,
        output_directory=output,
    )

    packaged = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
    }
    assert packaged == {
        "index.html",
        "styles.css",
        "app.js",
        "data/regime-results.json",
        "publication-manifest.json",
    }
    assert manifest["package_kind"] == "synthetic_demo_only"
    assert manifest["payload_mode"] == "demo"
    assert output.stat().st_mode & 0o777 == 0o755
    copied_payload = (output / "data/regime-results.json").read_bytes()
    assert manifest["files"]["data/regime-results.json"]["sha256"] == hashlib.sha256(
        copied_payload
    ).hexdigest()


def test_package_refuses_live_payload_without_creating_output(tmp_path: Path) -> None:
    output = tmp_path / "public-demo"
    with pytest.raises(package_public_demo.PackagingError, match="meta.mode=demo"):
        package_public_demo.package_public_demo(
            web_root=_web_root(tmp_path),
            payload_path=_payload(tmp_path, mode="live", synthetic=False),
            output_directory=output,
        )
    assert not output.exists()


def test_live_derived_package_requires_explicit_rights_acknowledgement(
    tmp_path: Path,
) -> None:
    output = tmp_path / "public-live"
    with pytest.raises(package_public_demo.PackagingError, match="rights acknowledgement"):
        package_public_demo.package_public_dashboard(
            web_root=_web_root(tmp_path),
            payload_path=LIVE_PUBLICATION,
            output_directory=output,
            publication_mode=package_public_demo.PUBLICATION_MODE_LIVE_DERIVED,
            rights_acknowledged=False,
        )
    assert not output.exists()


def test_package_copies_only_allowlisted_assets_and_live_derived_snapshot(
    tmp_path: Path,
) -> None:
    web_root = _web_root(tmp_path)
    (web_root / "provider.sqlite3").write_bytes(b"private database")
    output = tmp_path / "public-live"
    manifest = package_public_demo.package_public_dashboard(
        web_root=web_root,
        payload_path=LIVE_PUBLICATION,
        output_directory=output,
        publication_mode=package_public_demo.PUBLICATION_MODE_LIVE_DERIVED,
        rights_acknowledged=True,
    )

    assert manifest["package_kind"] == "personal_noncommercial_live_derived"
    assert manifest["payload_mode"] == "live"
    assert manifest["publication_scope"] == "personal_noncommercial_derived_results"
    assert manifest["contains_raw_observations"] is False
    assert manifest["source_ids"] == ["alfred", "alpha_vantage"]
    assert not (output / "provider.sqlite3").exists()
    payload = json.loads((output / "data/regime-results.json").read_text())
    assert payload["meta"]["mode"] == "live"
    assert len(payload["weekly"]) >= 52


def test_live_derived_package_refuses_raw_observation_material(tmp_path: Path) -> None:
    payload = json.loads(LIVE_PUBLICATION.read_text(encoding="utf-8"))
    payload["sources"][0]["observations"] = [{"date": "2026-08-07", "value": 1.0}]
    payload_path = tmp_path / "unsafe-live.json"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "public-live"

    with pytest.raises(package_public_demo.PackagingError, match="forbidden provider material"):
        package_public_demo.package_public_dashboard(
            web_root=_web_root(tmp_path),
            payload_path=payload_path,
            output_directory=output,
            publication_mode=package_public_demo.PUBLICATION_MODE_LIVE_DERIVED,
            rights_acknowledged=True,
        )
    assert not output.exists()


def test_live_derived_package_refuses_machine_local_paths(tmp_path: Path) -> None:
    payload = json.loads(LIVE_PUBLICATION.read_text(encoding="utf-8"))
    payload["meta"]["debug_path"] = "/Users/example/private-output.json"
    payload_path = tmp_path / "unsafe-path-live.json"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(package_public_demo.PackagingError, match="machine-local path"):
        package_public_demo.package_public_dashboard(
            web_root=_web_root(tmp_path),
            payload_path=payload_path,
            output_directory=tmp_path / "public-live",
            publication_mode=package_public_demo.PUBLICATION_MODE_LIVE_DERIVED,
            rights_acknowledged=True,
        )


def test_package_refuses_provider_data_mislabeled_as_demo(tmp_path: Path) -> None:
    output = tmp_path / "public-demo"
    with pytest.raises(package_public_demo.PackagingError, match="synthetic fixture"):
        package_public_demo.package_public_demo(
            web_root=_web_root(tmp_path),
            payload_path=_payload(tmp_path, mode="demo", synthetic=False),
            output_directory=output,
        )
    assert not output.exists()


def test_package_refuses_malformed_synthetic_payload(tmp_path: Path) -> None:
    payload_path = _payload(tmp_path)
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["weekly"][0]["next_week"]["probabilities"] = {
        "risk_on": 0.8,
        "transition": 0.8,
        "risk_off": 0.1,
    }
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "public-demo"

    with pytest.raises(
        package_public_demo.PackagingError,
        match="dashboard payload contract is invalid",
    ):
        package_public_demo.package_public_demo(
            web_root=_web_root(tmp_path),
            payload_path=payload_path,
            output_directory=output,
        )

    assert not output.exists()


def test_package_refuses_to_overwrite_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "public-demo"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(package_public_demo.PackagingError, match="refusing to overwrite"):
        package_public_demo.package_public_demo(
            web_root=_web_root(tmp_path),
            payload_path=_payload(tmp_path),
            output_directory=output,
        )
    assert marker.read_text(encoding="utf-8") == "keep"


def test_package_rejects_output_through_symlink_parent(tmp_path: Path) -> None:
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(ValueError, match="symbolic-link"):
        package_public_demo.package_public_demo(
            web_root=_web_root(tmp_path),
            payload_path=_payload(tmp_path),
            output_directory=linked_parent / "public-demo",
        )
    assert not (real_parent / "public-demo").exists()


def test_package_rejects_absolute_other_project_output_before_write(
    tmp_path: Path,
) -> None:
    other_project = ROOT.parent / "do-not-touch-regime-package" / "public-demo"
    with pytest.raises(ValueError, match="must stay below"):
        package_public_demo.package_public_demo(
            web_root=_web_root(tmp_path),
            payload_path=_payload(tmp_path),
            output_directory=other_project,
        )


def test_private_live_outputs_are_ignored_by_repository_contract() -> None:
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "data/*.sqlite3*" in ignored
    assert "web/data/regime-results.json" in ignored
    assert "artifacts/" in ignored
    assert "artifacts/latest/" in ignored


def test_pages_workflow_uploads_only_verified_live_derived_package() -> None:
    workflow = PAGES_WORKFLOW.read_text(encoding="utf-8")
    assert "regime-lab demo" not in workflow
    assert "publication/live/regime-results.json" in workflow
    assert "--publication-mode live-derived" in workflow
    assert "--acknowledge-personal-noncommercial-publication" in workflow
    assert "--output dist/public-dashboard" in workflow
    assert "verify_public_package.py dist/public-dashboard" in workflow
    assert "actions/upload-pages-artifact@v4" in workflow
    assert "path: dist/public-dashboard" in workflow
    assert "web/data/regime-results.json" not in workflow
    assert "data/regime.sqlite3" not in workflow
    assert "artifacts/latest" not in workflow
    assert "secrets." not in workflow
