from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SCRIPT = ROOT / "scripts" / "package_public_demo.py"
VERIFY_SCRIPT = ROOT / "scripts" / "verify_public_package.py"
LIVE_PUBLICATION = ROOT / "publication" / "live" / "regime-results.json"


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


package_public_demo = _load_script("package_public_demo", PACKAGE_SCRIPT)
verify_public_package = _load_script("verify_public_package", VERIFY_SCRIPT)


def _web_root(tmp_path: Path) -> Path:
    web = tmp_path / "web"
    web.mkdir()
    (web / "index.html").write_text("<main>demo</main>\n", encoding="utf-8")
    (web / "styles.css").write_text("main { color: black; }\n", encoding="utf-8")
    (web / "app.js").write_text("console.log('demo');\n", encoding="utf-8")
    return web


def _payload(tmp_path: Path) -> Path:
    estimate = {
        "state": "transition",
        "probabilities": {"risk_on": 0.25, "transition": 0.5, "risk_off": 0.25},
        "confidence": 0.5,
        "entropy": 0.95,
    }
    payload = {
        "meta": {
            "schema_version": "1.0.0",
            "generated_at": "2026-08-14T00:00:00Z",
            "mode": "demo",
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
                "id": "synthetic_market",
                "license_class": "synthetic_fixture",
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
    path = tmp_path / "payload.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _package(tmp_path: Path) -> Path:
    output = tmp_path / "public-demo"
    package_public_demo.package_public_demo(
        web_root=_web_root(tmp_path),
        payload_path=_payload(tmp_path),
        output_directory=output,
    )
    return output


def test_verifier_accepts_exact_synthetic_package(tmp_path: Path) -> None:
    result = verify_public_package.verify_public_package(_package(tmp_path))
    assert result["ok"] is True
    assert result["package_kind"] == "synthetic_demo_only"
    assert result["payload_mode"] == "demo"


def test_verifier_accepts_exact_live_derived_package(tmp_path: Path) -> None:
    output = tmp_path / "public-live"
    package_public_demo.package_public_dashboard(
        web_root=_web_root(tmp_path),
        payload_path=LIVE_PUBLICATION,
        output_directory=output,
        publication_mode=package_public_demo.PUBLICATION_MODE_LIVE_DERIVED,
        rights_acknowledged=True,
    )

    result = verify_public_package.verify_public_package(output)
    assert result["ok"] is True
    assert result["package_kind"] == "personal_noncommercial_live_derived"
    assert result["payload_mode"] == "live"


def test_verifier_refuses_extra_file(tmp_path: Path) -> None:
    output = _package(tmp_path)
    (output / "live.sqlite3").write_bytes(b"provider data")
    with pytest.raises(verify_public_package.VerificationError, match="extra"):
        verify_public_package.verify_public_package(output)


def test_verifier_refuses_tampered_allowlisted_asset(tmp_path: Path) -> None:
    output = _package(tmp_path)
    (output / "app.js").write_text("console.log('changed');\n", encoding="utf-8")
    with pytest.raises(verify_public_package.VerificationError, match="mismatch"):
        verify_public_package.verify_public_package(output)


def test_verifier_refuses_manifest_data_as_of_drift(tmp_path: Path) -> None:
    output = _package(tmp_path)
    manifest_path = output / "publication-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["payload_data_as_of"] = "1999-01-01T00:00:00Z"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(verify_public_package.VerificationError, match="data_as_of mismatch"):
        verify_public_package.verify_public_package(output)


def test_verifier_refuses_credential_like_material_even_with_matching_manifest(
    tmp_path: Path,
) -> None:
    output = _package(tmp_path)
    app = output / "app.js"
    raw = b"const FRED_API_KEY = 'must-not-publish';\n"
    app.write_bytes(raw)
    manifest_path = output / "publication-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["app.js"] = {
        "bytes": len(raw),
        "sha256": __import__("hashlib").sha256(raw).hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        verify_public_package.VerificationError,
        match="credential-like",
    ):
        verify_public_package.verify_public_package(output)
