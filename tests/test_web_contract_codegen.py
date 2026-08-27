from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from regime_lab.config import project_root
from regime_lab.operating_contract import canonical_sha256, load_operating_contract
from regime_lab.web_contract import (
    BROWSER_CONTRACT_SCHEMA,
    GENERATED_BROWSER_CONTRACT_PATH,
    browser_contract_document,
    render_browser_contract_javascript,
    validate_generated_browser_contract,
)


def test_browser_contract_is_derived_from_operating_contract() -> None:
    operating = load_operating_contract()
    document = browser_contract_document()

    assert document["schema_version"] == BROWSER_CONTRACT_SCHEMA
    assert document["operating_contract_canonical_sha256"] == canonical_sha256(
        operating.document
    )
    assert document["state_order"] == list(operating.state_order)
    assert document["state_meta"] == operating.document["state_meta"]
    assert document["models"] == operating.document["models"]
    assert document["selection_policy"] == operating.document["selection_policy"]


def test_checked_in_browser_contract_is_current_and_executable() -> None:
    raw = validate_generated_browser_contract()
    assert raw == render_browser_contract_javascript()
    path = project_root() / GENERATED_BROWSER_CONTRACT_PATH
    program = (
        "require(process.argv[1]);"
        "process.stdout.write(JSON.stringify(globalThis.REGIME_OPERATING_CONTRACT));"
    )
    completed = subprocess.run(
        ["node", "-e", program, str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == browser_contract_document()
    application = (project_root() / "web/app.js").read_text(encoding="utf-8")
    assert "globalThis.REGIME_OPERATING_CONTRACT" in application
    assert "OPERATING_CONTRACT?.state_order" in application
    assert "OPERATING_CONTRACT?.forecast?.transition_horizons_weeks" in application
    assert "OPERATING_CONTRACT?.models?.official_champion" in application


def test_generator_check_rejects_stale_output(tmp_path: Path) -> None:
    target = tmp_path / "contract.js"
    target.write_text("stale\n", encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(project_root() / "scripts/generate_web_contract.py"),
            "--output",
            str(target),
            "--check",
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "stale" in completed.stderr
