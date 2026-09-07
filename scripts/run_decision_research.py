#!/usr/bin/env python3
"""Reproduce decision research from explicit read-only local source snapshots."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from regime_lab.allocation.research_v2 import build_allocation_shadow_v2
from regime_lab.analysis.decision_research import build_decision_research_v2
from regime_lab.forecast_ledger import read_operational_diagnostics
from regime_lab.research.contract import validate_research_extensions


def _identity(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def run(
    *,
    canonical_cache: Path,
    source_artifacts: Path,
    source_payload: Path,
    ledger: Path,
    output_directory: Path,
) -> dict:
    paths = {
        "canonical_cache": canonical_cache,
        "source_payload": source_payload,
        "ledger": ledger,
        "oos_predictions": source_artifacts / "oos-predictions.csv",
        "transition_predictions": source_artifacts / "transition-oos-predictions.csv",
        "outcome_rows": source_artifacts / "model-conditioned-asset-outcomes.csv",
    }
    inputs = {name: _identity(path) for name, path in paths.items()}
    if output_directory.resolve() in {
        source_artifacts.resolve(),
        source_payload.parent.resolve(),
        ledger.parent.resolve(),
    }:
        raise ValueError(
            "research output must be separate from original source directories"
        )
    payload = json.loads(source_payload.read_text())
    prices = pd.read_pickle(canonical_cache)
    predictions = pd.read_csv(paths["oos_predictions"])
    transitions = pd.read_csv(paths["transition_predictions"])
    model = payload["model"]["champion"]
    cutoff = payload["model"]["selection_end"]
    allocation = build_allocation_shadow_v2(
        payload["weekly"],
        prices,
        predictions,
        forecast_model=model,
        selection_end=cutoff,
        current_signal=payload["research"]["prospective_decision_shadow"][
            "current_signal"
        ],
    )
    decision = build_decision_research_v2(
        predictions,
        transitions,
        prices,
        forecast_model=model,
        selection_end=cutoff,
        outcome_rows=pd.read_csv(paths["outcome_rows"]),
    )
    operational = read_operational_diagnostics(ledger)
    research = {
        "decision_research_v2": decision,
        "operational_diagnostics": operational,
        "prospective_decision_shadow": {"allocation_research_v2": allocation},
    }
    validate_research_extensions(research)
    expected = len(
        predictions.loc[
            predictions.model.eq(model) & predictions.evaluation_split.eq("holdout")
        ]
    )
    if (
        allocation["performance"]["weeks"] != expected
        or decision["diagnostic_origins"] != expected
    ):
        raise ValueError(
            "allocation and decision research lost matched holdout origins"
        )
    if inputs != {name: _identity(path) for name, path in paths.items()}:
        raise ValueError(
            "source snapshot changed during read-only research; outputs not replaced"
        )
    documents = {
        "allocation-v2.json": allocation,
        "decision-research-v2.json": decision,
        "operational-diagnostics.json": operational,
    }
    encoded = {
        name: (
            json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
        ).encode()
        for name, value in documents.items()
    }
    manifest = {
        "schema_version": "regime-decision-research-run/1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "read_only_local_snapshots_no_collection_no_issuance",
        "origin": payload["weekly"][-1]["date"],
        "selection_end": cutoff,
        "forecast_model": model,
        "matched_holdout_origins": expected,
        "source_inputs": inputs,
        "outputs": {
            name: {"sha256": hashlib.sha256(value).hexdigest(), "bytes": len(value)}
            for name, value in encoded.items()
        },
    }
    encoded["decision-research-run.json"] = (
        json.dumps(manifest, indent=2) + "\n"
    ).encode()
    output_directory.mkdir(parents=True, exist_ok=True)
    # All contracts and JSON serialisation pass before any result is replaced.
    # The manifest is replaced last and binds this complete set of results.
    for name, value in encoded.items():
        with tempfile.NamedTemporaryFile(
            dir=output_directory, prefix=f".{name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(value)
        try:
            temporary.replace(output_directory / name)
        finally:
            temporary.unlink(missing_ok=True)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-cache", type=Path, required=True)
    parser.add_argument("--source-artifacts", type=Path, required=True)
    parser.add_argument("--source-payload", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument(
        "--output-directory", type=Path, default=ROOT / "build/comprehensive/research"
    )
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
