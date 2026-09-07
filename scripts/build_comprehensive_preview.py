#!/usr/bin/env python3
"""Build a validated local research preview while preserving issued forecasts."""

from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import os
import tempfile
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd
from regime_lab.analysis.directional_coherence import (
    apply_full_directional_research,
)
from regime_lab.contract_v5 import validate_v5_payload
from regime_lab.dashboard_split import build_dashboard_split, build_history_chunks
from regime_lab.research.additional_sources import source_summary
from regime_lab.research.contract import validate_research_extensions


def build_preview(
    source: Path, artifacts: Path, output: Path, *, allow_partial: bool = False
) -> dict:
    required = (
        "model-economics/directional-result.pkl",
        "model-economics/label-sensitivity.json",
        "research/allocation-v2.json",
        "research/decision-research-v2.json",
        "research/operational-diagnostics.json",
        "downside.json",
        "diagnostics.json",
    )
    missing = [name for name in required if not (artifacts / name).is_file()]
    if missing and not allow_partial:
        raise ValueError(f"complete preview requires finished research: {missing}")
    raw = source.read_bytes()
    original = json.loads(raw)
    from regime_lab.operational_forecast import frame_sha256

    manifest_path = artifacts / "input/input-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["source_payload_sha256"] != hashlib.sha256(raw).hexdigest():
        raise ValueError("research inputs do not belong to the selected source payload")
    frames = {
        name: pd.read_pickle(artifacts / f"input/{name}.pkl")
        for name in ("canonical", "states", "features")
    }
    if any(
        frame_sha256(frame) != manifest["frames"][name]
        for name, frame in frames.items()
    ):
        raise ValueError("research input cache checksum mismatch")
    states = frames["states"]
    if states.index[-1] != pd.Timestamp(original["meta"]["data_as_of"]):
        raise ValueError("research input cutoff differs from source")
    model_root = artifacts / "model-economics"
    composed = model_root / "coherent-research-payload.json"
    payload = (
        json.loads(composed.read_text())
        if (model_root / "directional-result.pkl").exists() and composed.exists()
        else deepcopy(original)
    )
    payload["meta"]["publication_status"] = "unpublished"
    payload["meta"].pop("publication_review", None)
    payload["meta"].pop("generation_manifest_sha256", None)
    if "lifecycle" in payload["model"]:
        payload["model"]["lifecycle"]["publication"] = {"status": "unpublished"}
        payload["model"]["lifecycle"]["deployment"] = {"status": "candidate"}
    if (model_root / "directional-result.pkl").exists():
        payload = apply_full_directional_research(
            payload, states, pd.read_pickle(model_root / "directional-result.pkl")
        )
    if (model_root / "label-sensitivity.json").exists():
        payload["research"]["label_sensitivity"] = json.loads(
            (model_root / "label-sensitivity.json").read_text()
        )
    # Keep the issued 1w forecast, economic history and actual issuance time.
    assert payload["forecast"] == original["forecast"]
    assert all(
        a["next_week"] == b["next_week"]
        for a, b in zip(payload["weekly"], original["weekly"], strict=True)
    )
    for key, filename in [
        ("decision_research_v2", "decision-research-v2.json"),
        ("operational_diagnostics", "operational-diagnostics.json"),
    ]:
        path = artifacts / "research" / filename
        if path.exists():
            payload["research"][key] = json.loads(path.read_text())
    allocation = artifacts / "research/allocation-v2.json"
    if allocation.exists():
        payload["research"]["prospective_decision_shadow"]["allocation_research_v2"] = (
            json.loads(allocation.read_text())
        )
    extensions = {
        "schema_version": "regime-research-extensions/1",
        "additional_data": source_summary(artifacts / "additional-sources"),
    }
    for key in ("downside", "diagnostics"):
        path = artifacts / f"{key}.json"
        if path.exists():
            extensions[key] = json.loads(path.read_text())
    extensions["build"] = {
        "compiled_at": datetime.now(timezone.utc).isoformat(),
        "source_payload_sha256": hashlib.sha256(raw).hexdigest(),
        "purpose": "local_research_preview",
        "issued_forecast_preserved": True,
        "full_directional_evaluated": (model_root / "directional-result.pkl").exists(),
    }
    payload["research"]["extensions"] = extensions
    validate_research_extensions(payload["research"])
    validate_v5_payload(payload)
    data = (
        json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        + "\n"
    ).encode()
    core, research = build_dashboard_split(payload, payload_raw=data)
    history, _ = build_history_chunks(payload, payload_raw=data)
    files = {
        "regime-results.json": data,
        "regime-core.json": core,
        "regime-research.json": research,
        **{Path(k).name: v for k, v in history.items()},
    }
    return publish_preview_generation(output, files)


def publish_preview_generation(output: Path, files: dict[str, bytes]) -> dict:
    """Stage and verify every member before switching the current generation."""
    inventory = {
        k: {"bytes": len(v), "sha256": hashlib.sha256(v).hexdigest()}
        for k, v in files.items()
    }
    generations = output.parent / "generations"
    generations.mkdir(parents=True, exist_ok=True)
    generation = (
        generations
        / hashlib.sha256(b"".join(files[name] for name in sorted(files))).hexdigest()[
            :24
        ]
    )
    staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=generations))
    for name, value in files.items():
        (staging / name).write_bytes(value)
        if (
            hashlib.sha256((staging / name).read_bytes()).hexdigest()
            != inventory[name]["sha256"]
        ):
            raise ValueError("staged preview hash mismatch")
    (staging / "preview-inventory.json").write_text(
        json.dumps(inventory, indent=2) + "\n"
    )
    if generation.exists():
        raise ValueError("preview generation already exists")
    staging.rename(generation)
    # Preserve the previous complete directory; only the current pointer flips.
    if output.exists() and not output.is_symlink():
        output.rename(generations / ("previous-" + uuid.uuid4().hex[:12]))
    pointer = output.parent / (".current-" + uuid.uuid4().hex)
    pointer.symlink_to(
        os.path.relpath(generation, output.parent), target_is_directory=True
    )
    os.replace(pointer, output)
    return inventory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=Path, default=ROOT / "publication/live/regime-results.json"
    )
    parser.add_argument("--artifacts", type=Path, default=ROOT / "build/comprehensive")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "build/comprehensive/preview/data"
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="development only: show completed research while other runs finish",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            build_preview(
                args.source,
                args.artifacts,
                args.output,
                allow_partial=args.allow_partial,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
