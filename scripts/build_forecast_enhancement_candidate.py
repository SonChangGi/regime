#!/usr/bin/env python3
"""Generate the weekly comparison factory from frozen inputs, without collection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
import pandas as pd

from regime_lab.forecast_enhancement_publication import STATES_FILENAME, STATES_MANIFEST_FILENAME, bind_document, encode
from regime_lab.io import write_json_atomic
from regime_lab.operational_forecast import frame_sha256
from regime_lab.research.forecast_enhancement_generation import build_forecast_enhancement_candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--cache-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_root = (ROOT/"build").resolve()
    for path in (args.output, args.cache_directory):
        if build_root not in path.resolve().parents:
            raise ValueError("offline candidate outputs must stay in the local build directory")
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("choose a new candidate output; existing results are preserved")
    payload = json.loads(args.source.read_text())
    manifest = json.loads((args.input/"input-manifest.json").read_text())
    canonical, states = pd.read_pickle(args.input/"canonical.pkl"), pd.read_pickle(args.input/"states.pkl")
    for name, frame in (("canonical", canonical), ("states", states)):
        if frame_sha256(frame) != manifest["frames"][name]:
            raise ValueError("frozen input manifest differs")
    labels = pd.DataFrame({"date": states.index, "state": states.to_numpy()})
    transition = pd.read_csv(args.artifacts/"transition-oos-predictions.csv")
    future = pd.read_csv(args.artifacts/"transition-candidate-forecasts.csv")
    benchmark = SimpleNamespace(state_label_history=labels, transition_benchmark=SimpleNamespace(
        predictions=transition, latest_candidate_forecasts=lambda: future.copy(deep=True)))
    document = build_forecast_enhancement_candidate(payload, dataset=SimpleNamespace(canonical=canonical),
        benchmark=benchmark, cache_directory=args.cache_directory, progress=lambda message: print(message, flush=True))
    bound = bind_document(document, payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        handle.write(encode(bound))
    schema = bound["provenance"]["state_frame_schema"]
    state_snapshot = states.rename(schema["series_name"]).rename_axis(schema["index_name"])
    if frame_sha256(state_snapshot) != bound["provenance"]["input_frames"]["states"]:
        raise ValueError("factory state snapshot differs from frozen states")
    state_snapshot.to_pickle(args.output.parent/STATES_FILENAME)
    write_json_atomic(args.output.parent/STATES_MANIFEST_FILENAME, {
        "schema_version": "regime-forecast-enhancement-inputs/1", "data_as_of": bound["data_as_of"],
        "source_generation_id": bound["source_generation_id"], "frames": {"states": frame_sha256(state_snapshot)},
    })
    print(json.dumps({"status": "validated_local_candidate", "output": str(args.output.resolve()),
                      "generation_id": bound["source_generation_id"], "history_rows": len(bound["history"]),
                      "external_publication": False}), flush=True)


if __name__ == "__main__":
    main()
