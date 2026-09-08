#!/usr/bin/env python3
"""Collect and compare additional public macro signals in a local workspace."""
import argparse
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
import pandas as pd
from regime_lab.research.forecast_macro_information import build_macro_information


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, default=ROOT/"publication/live/regime-results.json")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT/"build/forecast-upgrade-20260907/information")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    allowed = (ROOT/"build/forecast-upgrade-20260907").resolve()
    if allowed not in args.output.resolve().parents:
        parser.error("output must be inside the isolated forecast-upgrade workspace")
    source_payload, source_features = args.payload.read_bytes(), args.features.read_bytes()
    def verify_inputs():
        if args.payload.read_bytes() != source_payload or args.features.read_bytes() != source_features:
            raise ValueError("macro source inputs changed during the run")
    result = build_macro_information(json.loads(source_payload), pd.read_pickle(io.BytesIO(source_features)),
        args.output, refresh=args.refresh, verify_inputs=verify_inputs)
    print(json.dumps({"sources": len(result["sources"]), "metrics": result["evaluation"]["rows"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
