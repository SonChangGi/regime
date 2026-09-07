#!/usr/bin/env python3
"""Run additional sources, purged downside challengers and paired ablation."""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd
from regime_lab.research.additional_sources import (
    collect_sources,
    cboe_weekly,
    read_source,
    sloos_history,
    cmdi_history,
    ads_vintage_features,
)
from regime_lab.research.downside import run_downside_research
from regime_lab.research.diagnostics import ablation_diagnostics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts", type=Path, default=ROOT / "build/comprehensive")
    p.add_argument("--source-artifacts", type=Path, required=True)
    p.add_argument("--refresh-additional", action="store_true")
    args = p.parse_args()
    base = args.artifacts
    canonical = pd.read_pickle(base / "input/canonical.pkl")
    source = base / "additional-sources"
    collect_sources(source, refresh=args.refresh_additional)
    cboe = cboe_weekly(source, canonical.index)
    cboe.to_pickle(base / "input/cboe-features.pkl")
    result = run_downside_research(canonical, additional=cboe)
    (base / "downside.json").write_text(
        json.dumps(result.summary, allow_nan=False, indent=2)
    )
    result.predictions.to_csv(base / "downside-oos.csv", index=False)
    a = pd.read_csv(args.source_artifacts / "feature-ablation-oos-predictions.csv")
    (base / "diagnostics.json").write_text(
        json.dumps(ablation_diagnostics(a), allow_nan=False, indent=2)
    )
    for name, parser in [("sloos", sloos_history), ("cmdi", cmdi_history)]:
        parser(read_source(source, name)).to_csv(
            source / f"{name}-parsed.csv", index=False
        )
    ads = ads_vintage_features(read_source(source, "ads"), canonical.index)
    ads.to_pickle(base / "input/ads-features.pkl")
    ads.to_csv(source / "ads-vintage-features.csv")
    print(
        f"Completed: {len(result.predictions)} downside forecasts; {len(ads)} ADS vintage-aligned weeks."
    )


if __name__ == "__main__":
    main()
