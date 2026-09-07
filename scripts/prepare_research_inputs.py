#!/usr/bin/env python3
"""Cache exact read-only research inputs once, bound to the source forecast."""

from __future__ import annotations
import argparse
from datetime import date, datetime
import hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from regime_lab.config import load_config
from regime_lab.collection import weekly_cutoffs
from regime_lab.data import SQLiteSnapshotStore
from regime_lab.dataset import build_weekly_dataset
from regime_lab.analysis.labels import CausalRegimeLabeler, RegimeLabelConfig
from regime_lab.operational_forecast import frame_sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--database", type=Path, required=True)
    p.add_argument(
        "--payload", type=Path, default=ROOT / "publication/live/regime-results.json"
    )
    p.add_argument("--output", type=Path, default=ROOT / "build/comprehensive/input")
    args = p.parse_args()
    payload = json.loads(args.payload.read_text())
    config = load_config(ROOT / "config/series.json")
    with SQLiteSnapshotStore(args.database, read_only=True) as store:
        observations = store.read_last_good_observations()
    dataset = build_weekly_dataset(
        config,
        weekly_cutoffs(
            date(2006, 1, 1), datetime.fromisoformat(payload["meta"]["data_as_of"])
        ),
        observations,
        availability_basis="reconstructed_market",
    )
    c = dataset.canonical.loc[dataset.canonical.spy_close.notna()].copy()
    fit = payload["label"]["fit_period"]
    n = fit["weeks"]
    if (
        c.index[0].date().isoformat() != fit["start"]
        or c.index[n - 1].date().isoformat() != fit["end"]
    ):
        raise ValueError("source label fit period differs")
    labeler = CausalRegimeLabeler(
        RegimeLabelConfig(price_column="spy_close", minimum_fit_observations=260)
    )
    labeler.fit(c.iloc[:n])
    states = labeler.transform(c)
    args.output.mkdir(parents=True, exist_ok=True)
    frames = {
        "canonical": c,
        "states": states,
        "features": dataset.features.reindex(c.index),
    }
    for name, frame in frames.items():
        frame.to_pickle(args.output / f"{name}.pkl")
    (args.output / "feature-group-manifest.json").write_text(
        json.dumps(dataset.feature_group_manifest, default=str, indent=2)
    )
    manifest = {
        "source_payload_sha256": hashlib.sha256(args.payload.read_bytes()).hexdigest(),
        "data_as_of": payload["meta"]["data_as_of"],
        "observations": len(observations),
        "frames": {k: frame_sha256(v) for k, v in frames.items()},
    }
    (args.output / "input-manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
