#!/usr/bin/env python3
"""Opt-in local forecast copy, recipe freeze, issuance, scoring and summary."""
from __future__ import annotations
import argparse
from io import BytesIO
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd
from regime_lab.integrity import canonical_json_sha256_v1
from regime_lab.local_forecast_workflow import (initialize_local_copy, issue_local_preparation,
    local_summary, score_local_copy, _write_once)
from regime_lab.operational_forecast import frame_sha256, preparation_recipe_lock


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("copy-init")
    init.add_argument("--source", type=Path, required=True)
    init.add_argument("--workspace", type=Path, required=True)
    freeze = commands.add_parser("freeze")
    for field in ("features", "states", "oos-predictions", "transition-predictions", "payload", "feature-manifest", "output"):
        freeze.add_argument("--" + field, type=Path, required=True)
    freeze.add_argument("--canonical", type=Path, help="Rebuild the existing boundary/transition feature assembly from a frozen base cache")
    freeze.add_argument("--source-input-manifest", type=Path)
    issue = commands.add_parser("issue")
    for field in ("workspace", "prepared", "recipe-lock", "payload"):
        issue.add_argument("--" + field, type=Path, required=True)
    score = commands.add_parser("score")
    for field in ("workspace", "states", "input-manifest", "payload"):
        score.add_argument("--" + field, type=Path, required=True)
    score.add_argument("--states-available-at")
    summary = commands.add_parser("summary")
    summary.add_argument("--workspace", type=Path, required=True)
    for command in (init, issue, score, summary):
        command.add_argument("--output", type=Path)
    args = parser.parse_args()
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))
    if args.command == "copy-init":
        result = initialize_local_copy(args.source, args.workspace)
    elif args.command == "freeze":
        payload = read(args.payload)
        feature_manifest = read(args.feature_manifest)
        feature_hash = canonical_json_sha256_v1({k: v for k, v in feature_manifest.items() if k != "sha256"})
        if feature_hash != payload["model"]["feature_manifest_sha256"] or feature_hash != feature_manifest["sha256"]:
            raise ValueError("feature manifest differs from the locked payload")
        frames = {"features": pd.read_pickle(args.features), "states": pd.read_pickle(args.states),
                  "oos_predictions": pd.read_csv(args.oos_predictions), "transition_predictions": pd.read_csv(args.transition_predictions)}
        states = frames["states"]
        if args.canonical:
            if not args.source_input_manifest:
                raise ValueError("feature reconstruction requires the original --source-input-manifest")
            from regime_lab.analysis.labels import CausalRegimeLabeler, RegimeLabelConfig
            from regime_lab.analysis.transitions import derive_causal_transition_features
            canonical = pd.read_pickle(args.canonical)
            source_manifest = read(args.source_input_manifest)
            for name, frame in (("features", frames["features"]), ("states", states), ("canonical", canonical)):
                if frame_sha256(frame) != source_manifest["frames"][name]:
                    raise ValueError(f"frozen source {name} hash differs")
            canonical = canonical.loc[canonical["spy_close"].notna()].copy()
            labeler = CausalRegimeLabeler(RegimeLabelConfig(price_column="spy_close", minimum_fit_observations=260))
            labeler.fit(canonical.iloc[:520])
            reconstructed_states = labeler.transform(canonical)
            if not reconstructed_states.equals(states):
                raise ValueError("feature assembly does not reproduce the frozen official states")
            scores = labeler.score_frame(canonical)
            features = frames["features"].reindex(canonical.index).copy()
            for column in scores:
                features[f"regime_boundary__{column}"] = scores[column]
            frames["features"] = derive_causal_transition_features(features, states,
                risk_score_col="regime_boundary__risk_score", lower_threshold=labeler.lower_threshold_,
                upper_threshold=labeler.upper_threshold_).drop(columns=["current_state"])
        if not frames["features"].index.equals(states.index) or states.index[-1] != pd.Timestamp(payload["meta"]["data_as_of"]):
            raise ValueError("frozen features/states and published cutoff differ")
        for week in payload["weekly"]:
            if states.loc[pd.Timestamp(week["data_as_of"])] != week["current"]["state"]:
                raise ValueError("frozen states differ from published official states")
        expected_names = {name for group in feature_manifest["groups"] for name in group["features"]}
        if set(frames["features"].columns) != expected_names:
            raise ValueError("frozen features differ from approved feature contract")
        manifest = {"schema_version": "regime-operational-input-bundle/1", "data_as_of": states.index[-1].isoformat(),
            "candidate_manifest_sha256": payload["model"]["candidate_manifest_sha256"], "feature_manifest_sha256": feature_hash,
            "frames": {name: frame_sha256(frame) for name, frame in frames.items()}}
        recipe = preparation_recipe_lock(payload)
        args.output.mkdir(parents=True, exist_ok=True)
        for name in ("features", "states"):
            buffer = BytesIO()
            frames[name].to_pickle(buffer)
            destination = args.output / f"{name}.pkl"
            encoded = buffer.getvalue()
            if destination.exists() and destination.read_bytes() != encoded:
                raise ValueError("frozen frame already exists with different content")
            if not destination.exists():
                with destination.open("xb") as handle:
                    handle.write(encoded)
        _write_once(args.output / "input-manifest.json", manifest)
        _write_once(args.output / "recipe-lock.json", recipe)
        result = {"input_manifest": str((args.output / "input-manifest.json").resolve()),
                  "recipe_lock": str((args.output / "recipe-lock.json").resolve()), "status": "frozen_for_local_review"}
    elif args.command == "issue":
        result = issue_local_preparation(args.workspace, prepared=read(args.prepared),
            recipe_lock=read(args.recipe_lock), locked_payload=read(args.payload))
    elif args.command == "score":
        result = score_local_copy(args.workspace, states=pd.read_pickle(args.states), state_manifest=read(args.input_manifest),
            label_spec_sha256=read(args.payload)["label"]["spec_sha256"],
            states_available_at=datetime.fromisoformat(args.states_available_at) if args.states_available_at else None)
    else:
        result = local_summary(args.workspace)
    if args.command != "freeze" and args.output:
        _write_once(args.output, result)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
