"""Offline reproducible model-economics research from immutable local frames."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from regime_lab.analysis.directional_coherence import upgrade_directional_payload
from regime_lab.analysis.label_sensitivity import run_label_sensitivity
from regime_lab.analysis.duration import duration_context
from regime_lab.analysis.labels import CausalRegimeLabeler
from regime_lab.analysis.transitions import derive_causal_transition_features
from regime_lab.io import write_json_atomic
from regime_lab.v5_artifacts import (
    V5_RESEARCH_ARTIFACTS,
    canonical_v5_artifact_csv_bytes,
)
from regime_lab.analysis.directional import (
    DirectionalBenchmarkResult,
    run_directional_transition_benchmark,
)


def _directional_horizon(arguments):
    horizon, feature_path, state_path, cache_path = arguments
    features = pd.read_pickle(feature_path)
    states = pd.read_pickle(state_path)
    return run_directional_transition_benchmark(
        features,
        states,
        horizons=(horizon,),
        minimum_train_weeks=520,
        selection_end="2023-01-01",
        cache_directory=cache_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("labels", "directional", "upgrade"))
    parser.add_argument("--input", type=Path, default=Path("build/comprehensive/input"))
    parser.add_argument(
        "--output", type=Path, default=Path("build/comprehensive/model-economics")
    )
    parser.add_argument(
        "--payload", type=Path, default=Path("publication/live/regime-results.json")
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    canonical = pd.read_pickle(args.input / "canonical.pkl")
    states = pd.read_pickle(args.input / "states.pkl")
    if args.mode == "labels":
        result = run_label_sensitivity(canonical, states=states)
        result.spec_metrics.to_csv(args.output / "label-grid-metrics.csv", index=False)
        result.representative_oos.to_csv(
            args.output / "label-representative-oos.csv", index=False
        )
        write_json_atomic(args.output / "label-sensitivity.json", result.summary)
        print(
            json.dumps(
                {
                    "mode": "labels",
                    "variants": result.summary["execution_summary"][
                        "evaluated_spec_count"
                    ],
                    "model_oos_rows": len(result.representative_oos),
                }
            ),
            flush=True,
        )
    elif args.mode == "directional":
        features = pd.read_pickle(args.input / "features.pkl")
        if "regime_boundary__risk_score" not in features:
            labeler = CausalRegimeLabeler().fit(canonical.iloc[:520])
            score = labeler.score_frame(canonical)
            features = features.copy()
            for column in score:
                features[f"regime_boundary__{column}"] = score[column]
            features = derive_causal_transition_features(
                features,
                states,
                risk_score_col="regime_boundary__risk_score",
                lower_threshold=labeler.lower_threshold_,
                upper_threshold=labeler.upper_threshold_,
            ).drop(columns=["current_state"])
        features.to_pickle(args.output / "directional-features.pkl")
        with ProcessPoolExecutor(max_workers=3) as pool:
            partial = list(
                pool.map(
                    _directional_horizon,
                    [
                        (
                            h,
                            args.output / "directional-features.pkl",
                            args.input / "states.pkl",
                            args.output / "directional-cache",
                        )
                        for h in (1, 4, 13)
                    ],
                )
            )
        result = DirectionalBenchmarkResult(
            leaderboard=pd.concat([r.leaderboard for r in partial], ignore_index=True),
            predictions=pd.concat([r.predictions for r in partial], ignore_index=True),
            split_audit=pd.concat([r.split_audit for r in partial], ignore_index=True),
            selection_diagnostics=pd.concat(
                [r.selection_diagnostics for r in partial], ignore_index=True
            ),
            champions_by_horizon={
                h: m for r in partial for h, m in r.champions_by_horizon.items()
            },
            latest_forecasts=pd.concat(
                [r.latest_forecasts for r in partial], ignore_index=True
            ),
            selection_end=partial[0].selection_end,
        )
        for name, frame in (
            ("directional-oos", result.predictions),
            ("directional-leaderboard", result.leaderboard),
            ("directional-splits", result.split_audit),
            ("directional-selection", result.selection_diagnostics),
            ("directional-forecasts", result.latest_forecasts),
        ):
            frame.to_csv(args.output / f"{name}.csv", index=False)
        result_path = args.output / "directional-result.pkl"
        pd.to_pickle(result, result_path)
        for key, frame in {
            "directional_oos_predictions": result.predictions,
            "directional_model_leaderboard": result.leaderboard,
            "directional_walk_forward_splits": result.split_audit,
            "directional_selection_diagnostics": result.selection_diagnostics,
            "directional_forecasts": result.latest_forecasts,
        }.items():
            (args.output / V5_RESEARCH_ARTIFACTS[key].path).write_bytes(
                canonical_v5_artifact_csv_bytes(key, frame)
            )
        write_json_atomic(
            args.output / "directional-execution.json",
            {
                "schema_version": "full-directional-research/2",
                "evidence_track": "reconstructed_oos",
                "profile": "standard",
                "maximum_selection_origins": None,
                "maximum_diagnostic_origins": None,
                "champions": {
                    str(k): v for k, v in result.champions_by_horizon.items()
                },
                "selection_end": "2023-01-01",
                "prediction_count": len(result.predictions),
                "feature_count": len(features.columns),
                "feature_frame_sha256": hashlib.sha256(
                    pd.util.hash_pandas_object(features, index=True).values.tobytes()
                ).hexdigest(),
                "state_frame_sha256": hashlib.sha256(
                    pd.util.hash_pandas_object(states, index=True).values.tobytes()
                ).hexdigest(),
                "automatic_promotion_eligible": False,
            },
        )
        print(
            json.dumps(
                {
                    "mode": "directional",
                    "prediction_rows": len(result.predictions),
                    "champions": dict(result.champions_by_horizon),
                }
            ),
            flush=True,
        )
    else:
        original = json.loads(args.payload.read_text())
        upgraded = upgrade_directional_payload(original, states)
        for week in upgraded["weekly"]:
            week["duration_context"] = duration_context(
                states,
                as_of=week.get("data_as_of", week["date"]),
                bootstrap_resamples=1999
                if week["date"] == upgraded["weekly"][-1]["date"]
                else 0,
            )
        write_json_atomic(args.output / "coherent-research-payload.json", upgraded)
        write_json_atomic(
            args.output / "directional-coherence-evidence.json",
            upgraded["model"]["directional_transition"]["coherence_evidence"],
        )
        print(
            json.dumps({"mode": "upgrade", "weeks": len(upgraded["weekly"])}),
            flush=True,
        )


if __name__ == "__main__":
    main()
