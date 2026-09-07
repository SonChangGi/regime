#!/usr/bin/env python3
"""Replay the real weekly research composer using immutable local evidence."""

from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

sys.dont_write_bytecode = True

import pandas as pd
import requests

from regime_lab.integrity import canonical_json_sha256_v1
from regime_lab.research.publication import compose_live_publication_research


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(paths: list[Path]) -> dict[str, str]:
    result = {}
    for path in paths:
        if path.is_dir():
            result.update({str(item.resolve()): sha256(item) for item in sorted(path.rglob("*")) if item.is_file()})
        else:
            result[str(path.resolve())] = sha256(path)
    return result


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-root", type=Path, default=Path("/Users/changgison/projects/regime-improvements/build/release-20260904"))
    parser.add_argument("--payload", type=Path, default=Path("publication/live/regime-results.json"))
    parser.add_argument("--expected-preview", type=Path, default=Path("build/forecast-performance/preview-root/regime/data/regime-results.json"))
    parser.add_argument("--output", type=Path, default=Path("build/forecast-performance/weekly-replay-cache"))
    args = parser.parse_args()
    base, output = args.frozen_root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "replay.lock").open("a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("A weekly replay is already running in this output directory") from exc
    lock.seek(0)
    lock.truncate()
    lock.write(json.dumps({"pid": os.getpid(), "script": str(Path(__file__).resolve())}) + "\n")
    lock.flush()

    source = base / "source"
    payload_path = args.payload.resolve()
    expected_path = args.expected_preview.resolve()
    payload = json.loads(payload_path.read_text())
    expected_preview = json.loads(expected_path.read_text())
    expected_forecast = expected_preview["research"]["forecast_improvement"]
    cutoff = str(payload["meta"]["data_as_of"])
    cutoff_date = pd.Timestamp(cutoff).date().isoformat()
    snapshot = base / "weekly-replay-cache" / "additional-sources" / cutoff_date
    if not snapshot.is_dir():
        raise ValueError("The verified immutable additional-source snapshot is missing")
    cache = output / "additional-sources"
    cache.mkdir(exist_ok=True)
    snapshot_link = cache / cutoff_date
    if snapshot_link.exists() or snapshot_link.is_symlink():
        if not snapshot_link.is_symlink() or snapshot_link.resolve() != snapshot.resolve():
            raise ValueError("Existing source-cache entry does not point to the frozen snapshot")
    else:
        snapshot_link.symlink_to(snapshot, target_is_directory=True)

    names = (
        "oos-predictions.csv", "transition-oos-predictions.csv",
        "model-conditioned-asset-outcomes.csv", "feature-ablation-oos-predictions.csv",
        "state-label-history.csv",
    )
    read = lambda name: pd.read_csv(source / "artifacts" / name)
    benchmark = SimpleNamespace(
        predictions=read(names[0]),
        transition_benchmark=SimpleNamespace(predictions=read(names[1])),
        model_conditioned_asset_outcomes=read(names[2]),
        feature_ablation=SimpleNamespace(predictions=read(names[3])),
        state_label_history=read(names[4]),
    )
    canonical = pd.read_pickle(base / "input/canonical.pkl")
    ledger = source / "forecast-ledger.sqlite3"
    protected_paths = [payload_path, expected_path, base / "input/canonical.pkl", ledger, snapshot]
    protected_paths += [source / "artifacts" / name for name in names]
    protected_paths += [Path("src/regime_lab/research/publication.py").resolve(), Path("src/regime_lab/research/forecast_improvement.py").resolve(), Path("src/regime_lab/research/forecast_assets.py").resolve()]
    hashes_before = inventory(protected_paths)
    write_json(output / "frozen-inputs.json", hashes_before)
    write_json(output / "expected-forecast-improvement.json", expected_forecast)

    # Recreate the state presented to the normal generation composer before its
    # existing publication review. These changes affect only this local copy.
    payload["meta"]["publication_status"] = "unpublished"
    payload["meta"].pop("publication_review", None)
    payload["meta"].pop("generation_manifest_sha256", None)
    payload["model"]["lifecycle"]["publication"] = {"status": "unpublished"}
    payload["model"]["lifecycle"]["deployment"] = {"status": "candidate"}
    before = deepcopy(payload)
    network_attempts = []

    def forbid_network(*_args, **_kwargs):
        network_attempts.append("requests.Session.request")
        raise AssertionError("The weekly frozen-source replay must not request network data")

    def forbid_external_writes(event, arguments):
        paths = []
        if event == "open":
            path, mode, flags = arguments
            writing = bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            if isinstance(mode, str):
                writing = writing or any(char in mode for char in "wax+")
            if writing:
                paths = [path]
        elif event in {"os.mkdir", "os.remove", "os.rmdir", "os.chmod"}:
            paths = [arguments[0]]
        elif event in {"os.rename", "os.link", "os.symlink"}:
            paths = list(arguments[:2])
        for path in paths:
            if isinstance(path, (str, bytes, os.PathLike)):
                resolved = Path(os.fsdecode(path)).resolve()
                if not resolved.is_relative_to(output):
                    raise AssertionError(f"Replay write outside its output directory is forbidden: {resolved}")

    requests.sessions.Session.request = forbid_network
    sys.addaudithook(forbid_external_writes)
    started = time.monotonic()
    print(f"Real weekly composer started; {len(canonical)} canonical weeks, {len(benchmark.state_label_history)} authoritative state rows", flush=True)
    try:
        result = compose_live_publication_research(
            payload,
            dataset=SimpleNamespace(canonical=canonical), benchmark=benchmark,
            contract_version="v5", profile_name="standard", ledger_path=ledger,
            cache_directory=cache, progress=lambda message: print(message, flush=True),
        )
        # Keep the actual composed evidence even if a later independent
        # comparison fails; a successful run is signalled only by verification.
        write_json(output / "weekly-composed-results.json", result)
        assert payload == before, "The caller's payload was mutated"
        for key in ("meta", "model", "selection", "forecast", "weekly"):
            assert result[key] == before[key], f"Official {key} changed"
        assert result["research"]["prospective_decision_shadow"]["current_signal"] == before["research"]["prospective_decision_shadow"]["current_signal"]
        actual = result["research"]
        improvement = actual["forecast_improvement"]
        exact_forecast_fields = ("models", "baselines", "asset_statistics", "selected_model", "selection", "data_as_of", "evidence_track", "schema_version")
        for field in exact_forecast_fields:
            assert improvement[field] == expected_forecast[field], f"Standalone forecast field differs: {field}"
        pairs = {
            "allocation": (actual["prospective_decision_shadow"]["allocation_research_v2"], base / "research/allocation-v2.json"),
            "decision": (actual["decision_research_v2"], base / "research/decision-research-v2.json"),
            "downside": (actual["extensions"]["downside"], base / "downside.json"),
            "diagnostics": (actual["extensions"]["diagnostics"], base / "diagnostics.json"),
        }
        for name, (value, path) in pairs.items():
            assert value == json.loads(path.read_text()), f"Existing standalone research differs: {name}"
        operational = deepcopy(actual["operational_diagnostics"])
        expected_operational = json.loads((base / "research/operational-diagnostics.json").read_text())
        operational.pop("as_of", None)
        expected_operational.pop("as_of", None)
        assert operational == expected_operational, "Operational ledger diagnostics differ"
        assert not network_attempts
        hashes_after = inventory(protected_paths)
        assert hashes_after == hashes_before, "Frozen source, source snapshot, ledger, or active implementation changed"
        report = {
            "ok": True, "elapsed_seconds": round(time.monotonic() - started, 3),
            "data_as_of": cutoff, "official_weeks_preserved": len(result["weekly"]),
            "official_meta_model_selection_forecast_weekly_exact": True,
            "official_current_signal_exact": True, "caller_payload_unchanged": True,
            "network_attempts": 0, "writes_restricted_to_replay_output": True,
            "source_snapshot_reused_via_read_only_symlink": str(snapshot_link),
            "all_frozen_source_hashes_unchanged": True, "issued_ledger_unchanged": True,
            "canonical_weeks": len(canonical), "authoritative_state_rows": len(benchmark.state_label_history),
            "standalone_forecast_fields_exact": list(exact_forecast_fields),
            "forecast_provenance_equal": improvement["provenance"] == expected_forecast["provenance"],
            "expected_preview": str(expected_path),
            "baseline_input": str(source / "artifacts" / names[0]),
            "baseline_reader": "pandas.read_csv default",
            "forecast_evidence_sha256": canonical_json_sha256_v1({field: improvement[field] for field in exact_forecast_fields}),
            "per_model_holdout_weeks": {model["id"]: model["metrics"]["holdout"]["n_predictions"] for model in improvement["models"]},
            "asset_statistic_rows": len(improvement["asset_statistics"]["rows"]),
            "existing_research_exact": list(pairs),
            "operational_diagnostics_exact_except_calculation_clock": True,
            "phase_seconds": actual["extensions"]["build"]["phase_seconds"],
            "build": actual["extensions"]["build"],
        }
        write_json(output / "verification.json", report)
        print(json.dumps({key: value for key, value in report.items() if key != "build"}, ensure_ascii=False, indent=2), flush=True)
    except BaseException as error:
        write_json(output / "failure.json", {"ok": False, "error": f"{type(error).__name__}: {error}", "elapsed_seconds": round(time.monotonic() - started, 3), "network_attempts": len(network_attempts), "frozen_sources_unchanged": inventory(protected_paths) == hashes_before})
        raise
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    main()
