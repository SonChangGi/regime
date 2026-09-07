#!/usr/bin/env python3
"""Run the frozen audit recipe from trusted saved derived inputs, without a DB.

The output directory must be new. Full output is installed only after validation;
an unsuccessful run cannot overwrite a previously validated result.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from regime_lab.analysis.forecast_audit_research import (
    AuditResearchProtocol, forecast_research_extension, json_safe, run_audit_research,
)
from regime_lab.operational_forecast import frame_sha256


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="trusted canonical.pkl, states.pkl, input-manifest.json directory")
    parser.add_argument("--source-oos", required=True, type=Path)
    parser.add_argument("--state-history", type=Path, help="optional saved official state-label-history.csv for parity")
    parser.add_argument("--output", required=True, type=Path, help="new local output directory")
    parser.add_argument("--expected-as-of", default="2026-09-04T20:00:00+00:00")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output already exists; use a new run directory")
    tracked = {name: args.input / name for name in ("canonical.pkl", "states.pkl", "input-manifest.json")}
    tracked["source_oos"] = args.source_oos
    if args.state_history:
        tracked["official_state_history"] = args.state_history
    code = ["src/regime_lab/analysis/boundary_forecast.py", "src/regime_lab/analysis/forecast_paths.py",
            "src/regime_lab/analysis/forecast_audit_research.py", "src/regime_lab/analysis/forecast_research_evaluation.py",
            "src/regime_lab/analysis/labels.py", "scripts/run_forecast_audit_research.py",
            "docs/forecast-audit-research-protocol.md"]
    tracked.update({name: ROOT / name for name in code})
    hashes = {name: digest(path) for name, path in tracked.items()}
    manifest = json.loads((args.input / "input-manifest.json").read_text())
    canonical, states = pd.read_pickle(args.input / "canonical.pkl"), pd.read_pickle(args.input / "states.pkl")
    expected = pd.Timestamp(args.expected_as_of)
    if expected.tzinfo is None or pd.Timestamp(manifest["data_as_of"]) != expected or canonical.index[-1] != expected or states.index[-1] != expected:
        raise ValueError("saved cache dates do not match expected_as_of")
    for name, frame in (("canonical", canonical), ("states", states)):
        if frame_sha256(frame) != manifest["frames"][name]:
            raise ValueError(f"{name} content differs from its saved manifest hash")
    if args.state_history:
        official = pd.read_csv(args.state_history)
        official.index = pd.to_datetime(official.date, utc=True, format="ISO8601")
        if not states.index.equals(official.index) or not (states.to_numpy() == official.state.to_numpy()).all():
            raise ValueError("saved cache states differ from current official derived state history")
    baseline = pd.read_csv(args.source_oos)
    protocol = AuditResearchProtocol()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output.name}-", dir=args.output.parent))
    started = time.monotonic()
    # Freeze before fitting: this includes recipe and all source/code hashes.
    write_json(staging / "frozen-protocol.json", {"protocol": protocol.record(), "source_hashes": hashes,
                                                "frozen_before_run_at": datetime.now(timezone.utc).isoformat()})
    try:
        result = run_audit_research(canonical, states, baseline, protocol=protocol,
                                    progress=lambda message: print(message, flush=True))
        result.document["generated_at"] = datetime.now(timezone.utc).isoformat()
        result.document["source_hashes"] = hashes
        result.document["source_context"] = {
            "input_manifest": manifest, "input_directory": str(args.input.resolve()),
            "source_oos": str(args.source_oos.resolve()), "raw_data_or_api_audit": False,
            "cache_manifest_hash_verified": True, "official_state_history_parity": args.state_history is not None,
            "runtime": {name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn")}}
        extension = forecast_research_extension(result.document)
        write_json(staging / "audit-research.json", result.document)
        write_json(staging / "forecast_research.json", extension)
        result.predictions.to_csv(staging / "oos-predictions.csv", index=False)
        write_json(staging / "path-predictions.json", result.path_predictions)
        result.path_scores.to_csv(staging / "path-scores.csv", index=False)
        result.alert_predictions.to_csv(staging / "alert-policies.csv", index=False)
        result.episode_details.to_csv(staging / "episode-details.csv", index=False)
        selected = [row for row in result.document["metrics"] if row.get("period") == "retrospective_2023_2026" and row["target"] == "next_state"]
        lines = ["# 국면 예측 감사 개선 실험 v1", "", "기존 라벨·챔피언을 유지한 로컬 연구 결과다. 2023년 이후는 이미 확인한 진단 구간이다.", "",
                 "| 모델 | 표본 | Log loss | Brier | 악화 포착/사건 | 회복 포착/사건 |", "|---|---:|---:|---:|---:|---:|"]
        for row in selected:
            lines.append(f"| {row['model']} | {row['weeks']} | {row['log_loss']:.4f} | {row['brier']:.4f} | {row['worsening_hits']}/{row['worsening_events']} | {row['recovery_hits']}/{row['recovery_events']} |")
        lines.extend(["", "- `forecast_research.json`: 로컬 UI/부모 조립용 확장. 자동 승격 없음.",
                      "- `audit-research.json`: 프로토콜·동일 표본 점수·경보 예산·에피소드·지속성·경제적 연관성·잔차 진단.",
                      "- `oos-predictions.csv`, `path-scores.csv`, `alert-policies.csv`, `episode-details.csv`: 재검산용 행 단위 근거.",
                      "- 경제적 검증은 미래 변동성·하방사건과 위험확률의 연관성을 보여주며 투자 수익이나 인과관계를 입증하지 않는다.",
                      "- 경로 예측에서는 상태와 지속기간이 갱신되고 경계·충격 공변량은 원점 값으로 유지된다.",
                      "- 선정 구간 오경보 예산과 이후 실현 오경보는 구분한다. 초과 여부도 JSON에 기록된다."])
        (staging / "research-report.md").write_text("\n".join(lines) + "\n")
        for name, path in tracked.items():
            if digest(path) != hashes[name]:
                raise RuntimeError(f"research source changed during run: {name}")
        write_json(staging / "run-manifest.json", {"status": "validated_local_research", "data_as_of": args.expected_as_of,
                                                 "protocol_sha256": protocol.record()["sha256"], "source_hashes": hashes,
                                                 "elapsed_seconds": time.monotonic() - started,
                                                 "files": {path.name: digest(path) for path in sorted(staging.iterdir()) if path.is_file()}})
        staging.rename(args.output)
        print(json.dumps({"output": str(args.output.resolve()), "elapsed_seconds": time.monotonic() - started,
                          "source_hashes_verified": True, "automatic_promotion": False}, indent=2), flush=True)
    except BaseException:
        # Preserve the frozen protocol for diagnosis, but never present partial
        # output under the final destination or replace another successful run.
        print(f"Run incomplete; staging retained at {staging}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
