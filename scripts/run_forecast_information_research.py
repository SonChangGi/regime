#!/usr/bin/env python3
"""Collect optional new information and run matched-origin forecast experiments.

All outputs stay under build/forecast-audit-improvements/new-information.
Existing snapshots, derived inputs and publication are read-only references.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import hashlib
from html import escape
import json
from pathlib import Path

import pandas as pd

from regime_lab.research.forecast_information_study import read_existing_cboe, evaluate_stored_block
from regime_lab.operational_forecast import frame_sha256
from regime_lab.research import forecast_new_information as research

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "build/forecast-audit-improvements/new-information"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_derived_input(directory: Path, payload: dict, history: pd.DataFrame) -> dict:
    manifest_path = directory / "input-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    data_as_of = research.utc(payload["meta"]["data_as_of"])
    if research.utc(manifest["data_as_of"]) != data_as_of:
        raise ValueError("derived input and publication data_as_of differ")
    proof = {"directory": str(directory.resolve()), "data_as_of": data_as_of,
             "manifest_sha256": digest(manifest_path), "files": {}}
    for key in ("canonical", "states", "features"):
        path = directory / f"{key}.pkl"
        frame = pd.read_pickle(path)  # explicit trusted local derived-input interface
        actual_sha = frame_sha256(frame)
        if actual_sha != manifest["frames"][key] or research.utc(frame.index[-1]) != data_as_of:
            raise ValueError(f"{key}: manifest checksum or terminal date mismatch")
        proof["files"][key] = {"file_sha256": digest(path), "frame_sha256": actual_sha, "rows": len(frame)}
        if key == "states":
            for _, row in history.iterrows():
                if frame.loc[row.origin_date] != row.current_state or frame.loc[row.target_date] != row.actual:
                    raise ValueError("boundary history differs from verified official states")
    proof["origin_and_target_states_match"] = True
    return proof


def preview(summary: dict, path: Path) -> None:
    columns = ["모델", "표본", "Log loss", "Brier", "악화 포착", "회복 포착"]
    table = []
    metrics = summary.get("experiments", {}).get("vix3m", {}).get("metrics", {}).get("holdout", {})
    labels = {"boundary_reference": "동결 경계 기준선", "control": "기준선 + 기존 변동성 정보", "candidate": "기준선 + 기존 정보 + VIX3M"}
    for key in labels:
        if key in metrics:
            item = metrics[key]
            cells = [labels[key], str(item["n"]), f'{item["log_loss"]:.4f}', f'{item["brier"]:.4f}',
                     f'{item["worsening"]["captured"]}/{item["worsening"]["events"]}',
                     f'{item["recovery"]["captured"]}/{item["recovery"]["events"]}']
            table.append("<tr>" + "".join(f"<td>{escape(cell)}</td>" for cell in cells) + "</tr>")
    block_labels = {"vix3m": "VIX3M 기간구조", "fomc": "FOMC 일정", "bls": "CPI·고용 일정",
                    "cftc_tff": "CFTC 금융선물 포지션", "board_ebp": "초과 채권 프리미엄 EBP"}
    status_labels = {"evaluated": "동일 표본 실험 완료", "prospective_only": "첫 수집 이후에만 사용 · 과거 이용시점 미확인",
                     "unavailable": "선택적 자료 미사용", "invalid_optional_block": "입력 검증 실패"}
    states = "".join(f'<li><b>{escape(block_labels.get(key, key))}</b>: {escape(status_labels.get(value.get("status"), str(value.get("status"))))}</li>'
                     for key, value in summary["blocks"].items())
    html = """<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>국면 예측 신규 정보 실험</title><style>body{font:16px/1.65 system-ui,sans-serif;background:#f4f6f8;color:#192331;margin:0}main{max-width:1050px;margin:40px auto;padding:32px;background:white;border-radius:16px}h1{font-size:27px}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:12px;border-bottom:1px solid #dce2e8;text-align:right}th:first-child,td:first-child{text-align:left}.table{overflow-x:auto}.note{background:#eef4fa;padding:16px;border-radius:8px}a{color:#175da5}li{margin:12px 0}@media(max-width:600px){main{margin:0;padding:20px}th,td{padding:8px}}</style>
<main><h1>국면 예측: 신규 정보의 추가 효과</h1><p>VIX3M을 기존 VIX·VIX9D·VVIX와 함께 사용했을 때 다음 주 국면 예측이 개선되는지 비교했습니다.</p>
<p class="note">2023년 이후는 이미 검토된 진단 구간입니다. 과거 시장자료의 공개시점 재구성이며 실제 당시 발행 성과가 아닙니다. 일정과 CFTC는 저장된 과거 버전이 없으면 과거 점수를 만들지 않습니다.</p>
<h2>동일한 주의 진단 비교</h2><div class="table"><table><thead><tr>"""
    html += "".join(f"<th>{escape(c)}</th>" for c in columns) + "</tr></thead><tbody>" + "".join(table) + "</tbody></table></div>"
    html += '<p>확률 점수는 낮을수록 좋습니다. 포착은 최빈 예측이 실제 도착 국면까지 맞힌 건수입니다. 동일 학습표본과 고정 설정을 사용합니다.</p>'
    alternate_note = ("BLS 일정은 공식 페이지에서 확보한 대체 표현을 사용합니다. "
                      if summary["blocks"].get("bls", {}).get("alternate_representation") else "")
    html += f'<h2>선택적 블록 상태</h2><ul>{states}</ul><p>{alternate_note}일정은 확인한 버전 이후에만 사용하며 오래된 버전은 미상으로 처리합니다.</p><p><a href="summary.json">전체 JSON·출처·검증 결과</a> · <a href="oos-predictions.csv">동일 주 예측 CSV</a></p><p>운영 모델 및 실시간 발행 결과는 변경하지 않았습니다.</p></main></html>'
    path.write_text(html)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, default=ROOT / "publication/live/regime-results.json")
    parser.add_argument("--input", type=Path, help="Read-only trusted derived directory containing input-manifest and three pickles")
    parser.add_argument("--existing-sources", type=Path, help="Read-only additional-sources manifest for existing VIX/VIX9D/VVIX")
    parser.add_argument("--config", type=Path, default=ROOT / "config/forecast-new-information.json")
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--offline", action="store_true", help="Use stored snapshots only; never fetch")
    parser.add_argument("--refresh", action="store_true", help="Append new source versions")
    parser.add_argument("--as-of", help="Runtime decision timestamp; never backdates source retrieval")
    parser.add_argument("--bls-web-extract", type=Path, help="Saved current official CPI+jobs web-tool page text; imported now, never backdated")
    parser.add_argument("--ebp-records", type=Path, help="Optional board_ebp ReleaseRecord JSON list, already collected elsewhere")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(OUTPUT_ROOT.resolve()):
        raise ValueError(f"outputs must stay within {OUTPUT_ROOT}")
    if args.offline and args.refresh:
        raise ValueError("offline and refresh are mutually exclusive")
    output.mkdir(parents=True, exist_ok=True)
    started = pd.Timestamp(datetime.now(timezone.utc))
    payload_sha = digest(args.payload)
    payload = json.loads(args.payload.read_text())
    history = research.load_boundary_history(payload)
    config = json.loads(args.config.read_text())
    if config["baseline_model"] != "boundary_filtered_history" or research.utc(config["selection_end"]) != research.SELECTION_END:
        raise ValueError("protocol baseline/cutoff disagrees with implementation")
    summary = {"schema_version": research.SCHEMA_VERSION, "started_at": started,
               "data_as_of": payload["meta"]["data_as_of"], "payload_sha256": payload_sha,
               "protocol": config, "protocol_sha256": digest(args.config),
               "module_sha256": digest(Path(research.__file__)), "script_sha256": digest(Path(__file__)),
               "promotion": "none", "blocks": {}, "sources": {}, "experiments": {}}
    if args.input:
        summary["derived_input_verification"] = verify_derived_input(args.input, payload, history)
    # Persist protocol before any fitting. This is a run registration, not a claim
    # that the developers have never looked at the diagnostic period.
    research.write_json(output / "run-registration.json", summary)
    from regime_lab.research.forecast_information_study import run_information_study
    study = run_information_study(history, data_as_of=payload["meta"]["data_as_of"],
        config=config, output=output, sources_dir=output / "sources",
        existing_sources=args.existing_sources, offline=args.offline, refresh=args.refresh,
        as_of=args.as_of, bls_web_extract=args.bls_web_extract,
        ebp_records_path=args.ebp_records, progress=print)
    summary.update(study)
    summary["source_payload_unchanged"] = digest(args.payload) == payload_sha
    if not summary["source_payload_unchanged"]:
        raise ValueError("source publication changed during research; rerun with an immutable input")
    if args.input:
        if summary["derived_input_verification"] != verify_derived_input(args.input, payload, history):
            raise ValueError("read-only derived inputs changed during research")
    summary["completed_at"] = pd.Timestamp(datetime.now(timezone.utc))
    adapter = research.build_forecast_information(summary)
    research.write_json(output / "forecast-information.json", adapter)
    summary["ui_adapter"] = {"file": "forecast-information.json", "schema_version": adapter["schema_version"],
                             "rows": len(adapter["rows"])}
    research.write_json(output / "summary.json", summary)
    preview(research.json_safe(summary), output / "preview.html")
    print(json.dumps(research.json_safe({"summary": str(output / "summary.json"),
                     "blocks": {k: v["status"] for k, v in summary["blocks"].items()}}), indent=2))


if __name__ == "__main__":
    main()
