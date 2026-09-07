#!/usr/bin/env python3
"""Explicit local research protocol/forecast/evaluation lifecycle.

Preview writes only the selected JSON output. No default ledger, live database,
clock override, deployment, execution-price dependency, or automatic promotion.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import shutil

from regime_lab.research.forecast_prospective import (
    ForecastResearchLedger, ResearchLedgerError, build_protocol_template, content_sha256,
    freeze_research_forecast, reconstructed_preview_information, capture_research_inputs,
    prepare_research_forecast, _utc, _utc_now,
)
from regime_lab.data.release_archive import weekly_decision_at


def _read(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _write_new(path: Path, value):
    """An explicit new artifact, atomically visible and never overwritten."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".research-preview-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write((json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    template = subparsers.add_parser("template", help="build actual-block preview and future protocol template; no DB or issuance")
    template.add_argument("--block", type=Path, required=True)
    template.add_argument("--label-spec", type=Path, required=True)
    template.add_argument("--version", required=True)
    template.add_argument("--monitoring-origins", type=int, default=104)
    template.add_argument("--criteria", type=Path, help="optional complete proposed criteria JSON before registration")
    template.add_argument("--output-dir", type=Path, required=True, help="new directory for the complete preview bundle")
    preview = subparsers.add_parser("preview", help="validate and freeze JSON; no database or issuance")
    preview.add_argument("--block", type=Path, required=True)
    preview.add_argument("--protocol", type=Path, required=True)
    preview.add_argument("--information-set", type=Path, required=True)
    preview.add_argument("--output", type=Path, required=True)
    capture = subparsers.add_parser("capture-inputs", help="capture timely actual local inputs and timestamps; no model result relabeling or issue")
    capture.add_argument("--protocol", type=Path, required=True)
    capture.add_argument("--input", type=Path, required=True, help="trusted current canonical.pkl, states.pkl, input-manifest.json")
    capture.add_argument("--source-oos", type=Path, required=True)
    capture.add_argument("--state-history", type=Path, required=True)
    capture.add_argument("--output-dir", type=Path, required=True, help="new immutable current input capture directory")
    prepare = subparsers.add_parser("prepare", help="run fixed models freshly from an actual input capture; no DB or issuance")
    prepare.add_argument("--protocol", type=Path, required=True)
    prepare.add_argument("--capture-dir", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True, help="new prepared forward forecast directory")
    initialize = subparsers.add_parser("init", help="explicitly initialize a separate research ledger")
    initialize.add_argument("--ledger", type=Path, required=True)
    register = subparsers.add_parser("register", help="freeze a protocol before its first scheduled origin")
    register.add_argument("--ledger", type=Path, required=True)
    register.add_argument("--protocol", type=Path, required=True)
    issue = subparsers.add_parser("issue", help="explicit real-time LOCAL research publication; never historical replay")
    issue.add_argument("--ledger", type=Path, required=True)
    issue.add_argument("--frozen", type=Path, required=True)
    issue.add_argument("--publication", type=Path, required=True,
                       help="new local publication file (exclusive create, no existing file replacement)")
    evaluate = subparsers.add_parser("evaluate", help="append matured official state-path scores without execution prices")
    evaluate.add_argument("--ledger", type=Path, required=True)
    evaluate.add_argument("--states", type=Path, required=True)
    summary = subparsers.add_parser("summary", help="read continuous evidence and manual-review readiness")
    summary.add_argument("--ledger", type=Path, required=True)
    summary.add_argument("--protocol-sha256", required=True)
    summary.add_argument("--output", type=Path, help="optional new JSON artifact; otherwise stdout only")
    args = parser.parse_args(argv)
    try:
        if args.command == "template":
            result = _template_bundle(args)
        elif args.command == "capture-inputs":
            result = capture_research_inputs(protocol=_read(args.protocol), input_directory=args.input,
                                             source_oos=args.source_oos, official_state_history=args.state_history,
                                             output_directory=args.output_dir)
        elif args.command == "prepare":
            if args.output_dir.exists():
                raise FileExistsError(f"output directory already exists: {args.output_dir}")
            prepared = prepare_research_forecast(protocol=_read(args.protocol), capture_directory=args.capture_dir)
            args.output_dir.parent.mkdir(parents=True, exist_ok=True)
            staging = Path(tempfile.mkdtemp(dir=args.output_dir.parent, prefix=".research-prepared-"))
            try:
                for name, value in (("producer-latest.json", prepared["producer_block"]),
                                    ("information-set.json", prepared["information_set"]), ("frozen.json", prepared["frozen"])):
                    _write_new(staging / name, value)
                staging.rename(args.output_dir)
            except BaseException:
                shutil.rmtree(staging, ignore_errors=True)
                raise
            result = {"output": str(args.output_dir.resolve()), "status": prepared["status"], "issued_forecasts": 0,
                      "forecast_sha256": prepared["frozen"]["forecast_sha256"],
                      "eligible_for_issue_at_preview": prepared["frozen"]["eligible_for_issue_at_preview"]}
        elif args.command == "preview":
            result = freeze_research_forecast(_read(args.block), protocol=_read(args.protocol),
                                              information_set=_read(args.information_set))
            _write_new(args.output, result)
        else:
            ledger = ForecastResearchLedger(args.ledger, create=args.command == "init")
            if args.command == "init":
                result = {"ledger": str(ledger.path), "initialized": True, "issued_forecasts": 0}
            elif args.command == "register":
                result = ledger.register_protocol(_read(args.protocol))
            elif args.command == "issue":
                result = ledger.issue(_read(args.frozen), publication_path=args.publication)
            elif args.command == "evaluate":
                result = ledger.evaluate_matured(_read(args.states))
            else:
                result = ledger.readiness(args.protocol_sha256)
                if args.output:
                    _write_new(args.output, result)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    except (ResearchLedgerError, OSError, json.JSONDecodeError) as exc:
        print(f"Research ledger error: {exc}", file=sys.stderr)
        return 2
    return 0


def _template_bundle(args):
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    block = _read(args.block)
    label_raw = args.label_spec.read_bytes()
    label = json.loads(label_raw)
    label_version = label["specs"][label["default_spec"]]["version"]
    kwargs = {"label_version": label_version, "label_sha256": hashlib.sha256(label_raw).hexdigest(),
              "monitoring_origins": args.monitoring_origins,
              "criteria": _read(args.criteria) if args.criteria else None}
    preview_protocol = build_protocol_template(block, version=args.version + "-preview", **kwargs)
    information = reconstructed_preview_information(block)
    frozen = freeze_research_forecast(block, protocol=preview_protocol, information_set=information)
    # Find the next configured cutoff, preserving the project's holiday/DST convention.
    origin = _utc(block["data_as_of"])
    from zoneinfo import ZoneInfo
    next_date = origin.astimezone(ZoneInfo("America/New_York")).date()
    while weekly_decision_at(next_date) <= _utc_now():
        next_date += timedelta(weeks=1)
    future = build_protocol_template(block, version=args.version + "-forward",
                                     first_origin=weekly_decision_at(next_date), **kwargs)
    manifest = {"status": "local_preview_only", "source_block": str(args.block.resolve()),
                "source_block_file_sha256": hashlib.sha256(args.block.read_bytes()).hexdigest(),
                "source_block_semantic_sha256": content_sha256(block),
                "generated_at": block["generated_at"], "evidence_track": block["evidence_track"],
                "forecast_sha256": frozen["forecast_sha256"], "future_protocol_sha256": content_sha256(future),
                "future_first_origin": future["calendar"][0], "registered_protocols": 0, "issued_forecasts": 0,
                "database_created": False, "automatic_promotion": False,
                "issue_disqualifiers": frozen["forecast"]["issue_disqualifiers"]}
    lines = ["# 전향 국면 예측 원장 — 실제 연구 결과 미리보기", "",
             "**실제 발행 0건 · 프로토콜 등록 0건 · DB 생성 없음.** 현재 연구 결과를 그대로 동결한 검토용 산출물입니다.", "",
             f"- 관측 기준: {block['data_as_of']}", f"- 실제 연구 결과 생성: {block['generated_at']}",
             f"- 증거 구분: `{block['evidence_track']}`. 전향 실적으로 사용할 수 없습니다.",
             "- 과거 입력의 실제 공개·수집 시점이 증명되지 않은 경우 null로 보존합니다.",
             f"- 발행 부적격 사유: {', '.join(manifest['issue_disqualifiers'])}", "",
             "| 모델 | 다음 주 위험선호 | 다음 주 전환 | 다음 주 위험회피 | 4주 내 위험회피 진입 | 13주 내 위험회피 진입 |",
             "|---|---:|---:|---:|---:|---:|"]
    for model in frozen["forecast"]["models"]:
        by_h = {p["horizon_weeks"]: p for p in model["paths"]}
        probability = by_h[1]["endpoint"]
        tail = [f"{by_h[h]['any_risk_off_entry']:.2%}" if h in by_h else "—" for h in (4, 13)]
        lines.append(f"| {model['id']} | {probability['risk_on']:.2%} | {probability['transition']:.2%} | {probability['risk_off']:.2%} | {' | '.join(tail)} |")
    lines.extend(["", "첫 이탈 목적지와 기간 말 국면, 기간 중 위험회피 진입·점유는 별도 목표로 저장·평가합니다.", "",
                  f"미래 프로토콜 템플릿의 첫 관측 기준은 **{future['calendar'][0]}**입니다. 금요일 뉴욕 16시 기준이며 DST를 반영합니다.",
                  "첫 관측 기준 이전의 명시적 등록이 필요합니다. 현재 미리보기 프로토콜은 이미 지난 기준일을 포함하므로 등록할 수 없습니다.", "",
                  "후보마다 같은 시점에 저장한 기준모형과 비교합니다. 매주 발행 누락, 성숙했지만 미평가된 표본, 악화·회복 사건 수 및 독립 위험회피 진입 수가 승격 검토 조건에 포함됩니다.",
                  "초기 조건은 검토 제안이며, 52개 짝지은 관측만으로 충분하다고 판단하지 않습니다. 모든 조건이 통과해도 결과는 수동 검토 준비일 뿐 자동 승격하지 않습니다.", "",
                  "- `protocol.preview.json`: 현재 결과의 정확한 날짜·모델·라벨·기준모형·검토 조건.",
                  "- `frozen-preview.json`: 원래 확률값을 변경하지 않은 동결 결과와 해시.",
                  "- `information-set.preview.json`: 확인되지 않은 과거 가용 시각을 null로 보존한 입력 출처.",
                  "- `protocol.future-template.json`: 미래 등록용 템플릿. 아직 등록되지 않았습니다.",
                  "- `preview-manifest.json`: 원본 파일·의미 해시와 발행 부적격 사유."])
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=args.output_dir.parent, prefix=".prospective-preview-"))
    try:
        for name, value in (("protocol.preview.json", preview_protocol), ("information-set.preview.json", information),
                            ("frozen-preview.json", frozen), ("protocol.future-template.json", future), ("preview-manifest.json", manifest)):
            _write_new(staging / name, value)
        (staging / "preview.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        if hashlib.sha256(args.block.read_bytes()).hexdigest() != manifest["source_block_file_sha256"]:
            raise ResearchLedgerError("source block changed while preparing the preview")
        if args.label_spec.read_bytes() != label_raw:
            raise ResearchLedgerError("label specification changed while preparing the preview")
        staging.rename(args.output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"output": str(args.output_dir.resolve()), **manifest}


if __name__ == "__main__":
    raise SystemExit(main())
