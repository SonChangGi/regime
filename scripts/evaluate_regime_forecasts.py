#!/usr/bin/env python3
"""Evaluate regime outcomes in an isolated SQLite snapshot; leave issued records intact."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import pandas as pd
from regime_lab.forecast_ledger import ForecastLedger, build_operational_diagnostics
from regime_lab.forecast_probability import mature_probability_evaluations
from regime_lab.io import write_json_atomic
from regime_lab.operational_forecast import frame_sha256


def evaluate_copy(source: Path, inputs: Path, payload_path: Path, output: Path) -> dict:
    source = source.resolve(strict=True)
    output = output.resolve()
    if output == source or output in source.parents:
        raise ValueError('evaluation output must not replace its source')
    output.mkdir(parents=True, exist_ok=True)
    target = output / 'forecast-ledger-evaluated.sqlite3'
    if target.exists():
        raise ValueError('choose a new evaluation directory to preserve prior evidence')
    payload = json.loads(payload_path.read_text())
    manifest = json.loads((inputs / 'input-manifest.json').read_text())
    states = pd.read_pickle(inputs / 'states.pkl')
    if frame_sha256(states) != manifest['frames']['states']:
        raise ValueError('state input hash differs')
    if states.index[-1] != pd.Timestamp(payload['meta']['data_as_of']):
        raise ValueError('state and published cutoff differ')
    for week in payload['weekly']:
        if states.loc[pd.Timestamp(week['data_as_of'])] != week['current']['state']:
            raise ValueError('input states differ from frozen published labels')
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    with sqlite3.connect(source.as_uri() + '?mode=ro', uri=True) as original, sqlite3.connect(target) as copied:
        original.backup(copied)
    clock = datetime.now(timezone.utc)
    with ForecastLedger(target) as ledger:
        entries = ledger.list_probability_forecasts()
        original_forecasts = [(e.key.as_sql_tuple(),e.forecast_sha256) for e in entries]
        original_evaluations = [(e.forecast_key.as_sql_tuple(),e.evaluation_sha256) for e in ledger.list_evaluations()]
        result = mature_probability_evaluations(ledger, states=states, evaluated_at=clock,
            label_spec_sha256=payload['label']['spec_sha256'])
        diagnostics = build_operational_diagnostics(entries, ledger.list_evaluations(),
            probability_evaluations=ledger.list_probability_evaluations(), as_of=clock)
        assert original_forecasts == [(e.key.as_sql_tuple(),e.forecast_sha256) for e in ledger.list_probability_forecasts()]
        assert original_evaluations == [(e.forecast_key.as_sql_tuple(),e.evaluation_sha256) for e in ledger.list_evaluations()]
    after = hashlib.sha256(source.read_bytes()).hexdigest()
    if before != after:
        raise ValueError('source ledger changed during isolated evaluation')
    write_json_atomic(output / 'operational-diagnostics.json', diagnostics)
    report = {'source_sha256':before,'source_unchanged':True,'source_label_spec_sha256':payload['label']['spec_sha256'],
        'source_payload_sha256':hashlib.sha256(payload_path.read_bytes()).hexdigest(),
        'appended_probability_evaluations':len(result['appended']), 'pending_count':result['pending_count'],
        'unresolved':result['unresolved'], 'probability_scores':diagnostics['probability_scores'],
        'issued_forecasts_unchanged':True, 'investment_evaluations_unchanged':True,
        'evaluation_track':'isolated_copy_of_actual_issued_ledger'}
    write_json_atomic(output / 'evaluation-run.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ledger',type=Path,required=True)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--payload',type=Path,default=ROOT/'publication/live/regime-results.json')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(evaluate_copy(args.ledger,args.inputs,args.payload,args.output),ensure_ascii=False,indent=2))


if __name__=='__main__': main()
