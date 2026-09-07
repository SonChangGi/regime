#!/usr/bin/env python3
"""Replay past-only calibration from archived derived forecasts into a local preview.

The base models and source/issued artifacts are never changed. Calibration
coefficients are reconstructed from prior selection blocks; diagnostics do not
choose calibration. Output is one atomically replaced JSON report.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from regime_lab.analysis.causal_calibration import (
    TransitionCalibrator, TRANSITION_CALIBRATION_VERSION,
    TRANSITION_CALIBRATION_BLOCK_WEEKS, TRANSITION_CALIBRATION_MAX_BLOCKS,
    TRANSITION_CALIBRATION_MIN_TRAIN_ROWS, TRANSITION_CALIBRATION_SHRINK_WEIGHT,
)
from regime_lab.v5 import _anchored_isotonic_transition_risk


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_output_path(output: Path, sources: list[Path]) -> None:
    target = output.resolve()
    live = (ROOT / 'publication/live').resolve()
    if target == live or live in target.parents:
        raise ValueError('calibration preview output cannot modify publication/live')
    for source in sources:
        if target == source.resolve() or (output.exists() and os.path.samefile(output, source)):
            raise ValueError('calibration preview output cannot replace an archived source')


def write_atomic_report(report: dict, output: Path) -> None:
    # Serialize before touching the destination. Invalid/nonfinite results can
    # never replace the last valid report; unique temp names support retries.
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    json.loads(serialized)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=output.parent,
                                         prefix=f'.{output.name}.', suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build_calibration_audit(*, source: Path, oos: Path, prospective: Path,
                            selection_end: str | None = None) -> dict:
    from regime_lab.research.forecast_calibration import build_calibration_audit_from_frames
    paths = [oos.resolve(), prospective.resolve(), source.resolve()]
    source_bytes = {p: p.read_bytes() for p in paths}
    data, future = (pd.read_csv(io.BytesIO(source_bytes[p])) for p in paths[:2])
    payload = json.loads(source_bytes[paths[2]])
    report = build_calibration_audit_from_frames(payload, data, future,
        selection_end=selection_end, sources=[
            {'path': str(p), 'sha256': hashlib.sha256(content).hexdigest()}
            for p, content in source_bytes.items()])
    require(all(p.read_bytes() == content for p, content in source_bytes.items()),
            'archived input changed during replay')
    return report


def main() -> None:
    artifacts = ROOT / 'build/weekly-automation/generation-v5/artifacts'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'publication/live/regime-results.json',
                        help='Read-only archived payload providing issued probabilities and the one-week anchor.')
    parser.add_argument('--oos', type=Path, default=artifacts / 'transition-oos-predictions.csv',
                        help='Derived resolved transition forecasts including raw probabilities and targets.')
    parser.add_argument('--prospective', type=Path, default=artifacts / 'transition-candidate-forecasts.csv',
                        help='Derived unresolved forecasts for every candidate and horizon.')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'build/forecast-audit-improvements/calibration/calibration-audit.json',
                        help='Destination JSON file; must not replace any source or live publication.')
    parser.add_argument('--selection-end', help='Frozen exclusive cutoff, defaulting to the source contract.')
    args = parser.parse_args()
    validate_output_path(args.output, [args.source, args.oos, args.prospective])
    report = build_calibration_audit(source=args.source, oos=args.oos, prospective=args.prospective,
                                     selection_end=args.selection_end)
    validate_output_path(args.output, [args.source, args.oos, args.prospective])
    write_atomic_report(report, args.output)
    print(json.dumps({'output': str(args.output), 'data_as_of': report['data_as_of'],
                      'rows': len(report['rows']), 'latest_rows': len(report['latest_rows']),
                      'verification': report['verification']}, indent=2))


if __name__ == '__main__':
    main()
