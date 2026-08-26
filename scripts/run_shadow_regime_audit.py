#!/usr/bin/env python3
"""Execute derived-only real-data regime shadows."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

from regime_lab.config import load_config, project_root
from regime_lab.path_safety import confined_mutable_path
from regime_lab.shadow_audit import run_real_data_shadow_audit, write_shadow_audit


def _timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("--as-of must include a timezone")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/series.json"))
    parser.add_argument("--database", type=Path, default=Path("data/regime.sqlite3"))
    parser.add_argument("--as-of", type=_timestamp, required=True)
    parser.add_argument("--direct-jump-origins", type=int, default=10)
    parser.add_argument("--dynamic-factor-origins", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/research-audits/shadow-regime-audit.json"),
    )
    args = parser.parse_args()
    root = project_root()
    database = args.database if args.database.is_absolute() else root / args.database
    output = confined_mutable_path(
        args.output,
        project_directory=root,
        label="shadow audit output",
    )
    report = run_real_data_shadow_audit(
        load_config(args.config),
        database=database,
        as_of=args.as_of,
        direct_jump_origins=args.direct_jump_origins,
        dynamic_factor_origins=args.dynamic_factor_origins,
    )
    write_shadow_audit(output, report)
    print(
        json.dumps(
            {
                "output": str(output),
                "report_sha256": report["report_sha256"],
                "methods": {
                    key: value["status"] for key, value in report["methods"].items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
