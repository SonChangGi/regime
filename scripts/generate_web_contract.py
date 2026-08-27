#!/usr/bin/env python3
"""Generate or verify the browser-facing operating contract."""

from __future__ import annotations

import argparse
from pathlib import Path

from regime_lab.config import project_root
from regime_lab.web_contract import (
    GENERATED_BROWSER_CONTRACT_PATH,
    render_browser_contract_javascript,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root() / GENERATED_BROWSER_CONTRACT_PATH,
    )
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    expected = render_browser_contract_javascript()
    if args.check:
        if args.output.is_symlink() or not args.output.is_file():
            raise SystemExit(f"generated browser contract is missing: {args.output}")
        if args.output.read_bytes() != expected:
            raise SystemExit("generated browser contract is stale")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(expected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
