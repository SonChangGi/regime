#!/usr/bin/env python3
"""Print non-destructive backup/checkpoint/preview recovery evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from regime_lab.recovery_inventory import (
    RecoveryInventoryError,
    build_recovery_inventory,
    run_restore_drill,
)
from regime_lab.recovery_policy import RecoveryPolicyError, load_recovery_policy


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup-directory", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--preview", type=Path, action="append", default=[])
    parser.add_argument("--policy", type=Path, default=None)
    parser.add_argument(
        "--restore-drill",
        action="store_true",
        help="also restore and validate the newest valid-current generation",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        policy = load_recovery_policy(args.policy)
        document = {
            "inventory": build_recovery_inventory(
                args.backup_directory,
                checkpoint_paths=args.checkpoint,
                preview_paths=args.preview,
                policy=policy,
            )
        }
        if args.restore_drill:
            document["restore_drill"] = run_restore_drill(
                args.backup_directory,
                policy=policy,
            )
    except (RecoveryInventoryError, RecoveryPolicyError, OSError) as exc:
        print(f"recovery inventory failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
