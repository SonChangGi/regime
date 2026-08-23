"""Fail-closed verification for the local v4 baseline used by v5 research."""

from __future__ import annotations

from collections.abc import Mapping
import csv
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any

from regime_lab.config import project_root


FROZEN_V4_BASELINE = MappingProxyType(
    {
        "result_version": "weekly-regime-result-v4",
        "label_version": "market-causal-3state-v1",
        "model_version": "weekly-nondl-structural-v4",
        "feature_set_version": "weekly-pit-structural-v4",
        "champion": "markov",
        "payload_sha256": (
            "e58eda3f5519e1c3c340c671e6c6c1c69279dae068f9c21f9bedfde22e03b96b"
        ),
        "artifacts_inventory_sha256": (
            "3b0ffe79dea816b2a47c22ecba7eebb9b8fa8f4e9e2bb4ccba30f982d69c7613"
        ),
        "captured_at": "2026-08-21",
        "profile": "standard",
        "generation_id": "20260813T190841.471317Z",
        "data_as_of": "2026-08-07T20:00:00+00:00",
        "payload_path": "publication/baselines/v4-20260821/regime-results.json",
        "artifacts_path": "artifacts/baselines/v4-20260821",
    }
)
FROZEN_V4_OOS_PREDICTIONS = MappingProxyType(
    {
        "path": "oos-predictions.csv",
        "row_count": 8_832,
        "sha256": (
            "8fe52c9952f5b86d475d50fb08c6b2d1ec1ddb2aa9a3c350f91a247c4067e9a0"
        ),
    }
)
FROZEN_V4_INVENTORY_FILE_COUNT = 23
_INVENTORY_LINE = re.compile(
    r"(?P<sha256>[0-9a-f]{64})  (?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)"
)


class FrozenV4BaselineError(RuntimeError):
    """The local baseline does not match the reviewed v4 generation."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FrozenV4BaselineError(f"frozen v4 {label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise FrozenV4BaselineError(f"frozen v4 {label} must be a JSON object")
    return value


def _inventory_entries(raw: bytes) -> dict[str, str]:
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise FrozenV4BaselineError("frozen v4 SHA256SUMS must be ASCII") from exc
    if not text.endswith("\n"):
        raise FrozenV4BaselineError("frozen v4 SHA256SUMS must end with a newline")
    entries: dict[str, str] = {}
    for line in text.splitlines():
        match = _INVENTORY_LINE.fullmatch(line)
        if match is None:
            raise FrozenV4BaselineError("frozen v4 SHA256SUMS has an invalid row")
        name = match.group("name")
        if name in entries or name == "SHA256SUMS":
            raise FrozenV4BaselineError("frozen v4 SHA256SUMS has duplicate entries")
        entries[name] = match.group("sha256")
    canonical = "".join(
        f"{entries[name]}  {name}\n" for name in sorted(entries)
    ).encode("ascii")
    if canonical != raw:
        raise FrozenV4BaselineError("frozen v4 SHA256SUMS is not canonical")
    return entries


def _require_metadata(
    payload: Mapping[str, Any],
    generation: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> None:
    meta = payload.get("meta")
    model = payload.get("model")
    if not isinstance(meta, Mapping) or not isinstance(model, Mapping):
        raise FrozenV4BaselineError("frozen v4 payload metadata is missing")
    expected_meta = {
        "result_version": contract["result_version"],
        "generation_id": contract["generation_id"],
        "data_as_of": contract["data_as_of"],
        "mode": "live",
    }
    for field, expected in expected_meta.items():
        if meta.get(field) != expected:
            raise FrozenV4BaselineError(
                f"frozen v4 payload meta.{field} does not match the contract"
            )
    expected_model = {
        "version": contract["model_version"],
        "label_version": contract["label_version"],
        "feature_set_version": contract["feature_set_version"],
        "champion": contract["champion"],
        "profile": contract["profile"],
    }
    for field, expected in expected_model.items():
        if model.get(field) != expected:
            raise FrozenV4BaselineError(
                f"frozen v4 payload model.{field} does not match the contract"
            )
    if generation != {"generation_id": contract["generation_id"]}:
        raise FrozenV4BaselineError(
            "frozen v4 build generation does not match the contract"
        )


def verify_frozen_v4_baseline(
    *,
    project_directory: Path | None = None,
) -> dict[str, Any]:
    """Verify every frozen byte before a v5 research build may start."""

    contract = FROZEN_V4_BASELINE
    root = (project_directory or project_root()).resolve()
    directory = root / str(contract["artifacts_path"])
    if directory.is_symlink() or not directory.is_dir():
        raise FrozenV4BaselineError(
            f"frozen v4 baseline directory is missing: {directory}"
        )

    inventory_path = directory / "SHA256SUMS"
    if inventory_path.is_symlink() or not inventory_path.is_file():
        raise FrozenV4BaselineError("frozen v4 SHA256SUMS is missing")
    inventory = inventory_path.read_bytes()
    inventory_sha256 = hashlib.sha256(inventory).hexdigest()
    if inventory_sha256 != contract["artifacts_inventory_sha256"]:
        raise FrozenV4BaselineError("frozen v4 SHA256SUMS hash does not match")
    entries = _inventory_entries(inventory)
    if len(entries) != FROZEN_V4_INVENTORY_FILE_COUNT:
        raise FrozenV4BaselineError("frozen v4 inventory file count does not match")

    actual_names = {path.name for path in directory.iterdir()}
    expected_names = {*entries, "SHA256SUMS"}
    if actual_names != expected_names:
        raise FrozenV4BaselineError("frozen v4 baseline file set does not match")
    for name, expected_sha256 in entries.items():
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise FrozenV4BaselineError(f"frozen v4 artifact is invalid: {name}")
        if _sha256(path) != expected_sha256:
            raise FrozenV4BaselineError(
                f"frozen v4 artifact hash does not match: {name}"
            )

    oos_path = directory / str(FROZEN_V4_OOS_PREDICTIONS["path"])
    if entries.get(oos_path.name) != FROZEN_V4_OOS_PREDICTIONS["sha256"]:
        raise FrozenV4BaselineError(
            "frozen v4 OOS predictions hash does not match the immutable record"
        )
    try:
        with oos_path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream)
            next(reader, None)
            row_count = sum(1 for _ in reader)
    except OSError as exc:
        raise FrozenV4BaselineError(
            "frozen v4 OOS predictions could not be read"
        ) from exc
    if row_count != FROZEN_V4_OOS_PREDICTIONS["row_count"]:
        raise FrozenV4BaselineError(
            "frozen v4 OOS predictions row count does not match the immutable record"
        )

    payload_path = directory / "regime-results.json"
    if entries.get(payload_path.name) != contract["payload_sha256"]:
        raise FrozenV4BaselineError("frozen v4 payload hash does not match")
    payload = _json_object(payload_path, label="payload")
    generation = _json_object(
        directory / "build-generation.json",
        label="build generation",
    )
    _require_metadata(payload, generation, contract)
    return dict(contract)


__all__ = [
    "FROZEN_V4_BASELINE",
    "FROZEN_V4_INVENTORY_FILE_COUNT",
    "FROZEN_V4_OOS_PREDICTIONS",
    "FrozenV4BaselineError",
    "verify_frozen_v4_baseline",
]
