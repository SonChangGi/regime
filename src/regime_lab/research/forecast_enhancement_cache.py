"""Exact numerical recipe bindings, including local imports and runtime versions."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess


MODEL_SOURCE = "src/regime_lab/research/forecast_enhancement_models.py"
MODEL_CONFIG = ("config/label-spec.json", "config/operating-contract.json")
# Numerical call graph: frozen score preprocessing, boundary features/state step,
# Markov fit, encoded XGBoost, and configuration loaded by those implementations.
# Broader imports inside unrelated research/evaluation functions are tracked in
# artifact provenance but do not invalidate these numerical fits.
MODEL_NUMERICAL_SOURCES = (MODEL_SOURCE, *(
    "src/regime_lab/analysis/" + name for name in
    ("labels.py", "label_spec.py", "boundary_forecast.py", "forecast_paths.py", "models.py")
), *("src/regime_lab/" + name for name in
      ("schema.py", "operating_contract.py", "config.py", "integrity.py")))


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local_source_hashes(root: Path, entries: list[str]) -> dict[str, str]:
    """Follow explicit local imports, including imports inside fit functions."""
    pending = [root / entry for entry in entries]
    found = {}
    while pending:
        path = pending.pop()
        relative = str(path.relative_to(root))
        if relative in found:
            continue
        found[relative] = file_hash(path)
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.level:
                package = list(path.relative_to(root / "src").parts[:-1])
                base = package[:len(package) - node.level + 1]
                names = [".".join([*base, node.module])] if node.module else [".".join([*base, alias.name]) for alias in node.names]
            else:
                names = ([node.module] if isinstance(node, ast.ImportFrom)
                         else [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
            for name in names:
                if name and name.startswith("regime_lab"):
                    dependency = root / "src" / (name.replace(".", "/") + ".py")
                    if dependency.is_file():
                        pending.append(dependency)
    return dict(sorted(found.items()))


def model_cache_key(root: Path, inputs: dict, protocol: dict, runtime: dict) -> dict:
    sources = {name: file_hash(root / name) for name in MODEL_NUMERICAL_SOURCES}
    sources.update({name: file_hash(root / name) for name in MODEL_CONFIG})
    return {"schema_version": "forecast-model-cache/2", "inputs": inputs,
            "protocol": protocol, "runtime": runtime, "source_sha256": dict(sorted(sources.items()))}


def cache_matches(path: Path, expected: dict) -> bool:
    """A cold cache requests fitting; a changed recipe must never load old fits."""
    if not path.exists():
        return False
    if json.loads(path.read_text()) != expected:
        raise ValueError("cached model recipe/source/runtime changed")
    return True


def legacy_binding_evidence(root: Path, old: dict, new: dict, previous_runtime: dict) -> dict:
    """Explicit one-time metadata expansion; never pretend old hashes existed.

    The caller must authorize this migration and verify the saved artifact hashes.
    Existing recorded fields must match exactly; added tracked dependencies must
    match Git HEAD. Newly recorded runtime fields remain explicitly identified.
    """
    legacy = {"inputs": new["inputs"], "code": new["source_sha256"][MODEL_SOURCE],
              "protocol": new["protocol"]}
    if old != legacy:
        raise ValueError("legacy numerical inputs/model/protocol changed")
    if any(new["runtime"].get(name) != value for name, value in previous_runtime.items()):
        raise ValueError("recorded legacy runtime changed")
    checked = {}
    for relative, digest in new["source_sha256"].items():
        if relative == MODEL_SOURCE:
            continue
        content = subprocess.run(["git", "show", f"HEAD:{relative}"], cwd=root,
                                 capture_output=True, check=True).stdout
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError(f"cannot extend legacy binding: dependency differs from HEAD: {relative}")
        checked[relative] = digest
    return {"operation": "explicit_legacy_metadata_expansion_without_numerical_refit",
            "legacy_inputs_model_protocol_unchanged": True,
            "previously_recorded_runtime_unchanged": previous_runtime,
            "new_runtime_fields_recorded_at_migration": {k: v for k, v in new["runtime"].items() if k not in previous_runtime},
            "added_sources_match_head_at_migration": checked,
            "evidence_limit": "Added dependency hashes were not recorded by the original fit; they are bound now after checking HEAD. Cached numerical file hashes are preserved."}
