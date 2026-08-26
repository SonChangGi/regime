"""Typed, hash-bound source of truth for the active Regime operating policy."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from regime_lab.config import project_root
from regime_lab.integrity import IntegrityError, canonical_json_sha256_v1


OPERATING_CONTRACT_SCHEMA = "regime-operating-contract/1"


class OperatingContractError(ValueError):
    """The active operating contract is missing, ambiguous, or inconsistent."""


def canonical_sha256(value: object) -> str:
    try:
        return canonical_json_sha256_v1(value)
    except IntegrityError as exc:
        raise OperatingContractError("operating contract is not canonical JSON") from exc


@dataclass(frozen=True)
class OperatingContract:
    document: Mapping[str, Any]
    sha256: str
    path: Path

    @property
    def state_order(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.document["state_order"])

    @property
    def state_definitions(self) -> tuple[dict[str, str], ...]:
        metadata = self.document["state_meta"]
        return tuple(
            {
                "id": state,
                "label": str(metadata[state]["label"]),
                "label_ko": str(metadata[state]["label_ko"]),
                "description": str(metadata[state]["description"]),
                "color": str(metadata[state]["color"]),
                "symbol": str(metadata[state]["symbol"]),
            }
            for state in self.state_order
        )

    @property
    def selection_policy(self) -> Mapping[str, Any]:
        return self.document["selection_policy"]

    @property
    def selection_policy_sha256(self) -> str:
        return canonical_sha256(self.selection_policy)

    @property
    def complexity_registry_sha256(self) -> str:
        return canonical_sha256(self.selection_policy["complexity_registry"])

    @property
    def weekly_base_models(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.document["models"]["weekly_base_models"])

    def historical_reviewed_roster(self, roster_id: str) -> Mapping[str, Any] | None:
        rosters = self.document["models"].get("historical_reviewed_rosters", {})
        value = rosters.get(str(roster_id))
        return value if isinstance(value, Mapping) else None

    def historical_reviewed_roster_by_manifest_sha256(
        self,
        manifest_sha256: str,
    ) -> Mapping[str, Any] | None:
        rosters = self.document["models"].get("historical_reviewed_rosters", {})
        for value in rosters.values():
            if (
                isinstance(value, Mapping)
                and value.get("candidate_manifest_sha256") == manifest_sha256
            ):
                return value
        return None


def default_operating_contract_path() -> Path:
    return project_root() / "config" / "operating-contract.json"


def _require_mapping(value: object, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OperatingContractError(f"{context} must be an object")
    return value


def _validate_document(document: Mapping[str, Any], *, root: Path) -> None:
    required = {
        "schema_version",
        "contract_version",
        "purpose",
        "state_order",
        "state_meta",
        "preregistration",
        "label",
        "forecast",
        "models",
        "selection_policy",
        "lifecycle",
        "ablation_tracks",
    }
    if set(document) != required:
        raise OperatingContractError("operating contract top-level fields are invalid")
    if document["schema_version"] != OPERATING_CONTRACT_SCHEMA:
        raise OperatingContractError("unsupported operating contract schema")
    state_order = tuple(document["state_order"])
    if state_order != ("risk_on", "transition", "risk_off"):
        raise OperatingContractError("operating state order must remain the canonical three states")
    state_meta = _require_mapping(document["state_meta"], context="state_meta")
    if set(state_meta) != set(state_order):
        raise OperatingContractError("state_meta must exactly match state_order")
    for state in state_order:
        if set(_require_mapping(state_meta[state], context=f"state_meta.{state}")) != {
            "label", "label_ko", "description", "color", "symbol"
        }:
            raise OperatingContractError(f"state_meta.{state} fields are invalid")

    forecast = _require_mapping(document["forecast"], context="forecast")
    if forecast.get("horizon_weeks") != 1 or forecast.get("official_gap_weeks") != 1:
        raise OperatingContractError("official forecast horizon and gap must both remain one week")
    if forecast.get("transition_horizons_weeks") != [1, 4, 13]:
        raise OperatingContractError("transition horizons must remain [1, 4, 13]")
    if forecast.get("evidence_tracks") != ["operational_oos", "reconstructed_oos"]:
        raise OperatingContractError("forecast evidence tracks are invalid")

    prereg = _require_mapping(document["preregistration"], context="preregistration")
    prereg_path = root / str(prereg.get("path", ""))
    if prereg.get("immutable") is not True or not prereg_path.is_file():
        raise OperatingContractError("immutable preregistration is unavailable")
    actual_prereg = hashlib.sha256(prereg_path.read_bytes()).hexdigest()
    if prereg.get("sha256") != actual_prereg:
        raise OperatingContractError("immutable preregistration hash mismatch")

    models = _require_mapping(document["models"], context="models")
    model_groups = (
        tuple(models.get("weekly_base_models", ())),
        tuple(models.get("core_candidates", ())),
        tuple(models.get("shadow_models", ())),
        tuple(models.get("frozen_reproduction_only", ())),
        tuple(models.get("ensemble_components", ())),
    )
    if any(not group or len(group) != len(set(group)) for group in model_groups):
        raise OperatingContractError("model groups must be non-empty and duplicate-free")
    if set(model_groups[0]).intersection(model_groups[3]):
        raise OperatingContractError("weekly and frozen-reproduction model groups overlap")
    historical = _require_mapping(
        models.get("historical_reviewed_rosters"),
        context="models.historical_reviewed_rosters",
    )
    allowed_historical_models = set().union(*map(set, model_groups))
    for roster_id, raw_roster in historical.items():
        roster = _require_mapping(
            raw_roster,
            context=f"models.historical_reviewed_rosters.{roster_id}",
        )
        if set(roster) != {
            "role",
            "candidate_manifest_sha256",
            "candidate_models",
            "forecast_comparison_models",
        }:
            raise OperatingContractError("historical reviewed roster fields are invalid")
        candidates = tuple(roster["candidate_models"])
        comparisons = tuple(roster["forecast_comparison_models"])
        manifest_sha256 = str(roster["candidate_manifest_sha256"])
        if (
            len(manifest_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in manifest_sha256
            )
            or not candidates
            or len(candidates) != len(set(candidates))
            or not comparisons
            or len(comparisons) != len(set(comparisons))
            or not set(comparisons).issubset(candidates)
            or not set(candidates).issubset(allowed_historical_models)
        ):
            raise OperatingContractError("historical reviewed roster is invalid")

    policy = _require_mapping(document["selection_policy"], context="selection_policy")
    if policy.get("simplicity_tolerance") != 0.01:
        raise OperatingContractError("simplicity_tolerance must be explicit and equal to 0.01")
    if policy.get("tie_break_order") != [
        "complexity_rank", "calibration_error", "log_loss", "model"
    ]:
        raise OperatingContractError("selection tie-break order is invalid")
    registry = _require_mapping(policy.get("complexity_registry"), context="complexity_registry")
    if any(type(value) is not int or value < 0 for value in registry.values()):
        raise OperatingContractError("complexity ranks must be non-negative integers")

    lifecycle = _require_mapping(document["lifecycle"], context="lifecycle")
    allowed = {tuple(row) for row in lifecycle.get("allowed_combinations", ())}
    if ("selected_by_gate", "operating", "reviewed_publication") not in allowed:
        raise OperatingContractError("operating reviewed lifecycle combination is unavailable")
    if any(
        publication == "reviewed_publication" and deployment != "operating"
        for _selection, deployment, publication in allowed
    ):
        raise OperatingContractError("reviewed publication may only be operating")

    if document["ablation_tracks"] != [
        "state_only",
        "label_mechanics",
        "market_ex_label_components",
        "macro_rates_credit",
        "full",
    ]:
        raise OperatingContractError("ablation tracks are invalid")


def load_operating_contract(path: str | Path | None = None) -> OperatingContract:
    selected = Path(path) if path is not None else default_operating_contract_path()
    try:
        raw = selected.read_bytes()
        document = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise OperatingContractError(f"operating contract is unavailable: {selected}") from exc
    if not isinstance(document, Mapping):
        raise OperatingContractError("operating contract must be a JSON object")
    _validate_document(document, root=project_root())
    return OperatingContract(document=dict(document), sha256=hashlib.sha256(raw).hexdigest(), path=selected)


__all__ = [
    "OPERATING_CONTRACT_SCHEMA",
    "OperatingContract",
    "OperatingContractError",
    "canonical_sha256",
    "default_operating_contract_path",
    "load_operating_contract",
]
