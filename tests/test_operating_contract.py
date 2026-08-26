from __future__ import annotations

import json
from pathlib import Path

import pytest

from regime_lab.operating_contract import (
    OperatingContractError,
    load_operating_contract,
)


def test_active_operating_contract_is_hash_bound_and_complete() -> None:
    contract = load_operating_contract()
    assert contract.state_order == ("risk_on", "transition", "risk_off")
    assert tuple(row["id"] for row in contract.state_definitions) == contract.state_order
    assert contract.document["forecast"]["official_gap_weeks"] == 1
    assert contract.document["label"]["membership_semantics"] == (
        "distance_to_threshold_anchors_not_posterior"
    )
    assert contract.document["ablation_tracks"] == [
        "state_only",
        "label_mechanics",
        "market_ex_label_components",
        "macro_rates_credit",
        "full",
    ]
    assert len(contract.sha256) == len(contract.selection_policy_sha256) == 64


def test_operating_contract_rejects_reviewed_nonoperating_lifecycle(
    tmp_path: Path,
) -> None:
    source = load_operating_contract()
    value = json.loads(source.path.read_text(encoding="utf-8"))
    value["lifecycle"]["allowed_combinations"].append(
        ["selected_by_gate", "candidate", "reviewed_publication"]
    )
    path = tmp_path / "operating-contract.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(OperatingContractError, match="only be operating"):
        load_operating_contract(path)


def test_operating_contract_rejects_preregistration_hash_drift(tmp_path: Path) -> None:
    source = load_operating_contract()
    value = json.loads(source.path.read_text(encoding="utf-8"))
    value["preregistration"]["sha256"] = "0" * 64
    path = tmp_path / "operating-contract.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(OperatingContractError, match="preregistration hash mismatch"):
        load_operating_contract(path)
