from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from regime_lab.integrity import canonical_json_sha256_v1
from regime_lab.shadow_audit import (
    SHADOW_AUDIT_SCHEMA_VERSION,
    build_shadow_audit_from_inputs,
    write_shadow_audit,
)


UTC = timezone.utc


def test_shadow_audit_executes_each_method_without_promotion(tmp_path) -> None:
    index = pd.date_range("2010-01-01", periods=600, freq="W-FRI", tz="UTC")
    position = np.arange(len(index), dtype=float)
    features = pd.DataFrame(
        {
            "market": np.sin(position / 11.0) + position / 500.0,
            "breadth": np.cos(position / 17.0),
            "credit": np.sin(position / 23.0),
            "macro": np.cos(position / 31.0),
        },
        index=index,
    )
    cycle = ["risk_on"] * 13 + ["transition"] * 4 + ["risk_off"] * 9
    states = pd.Series(
        [cycle[item % len(cycle)] for item in range(len(index))],
        index=index,
        name="state",
    )
    memberships = pd.DataFrame(0.02, index=index, columns=tuple(
        ("risk_on", "transition", "risk_off")
    ))
    for at, state in states.items():
        memberships.loc[at, state] = 0.96
    risk_score = pd.Series(np.sin(position / 9.0), index=index)
    months = pd.date_range("2010-01-31", periods=132, freq="ME")
    month_position = np.arange(len(months), dtype=float)
    monthly_raw_price = pd.Series(
        100.0 + month_position * 0.2 + 20.0 * np.sin(month_position / 4.0),
        index=months,
    )

    report = build_shadow_audit_from_inputs(
        features=features,
        states=states,
        memberships=memberships,
        risk_score=risk_score,
        monthly_raw_price=monthly_raw_price,
        data_as_of=datetime(2021, 6, 4, 20, tzinfo=UTC),
        input_metadata={"feature_count": 4, "weekly_row_count": 600},
        source_fingerprint_sha256="a" * 64,
        minimum_train_weeks=100,
        direct_jump_origins=2,
        dynamic_factor_origins=2,
    )

    assert report["schema_version"] == SHADOW_AUDIT_SCHEMA_VERSION
    assert report["evidence_track"] == "reconstructed_oos"
    assert report["automatic_promotion_eligible"] is False
    assert report["canonical_target"] is False
    assert set(report["methods"]) == {
        "filtered_hsmm",
        "bayesian_online_changepoint",
        "direct_jump_tvtp_hurdle",
        "dynamic_factor_tvtp",
        "pagan_sossounov",
    }
    assert all(
        method["status"].startswith("executed_")
        for method in report["methods"].values()
    )
    assert report["methods"]["direct_jump_tvtp_hurdle"]["origin_count"] == 2
    assert report["methods"]["dynamic_factor_tvtp"]["origin_count"] == 2
    assert report["methods"]["pagan_sossounov"]["uses_future_observations"] is True
    body = dict(report)
    expected_hash = body.pop("report_sha256")
    assert canonical_json_sha256_v1(body) == expected_hash

    output = write_shadow_audit(tmp_path / "shadow.json", report)
    assert output.is_file()
    assert output.read_text(encoding="utf-8").endswith("\n")
