from __future__ import annotations

import json

import pytest

from regime_lab.config import ConfigurationError, default_config_path, load_config


def test_default_config_keeps_one_week_destination_and_exact_transition_horizons() -> None:
    config = load_config()

    assert config["model"]["horizon_weeks"] == 1
    assert config["model"]["transition_horizons_weeks"] == [1, 4, 13]
    assert config["model"]["state_order"] == [
        "risk_on",
        "transition",
        "risk_off",
    ]


def test_default_structural_data_contract_preserves_alpha_reserve_and_groups() -> None:
    config = load_config()
    alpha = config["alpha_vantage"]
    symbols = alpha["symbols"]
    groups = alpha["symbol_groups"]

    assert len(symbols) == 23
    assert alpha["daily_request_cap"] - len(symbols) == 2
    assert groups == {
        "gics_sector": [
            "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE",
            "XLU", "XLV", "XLY",
        ],
        "broad_size_style": ["SPY", "QQQ", "IWM", "DIA", "RSP"],
        "cross_asset": ["SHY", "IEF", "TLT", "HYG", "LQD", "GLD", "UUP"],
    }
    flattened = [symbol for members in groups.values() for symbol in members]
    assert len(flattened) == len(set(flattened)) == len(symbols)
    assert set(flattened) == set(symbols)


def test_default_structural_alfred_and_feature_contract_is_complete() -> None:
    config = load_config()
    alfred_ids = {item["id"] for item in config["alfred"]["series"]}

    assert {
        "DGS1", "DGS5", "DGS7", "DGS20", "DGS30", "ANFCI", "TOTBKCR",
        "TOTCI", "DPSACBW027SBOG", "H8B3094NCBA",
    }.issubset(alfred_ids)
    engineering = config["feature_engineering"]
    curve = engineering["nelson_siegel"]
    assert curve["lambda_per_month"] == 0.0609
    assert curve["minimum_maturities"] == 4
    assert set(curve["series_months"]) == {
        "DGS3MO", "DGS1", "DGS2", "DGS5", "DGS7", "DGS10", "DGS20", "DGS30",
    }
    assert engineering["release_innovation"] == {
        "series": [
            "UNRATE", "PAYEMS", "INDPRO", "RSAFS", "HOUST", "CPIAUCSL",
            "PCEPI", "GDPC1",
        ],
        "prior_release_window": 12,
        "minimum_prior_releases": 4,
    }
    assert engineering["financial_conditions"] == {
        "series": "ANFCI",
        "features": ["level", "change_1w", "change_4w", "z_52w"],
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("horizon_weeks", 4),
        ("transition_horizons_weeks", [1, 13]),
        ("state_order", ["risk_on", "risk_off", "transition"]),
    ],
)
def test_config_rejects_drift_from_fixed_three_state_horizon_contract(
    tmp_path, field: str, value: object
) -> None:
    config = json.loads(default_config_path().read_text(encoding="utf-8"))
    config["model"][field] = value
    path = tmp_path / "series.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ConfigurationError):
        load_config(path)
