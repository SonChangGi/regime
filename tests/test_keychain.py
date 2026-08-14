from __future__ import annotations

import os
from unittest.mock import patch

from regime_lab.keychain import provider_environment_from_keychain


def test_keychain_environment_is_restored() -> None:
    os.environ.pop("ALPHA_VANTAGE_API_KEY", None)
    os.environ["FRED_API_KEY"] = "old"
    with patch(
        "regime_lab.keychain.load_provider_secrets",
        return_value={"FRED_API_KEY": "secret-a", "ALPHA_VANTAGE_API_KEY": "secret-b"},
    ):
        with provider_environment_from_keychain(rights_acknowledged=True):
            assert os.environ["FRED_API_KEY"] == "secret-a"
            assert os.environ["ALPHA_VANTAGE_API_KEY"] == "secret-b"
            assert os.environ["ALFRED_ML_RIGHTS_ACK"] == "1"
    assert os.environ["FRED_API_KEY"] == "old"
    assert "ALPHA_VANTAGE_API_KEY" not in os.environ
    assert "ALFRED_ML_RIGHTS_ACK" not in os.environ
