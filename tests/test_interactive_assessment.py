import math
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT))

from dashboard.assessment import build_assessment_features  # noqa: E402
from pension_capstone.config import (  # noqa: E402
    ASSET_COLUMNS,
    CLASSIFICATION_FEATURES,
    SHARE_MAPPING,
)


def _previous_record() -> pd.Series:
    amounts = {
        "Cash, Demand & Fixed Deposits": 10.0,
        "Commercial Paper and Corporate Bonds": 0.0,
        "Guaranteed Funds": 60.0,
        "Immovable Property": 0.0,
        "Kenya Government Securities": 30.0,
        "Offshore Investments": 0.0,
        "Property Unit Trusts": 0.0,
        "Quoted Equity": 0.0,
        "Unquoted Equity": 0.0,
    }
    total = sum(amounts.values())
    shares = {share: amounts[asset] / total for asset, share in SHARE_MAPPING.items()}
    dominant = max(SHARE_MAPPING, key=lambda asset: shares[SHARE_MAPPING[asset]])
    row = {
        **amounts,
        "Ref": 123,
        "Year": 2024,
        "Total_Investments": total,
        "Active_Asset_Classes": sum(value > 0 for value in amounts.values()),
        "Dominant_Asset_Class": dominant,
        "Portfolio_HHI": sum(value * value for value in shares.values()),
        **shares,
    }
    return pd.Series(row)


def test_builds_exact_26_predictors_and_current_historical_comparison():
    previous = _previous_record()
    current = {asset: 0.0 for asset in ASSET_COLUMNS}
    current["Guaranteed Funds"] = 70.0
    current["Kenya Government Securities"] = 30.0

    features, comparison = build_assessment_features(previous, current, 2025)

    assert list(features) == list(CLASSIFICATION_FEATURES)
    assert len(features) == 26
    assert math.isclose(sum(features[name] for name in SHARE_MAPPING.values()), 1.0)
    assert features["Active_Asset_Classes"] == 2
    assert features["Dominant_Asset_Class"] == "Guaranteed Funds"
    assert features["Year"] == 2025
    assert comparison["previous_year"] == 2024
    assert comparison["current_year"] == 2025
    assert comparison["current_active_classes"] == 2
    assert math.isclose(comparison["current_total"], 100.0)


def test_requires_consecutive_previous_year():
    previous = _previous_record()
    previous["Year"] = 2023
    current = {asset: 0.0 for asset in ASSET_COLUMNS}
    current["Guaranteed Funds"] = 100.0

    with pytest.raises(ValueError, match="consecutive previous-year record"):
        build_assessment_features(previous, current, 2025)


def test_rejects_zero_portfolio():
    previous = _previous_record()
    current = {asset: 0.0 for asset in ASSET_COLUMNS}

    with pytest.raises(ValueError, match="positive investment amount"):
        build_assessment_features(previous, current, 2025)
