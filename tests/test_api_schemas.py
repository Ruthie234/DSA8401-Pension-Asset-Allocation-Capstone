import pytest
from pydantic import ValidationError

from api.schemas import PortfolioFeatures


def valid_portfolio_payload():
    """Return a portfolio with mathematically consistent predictors."""

    return {
        "Cash_Share": 0.0,
        "Corporate_Bonds_Share": 0.0,
        "Guaranteed_Funds_Share": 0.25,
        "Immovable_Property_Share": 0.0,
        "Government_Securities_Share": 0.75,
        "Offshore_Investments_Share": 0.0,
        "Property_Unit_Trusts_Share": 0.0,
        "Quoted_Equity_Share": 0.0,
        "Unquoted_Equity_Share": 0.0,

        "Change_Cash_Share": 0.0,
        "Change_Corporate_Bonds_Share": 0.0,
        "Change_Guaranteed_Funds_Share": -0.25,
        "Change_Immovable_Property_Share": 0.0,
        "Change_Government_Securities_Share": 0.25,
        "Change_Offshore_Investments_Share": 0.0,
        "Change_Property_Unit_Trusts_Share": 0.0,
        "Change_Quoted_Equity_Share": 0.0,
        "Change_Unquoted_Equity_Share": 0.0,

        "Portfolio_HHI": 0.625,
        "Active_Asset_Classes": 2,
        "HHI_Change": 0.125,
        "Dominant_Class_Changed": 1,
        "Log_Total_Investments": 17.0,
        "Log_Portfolio_Growth": 0.2,
        "Year": 2025,
        "Dominant_Asset_Class": "Kenya Government Securities",
    }


def test_accepts_consistent_portfolio():
    portfolio = PortfolioFeatures(**valid_portfolio_payload())

    assert portfolio.Portfolio_HHI == 0.625
    assert portfolio.Active_Asset_Classes == 2


def test_rejects_incorrect_hhi():
    payload = valid_portfolio_payload()
    payload["Portfolio_HHI"] = 0.60

    with pytest.raises(ValidationError, match="Portfolio_HHI"):
        PortfolioFeatures(**payload)


def test_rejects_incorrect_active_asset_count():
    payload = valid_portfolio_payload()
    payload["Active_Asset_Classes"] = 3

    with pytest.raises(ValidationError, match="Active_Asset_Classes"):
        PortfolioFeatures(**payload)


def test_rejects_incorrect_hhi_change():
    payload = valid_portfolio_payload()
    payload["HHI_Change"] = 0.20

    with pytest.raises(ValidationError, match="HHI_Change"):
        PortfolioFeatures(**payload)


def test_rejects_incorrect_dominant_class_change():
    payload = valid_portfolio_payload()
    payload["Dominant_Class_Changed"] = 0

    with pytest.raises(ValidationError, match="Dominant_Class_Changed"):
        PortfolioFeatures(**payload)