"""Request and response schemas for the pension allocation prediction API."""

from __future__ import annotations

from math import isfinite
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


ASSET_SHARE_FIELDS = (
    "Cash_Share",
    "Corporate_Bonds_Share",
    "Guaranteed_Funds_Share",
    "Immovable_Property_Share",
    "Government_Securities_Share",
    "Offshore_Investments_Share",
    "Property_Unit_Trusts_Share",
    "Quoted_Equity_Share",
    "Unquoted_Equity_Share",
)

CHANGE_FIELDS = (
    "Change_Cash_Share",
    "Change_Corporate_Bonds_Share",
    "Change_Guaranteed_Funds_Share",
    "Change_Immovable_Property_Share",
    "Change_Government_Securities_Share",
    "Change_Offshore_Investments_Share",
    "Change_Property_Unit_Trusts_Share",
    "Change_Quoted_Equity_Share",
    "Change_Unquoted_Equity_Share",
)

DOMINANT_ASSET_CLASSES = (
    "Cash, Demand & Fixed Deposits",
    "Commercial Paper and Corporate Bonds",
    "Guaranteed Funds",
    "Immovable Property",
    "Kenya Government Securities",
    "Offshore Investments",
    "Property Unit Trusts",
    "Quoted Equity",
    "Unquoted Equity",
)


class PortfolioFeatures(BaseModel):
    """One scheme-year row containing exactly the 26 model predictors."""

    model_config = ConfigDict(extra="forbid")

    Cash_Share: float = Field(..., ge=0, le=1, description="Cash, demand and fixed deposits as a proportion of total investments.")
    Corporate_Bonds_Share: float = Field(..., ge=0, le=1)
    Guaranteed_Funds_Share: float = Field(..., ge=0, le=1)
    Immovable_Property_Share: float = Field(..., ge=0, le=1)
    Government_Securities_Share: float = Field(..., ge=0, le=1)
    Offshore_Investments_Share: float = Field(..., ge=0, le=1)
    Property_Unit_Trusts_Share: float = Field(..., ge=0, le=1)
    Quoted_Equity_Share: float = Field(..., ge=0, le=1)
    Unquoted_Equity_Share: float = Field(..., ge=0, le=1)

    Change_Cash_Share: float = Field(..., ge=-1, le=1)
    Change_Corporate_Bonds_Share: float = Field(..., ge=-1, le=1)
    Change_Guaranteed_Funds_Share: float = Field(..., ge=-1, le=1)
    Change_Immovable_Property_Share: float = Field(..., ge=-1, le=1)
    Change_Government_Securities_Share: float = Field(..., ge=-1, le=1)
    Change_Offshore_Investments_Share: float = Field(..., ge=-1, le=1)
    Change_Property_Unit_Trusts_Share: float = Field(..., ge=-1, le=1)
    Change_Quoted_Equity_Share: float = Field(..., ge=-1, le=1)
    Change_Unquoted_Equity_Share: float = Field(..., ge=-1, le=1)

    Portfolio_HHI: float = Field(..., ge=(1 / 9) - 1e-6, le=1)
    Active_Asset_Classes: int = Field(..., ge=1, le=9)
    HHI_Change: float = Field(..., ge=-1, le=1)
    Dominant_Class_Changed: int = Field(..., ge=0, le=1)
    Log_Total_Investments: float
    Log_Portfolio_Growth: float
    Year: int = Field(..., ge=2013, description="Reporting year of the portfolio features, not the future outcome year.")
    Dominant_Asset_Class: str

    @field_validator("Dominant_Asset_Class")
    @classmethod
    def validate_dominant_asset_class(cls, value: str) -> str:
        if value not in DOMINANT_ASSET_CLASSES:
            raise ValueError(
                "Dominant_Asset_Class must be one of the nine asset classes used by the model."
            )
        return value

    @field_validator("Log_Total_Investments", "Log_Portfolio_Growth", mode="before")
    @classmethod
    def validate_finite_numeric_values(cls, value):
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("Value must be numeric and finite.") from exc
        if not isfinite(numeric_value):
            raise ValueError("Value must be numeric and finite.")
        return numeric_value

    @model_validator(mode="after")
    def validate_portfolio_consistency(self):
        shares = [getattr(self, field) for field in ASSET_SHARE_FIELDS]
        changes = [getattr(self, field) for field in CHANGE_FIELDS]

        tolerance = 1e-6

        # Check 1: Current asset shares must sum to 100%.
        if abs(sum(shares) - 1.0) > 1e-4:
            raise ValueError(
                "The nine asset-class shares must sum to 1.0 (100%)."
            )

        # Check 2: HHI must match the current asset shares.
        calculated_hhi = sum(share ** 2 for share in shares)

        if abs(self.Portfolio_HHI - calculated_hhi) > tolerance:
            raise ValueError(
                "Portfolio_HHI must equal the sum of squared current asset shares."
            )

        # Check 3: Active asset classes must match positive shares.
        calculated_active_classes = sum(share > 0 for share in shares)

        if self.Active_Asset_Classes != calculated_active_classes:
            raise ValueError(
                "Active_Asset_Classes must equal the number of asset classes "
                "with positive investment shares."
            )

        # Check 4: Allocation changes must sum to approximately zero.
        if abs(sum(changes)) > 1e-4:
            raise ValueError(
                "The nine allocation changes must sum to approximately zero."
            )

        # Check 5: Reconstruct previous-year asset shares.
        previous_shares = [
            current - change
            for current, change in zip(shares, changes)
        ]

        if any(
            share < -1e-4 or share > 1.0 + 1e-4
            for share in previous_shares
        ):
            raise ValueError(
                "The allocation changes imply invalid previous-year asset shares."
            )

        if abs(sum(previous_shares) - 1.0) > 1e-4:
            raise ValueError(
                "Previous-year asset shares must sum to 100%."
            )

        # Check 6: HHI change must match the reconstructed previous portfolio.
        previous_hhi = sum(share ** 2 for share in previous_shares)
        expected_hhi_change = calculated_hhi - previous_hhi

        if abs(self.HHI_Change - expected_hhi_change) > tolerance:
            raise ValueError(
                "HHI_Change must equal current HHI minus previous-year HHI."
            )

        # Check 7: Verify the current dominant asset class.
        current_dominant_index = max(
            range(len(shares)),
            key=shares.__getitem__
        )

        expected_current_dominant = DOMINANT_ASSET_CLASSES[
            current_dominant_index
        ]

        if self.Dominant_Asset_Class != expected_current_dominant:
            raise ValueError(
                "Dominant_Asset_Class does not match the largest current share."
            )

        # Check 8: Verify whether the dominant class changed.
        previous_dominant_index = max(
            range(len(previous_shares)),
            key=previous_shares.__getitem__
        )

        expected_previous_dominant = DOMINANT_ASSET_CLASSES[
            previous_dominant_index
        ]

        expected_dominant_changed = int(
            expected_current_dominant != expected_previous_dominant
        )

        if self.Dominant_Class_Changed != expected_dominant_changed:
            raise ValueError(
                "Dominant_Class_Changed does not match the previous-year comparison."
            )

        return self


class FeatureContribution(BaseModel):
    feature: str
    shap_value: float
    effect: Literal["pushes_toward_material_change", "pushes_toward_no_material_change", "approximately_neutral"]


class PredictionResponse(BaseModel):
    selected_model: str
    predicted_probability_material_change: float = Field(..., ge=0, le=1)
    predicted_material_change: int = Field(..., ge=0, le=1)
    decision_threshold: float = Field(..., ge=0, le=1)
    target_definition: str
    interpretation: str
    top_contributing_factors: list[FeatureContribution]
    explanation_note: str
