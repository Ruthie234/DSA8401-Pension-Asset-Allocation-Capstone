"""Shared project configuration for the pension asset-allocation capstone."""

from __future__ import annotations

RANDOM_STATE = 42
MATERIAL_CHANGE_THRESHOLD = 0.05
CALIBRATED_DECISION_THRESHOLD = 0.50
TEMPORAL_CV_YEARS = [2020, 2021, 2022]
CALIBRATION_OUTCOME_YEAR = 2023
TEST_OUTCOME_YEARS = [2024, 2025]
SENSITIVITY_THRESHOLDS = [0.03, 0.04, 0.05, 0.10, 0.15, 0.20]

DATA_FILENAME = "COPY 1 FINAL.csv"

ASSET_INCOME_PAIRS = {
    "Cash, Demand & Fixed Deposits": "Cash, Demand & Fixed Deposits - Income",
    "Commercial Paper and Corporate Bonds": "Commercial Paper and Corporate Bonds - Income",
    "Guaranteed Funds": "Guaranteed Funds - Income",
    "Immovable Property": "Immovable Property - Income",
    "Kenya Government Securities": "Kenya Government Securities - Income",
    "Offshore Investments": "Offshore Investments - Income",
    "Property Unit Trusts": "Property Unit Trusts - Income",
    "Quoted Equity": "Quoted Equity - Income",
    "Unquoted Equity": "Unquoted Equity - Income",
}

ASSET_COLUMNS = list(ASSET_INCOME_PAIRS.keys())
INCOME_COLUMNS = list(ASSET_INCOME_PAIRS.values())
FINANCIAL_COLUMNS = ASSET_COLUMNS + INCOME_COLUMNS

CASH_COMBINED_COLUMN = "Cash, Demand & Fixed Deposits"
CASH_COMPONENT_COLUMNS = ["Cash and Demand Deposits", "Fixed and Time Deposits"]
FIXED_DEPOSIT_INCOME_COLUMN = "Fixed and Time Deposits - Income"

SHARE_MAPPING = {
    "Cash, Demand & Fixed Deposits": "Cash_Share",
    "Commercial Paper and Corporate Bonds": "Corporate_Bonds_Share",
    "Guaranteed Funds": "Guaranteed_Funds_Share",
    "Immovable Property": "Immovable_Property_Share",
    "Kenya Government Securities": "Government_Securities_Share",
    "Offshore Investments": "Offshore_Investments_Share",
    "Property Unit Trusts": "Property_Unit_Trusts_Share",
    "Quoted Equity": "Quoted_Equity_Share",
    "Unquoted Equity": "Unquoted_Equity_Share",
}
SHARE_COLUMNS = list(SHARE_MAPPING.values())
CHANGE_COLUMNS = [f"Change_{share}" for share in SHARE_COLUMNS]

CLASSIFICATION_TARGET = "Material_Allocation_Change_Next_Year"
MAXIMUM_CHANGE_COLUMN = "Maximum_Future_Asset_Class_Change"

CLASSIFICATION_NUMERIC_FEATURES = (
    SHARE_COLUMNS
    + CHANGE_COLUMNS
    + [
        "Portfolio_HHI",
        "Active_Asset_Classes",
        "HHI_Change",
        "Dominant_Class_Changed",
        "Log_Total_Investments",
        "Log_Portfolio_Growth",
        "Year",
    ]
)
CLASSIFICATION_CATEGORICAL_FEATURES = ["Dominant_Asset_Class"]
CLASSIFICATION_FEATURES = (
    CLASSIFICATION_NUMERIC_FEATURES + CLASSIFICATION_CATEGORICAL_FEATURES
)

REQUIRED_SOURCE_COLUMNS = [
    "Ref", "Period", "Month", "Year",
    CASH_COMBINED_COLUMN,
    *CASH_COMPONENT_COLUMNS[:1],
    "Commercial Paper and Corporate Bonds",
    CASH_COMPONENT_COLUMNS[1],
    "Guaranteed Funds",
    "Immovable Property",
    "Kenya Government Securities",
    "Offshore Investments",
    "Property Unit Trusts",
    "Quoted Equity",
    "Unquoted Equity",
    *INCOME_COLUMNS,
    FIXED_DEPOSIT_INCOME_COLUMN,
]

PROJECT_TITLE = (
    "Analysing and Predicting Material Asset-Allocation Transitions "
    "among Retirement Benefits Schemes in Kenya"
)
