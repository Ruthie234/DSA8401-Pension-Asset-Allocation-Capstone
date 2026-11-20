"""Raw RBA CSV ingestion, initial cleaning and financial quality preparation.

This module follows the data rules documented in the capstone notebook. It does
not silently overwrite the combined cash value, turn missing financial values
into zero, or automatically remove negative-income observations.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    ASSET_COLUMNS,
    ASSET_INCOME_PAIRS,
    CASH_COMPONENT_COLUMNS,
    CASH_COMBINED_COLUMN,
    FINANCIAL_COLUMNS,
    FIXED_DEPOSIT_INCOME_COLUMN,
    INCOME_COLUMNS,
    REQUIRED_SOURCE_COLUMNS,
)


@dataclass
class DataPreparationResult:
    """All key data objects and audit summaries produced during preparation."""

    df_raw: pd.DataFrame
    df_all_reports: pd.DataFrame
    df_earlier_reports: pd.DataFrame
    df_annual: pd.DataFrame
    df_without_reporting_month_year: pd.DataFrame
    df_missing_financial: pd.DataFrame
    df_partial_financial: pd.DataFrame
    df_negative_investment: pd.DataFrame
    df_negative_income: pd.DataFrame
    df_nonpositive_portfolio: pd.DataFrame
    df_modelling: pd.DataFrame
    cash_reconciliation: pd.DataFrame
    negative_income_detail_table: pd.DataFrame
    unusual_positive_cases: pd.DataFrame
    data_quality_summary: pd.DataFrame


def resolve_data_path(data_path: str | Path | None = None) -> Path:
    """Resolve the raw CSV path.

    Priority:
      1. Explicit ``data_path`` argument (recommended for reproducibility).
      2. ``RBA_DATA_PATH`` environment variable.
      3. Common project-relative locations for ``COPY 1 FINAL.csv``.

    The raw administrative dataset is deliberately not bundled with this code.
    """
    candidate_value = data_path or os.environ.get("RBA_DATA_PATH")
    if candidate_value:
        candidate = Path(candidate_value).expanduser()
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"The specified RBA CSV does not exist: {candidate}")
        return candidate.resolve()

    cwd = Path.cwd().resolve()
    roots = [cwd, *cwd.parents]
    checked: list[Path] = []
    for root in roots:
        checked.extend([
            root / "COPY 1 FINAL.csv",
            root / "data" / "COPY 1 FINAL.csv",
            root / "data" / "raw" / "COPY 1 FINAL.csv",
        ])
    for candidate in checked:
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not locate the authorised RBA dataset. Pass its path with "
        "--data, set RBA_DATA_PATH, or place it at ./COPY 1 FINAL.csv, "
        "./data/COPY 1 FINAL.csv, or ./data/raw/COPY 1 FINAL.csv."
    )


def load_raw_data(data_path: str | Path | None = None) -> tuple[pd.DataFrame, Path]:
    """Load and validate the expected raw source columns."""
    path = resolve_data_path(data_path)
    raw = pd.read_csv(path)
    missing = [column for column in REQUIRED_SOURCE_COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError(f"The source CSV is missing required columns: {missing}")
    return raw, path


def prepare_annual_panel(df_raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Clean reports and retain the latest report for each scheme-year.

    Mirrors notebook sections 3 and 4. Earlier reports and undated rows are
    preserved in separate outputs for audit. A tie at the latest reporting
    month is treated as an error requiring manual review.
    """
    df = df_raw.copy()

    # Remove rows that contain no information at all.
    df = df.dropna(how="all").copy().reset_index(drop=True)

    # A usable annual panel requires both reporting month and year.
    missing_date_mask = df[["Month", "Year"]].isna().any(axis=1)
    df_without_reporting_month_year = df.loc[missing_date_mask].copy()
    df = df.loc[~missing_date_mask].copy().reset_index(drop=True)

    # Validate reference, month and year before integer conversion.
    for column in ["Ref", "Month", "Year"]:
        df[column] = pd.to_numeric(df[column], errors="raise")
        if df[column].isna().any():
            raise ValueError(f"{column} contains missing values among dated records.")
        if not np.equal(df[column], np.floor(df[column])).all():
            raise ValueError(f"{column} contains non-integer values.")

    if not df["Month"].between(1, 12).all():
        invalid_months = sorted(df.loc[~df["Month"].between(1, 12), "Month"].unique())
        raise ValueError(f"Month contains values outside 1–12: {invalid_months}")

    df = df.astype({"Ref": "int64", "Month": "int64", "Year": "int64"})
    df["Reporting_Period"] = pd.to_datetime(
        {"year": df["Year"], "month": df["Month"], "day": 1}, errors="raise"
    )

    df = df.sort_values(
        ["Ref", "Year", "Month"], ascending=[True, True, True], kind="stable"
    ).reset_index(drop=True)

    max_month = df.groupby(["Ref", "Year"])["Month"].transform("max")
    latest_month_rows = df.loc[df["Month"].eq(max_month)]
    latest_month_counts = latest_month_rows.groupby(["Ref", "Year"]).size()
    latest_month_ties = latest_month_counts.loc[latest_month_counts > 1]
    if not latest_month_ties.empty:
        raise ValueError(
            "Some scheme-years contain multiple records in the latest reporting month. "
            "Resolve these ties before annualisation. Affected groups: "
            f"{latest_month_ties.to_dict()}"
        )

    df_all_reports = df.copy()
    earlier_mask = df_all_reports.duplicated(subset=["Ref", "Year"], keep="last")
    df_earlier_reports = df_all_reports.loc[earlier_mask].copy()
    df_annual = df_all_reports.loc[~earlier_mask].copy().reset_index(drop=True)

    if df_annual.duplicated(["Ref", "Year"]).any():
        raise AssertionError("Annual dataset still contains duplicate scheme-year keys.")

    return {
        "df_all_reports": df_all_reports,
        "df_earlier_reports": df_earlier_reports,
        "df_annual": df_annual,
        "df_without_reporting_month_year": df_without_reporting_month_year,
    }


def prepare_financial_dataset(df_annual: pd.DataFrame) -> dict[str, Any]:
    """Apply the notebook's financial quality rules and create ``df_modelling``."""
    nonzero_fixed_income = int(
        df_annual[FIXED_DEPOSIT_INCOME_COLUMN].fillna(0).ne(0).sum()
    )
    if nonzero_fixed_income > 0:
        raise ValueError(
            f"{FIXED_DEPOSIT_INCOME_COLUMN} contains {nonzero_fixed_income} non-zero "
            "values. Review the income mapping before proceeding."
        )

    cash_components_total = df_annual[CASH_COMPONENT_COLUMNS].sum(axis=1, min_count=2)
    cash_difference = df_annual[CASH_COMBINED_COLUMN] - cash_components_total
    cash_mismatch_mask = cash_difference.abs() > 1.0
    cash_reconciliation = df_annual.loc[
        cash_mismatch_mask,
        ["Ref", "Year", "Month", CASH_COMBINED_COLUMN] + CASH_COMPONENT_COLUMNS,
    ].copy()
    cash_reconciliation["Cash_Difference"] = cash_difference.loc[cash_mismatch_mask]

    df_clean = df_annual.drop(
        columns=CASH_COMPONENT_COLUMNS + [FIXED_DEPOSIT_INCOME_COLUMN]
    ).copy()

    all_financial_missing_mask = df_clean[FINANCIAL_COLUMNS].isna().all(axis=1)
    partial_financial_missing_mask = (
        df_clean[FINANCIAL_COLUMNS].isna().any(axis=1) & ~all_financial_missing_mask
    )
    df_missing_financial = df_clean.loc[all_financial_missing_mask].copy()
    df_partial_financial = df_clean.loc[partial_financial_missing_mask].copy()
    df_clean = df_clean.loc[
        ~(all_financial_missing_mask | partial_financial_missing_mask)
    ].copy().reset_index(drop=True)

    df_clean["Total_Investments"] = df_clean[ASSET_COLUMNS].sum(
        axis=1, min_count=len(ASSET_COLUMNS)
    )
    df_clean["Total_Income"] = df_clean[INCOME_COLUMNS].sum(
        axis=1, min_count=len(INCOME_COLUMNS)
    )
    if df_clean[["Total_Investments", "Total_Income"]].isna().any().any():
        raise AssertionError("Complete financial totals unexpectedly contain missing values.")

    # Income exceeding an investment amount is flagged for review, not removed.
    unusual_parts = []
    for asset_column, income_column in ASSET_INCOME_PAIRS.items():
        mask = (
            df_clean[asset_column].gt(0)
            & df_clean[income_column].gt(df_clean[asset_column])
        )
        if mask.any():
            part = df_clean.loc[
                mask, ["Ref", "Year", "Reporting_Period", asset_column, income_column]
            ].copy().rename(columns={
                asset_column: "Investment_Amount",
                income_column: "Income_Amount",
            })
            part["Asset_Class"] = asset_column
            part["Income_to_Investment_Ratio"] = (
                part["Income_Amount"] / part["Investment_Amount"]
            )
            unusual_parts.append(part)
    unusual_positive_cases = (
        pd.concat(unusual_parts, ignore_index=True).sort_values(
            "Income_to_Investment_Ratio", ascending=False
        ).reset_index(drop=True)
        if unusual_parts else
        pd.DataFrame(columns=[
            "Ref", "Year", "Reporting_Period", "Investment_Amount",
            "Income_Amount", "Asset_Class", "Income_to_Investment_Ratio",
        ])
    )

    # Preserve negative investment observations separately; negative income is
    # reviewed and retained only after negative-investment rows are set aside,
    # matching the notebook's sequence.
    negative_investment_mask = (df_clean[ASSET_COLUMNS] < 0).any(axis=1)
    df_negative_investment = df_clean.loc[negative_investment_mask].copy()

    df_after_negative_investment = df_clean.loc[~negative_investment_mask].copy().reset_index(drop=True)
    df_negative_income = df_after_negative_investment.loc[
        (df_after_negative_investment[INCOME_COLUMNS] < 0).any(axis=1)
    ].copy()
    df_after_negative_investment["Total_Investments"] = (
        df_after_negative_investment[ASSET_COLUMNS].sum(
            axis=1, min_count=len(ASSET_COLUMNS)
        )
    )
    df_after_negative_investment["Total_Income"] = (
        df_after_negative_investment[INCOME_COLUMNS].sum(
            axis=1, min_count=len(INCOME_COLUMNS)
        )
    )

    nonpositive_mask = df_after_negative_investment["Total_Investments"] <= 0
    df_nonpositive_portfolio = df_after_negative_investment.loc[nonpositive_mask].copy()
    df_modelling = df_after_negative_investment.loc[~nonpositive_mask].copy().reset_index(drop=True)

    # Diagnostics mirroring the notebook, without exporting scheme-level details.
    data_quality_summary = pd.DataFrame([
        {"Measure": "Raw source rows", "Count": len(df_annual) + 0},
        {"Measure": "Rows with all mapped financial fields missing", "Count": len(df_missing_financial)},
        {"Measure": "Rows with partially missing mapped financial fields", "Count": len(df_partial_financial)},
        {"Measure": "Negative-investment scheme-years set aside", "Count": len(df_negative_investment)},
        {"Measure": "Non-positive portfolios set aside", "Count": len(df_nonpositive_portfolio)},
        {"Measure": "Positive-investment modelling observations", "Count": len(df_modelling)},
        {"Measure": "Schemes in modelling dataset", "Count": df_modelling["Ref"].nunique()},
        {"Measure": "Negative-income observations retained", "Count": len(df_negative_income)},
        {"Measure": "Cash reconciliation mismatches (absolute difference > 1)", "Count": len(cash_reconciliation)},
    ])

    return {
        "df_clean": df_clean,
        "df_missing_financial": df_missing_financial,
        "df_partial_financial": df_partial_financial,
        "df_negative_investment": df_negative_investment,
        "df_negative_income": df_negative_income,
        "df_nonpositive_portfolio": df_nonpositive_portfolio,
        "df_modelling": df_modelling,
        "cash_reconciliation": cash_reconciliation,
        "unusual_positive_cases": unusual_positive_cases,
        "data_quality_summary": data_quality_summary,
    }


def run_data_preparation(data_path: str | Path | None = None) -> DataPreparationResult:
    """Execute raw loading, annualisation and financial preparation."""
    df_raw, _ = load_raw_data(data_path)
    annual_objects = prepare_annual_panel(df_raw)
    financial_objects = prepare_financial_dataset(annual_objects["df_annual"])

    summary = pd.DataFrame([
        {"Measure": "Raw source rows", "Count": len(df_raw)},
        {"Measure": "Completely empty rows removed", "Count": int(df_raw.isna().all(axis=1).sum())},
        {"Measure": "Undated rows set aside", "Count": len(annual_objects["df_without_reporting_month_year"])},
        {"Measure": "Valid dated reports", "Count": len(annual_objects["df_all_reports"])},
        {"Measure": "Earlier scheme-year reports set aside", "Count": len(annual_objects["df_earlier_reports"])},
        {"Measure": "Annual scheme-year records", "Count": len(annual_objects["df_annual"])},
        *financial_objects["data_quality_summary"].iloc[1:].to_dict("records"),
    ])

    return DataPreparationResult(
        df_raw=df_raw,
        df_all_reports=annual_objects["df_all_reports"],
        df_earlier_reports=annual_objects["df_earlier_reports"],
        df_annual=annual_objects["df_annual"],
        df_without_reporting_month_year=annual_objects["df_without_reporting_month_year"],
        df_missing_financial=financial_objects["df_missing_financial"],
        df_partial_financial=financial_objects["df_partial_financial"],
        df_negative_investment=financial_objects["df_negative_investment"],
        df_negative_income=financial_objects["df_negative_income"],
        df_nonpositive_portfolio=financial_objects["df_nonpositive_portfolio"],
        df_modelling=financial_objects["df_modelling"],
        cash_reconciliation=financial_objects["cash_reconciliation"],
        negative_income_detail_table=pd.DataFrame(),
        unusual_positive_cases=financial_objects["unusual_positive_cases"],
        data_quality_summary=summary,
    )
