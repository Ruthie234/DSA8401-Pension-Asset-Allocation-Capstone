"""Construct portfolio features, historical transitions and the future binary target."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    ASSET_COLUMNS,
    CLASSIFICATION_CATEGORICAL_FEATURES,
    CLASSIFICATION_FEATURES,
    CLASSIFICATION_NUMERIC_FEATURES,
    CLASSIFICATION_TARGET,
    MATERIAL_CHANGE_THRESHOLD,
    MAXIMUM_CHANGE_COLUMN,
    SHARE_COLUMNS,
    SHARE_MAPPING,
)


def build_portfolio_features(df_modelling: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Add allocation shares, HHI, diversification and historical lag features."""
    data = df_modelling.copy()

    for asset_column, share_column in SHARE_MAPPING.items():
        data[share_column] = data[asset_column] / data["Total_Investments"]

    data["Share_Sum"] = data[SHARE_COLUMNS].sum(axis=1)
    invalid_share_sums = ~np.isclose(data["Share_Sum"], 1.0, atol=1e-6)
    if invalid_share_sums.any():
        raise ValueError(
            f"{int(invalid_share_sums.sum())} rows have nine asset shares that do not sum to one."
        )

    share_to_asset = {share: asset for asset, share in SHARE_MAPPING.items()}
    data["Active_Asset_Classes"] = (data[ASSET_COLUMNS] > 0).sum(axis=1)
    data["Dominant_Asset_Class"] = data[SHARE_COLUMNS].idxmax(axis=1).map(share_to_asset)
    data["Portfolio_HHI"] = (data[SHARE_COLUMNS] ** 2).sum(axis=1)
    data["Diversification_Index"] = 1 - data["Portfolio_HHI"]

    if not data["Active_Asset_Classes"].between(1, 9).all():
        raise AssertionError("Active asset-class count is outside the expected 1–9 range.")
    if not data["Portfolio_HHI"].between(1 / 9 - 1e-8, 1 + 1e-8).all():
        raise AssertionError("Portfolio HHI falls outside its theoretical range.")

    data = data.sort_values(["Ref", "Year", "Month"], kind="stable").reset_index(drop=True)
    data["Previous_Year"] = data.groupby("Ref")["Year"].shift(1)
    data["Year_Gap"] = data["Year"] - data["Previous_Year"]
    data["Consecutive_Year"] = data["Year_Gap"].eq(1)

    change_columns: list[str] = []
    for share_column in SHARE_COLUMNS:
        previous_column = f"Previous_{share_column}"
        change_column = f"Change_{share_column}"
        change_columns.append(change_column)
        data[previous_column] = data.groupby("Ref")[share_column].shift(1)
        data[change_column] = data[share_column] - data[previous_column]
        data.loc[~data["Consecutive_Year"], [previous_column, change_column]] = np.nan

    for current_column in ["Total_Investments", "Portfolio_HHI", "Dominant_Asset_Class"]:
        previous_column = f"Previous_{current_column}"
        data[previous_column] = data.groupby("Ref")[current_column].shift(1)
        data.loc[~data["Consecutive_Year"], previous_column] = np.nan

    transitions = data.loc[data["Consecutive_Year"]].copy()
    transitions["Total_Allocation_Change"] = transitions[change_columns].sum(axis=1)
    if not np.isclose(transitions["Total_Allocation_Change"], 0.0, atol=1e-6).all():
        raise AssertionError("Historical asset-allocation changes do not sum to approximately zero.")

    transitions["Dominant_Class_Changed"] = (
        transitions["Dominant_Asset_Class"] != transitions["Previous_Dominant_Asset_Class"]
    )
    transitions["HHI_Change"] = transitions["Portfolio_HHI"] - transitions["Previous_Portfolio_HHI"]
    transitions["Diversification_Change"] = -transitions["HHI_Change"]
    transitions["Portfolio_Growth"] = (
        transitions["Total_Investments"] / transitions["Previous_Total_Investments"] - 1
    )

    change_to_asset = {
        f"Change_{share}": asset for asset, share in SHARE_MAPPING.items()
    }
    transitions["Largest_Increase_Class"] = transitions[change_columns].idxmax(axis=1).map(change_to_asset)
    transitions["Largest_Decrease_Class"] = transitions[change_columns].idxmin(axis=1).map(change_to_asset)
    transitions["Largest_Increase_Amount"] = transitions[change_columns].max(axis=1)
    transitions["Largest_Decrease_Amount"] = transitions[change_columns].min(axis=1)
    no_change = np.isclose(transitions["Largest_Increase_Amount"], 0.0, atol=1e-8)
    transitions.loc[no_change, ["Largest_Increase_Class", "Largest_Decrease_Class"]] = "No allocation change"
    transitions["Meaningful_Government_Increase"] = (
        transitions["Change_Government_Securities_Share"] >= 0.10
    )
    transitions["Meaningful_Guaranteed_Funds_Increase"] = (
        transitions["Change_Guaranteed_Funds_Share"] >= 0.10
    )

    # Future values are used only to construct the outcome, never as predictors.
    data["Next_Year"] = data.groupby("Ref")["Year"].shift(-1)
    data["Next_Year_Consecutive"] = (data["Next_Year"] - data["Year"]).eq(1)
    future_share_columns = []
    for share_column in SHARE_COLUMNS:
        future_column = f"Next_{share_column}"
        future_share_columns.append(future_column)
        data[future_column] = data.groupby("Ref")[share_column].shift(-1)
        data.loc[~data["Next_Year_Consecutive"], future_column] = np.nan

    # Preserve the transition table's historical features, then attach lead values.
    transitions["Next_Year"] = data["Next_Year"].reindex(transitions.index)
    transitions["Next_Year_Consecutive"] = data["Next_Year_Consecutive"].reindex(transitions.index)
    for future_column in future_share_columns:
        transitions[future_column] = data[future_column].reindex(transitions.index)

    absolute_future_changes = pd.DataFrame(index=transitions.index)
    for current_share, future_share in zip(SHARE_COLUMNS, future_share_columns):
        absolute_future_changes[current_share] = (
            transitions[future_share] - transitions[current_share]
        ).abs()

    transitions[MAXIMUM_CHANGE_COLUMN] = absolute_future_changes.max(axis=1, skipna=False)
    transitions["Future_Allocation_Turnover"] = 0.5 * absolute_future_changes.sum(
        axis=1, min_count=len(SHARE_COLUMNS)
    )

    classification = transitions.loc[
        transitions["Next_Year_Consecutive"] & transitions[MAXIMUM_CHANGE_COLUMN].notna()
    ].copy()
    classification[CLASSIFICATION_TARGET] = (
        classification[MAXIMUM_CHANGE_COLUMN] >= MATERIAL_CHANGE_THRESHOLD
    ).astype("int8")

    classification["Log_Total_Investments"] = np.log1p(classification["Total_Investments"])
    classification["Log_Portfolio_Growth"] = np.log(
        classification["Total_Investments"] / classification["Previous_Total_Investments"]
    )
    classification["Dominant_Class_Changed"] = classification["Dominant_Class_Changed"].astype("int8")

    missing = classification[CLASSIFICATION_FEATURES].isna().sum()
    missing = missing[missing > 0]
    numeric_values = classification[CLASSIFICATION_NUMERIC_FEATURES].to_numpy(dtype=float)
    if not missing.empty:
        raise ValueError(f"Predictors contain missing values: {missing.to_dict()}")
    if not np.isfinite(numeric_values).all():
        raise ValueError("Numeric predictors contain infinite or non-finite values.")
    if len(CLASSIFICATION_FEATURES) != 26:
        raise AssertionError("Expected the 26 proposal-notebook predictors.")
    if classification[CLASSIFICATION_TARGET].nunique() != 2:
        raise ValueError("The classification target does not contain both classes.")

    return data, transitions, classification


def feature_manifest() -> dict[str, list[str] | float | str]:
    """Return a serialisable description of the model features and target."""
    return {
        "target": CLASSIFICATION_TARGET,
        "target_definition": (
            "1 when at least one individual asset-class share changes in absolute value "
            "by at least five percentage points in the following consecutive reporting year."
        ),
        "material_change_threshold": MATERIAL_CHANGE_THRESHOLD,
        "numeric_features": CLASSIFICATION_NUMERIC_FEATURES,
        "categorical_features": CLASSIFICATION_CATEGORICAL_FEATURES,
        "all_features": CLASSIFICATION_FEATURES,
    }
