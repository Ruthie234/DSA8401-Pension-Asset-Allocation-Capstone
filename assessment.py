"""Feature construction for one current-year pension portfolio assessment.

The previous-year portfolio is retrieved from the authorised local RBA history.
Current-year amounts are supplied by the user. This module reproduces the
26-predictor definitions used by pension_capstone.feature_engineering.
"""
from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import pandas as pd

from pension_capstone.config import (
    ASSET_COLUMNS,
    CLASSIFICATION_FEATURES,
    SHARE_COLUMNS,
    SHARE_MAPPING,
)


def build_assessment_features(
    previous_record: pd.Series,
    current_amounts: Mapping[str, float],
    reporting_year: int,
) -> tuple[dict[str, object], dict[str, object]]:
    """Build the exact 26 model features for a manually entered current portfolio.

    Args:
        previous_record: Cleaned/annualised row from the project history, for
            the same anonymised scheme and exactly reporting_year - 1.
        current_amounts: Mapping from the nine source asset-class names to
            current-year amounts. Unlisted classes are treated as zero.
        reporting_year: Current portfolio reporting year (the model feature year).

    Returns:
        (predictor_payload, comparison_summary)

    Raises:
        ValueError: for invalid amounts, missing/non-consecutive history, or
            a portfolio that cannot be assessed.
    """
    year = int(reporting_year)
    previous_year = int(previous_record["Year"])
    if previous_year != year - 1:
        raise ValueError(
            f"A consecutive previous-year record is required. Expected {year - 1}, "
            f"but the retrieved record is for {previous_year}."
        )

    values: dict[str, float] = {}
    for asset in ASSET_COLUMNS:
        raw_value = current_amounts.get(asset, 0.0)
        try:
            amount = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Investment amount for {asset} must be numeric.") from exc
        if not math.isfinite(amount) or amount < 0:
            raise ValueError(f"Investment amount for {asset} must be finite and non-negative.")
        values[asset] = amount

    current_total = float(sum(values.values()))
    previous_total = float(previous_record["Total_Investments"])
    if not math.isfinite(current_total) or current_total <= 0:
        raise ValueError("Enter a positive investment amount for at least one asset class.")
    if not math.isfinite(previous_total) or previous_total <= 0:
        raise ValueError("The retrieved previous-year portfolio has no valid positive total.")

    current_shares: dict[str, float] = {
        share: values[asset] / current_total
        for asset, share in SHARE_MAPPING.items()
    }
    previous_shares: dict[str, float] = {
        share: float(previous_record[share])
        for share in SHARE_COLUMNS
    }

    share_sum = sum(current_shares.values())
    if not math.isclose(share_sum, 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError("Current asset-class shares do not sum to 100%. Check the entered amounts.")

    # Match the source notebook's pandas idxmax behaviour: the first asset in
    # SHARE_MAPPING order wins when two classes have exactly equal shares.
    dominant_asset = max(SHARE_MAPPING, key=lambda asset: current_shares[SHARE_MAPPING[asset]])
    previous_dominant_asset = str(previous_record["Dominant_Asset_Class"])
    current_hhi = float(sum(value * value for value in current_shares.values()))
    previous_hhi = float(previous_record["Portfolio_HHI"])

    features: dict[str, object] = {}
    for share in SHARE_COLUMNS:
        features[share] = float(current_shares[share])
    for share in SHARE_COLUMNS:
        features[f"Change_{share}"] = float(current_shares[share] - previous_shares[share])

    features.update({
        "Portfolio_HHI": current_hhi,
        "Active_Asset_Classes": int(sum(amount > 0 for amount in values.values())),
        "HHI_Change": float(current_hhi - previous_hhi),
        "Dominant_Class_Changed": int(dominant_asset != previous_dominant_asset),
        "Log_Total_Investments": float(np.log1p(current_total)),
        "Log_Portfolio_Growth": float(np.log(current_total / previous_total)),
        "Year": year,
        "Dominant_Asset_Class": dominant_asset,
    })

    if len(features) != 26 or set(features) != set(CLASSIFICATION_FEATURES):
        missing = sorted(set(CLASSIFICATION_FEATURES) - set(features))
        extra = sorted(set(features) - set(CLASSIFICATION_FEATURES))
        raise RuntimeError(f"Predictor construction mismatch. Missing={missing}; extra={extra}")

    # Preserve the training metadata order in the outbound API payload.
    features = {name: features[name] for name in CLASSIFICATION_FEATURES}

    comparison: dict[str, object] = {
        "previous_year": previous_year,
        "current_year": year,
        "previous_total": previous_total,
        "current_total": current_total,
        "previous_shares": previous_shares,
        "current_shares": current_shares,
        "previous_hhi": previous_hhi,
        "current_hhi": current_hhi,
        "previous_diversification": 1.0 - previous_hhi,
        "current_diversification": 1.0 - current_hhi,
        "previous_active_classes": int(previous_record["Active_Asset_Classes"]),
        "current_active_classes": int(sum(amount > 0 for amount in values.values())),
        "previous_dominant_asset": previous_dominant_asset,
        "current_dominant_asset": dominant_asset,
        "previous_amounts": {asset: float(previous_record[asset]) for asset in ASSET_COLUMNS},
        "current_amounts": values,
        "total_investment_growth_pct": (current_total / previous_total - 1.0) * 100.0,
    }
    return features, comparison
