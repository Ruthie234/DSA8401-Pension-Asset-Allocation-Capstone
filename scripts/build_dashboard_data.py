"""Build aggregate-only dashboard datasets from the authorised RBA data.

Run from the project root after scripts/run_training.py has completed:
    python scripts/build_dashboard_data.py --data "COPY 1 FINAL.csv"

Only grouped/aggregate outputs are written. Scheme references and row-level
records are never exported to dashboard_data/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from pension_capstone.config import (  # noqa: E402
    CLASSIFICATION_FEATURES,
    CLASSIFICATION_TARGET,
    MATERIAL_CHANGE_THRESHOLD,
    SHARE_COLUMNS,
    SHARE_MAPPING,
)
from pension_capstone.data_pipeline import run_data_preparation  # noqa: E402
from pension_capstone.feature_engineering import build_portfolio_features  # noqa: E402

MODEL_PATH = PROJECT_ROOT / "models" / "material_allocation_change_model.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
REPORTS_DIR = PROJECT_ROOT / "reports"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "dashboard_data"

ASSET_LABELS = {
    "Cash_Share": "Cash, demand and fixed deposits",
    "Corporate_Bonds_Share": "Corporate bonds",
    "Guaranteed_Funds_Share": "Guaranteed funds",
    "Immovable_Property_Share": "Immovable property",
    "Government_Securities_Share": "Kenya government securities",
    "Offshore_Investments_Share": "Offshore investments",
    "Property_Unit_Trusts_Share": "Property unit trusts",
    "Quoted_Equity_Share": "Quoted equity",
    "Unquoted_Equity_Share": "Unquoted equity",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create safe aggregate-only dashboard data.")
    parser.add_argument("--data", type=str, default=None, help="Authorised raw RBA CSV path.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write only aggregate tables; reject scheme identifiers and row-level output."""
    forbidden = {"Ref", "Scheme", "Scheme_ID", "Scheme_Ref", "Reference_Number"}
    bad = forbidden.intersection(frame.columns)
    if bad:
        raise ValueError(f"Refusing to export scheme-level identifiers: {sorted(bad)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _feature_group(feature: str) -> str:
    name = str(feature).replace("numeric__", "").replace("categorical__", "")
    if name in SHARE_COLUMNS:
        return "Current asset allocation shares"
    if name.startswith("Change_"):
        return "Historical allocation changes"
    if name in {"Portfolio_HHI", "HHI_Change"}:
        return "Concentration and diversification"
    if name in {"Active_Asset_Classes", "Dominant_Class_Changed"} or name.startswith("Dominant_Asset_Class_"):
        return "Portfolio composition"
    if name in {"Log_Total_Investments", "Log_Portfolio_Growth"}:
        return "Portfolio size and growth"
    if name == "Year":
        return "Time"
    return "Other predictors"


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.is_file() or not METADATA_PATH.is_file():
        raise FileNotFoundError(
            "Trained model artifacts were not found. Run `python scripts/run_training.py` first."
        )

    print("Preparing source data and aggregate portfolio features...")
    prepared = run_data_preparation(args.data)
    portfolio, transitions, classification = build_portfolio_features(prepared.df_modelling)

    # 1. Annual scheme-average allocation and portfolio composition.
    annual_agg: dict[str, tuple[str, Any]] = {
        "Number_of_Schemes": ("Ref", "nunique"),
        "Average_Portfolio_HHI": ("Portfolio_HHI", "mean"),
        "Average_Diversification_Index": ("Diversification_Index", "mean"),
        "Average_Active_Asset_Classes": ("Active_Asset_Classes", "mean"),
    }
    for share in SHARE_COLUMNS:
        annual_agg[share] = (share, "mean")
    annual = portfolio.groupby("Year", as_index=False).agg(**annual_agg).sort_values("Year")
    for share in SHARE_COLUMNS:
        annual[share] = annual[share] * 100.0
    annual = annual.rename(columns=ASSET_LABELS)
    annual["Average_Portfolio_HHI"] = annual["Average_Portfolio_HHI"].round(4)
    annual["Average_Diversification_Index"] = annual["Average_Diversification_Index"].round(4)
    annual["Average_Active_Asset_Classes"] = annual["Average_Active_Asset_Classes"].round(2)
    _write_csv(annual, output_dir / "allocation_and_portfolio_by_year.csv")

    # 2. Observed transitions grouped by the year in which the outcome occurred.
    target = CLASSIFICATION_TARGET
    transition_summary = classification.groupby("Next_Year").agg(
        Number_of_Transitions=(target, "size"),
        Observed_Material_Changes=(target, "sum"),
        Observed_Material_Change_Rate=(target, "mean"),
        Average_Maximum_Asset_Class_Change=("Maximum_Future_Asset_Class_Change", "mean"),
        Average_Allocation_Turnover=("Future_Allocation_Turnover", "mean"),
    ).reset_index().rename(columns={"Next_Year": "Outcome_Year"})
    transition_summary["Observed_Material_Change_Rate_pct"] = transition_summary.pop(
        "Observed_Material_Change_Rate"
    ) * 100.0
    transition_summary["Average_Maximum_Asset_Class_Change_pp"] = transition_summary.pop(
        "Average_Maximum_Asset_Class_Change"
    ) * 100.0
    transition_summary["Average_Allocation_Turnover_pp"] = transition_summary.pop(
        "Average_Allocation_Turnover"
    ) * 100.0
    transition_summary = transition_summary.round(3)
    _write_csv(transition_summary, output_dir / "observed_transitions_by_outcome_year.csv")

    # 3. Held-out test-period predictions, grouped by outcome year.
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    features = metadata.get("features", CLASSIFICATION_FEATURES)
    if not isinstance(features, list) or len(features) != 26:
        raise ValueError("Model metadata does not contain the expected 26 predictors.")
    missing = sorted(set(features).difference(classification.columns))
    if missing:
        raise ValueError(f"Classification data is missing model features: {missing}")

    model = joblib.load(MODEL_PATH)
    probabilities = model.predict_proba(classification.loc[:, features])[:, 1]
    threshold = float(metadata.get("decision_threshold", 0.50))
    prediction_rows = pd.DataFrame({
        "Outcome_Year": classification["Next_Year"].astype(int).to_numpy(),
        "Predicted_Probability": probabilities,
        "Predicted_Class": (probabilities >= threshold).astype(int),
        "Observed_Class": classification[target].astype(int).to_numpy(),
    })
    test_years = [int(y) for y in metadata.get("test_outcome_years", [2024, 2025])]
    prediction_rows = prediction_rows[prediction_rows["Outcome_Year"].isin(test_years)]
    if prediction_rows.empty:
        raise ValueError(f"No prediction rows found for configured test years: {test_years}")
    predictions = prediction_rows.groupby("Outcome_Year").agg(
        Number_of_Test_Observations=("Observed_Class", "size"),
        Mean_Predicted_Probability=("Predicted_Probability", "mean"),
        Predicted_Material_Change_Count=("Predicted_Class", "sum"),
        Predicted_Material_Change_Rate=("Predicted_Class", "mean"),
        Observed_Material_Change_Count=("Observed_Class", "sum"),
        Observed_Material_Change_Rate=("Observed_Class", "mean"),
    ).reset_index()
    for col in ["Mean_Predicted_Probability", "Predicted_Material_Change_Rate", "Observed_Material_Change_Rate"]:
        predictions[col] = predictions[col] * 100.0
    predictions = predictions.round(3)
    _write_csv(predictions, output_dir / "heldout_predictions_by_year.csv")

    # 4. Global SHAP feature importance, with a transparent predictor-group label.
    shap_path = REPORTS_DIR / "shap_feature_importance.csv"
    if shap_path.is_file():
        shap = pd.read_csv(shap_path)
        required_cols = {"Feature", "Mean_Absolute_SHAP"}
        if not required_cols.issubset(shap.columns):
            raise ValueError(f"Unexpected SHAP report columns; expected {sorted(required_cols)}")
        shap["Predictor_Group"] = shap["Feature"].map(_feature_group)
        shap = shap.sort_values("Mean_Absolute_SHAP", ascending=False)
    else:
        shap = pd.DataFrame(columns=["Feature", "Mean_Absolute_SHAP", "Predictor_Group"])
    _write_csv(shap, output_dir / "shap_feature_importance_aggregate.csv")

    # 5. Aggregate data-quality table (measure and count only).
    quality = prepared.data_quality_summary.copy()
    _write_csv(quality, output_dir / "data_quality_summary.csv")

    dashboard_metadata = {
        "project": metadata.get("project", "Pension Asset-Allocation Change Monitoring"),
        "selected_model": metadata.get("selected_model", "Unknown"),
        "decision_threshold": threshold,
        "material_change_threshold": MATERIAL_CHANGE_THRESHOLD,
        "target_definition": (
            "A material change is an absolute change of at least five percentage points "
            "in at least one individual asset-class share between consecutive reporting years."
        ),
        "test_outcome_years": test_years,
        "aggregation_note": (
            "All dashboard datasets are aggregate summaries. No scheme reference identifiers "
            "or row-level scheme observations are exported. Asset shares are unweighted averages "
            "across schemes in each reporting year."
        ),
        "shap_note": (
            "SHAP values describe the underlying base classifier before the separate sigmoid "
            "calibration layer. They are not causal effects or changes in calibrated probability."
        ),
    }
    (output_dir / "dashboard_metadata.json").write_text(
        json.dumps(dashboard_metadata, indent=2), encoding="utf-8"
    )

    # Defensive output check: none of the generated tables may contain scheme identifiers.
    for csv_path in output_dir.glob("*.csv"):
        columns = set(pd.read_csv(csv_path, nrows=0).columns)
        if "Ref" in columns:
            raise AssertionError(f"Scheme reference leaked into dashboard export: {csv_path.name}")

    print(f"Dashboard data created in: {output_dir}")
    print("Exports: allocation trends, HHI/diversification, observed transitions,")
    print("held-out 2024–2025 prediction summaries, SHAP importance and data quality.")
    print("Privacy check: no scheme reference column exported.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
