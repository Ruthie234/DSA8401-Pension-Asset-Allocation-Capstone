r"""Run the reproducible pension asset-allocation ML pipeline from raw CSV to model.

Example (Windows PowerShell):
    python scripts/run_training.py --data "C:\path\to\COPY 1 FINAL.csv"

Example (from project root when the CSV is in data/raw):
    python scripts/run_training.py --data "data/raw/COPY 1 FINAL.csv"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from pension_capstone.config import (  # noqa: E402
    CALIBRATED_DECISION_THRESHOLD,
    CALIBRATION_OUTCOME_YEAR,
    CLASSIFICATION_FEATURES,
    CLASSIFICATION_TARGET,
    MATERIAL_CHANGE_THRESHOLD,
    PROJECT_TITLE,
    TEMPORAL_CV_YEARS,
    TEST_OUTCOME_YEARS,
)
from pension_capstone.data_pipeline import run_data_preparation  # noqa: E402
from pension_capstone.feature_engineering import (  # noqa: E402
    build_portfolio_features,
    feature_manifest,
)
from pension_capstone.modeling import run_modeling  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the pension asset-allocation change classifier.")
    parser.add_argument(
        "--data", type=str, default=None,
        help="Path to the authorised raw RBA CSV. Can also be set with RBA_DATA_PATH.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(PROJECT_ROOT),
        help="Project output directory (default: project root).",
    )
    parser.add_argument(
        "--skip-shap", action="store_true",
        help="Skip optional SHAP plots and feature-importance export.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_dir).expanduser().resolve()
    models_dir = output_root / "models"
    reports_dir = output_root / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(PROJECT_TITLE)
    print("=" * 72)
    print("1/5: Loading, cleaning and validating source data...")
    prepared = run_data_preparation(args.data)
    print(prepared.data_quality_summary.to_string(index=False))

    print("\n2/5: Constructing portfolio features and next-year classification target...")
    df_modelling, df_transitions, classification_data = build_portfolio_features(
        prepared.df_modelling
    )
    print(f"Positive-investment scheme-year rows: {len(df_modelling):,}")
    print(f"Consecutive-year transitions: {len(df_transitions):,}")
    print(f"Classification-ready observations: {len(classification_data):,}")
    print("Target distribution:")
    print(classification_data[CLASSIFICATION_TARGET].value_counts().sort_index().to_string())

    print("\n3/5: Running expanding-window validation and fitting/calibrating models...")
    result = run_modeling(classification_data)
    print("Temporal validation summary:")
    print(result.temporal_cv_summary.round(4).to_string(index=False))
    print(f"\nSelected model: {result.selected_model_name}")

    print("\n4/5: Saving model artifacts and aggregate evaluation reports...")
    model_path = models_dir / "material_allocation_change_model.joblib"
    metadata_path = models_dir / "model_metadata.json"
    joblib.dump(result.calibrated_model, model_path)

    metadata = {
        "project": PROJECT_TITLE,
        "selected_model": result.selected_model_name,
        "model_selection_metric": "Mean Average Precision (PR-AUC) across expanding-window temporal validation years",
        "temporal_validation_outcome_years": TEMPORAL_CV_YEARS,
        "calibration_outcome_year": CALIBRATION_OUTCOME_YEAR,
        "test_outcome_years": TEST_OUTCOME_YEARS,
        "target": CLASSIFICATION_TARGET,
        "material_change_threshold": MATERIAL_CHANGE_THRESHOLD,
        "decision_threshold": CALIBRATED_DECISION_THRESHOLD,
        "threshold_rule": "Pre-specified 0.50 threshold; not optimised on calibration or test data.",
        "features": CLASSIFICATION_FEATURES,
        "feature_manifest": feature_manifest(),
        "training_observations": int(len(result.train_data)),
        "calibration_observations": int(len(result.calibration_data)),
        "test_observations": int(len(result.test_data)),
        "test_set_used_for_model_selection_or_fitting": False,
        "interpretation": "Decision-support monitoring signal; not a determination that a change is harmful or non-compliant.",
        "prediction_input_note": "Supply one row per scheme-year with exactly the 26 predictor columns listed in features. Predictor values must be calculated from information available at the prediction date.",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    result.temporal_cv_results.to_csv(reports_dir / "temporal_validation_results.csv", index=False)
    result.temporal_cv_summary.to_csv(reports_dir / "temporal_validation_summary.csv", index=False)
    result.heldout_test_results.to_csv(reports_dir / "heldout_test_results.csv", index=False)
    result.confusion_summary.to_csv(reports_dir / "confusion_summary.csv", index=False)
    result.calibration_results.to_csv(reports_dir / "calibration_results.csv", index=False)
    result.sensitivity_results.to_csv(reports_dir / "sensitivity_analysis.csv", index=False)
    prepared.data_quality_summary.to_csv(reports_dir / "data_quality_summary.csv", index=False)

    fold_and_partition = []
    for _, row in result.temporal_cv_results.drop_duplicates("Validation_Outcome_Year").iterrows():
        fold_and_partition.append({
            "Validation_Outcome_Year": int(row["Validation_Outcome_Year"]),
            "Fold_Training_Observations": int(row["Training_Observations"]),
            "Fold_Validation_Observations": int(row["Validation_Observations"]),
        })
    (reports_dir / "temporal_fold_sizes.json").write_text(
        json.dumps(fold_and_partition, indent=2), encoding="utf-8"
    )

    print(f"Saved calibrated model: {model_path}")
    print(f"Saved metadata: {metadata_path}")

    print("\n5/5: Reloading the serialized model and testing predictions...")
    reloaded = joblib.load(model_path)
    smoke_probabilities = reloaded.predict_proba(result.X_test.iloc[:5])[:, 1]
    if smoke_probabilities.shape[0] != min(5, len(result.X_test)):
        raise AssertionError("Serialized-model smoke test returned an unexpected number of predictions.")
    if not np.isfinite(smoke_probabilities).all() or ((smoke_probabilities < 0) | (smoke_probabilities > 1)).any():
        raise AssertionError("Serialized-model smoke test returned invalid probabilities.")
    print("Serialized-model reload and prediction smoke test: PASSED")
    print("Example probabilities:", np.round(smoke_probabilities, 4).tolist())

    if not args.skip_shap:
        print("\nOptional SHAP analysis...")
        try:
            from pension_capstone.interpretability import run_shap_analysis
            importance = run_shap_analysis(
                result.selected_base_model,
                result.X_train,
                result.X_test,
                reports_dir,
            )
            print("SHAP feature importance saved:")
            print(importance.head(10).to_string(index=False))
        except Exception as exc:  # The main pipeline must remain usable if SHAP is unavailable.
            print(f"SHAP analysis was not completed: {exc}")
            print("Install the optional dependency with: python -m pip install shap")

    print("\nPipeline completed successfully.")
    print(f"Selected model: {result.selected_model_name}")
    print(f"Final test report: {reports_dir / 'heldout_test_results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
