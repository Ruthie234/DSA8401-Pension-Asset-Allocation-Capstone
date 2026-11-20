"""Generate predictions from a saved calibrated model.

The input CSV must contain one or more rows with the 26 predictor columns
listed in models/model_metadata.json. It must not contain target/future fields.

Example:
    python scripts/predict.py --input new_scheme_year_features.csv --output predictions.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict next-year material asset-allocation changes.")
    parser.add_argument("--input", required=True, help="CSV containing model predictor columns.")
    parser.add_argument(
        "--model", default=str(PROJECT_ROOT / "models" / "material_allocation_change_model.joblib"),
        help="Path to the serialized calibrated model.",
    )
    parser.add_argument(
        "--metadata", default=str(PROJECT_ROOT / "models" / "model_metadata.json"),
        help="Path to model metadata JSON.",
    )
    parser.add_argument("--output", default="predictions.csv", help="Output CSV path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = Path(args.model).expanduser().resolve()
    metadata_path = Path(args.metadata).expanduser().resolve()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    for path in [model_path, metadata_path, input_path]:
        if not path.is_file():
            raise FileNotFoundError(f"Required file does not exist: {path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_names = metadata["features"]
    threshold = float(metadata.get("decision_threshold", 0.50))

    input_data = pd.read_csv(input_path)
    missing = [column for column in feature_names if column not in input_data.columns]
    if missing:
        raise ValueError(f"Prediction input is missing required features: {missing}")

    X = input_data.loc[:, feature_names].copy()
    if X.empty:
        raise ValueError("Prediction input contains no observations.")

    model = joblib.load(model_path)
    probabilities = model.predict_proba(X)[:, 1]
    if not np.isfinite(probabilities).all():
        raise ValueError("Model generated non-finite prediction probabilities.")

    output = pd.DataFrame({
        "Predicted_Probability_Material_Change": probabilities,
        "Predicted_Material_Change": (probabilities >= threshold).astype(int),
        "Decision_Threshold": threshold,
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    print(f"Loaded model: {model_path}")
    print(f"Input observations: {len(X)}")
    print(f"Output saved: {output_path}")
    print("Interpretation: 1 indicates a predicted material change; 0 indicates no predicted material change.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
