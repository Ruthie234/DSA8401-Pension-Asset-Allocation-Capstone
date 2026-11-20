"""FastAPI service for predicting next-year material pension allocation changes.

Run from the project root:
    python -m uvicorn api.main:app --reload

The interactive API documentation is then available at http://127.0.0.1:8000/docs.
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from sklearn.pipeline import Pipeline

from api.schemas import FeatureContribution, PortfolioFeatures, PredictionResponse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "material_allocation_change_model.joblib"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
TARGET_DEFINITION = (
    "A material change is an absolute change of at least five percentage points "
    "in the share of at least one individual asset class between consecutive reporting years."
)


def _configured_path(environment_name: str, default: Path) -> Path:
    value = os.environ.get(environment_name)
    return Path(value).expanduser().resolve() if value else default.resolve()


def _extract_base_pipeline(calibrated_model: Any) -> Pipeline:
    """Get the frozen base Pipeline inside a fitted CalibratedClassifierCV."""
    calibrated_classifiers = getattr(calibrated_model, "calibrated_classifiers_", None)
    if not calibrated_classifiers:
        raise RuntimeError(
            "The saved model does not expose fitted calibrated classifiers; "
            "rerun scripts/run_training.py to regenerate the model artifact."
        )

    estimator = calibrated_classifiers[0].estimator
    # With recent scikit-learn, the fitted base estimator is wrapped in FrozenEstimator.
    if hasattr(estimator, "estimator"):
        estimator = estimator.estimator

    if not isinstance(estimator, Pipeline):
        raise RuntimeError(
            "Could not locate the fitted preprocessing/model Pipeline inside the calibrated model."
        )
    return estimator


def _normalise_shap_values(raw_values: Any) -> np.ndarray:
    """Return SHAP values for the positive (material-change) class."""
    if isinstance(raw_values, list):
        values = np.asarray(raw_values[-1])
    else:
        values = np.asarray(raw_values)
        if values.ndim == 3 and values.shape[-1] == 2:
            values = values[:, :, 1]
        elif values.ndim == 3 and values.shape[0] == 2:
            values = values[1]

    if values.ndim == 2 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 1:
        raise RuntimeError(f"Unexpected SHAP output shape: {values.shape}")
    return values.astype(float)


def _load_runtime() -> dict[str, Any]:
    model_path = _configured_path("CAPSTONE_MODEL_PATH", DEFAULT_MODEL_PATH)
    metadata_path = _configured_path("CAPSTONE_METADATA_PATH", DEFAULT_METADATA_PATH)

    if not model_path.is_file():
        raise RuntimeError(
            f"Trained model not found: {model_path}. Run `python scripts/run_training.py` first."
        )
    if not metadata_path.is_file():
        raise RuntimeError(
            f"Model metadata not found: {metadata_path}. Run `python scripts/run_training.py` first."
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    features = metadata.get("features")
    if not isinstance(features, list) or len(features) != 26:
        raise RuntimeError("Model metadata must contain the expected 26 predictor names.")

    model = joblib.load(model_path)
    if not hasattr(model, "predict_proba"):
        raise RuntimeError("The saved model does not implement predict_proba().")

    threshold = float(metadata.get("decision_threshold", 0.50))
    if not 0 <= threshold <= 1:
        raise RuntimeError("The model metadata contains an invalid decision threshold.")

    runtime: dict[str, Any] = {
        "model": model,
        "metadata": metadata,
        "features": features,
        "threshold": threshold,
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "shap_explainer": None,
        "shap_feature_names": None,
        "shap_error": None,
    }

    # SHAP explanations are part of the proposal. Load the same base estimator
    # and feature transformation used by the training pipeline. If SHAP cannot
    # initialise, predictions remain available but the API reports this clearly.
    try:
        import shap

        base_pipeline = _extract_base_pipeline(model)
        preprocessor = base_pipeline.named_steps["preprocessor"]
        classifier = base_pipeline.named_steps["classifier"]
        feature_names = list(preprocessor.get_feature_names_out())

        if hasattr(classifier, "estimators_"):
            explainer = shap.TreeExplainer(classifier)
        elif hasattr(classifier, "coef_"):
            explainer = shap.LinearExplainer(classifier, np.zeros((1, len(feature_names))))
        else:
            raise RuntimeError(
                f"SHAP explanation is not configured for {type(classifier).__name__}."
            )

        runtime["base_pipeline"] = base_pipeline
        runtime["shap_explainer"] = explainer
        runtime["shap_feature_names"] = feature_names
    except Exception as exc:  # Keep the prediction service available if explanation setup fails.
        runtime["shap_error"] = f"{type(exc).__name__}: {exc}"

    return runtime


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.runtime = _load_runtime()
    yield
    app.state.runtime = None


app = FastAPI(
    title="Pension Asset-Allocation Change Prediction API",
    description=(
        "A decision-support API that estimates the probability of a material "
        "asset-allocation change in the following consecutive reporting year. "
        "A positive prediction is not a finding of harm or non-compliance."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", tags=["Service"])
def root():
    return {
        "service": "Pension Asset-Allocation Change Prediction API",
        "status": "available",
        "documentation": "/docs",
        "health_check": "/health",
    }


@app.get("/health", tags=["Service"])
def health(request: Request):
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "model_loaded": False},
        )
    return {
        "status": "healthy",
        "model_loaded": True,
        "selected_model": runtime["metadata"].get("selected_model", "Unknown"),
        "shap_explanations_available": runtime["shap_explainer"] is not None,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(features: PortfolioFeatures, request: Request) -> PredictionResponse:
    """Predict the probability of a material change for one scheme-year row.

    Submit only the 26 predictor features. Do not submit the target, next-year
    allocation shares, or any information that would not be known at prediction time.
    """
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="Prediction model is not loaded.")

    payload = features.model_dump()
    feature_names = runtime["features"]
    missing = [name for name in feature_names if name not in payload]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing model features: {missing}")

    input_frame = pd.DataFrame([payload], columns=feature_names)
    try:
        probability = float(runtime["model"].predict_proba(input_frame)[0, 1])
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"The model could not generate a prediction: {type(exc).__name__}: {exc}",
        ) from exc

    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise HTTPException(status_code=500, detail="The model returned an invalid probability.")

    threshold = runtime["threshold"]
    predicted_class = int(probability >= threshold)

    contributions: list[FeatureContribution] = []
    explanation_note = (
        "SHAP contributions explain the underlying base classifier output before the "
        "separate sigmoid calibration layer. They are not causal effects and should not "
        "be interpreted as percentage-point changes in the calibrated probability."
    )

    explainer = runtime["shap_explainer"]
    if explainer is not None:
        try:
            base_pipeline = runtime["base_pipeline"]
            transformed = base_pipeline.named_steps["preprocessor"].transform(input_frame)
            shap_raw = explainer.shap_values(transformed)
            shap_values = _normalise_shap_values(shap_raw)
            names = runtime["shap_feature_names"]
            if len(shap_values) != len(names):
                raise RuntimeError("SHAP values do not align with transformed feature names.")

            top_indices = np.argsort(np.abs(shap_values))[::-1][:5]
            for index in top_indices:
                value = float(shap_values[index])
                if value > 1e-10:
                    effect = "pushes_toward_material_change"
                elif value < -1e-10:
                    effect = "pushes_toward_no_material_change"
                else:
                    effect = "approximately_neutral"
                feature_name = str(names[index]).replace("numeric__", "").replace("categorical__", "")
                contributions.append(
                    FeatureContribution(
                        feature=feature_name,
                        shap_value=value,
                        effect=effect,
                    )
                )
        except Exception as exc:
            # Do not silently imply explanations were produced when they were not.
            explanation_note = (
                "Prediction was generated, but per-observation SHAP explanations were unavailable: "
                f"{type(exc).__name__}: {exc}"
            )
    else:
        explanation_note = (
            "Prediction was generated, but SHAP could not be initialised when the API started: "
            f"{runtime['shap_error']}"
        )

    interpretation = (
        "1 indicates a predicted material allocation change; 0 indicates no predicted material change "
        "at the configured decision threshold. This is a decision-support signal, not a determination "
        "that a portfolio change is harmful or non-compliant."
    )

    return PredictionResponse(
        selected_model=str(runtime["metadata"].get("selected_model", "Unknown")),
        predicted_probability_material_change=probability,
        predicted_material_change=predicted_class,
        decision_threshold=threshold,
        target_definition=TARGET_DEFINITION,
        interpretation=interpretation,
        top_contributing_factors=contributions,
        explanation_note=explanation_note,
    )
