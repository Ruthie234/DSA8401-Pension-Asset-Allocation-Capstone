# FastAPI setup — Pension Asset-Allocation Capstone

This API loads the calibrated model and metadata created by `scripts/run_training.py`.
It accepts one scheme-year observation containing the 26 model predictors and returns
its calibrated probability, binary prediction, and (when SHAP initialises successfully)
the five largest per-observation SHAP contributions.

## 1. Install API dependencies

Run from the project root in the same Python environment used for training:

```bash
python -m pip install -r api_requirements.txt
```

## 2. Confirm the trained model artifacts exist

The following files must exist:

```text
models/material_allocation_change_model.joblib
models/model_metadata.json
```

If they do not exist, run the training pipeline first:

```bash
python scripts/run_training.py
```

## 3. Start the API

Run this from the project root (the folder that contains `api`, `models`, `scripts`, and `src`):

```bash
python -m uvicorn api.main:app --reload
```

Open these addresses in a browser:

- API landing page: http://127.0.0.1:8000/
- Health check: http://127.0.0.1:8000/health
- Interactive API documentation: http://127.0.0.1:8000/docs

Use the `POST /predict` endpoint in `/docs`. Submit one JSON object containing the 26 required predictor fields. The request schema is displayed by FastAPI in the documentation.

## 4. Request design

The request must contain the 26 predictors listed in `models/model_metadata.json`. Do not send the target column, next-year asset shares, or any other future information.

The nine allocation-share fields must be proportions between 0 and 1 and must sum to approximately 1. The dominant asset-class field must identify an asset class with the largest submitted share.

## 5. Response design

The response contains:

- selected model name;
- calibrated probability of a material allocation change;
- predicted class using the threshold stored in model metadata;
- decision threshold;
- target definition and interpretation;
- up to five leading per-observation SHAP contributions, when available.

The SHAP values explain the underlying base classifier output before the separate sigmoid calibration layer. They are neither causal effects nor percentage-point changes in calibrated probability.

## 6. Important data-use note

The API does not embed or publish the raw RBA dataset. It loads only the locally generated model artifacts. Review RBA/university permissions before distributing model artifacts, sample requests, or any outputs derived from administrative records.

## 7. Model compatibility

The model was produced by the project's training scripts. For reliable loading, train and serve with compatible Python and scikit-learn versions. The API uses the model and feature manifest generated together by the training pipeline.
