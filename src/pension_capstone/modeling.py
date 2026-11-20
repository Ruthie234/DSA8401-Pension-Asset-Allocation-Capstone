"""Temporal model validation, final fitting, calibration and held-out evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from .config import (
    CALIBRATED_DECISION_THRESHOLD,
    CALIBRATION_OUTCOME_YEAR,
    CLASSIFICATION_CATEGORICAL_FEATURES,
    CLASSIFICATION_FEATURES,
    CLASSIFICATION_NUMERIC_FEATURES,
    CLASSIFICATION_TARGET,
    MAXIMUM_CHANGE_COLUMN,
    RANDOM_STATE,
    SENSITIVITY_THRESHOLDS,
    TEMPORAL_CV_YEARS,
    TEST_OUTCOME_YEARS,
)


@dataclass
class ModelingResult:
    temporal_cv_results: pd.DataFrame
    temporal_cv_summary: pd.DataFrame
    selected_model_name: str
    fitted_models: dict[str, Pipeline]
    selected_base_model: Pipeline
    calibrated_model: Any
    baseline_model: DummyClassifier
    train_data: pd.DataFrame
    calibration_data: pd.DataFrame
    test_data: pd.DataFrame
    X_train: pd.DataFrame
    y_train: pd.Series
    X_calibration: pd.DataFrame
    y_calibration: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    heldout_test_results: pd.DataFrame
    confusion_summary: pd.DataFrame
    calibration_results: pd.DataFrame
    sensitivity_results: pd.DataFrame
    test_predictions: pd.DataFrame
    threshold: float = CALIBRATED_DECISION_THRESHOLD


def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn before sparse_output was introduced
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(steps=numeric_steps)
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", _make_one_hot_encoder()),
    ])
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, CLASSIFICATION_NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CLASSIFICATION_CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def make_candidate_models() -> dict[str, Pipeline]:
    """Create the three pre-specified candidate models from the notebook."""
    logistic_preprocessor = make_preprocessor(scale_numeric=True)
    tree_preprocessor = make_preprocessor(scale_numeric=False)

    return {
        "Logistic Regression": Pipeline(steps=[
            ("preprocessor", clone(logistic_preprocessor)),
            ("classifier", LogisticRegression(
                C=1.0, solver="lbfgs", max_iter=3000,
                class_weight="balanced", random_state=RANDOM_STATE,
            )),
        ]),
        "Random Forest": Pipeline(steps=[
            ("preprocessor", clone(tree_preprocessor)),
            ("classifier", RandomForestClassifier(
                n_estimators=300, max_depth=12, min_samples_leaf=5,
                max_features=0.7, class_weight="balanced",
                random_state=RANDOM_STATE, n_jobs=-1,
            )),
        ]),
        "Gradient Boosting": Pipeline(steps=[
            ("preprocessor", clone(tree_preprocessor)),
            ("classifier", GradientBoostingClassifier(
                n_estimators=150, learning_rate=0.03, max_depth=2,
                min_samples_leaf=10, subsample=0.8, random_state=RANDOM_STATE,
            )),
        ]),
    }


def evaluate_binary_classifier(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.50,
) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Evaluation requires both target classes (0 and 1).")
    if not np.isfinite(probabilities).all():
        raise ValueError("Predicted probabilities contain non-finite values.")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Predicted probabilities must be between zero and one.")

    predictions = (probabilities >= threshold).astype(int)
    return {
        "Threshold": float(threshold),
        "Accuracy": float(accuracy_score(y, predictions)),
        "Balanced_Accuracy": float(balanced_accuracy_score(y, predictions)),
        "Precision": float(precision_score(y, predictions, zero_division=0)),
        "Recall": float(recall_score(y, predictions, zero_division=0)),
        "F1": float(f1_score(y, predictions, zero_division=0)),
        "ROC_AUC": float(roc_auc_score(y, probabilities)),
        "PR_AUC_Average_Precision": float(average_precision_score(y, probabilities)),
        "Brier_Score": float(brier_score_loss(y, probabilities)),
        "Log_Loss": float(log_loss(y, probabilities, labels=[0, 1])),
    }


def _fit_model(model: Pipeline, model_name: str, X: pd.DataFrame, y: pd.Series) -> Pipeline:
    fitted = clone(model)
    if model_name == "Gradient Boosting":
        weights = compute_sample_weight(class_weight="balanced", y=y)
        fitted.fit(X, y, classifier__sample_weight=weights)
    else:
        fitted.fit(X, y)
    return fitted


def _fit_sigmoid_calibrator(base_model: Pipeline, X_cal: pd.DataFrame, y_cal: pd.Series):
    """Fit sigmoid calibration on a separate period while freezing the base model."""
    try:
        from sklearn.frozen import FrozenEstimator
        frozen = FrozenEstimator(base_model)
        calibrator = CalibratedClassifierCV(estimator=frozen, method="sigmoid")
        calibration_api = "FrozenEstimator"
    except ImportError:  # Compatibility with earlier scikit-learn versions.
        calibrator = CalibratedClassifierCV(
            estimator=base_model, method="sigmoid", cv="prefit"
        )
        calibration_api = "cv='prefit' compatibility mode"
    calibrator.fit(X_cal, y_cal)
    return calibrator, calibration_api


def build_temporal_partitions(classification_data: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create three expanding validation folds and final train/calibration/test sets."""
    train = classification_data.loc[
        classification_data["Next_Year"] <= max(TEMPORAL_CV_YEARS)
    ].copy()
    calibration = classification_data.loc[
        classification_data["Next_Year"] == CALIBRATION_OUTCOME_YEAR
    ].copy()
    test = classification_data.loc[
        classification_data["Next_Year"].isin(TEST_OUTCOME_YEARS)
    ].copy()

    folds = {}
    fold_rows = []
    for validation_year in TEMPORAL_CV_YEARS:
        fold_train = classification_data.loc[
            classification_data["Next_Year"] < validation_year
        ].copy()
        fold_validation = classification_data.loc[
            classification_data["Next_Year"] == validation_year
        ].copy()
        if fold_train.empty or fold_validation.empty:
            raise ValueError(f"Temporal fold {validation_year} has an empty partition.")
        for part_name, part in [("training", fold_train), ("validation", fold_validation)]:
            if part[CLASSIFICATION_TARGET].nunique() != 2:
                raise ValueError(
                    f"Temporal fold {validation_year} {part_name} partition does not contain both classes."
                )
        if fold_train["Next_Year"].max() >= fold_validation["Next_Year"].min():
            raise AssertionError("Temporal leakage detected in a validation fold.")
        folds[validation_year] = (fold_train, fold_validation)
        fold_rows.append({
            "Validation_Outcome_Year": validation_year,
            "Training_Observations": len(fold_train),
            "Training_Outcome_Year_Min": int(fold_train["Next_Year"].min()),
            "Training_Outcome_Year_Max": int(fold_train["Next_Year"].max()),
            "Training_Positive_Count": int(fold_train[CLASSIFICATION_TARGET].sum()),
            "Validation_Observations": len(fold_validation),
            "Validation_Positive_Count": int(fold_validation[CLASSIFICATION_TARGET].sum()),
            "Validation_Negative_Count": int((fold_validation[CLASSIFICATION_TARGET] == 0).sum()),
            "Validation_Positive_Rate": float(fold_validation[CLASSIFICATION_TARGET].mean()),
        })

    for name, part in [("final training", train), ("calibration", calibration), ("test", test)]:
        if part.empty or part[CLASSIFICATION_TARGET].nunique() != 2:
            raise ValueError(f"The {name} partition is empty or does not contain both classes.")

    if not (train["Next_Year"].max() < calibration["Next_Year"].min()
            < test["Next_Year"].min()):
        raise AssertionError("Final train, calibration and test periods are not chronologically separated.")

    fold_summary = pd.DataFrame(fold_rows)
    return folds, fold_summary, train, calibration, test


def run_modeling(classification_data: pd.DataFrame) -> ModelingResult:
    """Run temporal validation, model selection, calibration and final evaluation."""
    folds, fold_summary, train_data, calibration_data, test_data = build_temporal_partitions(
        classification_data
    )

    X_train = train_data[CLASSIFICATION_FEATURES].copy()
    y_train = train_data[CLASSIFICATION_TARGET].astype(int).copy()
    X_cal = calibration_data[CLASSIFICATION_FEATURES].copy()
    y_cal = calibration_data[CLASSIFICATION_TARGET].astype(int).copy()
    X_test = test_data[CLASSIFICATION_FEATURES].copy()
    y_test = test_data[CLASSIFICATION_TARGET].astype(int).copy()

    candidate_models = make_candidate_models()
    fold_rows: list[dict[str, Any]] = []

    # Expanding-window validation: each fold uses only earlier outcome years.
    for validation_year, (fold_train, fold_validation) in folds.items():
        X_fold_train = fold_train[CLASSIFICATION_FEATURES]
        y_fold_train = fold_train[CLASSIFICATION_TARGET].astype(int)
        X_fold_validation = fold_validation[CLASSIFICATION_FEATURES]
        y_fold_validation = fold_validation[CLASSIFICATION_TARGET].astype(int)

        dummy = DummyClassifier(strategy="prior")
        dummy.fit(X_fold_train, y_fold_train)
        dummy_prob = dummy.predict_proba(X_fold_validation)[:, 1]
        fold_rows.append({
            "Model": "Dummy Classifier",
            "Validation_Outcome_Year": validation_year,
            "Training_Observations": len(fold_train),
            "Validation_Observations": len(fold_validation),
            "Validation_Positive_Rate": float(y_fold_validation.mean()),
            **evaluate_binary_classifier(y_fold_validation, dummy_prob),
        })

        for model_name, template in candidate_models.items():
            fitted_fold_model = _fit_model(template, model_name, X_fold_train, y_fold_train)
            probabilities = fitted_fold_model.predict_proba(X_fold_validation)[:, 1]
            fold_rows.append({
                "Model": model_name,
                "Validation_Outcome_Year": validation_year,
                "Training_Observations": len(fold_train),
                "Validation_Observations": len(fold_validation),
                "Validation_Positive_Rate": float(y_fold_validation.mean()),
                **evaluate_binary_classifier(y_fold_validation, probabilities),
            })

    cv_results = pd.DataFrame(fold_rows).sort_values(
        ["Model", "Validation_Outcome_Year"]
    ).reset_index(drop=True)
    if not cv_results.groupby("Model")["Validation_Outcome_Year"].nunique().eq(3).all():
        raise AssertionError("Not every model was evaluated in all three temporal folds.")

    cv_summary = (
        cv_results.groupby("Model")
        .agg(
            Validation_Folds=("Validation_Outcome_Year", "nunique"),
            Mean_PR_AUC=("PR_AUC_Average_Precision", "mean"),
            SD_PR_AUC=("PR_AUC_Average_Precision", "std"),
            Mean_ROC_AUC=("ROC_AUC", "mean"),
            SD_ROC_AUC=("ROC_AUC", "std"),
            Mean_Balanced_Accuracy=("Balanced_Accuracy", "mean"),
            Mean_Recall_at_0_50=("Recall", "mean"),
            Mean_Precision_at_0_50=("Precision", "mean"),
        )
        .reset_index()
        .sort_values("Mean_PR_AUC", ascending=False)
        .reset_index(drop=True)
    )
    candidate_summary = cv_summary.loc[cv_summary["Model"].isin(candidate_models)].copy()
    selected_model_name = str(candidate_summary.iloc[0]["Model"])

    # Refit candidate models on all outcomes through 2022, preserving the 2023
    # calibration and 2024–2025 final test periods as separate later periods.
    fitted_models = {
        name: _fit_model(template, name, X_train, y_train)
        for name, template in candidate_models.items()
    }
    selected_base_model = fitted_models[selected_model_name]

    calibrated_model, calibration_api = _fit_sigmoid_calibrator(
        selected_base_model, X_cal, y_cal
    )

    baseline = DummyClassifier(strategy="prior")
    baseline.fit(X_train, y_train)

    # Final test results are generated only after all model selection and
    # calibration decisions have been completed.
    test_probabilities: dict[str, np.ndarray] = {}
    test_predictions: dict[str, np.ndarray] = {}
    test_rows = []
    for name, model in fitted_models.items():
        probability = model.predict_proba(X_test)[:, 1]
        prediction = (probability >= 0.50).astype(int)
        test_probabilities[name] = probability
        test_predictions[name] = prediction
        test_rows.append({
            "Model": name,
            "Probability_Version": "Uncalibrated",
            **evaluate_binary_classifier(y_test, probability, threshold=0.50),
        })

    dummy_probability = baseline.predict_proba(X_test)[:, 1]
    test_probabilities["Dummy Classifier"] = dummy_probability
    test_predictions["Dummy Classifier"] = (dummy_probability >= 0.50).astype(int)
    test_rows.append({
        "Model": "Dummy Classifier",
        "Probability_Version": "Prior baseline",
        **evaluate_binary_classifier(y_test, dummy_probability, threshold=0.50),
    })

    calibrated_probability = calibrated_model.predict_proba(X_test)[:, 1]
    calibrated_prediction = (calibrated_probability >= CALIBRATED_DECISION_THRESHOLD).astype(int)
    calibrated_label = f"{selected_model_name} (sigmoid-calibrated)"
    test_probabilities[calibrated_label] = calibrated_probability
    test_predictions[calibrated_label] = calibrated_prediction
    test_rows.append({
        "Model": selected_model_name,
        "Probability_Version": "Sigmoid-calibrated",
        **evaluate_binary_classifier(
            y_test, calibrated_probability,
            threshold=CALIBRATED_DECISION_THRESHOLD,
        ),
    })
    heldout_test_results = pd.DataFrame(test_rows).sort_values(
        ["Probability_Version", "PR_AUC_Average_Precision"],
        ascending=[True, False],
    ).reset_index(drop=True)

    confusion_rows = []
    for name, prediction in test_predictions.items():
        tn, fp, fn, tp = confusion_matrix(y_test, prediction, labels=[0, 1]).ravel()
        confusion_rows.append({
            "Model": name,
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "Sensitivity_Recall": tp / (tp + fn) if tp + fn else np.nan,
            "Specificity": tn / (tn + fp) if tn + fp else np.nan,
            "False_Positive_Rate": fp / (fp + tn) if fp + tn else np.nan,
            "False_Negative_Rate": fn / (fn + tp) if fn + tp else np.nan,
        })
    confusion_summary = pd.DataFrame(confusion_rows)

    raw_selected_probability = test_probabilities[selected_model_name]
    calibration_results = pd.DataFrame([
        {
            "Probability_Model": "Selected model before calibration",
            "Brier_Score": brier_score_loss(y_test, raw_selected_probability),
            "Log_Loss": log_loss(y_test, raw_selected_probability, labels=[0, 1]),
        },
        {
            "Probability_Model": "Selected model after sigmoid calibration",
            "Brier_Score": brier_score_loss(y_test, calibrated_probability),
            "Log_Loss": log_loss(y_test, calibrated_probability, labels=[0, 1]),
        },
    ])

    # Sensitivity analysis mirrors the notebook: Logistic Regression is used
    # for each alternative materiality definition, calibrated on 2023, assessed
    # on 2024–2025. These results do not alter the primary selected model.
    sensitivity_rows = []
    sensitivity_template = candidate_models["Logistic Regression"]
    for cutoff in SENSITIVITY_THRESHOLDS:
        y_train_alt = (train_data[MAXIMUM_CHANGE_COLUMN] >= cutoff).astype(int)
        y_cal_alt = (calibration_data[MAXIMUM_CHANGE_COLUMN] >= cutoff).astype(int)
        y_test_alt = (test_data[MAXIMUM_CHANGE_COLUMN] >= cutoff).astype(int)
        if any(y.nunique() < 2 for y in [y_train_alt, y_cal_alt, y_test_alt]):
            sensitivity_rows.append({
                "Materiality_Cutoff": cutoff,
                "Status": "Not evaluated: a partition contains only one class",
            })
            continue

        sensitivity_model = _fit_model(
            sensitivity_template, "Logistic Regression", X_train, y_train_alt
        )
        sensitivity_calibrator, _ = _fit_sigmoid_calibrator(
            sensitivity_model, X_cal, y_cal_alt
        )
        sensitivity_probability = sensitivity_calibrator.predict_proba(X_test)[:, 1]
        sensitivity_metrics = evaluate_binary_classifier(
            y_test_alt, sensitivity_probability,
            threshold=CALIBRATED_DECISION_THRESHOLD,
        )
        tn, fp, fn, tp = confusion_matrix(
            y_test_alt,
            (sensitivity_probability >= CALIBRATED_DECISION_THRESHOLD).astype(int),
            labels=[0, 1],
        ).ravel()
        sensitivity_rows.append({
            "Materiality_Cutoff": cutoff,
            "Status": "Evaluated",
            "Training_Positive_Rate": float(y_train_alt.mean()),
            "Calibration_Positive_Rate": float(y_cal_alt.mean()),
            "Test_Positive_Rate": float(y_test_alt.mean()),
            "Decision_Threshold": CALIBRATED_DECISION_THRESHOLD,
            "Test_Balanced_Accuracy": sensitivity_metrics["Balanced_Accuracy"],
            "Test_Precision": sensitivity_metrics["Precision"],
            "Test_Recall": sensitivity_metrics["Recall"],
            "Test_F1": sensitivity_metrics["F1"],
            "Test_ROC_AUC": sensitivity_metrics["ROC_AUC"],
            "Test_PR_AUC_Average_Precision": sensitivity_metrics["PR_AUC_Average_Precision"],
            "Test_Brier_Score": sensitivity_metrics["Brier_Score"],
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        })
    sensitivity_results = pd.DataFrame(sensitivity_rows)

    # Assemble a compact, non-identifying test-prediction file for audit. The
    # scheme reference is intentionally omitted from this export.
    test_predictions_frame = pd.DataFrame({
        "Reporting_Year": test_data["Year"].to_numpy(),
        "Outcome_Year": test_data["Next_Year"].to_numpy(),
        "Actual_Material_Change": y_test.to_numpy(),
        "Predicted_Probability": calibrated_probability,
        "Predicted_Class": calibrated_prediction,
    })

    return ModelingResult(
        temporal_cv_results=cv_results,
        temporal_cv_summary=cv_summary,
        selected_model_name=selected_model_name,
        fitted_models=fitted_models,
        selected_base_model=selected_base_model,
        calibrated_model=calibrated_model,
        baseline_model=baseline,
        train_data=train_data,
        calibration_data=calibration_data,
        test_data=test_data,
        X_train=X_train, y_train=y_train,
        X_calibration=X_cal, y_calibration=y_cal,
        X_test=X_test, y_test=y_test,
        heldout_test_results=heldout_test_results,
        confusion_summary=confusion_summary,
        calibration_results=calibration_results,
        sensitivity_results=sensitivity_results,
        test_predictions=test_predictions_frame,
    )
