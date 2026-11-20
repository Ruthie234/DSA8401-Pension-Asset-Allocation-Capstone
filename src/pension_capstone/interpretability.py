"""Optional SHAP analysis for the selected uncalibrated base estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd


def run_shap_analysis(model, X_train: pd.DataFrame, X_test: pd.DataFrame,
                      output_dir, random_state: int = 42,
                      background_size: int = 100, explanation_size: int = 250):
    """Compute SHAP importance and plots for the selected base estimator.

    SHAP explains the fitted base estimator before the separate sigmoid
    calibration layer, as described in the project notebook.
    """
    try:
        import shap
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "SHAP is not installed. Install it with `python -m pip install shap`."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    background_raw = X_train.sample(
        n=min(background_size, len(X_train)), random_state=random_state
    )
    explain_raw = X_test.sample(
        n=min(explanation_size, len(X_test)), random_state=random_state
    )
    background_transformed = preprocessor.transform(background_raw)
    explain_transformed = preprocessor.transform(explain_raw)
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(classifier, "coef_"):
        explainer = shap.LinearExplainer(classifier, background_transformed)
    else:
        explainer = shap.TreeExplainer(classifier)
    raw_values = explainer.shap_values(explain_transformed)

    if isinstance(raw_values, list):
        values = np.asarray(raw_values[1])
    else:
        values = np.asarray(raw_values)
        if values.ndim == 3 and values.shape[-1] == 2:
            values = values[:, :, 1]
        elif values.ndim == 3 and values.shape[0] == 2:
            values = values[1]

    if values.ndim != 2 or values.shape[1] != len(feature_names):
        raise ValueError(
            f"Unexpected SHAP shape {values.shape}; expected (n, {len(feature_names)})."
        )

    transformed_frame = pd.DataFrame(
        np.asarray(explain_transformed),
        columns=feature_names,
        index=explain_raw.index,
    )
    importance = pd.DataFrame({
        "Feature": feature_names,
        "Mean_Absolute_SHAP": np.abs(values).mean(axis=0),
    }).sort_values("Mean_Absolute_SHAP", ascending=False).reset_index(drop=True)
    importance.to_csv(output_dir / "shap_feature_importance.csv", index=False)

    shap.summary_plot(values, transformed_frame, plot_type="bar", max_display=20, show=False)
    plt.title("Global SHAP importance: selected base estimator")
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary_bar.png", dpi=180, bbox_inches="tight")
    plt.close()

    shap.summary_plot(values, transformed_frame, max_display=20, show=False)
    plt.title("SHAP summary: selected base estimator")
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary_beeswarm.png", dpi=180, bbox_inches="tight")
    plt.close()

    return importance
