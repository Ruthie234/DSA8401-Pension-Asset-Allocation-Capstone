"""Lightweight unit tests for stable, data-independent pipeline components."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pension_capstone.config import (  # noqa: E402
    CLASSIFICATION_FEATURES,
    MATERIAL_CHANGE_THRESHOLD,
    TEMPORAL_CV_YEARS,
)
from pension_capstone.modeling import evaluate_binary_classifier  # noqa: E402


def test_feature_manifest_shape():
    assert len(CLASSIFICATION_FEATURES) == 26
    assert len(set(CLASSIFICATION_FEATURES)) == 26


def test_proposal_target_definition_and_cv_years():
    assert MATERIAL_CHANGE_THRESHOLD == 0.05
    assert TEMPORAL_CV_YEARS == [2020, 2021, 2022]


def test_binary_metrics_are_computed():
    y_true = [0, 0, 1, 1]
    probabilities = [0.1, 0.4, 0.6, 0.9]
    metrics = evaluate_binary_classifier(y_true, probabilities, threshold=0.5)
    assert metrics["Balanced_Accuracy"] == 1.0
    assert metrics["Precision"] == 1.0
    assert metrics["Recall"] == 1.0
    assert metrics["PR_AUC_Average_Precision"] == 1.0
