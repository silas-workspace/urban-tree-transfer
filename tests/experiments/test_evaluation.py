"""Unit tests for Phase 3 evaluation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from urban_tree_transfer.experiments.evaluation import (
    bootstrap_confidence_interval,
    compute_cohens_d,
    compute_metrics,
    compute_per_class_metrics,
)


def test_compute_metrics_matches_sklearn() -> None:
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])

    metrics = compute_metrics(y_true, y_pred, average="weighted")

    assert metrics["accuracy"] == accuracy_score(y_true, y_pred)
    assert metrics["f1_score"] == f1_score(y_true, y_pred, average="weighted", zero_division=0)
    assert metrics["precision"] == precision_score(
        y_true, y_pred, average="weighted", zero_division=0
    )
    assert metrics["recall"] == recall_score(y_true, y_pred, average="weighted", zero_division=0)


def test_compute_per_class_metrics() -> None:
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    class_names = ["ACER", "TILIA"]

    df = compute_per_class_metrics(y_true, y_pred, class_names)

    assert isinstance(df, pd.DataFrame)
    assert list(df["genus"]) == class_names
    acer = df.iloc[0]
    tili = df.iloc[1]
    assert acer["support"] == 2
    assert tili["support"] == 2
    assert np.isclose(acer["precision"], 1.0)
    assert np.isclose(acer["recall"], 0.5)
    assert np.isclose(tili["precision"], 2 / 3)
    assert np.isclose(tili["recall"], 1.0)


def test_bootstrap_confidence_interval_bounds() -> None:
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 1])

    def metric_fn(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return accuracy_score(y_true, y_pred)

    point, lower, upper = bootstrap_confidence_interval(
        y_true, y_pred, metric_fn, n_bootstrap=200, confidence_level=0.9, random_seed=7
    )

    assert 0.0 <= lower <= upper <= 1.0
    assert lower <= point <= upper


def test_bootstrap_confidence_interval_invalid_inputs() -> None:
    y_true = np.array([])
    y_pred = np.array([])

    def metric_fn(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return accuracy_score(y_true, y_pred)

    with pytest.raises(ValueError, match="y_true is empty"):
        bootstrap_confidence_interval(y_true, y_pred, metric_fn)


def test_compute_cohens_d_equal_means() -> None:
    """Test Cohen's d when two groups have identical means."""
    values_a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    values_b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    d = compute_cohens_d(values_a, values_b)

    assert np.isclose(d, 0.0, atol=1e-10), "Cohen's d should be 0 for identical distributions"


def test_compute_cohens_d_positive_effect() -> None:
    """Test Cohen's d when group A has higher mean than group B."""
    values_a = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
    values_b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    d = compute_cohens_d(values_a, values_b)

    # Expected: mean_a=7, mean_b=3, diff=4, pooled_std≈2.24, d≈1.79
    assert d > 0.8, "Cohen's d should indicate large effect (|d| > 0.8)"
    assert d > 0, "Cohen's d should be positive when values_a > values_b"


def test_compute_cohens_d_negative_effect() -> None:
    """Test Cohen's d when group A has lower mean than group B."""
    values_a = np.array([1.0, 2.0, 3.0])
    values_b = np.array([5.0, 6.0, 7.0])

    d = compute_cohens_d(values_a, values_b)

    assert d < 0, "Cohen's d should be negative when values_a < values_b"
    assert abs(d) > 0.8, "Cohen's d should indicate large effect"


def test_compute_cohens_d_invalid_inputs() -> None:
    """Test Cohen's d error handling for invalid inputs."""
    # Empty arrays
    with pytest.raises(ValueError, match="Both samples must be non-empty"):
        compute_cohens_d(np.array([]), np.array([1.0, 2.0]))

    # Insufficient samples
    with pytest.raises(ValueError, match="Each sample must have at least 2 values"):
        compute_cohens_d(np.array([1.0]), np.array([2.0, 3.0]))

    # Zero variance (constant values)
    with pytest.raises(ValueError, match="Pooled standard deviation is zero"):
        compute_cohens_d(np.array([5.0, 5.0, 5.0]), np.array([5.0, 5.0, 5.0]))
