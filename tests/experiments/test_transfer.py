"""Unit tests for transfer evaluation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from urban_tree_transfer.experiments import transfer


def test_compute_transfer_gap() -> None:
    gap = transfer.compute_transfer_gap(0.7, 0.56)
    assert gap.absolute_drop == pytest.approx(0.14, abs=1e-10)
    assert gap.relative_drop > 0.0


def test_classify_transfer_robustness() -> None:
    source = {"TILIA": 0.8, "ACER": 0.6}
    target = {"TILIA": 0.78, "ACER": 0.45}
    result = transfer.classify_transfer_robustness(
        source, target, robust_threshold=0.05, medium_threshold=0.2
    )
    assert result["TILIA"]["label"] == "robust"
    assert result["ACER"]["label"] in {"medium", "poor"}


def test_compute_transfer_metrics(class_labels: list[str]) -> None:
    y_true = np.array([0, 1, 2, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 2])
    result = transfer.compute_transfer_metrics(y_true, y_pred, class_labels)
    assert "metrics" in result
    assert "f1_score" in result["metrics"]


def test_compute_feature_stability() -> None:
    source = pd.DataFrame({"feature": ["A", "B", "C"], "importance": [0.3, 0.2, 0.1]})
    target = pd.DataFrame({"feature": ["A", "B", "C"], "importance": [0.31, 0.19, 0.11]})
    result = transfer.compute_feature_stability(source, target)
    assert "spearman_rho" in result
    assert result["n_features"] == 3


def test_mcnemar_test() -> None:
    y_true = np.array([0, 1, 0, 1])
    y_pred_a = np.array([0, 1, 1, 1])
    y_pred_b = np.array([0, 0, 0, 1])
    result = transfer.mcnemar_test(y_true, y_pred_a, y_pred_b)
    assert "p_value" in result
    assert 0.0 <= result["p_value"] <= 1.0


def test_hypothesis_kruskal() -> None:
    genus_data = pd.DataFrame(
        {
            "genus": ["A"] * 20 + ["B"] * 20 + ["C"] * 20 + ["D"] * 20 + ["E"] * 20,
            "correct": (
                [1] * 18
                + [0] * 2
                + [1] * 14
                + [0] * 6
                + [1] * 10
                + [0] * 10
                + [1] * 8
                + [0] * 12
                + [1] * 4
                + [0] * 16
            ),
        }
    )

    result = transfer.test_hypothesis(
        {
            "id": "H1",
            "description": "Transfer loss varies significantly across genera",
            "test_type": "kruskal",
            "metric_variable": "correct",
            "group_variable": "genus",
        },
        genus_data=genus_data,
    )

    assert "statistic" in result
    assert "p_value" in result
    assert "effect_size" in result
    assert "conclusion" in result


def test_hypothesis_effect_sizes() -> None:
    mann_whitney_data = pd.DataFrame(
        {
            "is_conifer": [True, True, True, False, False, False],
            "transfer_gap": [0.05, 0.08, 0.07, 0.15, 0.17, 0.19],
        }
    )
    mann_whitney_result = transfer.test_hypothesis(
        {
            "id": "H2",
            "test_type": "mann_whitney",
            "group_variable": "is_conifer",
            "metric_variable": "transfer_gap",
            "group1_value": True,
            "group2_value": False,
        },
        genus_data=mann_whitney_data,
    )

    spearman_data = pd.DataFrame(
        {
            "mean_jm_distance": [0.4, 0.8, 1.2, 1.6, 1.9],
            "leipzig_f1": [0.45, 0.55, 0.62, 0.74, 0.81],
        }
    )
    spearman_result = transfer.test_hypothesis(
        {
            "id": "H3",
            "test_type": "spearman",
            "x_variable": "mean_jm_distance",
            "y_variable": "leipzig_f1",
        },
        genus_data=spearman_data,
    )

    assert mann_whitney_result["effect_size"] == pytest.approx(-1.0)
    assert "effect_size" in spearman_result


def test_hypothesis_mann_whitney_missing_metric_column() -> None:
    genus_data = pd.DataFrame({"is_conifer": [True, False, True, False]})

    result = transfer.test_hypothesis(
        {
            "id": "H2",
            "test_type": "mann_whitney",
            "group_variable": "is_conifer",
            "metric_variable": "transfer_gap",
            "group1_value": True,
            "group2_value": False,
        },
        genus_data=genus_data,
    )

    assert result["conclusion"] == "Missing required columns: is_conifer, transfer_gap"
