"""Transfer evaluation utilities for Phase 3 experiments."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Any

import numpy as np
import pandas as pd

from urban_tree_transfer.experiments.evaluation import (
    bootstrap_confidence_interval,
    compute_confusion_matrix,
    compute_metrics,
    compute_per_class_metrics,
)


@dataclass(frozen=True)
class TransferGap:
    """Transfer gap summary."""

    source_f1: float
    target_f1: float
    absolute_drop: float
    relative_drop: float


def compute_transfer_gap(source_f1: float, target_f1: float) -> TransferGap:
    """Compute absolute and relative transfer gap.

    Args:
        source_f1: F1 on source domain (Berlin).
        target_f1: F1 on target domain (Leipzig).

    Returns:
        TransferGap dataclass.
    """
    absolute_drop = float(source_f1 - target_f1)
    relative_drop = 0.0 if source_f1 <= 0 else float(absolute_drop / source_f1)
    return TransferGap(
        source_f1=float(source_f1),
        target_f1=float(target_f1),
        absolute_drop=absolute_drop,
        relative_drop=relative_drop,
    )


def compute_transfer_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: list[str],
    average: str = "weighted",
    include_per_class: bool = True,
    include_confusion: bool = True,
    include_ci: bool = False,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Compute transfer evaluation metrics for a model.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        class_labels: List of class labels in encoded order.
        average: Averaging strategy for aggregate metrics.
        include_per_class: Include per-class metrics table.
        include_confusion: Include confusion matrix.
        include_ci: Include bootstrap confidence intervals for primary metrics.
        n_bootstrap: Bootstrap iterations.
        confidence_level: Confidence level for CIs.

    Returns:
        Dict with metrics, per_class, confusion_matrix, confidence_intervals.
    """
    metrics = compute_metrics(y_true, y_pred, average=average)

    result: dict[str, Any] = {"metrics": metrics}

    if include_per_class:
        per_class = compute_per_class_metrics(y_true, y_pred, class_labels)
        result["per_class"] = per_class

    if include_confusion:
        cm = compute_confusion_matrix(y_true, y_pred, labels=list(range(len(class_labels))))
        result["confusion_matrix"] = cm

    if include_ci:
        ci: dict[str, tuple[float, float, float]] = {}
        for key in ("f1_score", "precision", "recall", "accuracy"):
            metric_fn = _metric_fn_factory(key, average)
            ci[key] = bootstrap_confidence_interval(
                y_true,
                y_pred,
                metric_fn,
                n_bootstrap=n_bootstrap,
                confidence_level=confidence_level,
            )
        result["confidence_intervals"] = ci

    return result


def _metric_fn_factory(metric_name: str, average: str) -> Any:
    def _metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        metrics = compute_metrics(y_true, y_pred, average=average)
        return float(metrics[metric_name])

    return _metric


def compute_feature_stability(
    source_importance: pd.DataFrame,
    target_importance: pd.DataFrame,
    feature_col: str = "feature",
    importance_col: str = "importance",
) -> dict[str, float]:
    """Compute Spearman correlation between feature importances.

    Args:
        source_importance: DataFrame with feature importance for source domain.
        target_importance: DataFrame with feature importance for target domain.
        feature_col: Feature column name.
        importance_col: Importance column name.

    Returns:
        Dict with spearman_rho and n_features.
    """
    merged = pd.merge(
        source_importance[[feature_col, importance_col]],
        target_importance[[feature_col, importance_col]],
        on=feature_col,
        suffixes=("_source", "_target"),
    )
    if merged.empty:
        raise ValueError("No overlapping features between source and target.")

    source_rank = merged[f"{importance_col}_source"].rank(method="average")
    target_rank = merged[f"{importance_col}_target"].rank(method="average")

    if source_rank.std() == 0 or target_rank.std() == 0:
        rho = 0.0
    else:
        rho = float(np.corrcoef(source_rank, target_rank)[0, 1])

    return {"spearman_rho": rho, "n_features": float(len(merged))}


def classify_transfer_robustness(
    source_per_genus: dict[str, float],
    target_per_genus: dict[str, float],
    robust_threshold: float = 0.05,
    medium_threshold: float = 0.15,
) -> dict[str, dict[str, Any]]:
    """Classify per-genus transfer robustness based on F1 drops.

    Args:
        source_per_genus: F1 scores in source domain by genus.
        target_per_genus: F1 scores in target domain by genus.
        robust_threshold: Relative drop threshold for robust.
        medium_threshold: Relative drop threshold for medium.

    Returns:
        Mapping genus -> {source_f1, target_f1, drop, relative_drop, label}.
    """
    results: dict[str, dict[str, Any]] = {}
    for genus, source_f1 in source_per_genus.items():
        target_f1 = float(target_per_genus.get(genus, 0.0))
        gap = compute_transfer_gap(float(source_f1), target_f1)

        if gap.relative_drop <= robust_threshold:
            label = "robust"
        elif gap.relative_drop <= medium_threshold:
            label = "medium"
        else:
            label = "poor"

        results[genus] = {
            "source_f1": gap.source_f1,
            "target_f1": gap.target_f1,
            "absolute_drop": gap.absolute_drop,
            "relative_drop": gap.relative_drop,
            "label": label,
        }

    return results


def compute_transfer_robustness_ranking(
    transfer_data: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Rank genera by transfer robustness.

    Args:
        transfer_data: Output of classify_transfer_robustness.

    Returns:
        DataFrame sorted by relative_drop ascending.
    """
    rows = []
    for genus, metrics in transfer_data.items():
        rows.append({"genus": genus, **metrics})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("relative_drop", ascending=True).reset_index(drop=True)


def mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
) -> dict[str, float]:
    """McNemar exact test for paired classifier comparison.

    Args:
        y_true: True labels.
        y_pred_a: Predictions from model A.
        y_pred_b: Predictions from model B.

    Returns:
        Dict with b, c, and p_value.
    """
    if y_true.shape != y_pred_a.shape or y_true.shape != y_pred_b.shape:
        raise ValueError("y_true and predictions must have same shape.")

    correct_a = y_pred_a == y_true
    correct_b = y_pred_b == y_true

    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))

    n = b + c
    if n == 0:
        p_value = 1.0
    else:
        p_value = 2.0 * _binom_cdf(min(b, c), n, 0.5)
        p_value = min(p_value, 1.0)

    return {"b": float(b), "c": float(c), "p_value": float(p_value)}


def _binom_cdf(k: int, n: int, p: float) -> float:
    total = 0.0
    for i in range(k + 1):
        total += comb(n, i) * (p**i) * ((1 - p) ** (n - i))
    return total


def summarize_hypotheses(
    hypotheses: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    """Evaluate hypothesis list against computed metrics.

    Args:
        hypotheses: List of hypothesis definitions from config.
        metrics: Metric mapping with keys used by hypotheses.

    Returns:
        List of hypothesis results with pass/fail and observed values.
    """
    results: list[dict[str, Any]] = []
    for hypothesis in hypotheses:
        metric_key = hypothesis.get("metric")
        metric_name = metric_key if isinstance(metric_key, str) else None
        direction = hypothesis.get("direction", "greater")
        threshold = hypothesis.get("threshold")
        observed = metrics.get(metric_name) if metric_name is not None else None

        passed = None
        if observed is not None and threshold is not None:
            if direction == "greater":
                passed = bool(observed > threshold)
            elif direction == "less":
                passed = bool(observed < threshold)
            else:
                passed = bool(observed == threshold)

        results.append(
            {
                "id": hypothesis.get("id"),
                "description": hypothesis.get("description"),
                "metric": metric_name,
                "observed": observed,
                "threshold": threshold,
                "direction": direction,
                "passed": passed,
            }
        )

    return results
