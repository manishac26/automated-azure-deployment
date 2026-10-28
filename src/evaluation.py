"""Shared metric, threshold, serialization, and plotting helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src import config

LOGGER = logging.getLogger(__name__)


def binary_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute binary classification metrics with ATTACK as the positive class."""

    truth = np.asarray(y_true, dtype=np.uint8).ravel()
    probabilities = np.asarray(y_prob, dtype=np.float64).ravel()
    if truth.shape != probabilities.shape:
        raise ValueError("y_true and y_prob must have the same shape.")
    if not np.isfinite(probabilities).all():
        raise ValueError("Predicted probabilities contain non-finite values.")

    predictions = (probabilities >= threshold).astype(np.uint8)
    tn, fp, fn, tp = confusion_matrix(truth, predictions, labels=[0, 1]).ravel()

    metrics: dict[str, Any] = {
        "threshold": round(float(threshold), 6),
        "n": int(truth.size),
        "positives": int((truth == 1).sum()),
        "negatives": int((truth == 0).sum()),
        "predicted_positive": int(predictions.sum()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": float(accuracy_score(truth, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predictions)),
        "precision": float(
            precision_score(truth, predictions, zero_division=0)
        ),
        "recall": float(recall_score(truth, predictions, zero_division=0)),
        "f1": float(f1_score(truth, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "mcc": float(matthews_corrcoef(truth, predictions))
        if len(np.unique(truth)) > 1
        else 0.0,
        "brier": float(brier_score_loss(truth, probabilities)),
    }
    if len(np.unique(truth)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(truth, probabilities))
        metrics["pr_auc"] = float(average_precision_score(truth, probabilities))
        metrics["log_loss"] = float(log_loss(truth, probabilities, labels=[0, 1]))
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
        metrics["log_loss"] = None

    for key, value in metrics.items():
        if isinstance(value, float):
            metrics[key] = round(value, 6)
    return metrics


def tune_threshold(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
) -> tuple[float, dict[str, Any]]:
    """Choose the validation threshold that maximises F1.

    Ties are resolved by the threshold closest to 0.5, which avoids choosing an
    extreme cutoff when several thresholds produce the same validation F1.
    """

    truth = np.asarray(y_true, dtype=np.uint8).ravel()
    probabilities = np.asarray(y_prob, dtype=np.float64).ravel()
    precision, recall, thresholds = precision_recall_curve(truth, probabilities)
    f1_values = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision, dtype=np.float64),
        where=(precision + recall) > 0,
    )
    if thresholds.size == 0:
        return 0.5, binary_metrics(truth, probabilities, 0.5)
    candidate_scores = f1_values[:-1]
    best = np.flatnonzero(np.isclose(candidate_scores, candidate_scores.max()))
    best_threshold = float(
        thresholds[best[np.argmin(np.abs(thresholds[best] - 0.5))]]
    )
    return best_threshold, binary_metrics(truth, probabilities, best_threshold)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON document without NumPy scalar types."""

    path.parent.mkdir(parents=True, exist_ok=True)

    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return clean(value.tolist())
        return value

    with path.open("w", encoding="utf-8") as handle:
        json.dump(clean(payload), handle, indent=2, ensure_ascii=False)
    LOGGER.info("Wrote %s", path)


def plot_confusion_matrix(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    title: str,
    path: Path,
) -> None:
    """Save a labelled confusion-matrix heatmap."""

    from sklearn.metrics import ConfusionMatrixDisplay

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix, display_labels=["BENIGN", "ATTACK"]
    )
    figure, axis = plt.subplots(figsize=(5.2, 4.6))
    display.plot(ax=axis, cmap="Blues", colorbar=False, values_format=",d")
    axis.set_title(title)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_roc_pr(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    title: str,
    path: Path,
) -> None:
    """Save side-by-side ROC and precision-recall curves."""

    truth = np.asarray(y_true, dtype=np.uint8).ravel()
    probabilities = np.asarray(y_prob, dtype=np.float64).ravel()
    fpr, tpr, _ = roc_curve(truth, probabilities)
    precision, recall, _ = precision_recall_curve(truth, probabilities)
    roc_auc = roc_auc_score(truth, probabilities)
    pr_auc = average_precision_score(truth, probabilities)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    axes[0].plot(fpr, tpr, color="#16537e", linewidth=2)
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title(f"ROC curve (AUC={roc_auc:.3f})")
    axes[0].grid(alpha=0.25)

    axes[1].plot(recall, precision, color="#9c2f2f", linewidth=2)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"Precision-recall curve (AP={pr_auc:.3f})")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1.02)
    axes[1].grid(alpha=0.25)

    figure.suptitle(title)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_feature_importance(
    importances: Sequence[float] | np.ndarray,
    feature_names: Sequence[str],
    title: str,
    path: Path,
    top_n: int = 20,
) -> None:
    """Save a horizontal bar chart of the strongest feature importances."""

    values = np.asarray(importances, dtype=np.float64).ravel()
    names = np.asarray(feature_names, dtype=object).ravel()
    if values.size != names.size:
        raise ValueError("Importances and feature names must have equal length.")
    order = np.argsort(values)[::-1][:top_n][::-1]

    figure, axis = plt.subplots(figsize=(7.4, max(3.6, 0.28 * len(order))))
    axis.barh(names[order], values[order], color="#3274a1")
    axis.set_xlabel("Importance")
    axis.set_title(title)
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_model_comparison(
    labels: Sequence[str],
    values: Sequence[float],
    title: str,
    xlabel: str,
    path: Path,
) -> None:
    """Save a simple model-comparison bar chart."""

    figure, axis = plt.subplots(figsize=(8.2, 4.4))
    bars = axis.bar(range(len(labels)), values, color="#3b7a57")
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=18, ha="right")
    axis.set_ylabel(xlabel)
    axis.set_title(title)
    axis.set_ylim(0, min(1.0, max(values) * 1.18 + 0.01))
    axis.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def summarize_test_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Convenience wrapper for the selected test threshold."""

    return binary_metrics(y_true, y_prob, threshold)


def ensure_results_directories() -> None:
    """Create the standard result directories."""

    config.ensure_directories()
    (config.RESULTS_DIR / "figures").mkdir(parents=True, exist_ok=True)
