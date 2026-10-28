"""Shared model construction and experiment bookkeeping."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src import config
from src.data_loader import BenchmarkData, load_benchmark_arrays
from src.evaluation import (
    binary_metrics,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_pr,
    save_json,
    tune_threshold,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_RF_PARAMS: dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": None,
    "max_features": "sqrt",
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "bootstrap": True,
    "class_weight": None,
    "n_jobs": config.RF_N_JOBS,
}


def build_rf_pipeline(
    seed: int = config.RANDOM_STATE,
    **params: Any,
) -> Pipeline:
    """Build median imputation plus a random-forest classifier."""

    classifier_params = {**DEFAULT_RF_PARAMS, **params}
    classifier_params.setdefault("random_state", seed)
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(**classifier_params)),
        ]
    )


def benchmark_data() -> BenchmarkData:
    """Load the fixed 250k/50k/50k benchmark."""

    return load_benchmark_arrays()


def append_comparison(metrics: dict[str, Any]) -> None:
    """Append or replace one model row in the comparison table."""

    test = metrics["test_tuned"]
    row = {
        "model": metrics["model"],
        "dataset": metrics["dataset"],
        "train_rows": metrics["train_rows"],
        "threshold": metrics["selected_threshold"],
        "val_f1": metrics["validation_tuned"]["f1"],
        "test_f1": test["f1"],
        "test_precision": test["precision"],
        "test_recall": test["recall"],
        "test_accuracy": test["accuracy"],
        "test_roc_auc": test["roc_auc"],
        "test_pr_auc": test["pr_auc"],
        "test_false_positive_rate": test["false_positive_rate"],
        "cv_f1_mean": metrics["cv_f1_mean"],
        "cv_f1_std": metrics["cv_f1_std"],
        "fit_seconds": metrics["fit_seconds"],
        "predict_seconds": metrics["predict_seconds"],
        "best_params": pd.Series(metrics["best_params"]).to_json(),
    }
    path = config.RESULTS_DIR / "model_comparison.csv"
    if path.exists():
        existing = pd.read_csv(path)
        existing = existing[existing["model"] != metrics["model"]]
        frame = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    else:
        frame = pd.DataFrame([row])
    frame.to_csv(path, index=False)
    LOGGER.info("Updated %s", path)


def evaluate_and_plot(
    model: Pipeline,
    data: BenchmarkData,
    model_name: str,
    *,
    dataset_name: str = "benchmark",
    train_rows: int,
    fit_seconds: float,
    cv_f1_mean: float | None = None,
    cv_f1_std: float | None = None,
    best_params: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate on validation/test, tune the threshold, and persist artifacts."""

    started = time.perf_counter()
    validation_probabilities = model.predict_proba(data.X_val)[:, 1]
    test_probabilities = model.predict_proba(data.X_test)[:, 1]
    predict_seconds = time.perf_counter() - started

    validation_default = binary_metrics(data.y_val, validation_probabilities, 0.5)
    selected_threshold, validation_tuned = tune_threshold(
        data.y_val, validation_probabilities
    )
    test_default = binary_metrics(data.y_test, test_probabilities, 0.5)
    test_tuned = binary_metrics(data.y_test, test_probabilities, selected_threshold)

    model_path = config.MODELS_DIR / f"{model_name}.joblib"
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path, compress=3)

    classifier = model.named_steps["classifier"]
    try:
        raw_params = classifier.get_params(deep=False)
    except Exception:  # pragma: no cover - third-party estimators vary
        raw_params = {}
    classifier_params = {
        str(key): value
        if isinstance(value, (str, int, float, bool, type(None)))
        else str(value)
        for key, value in raw_params.items()
    }
    classifier_params["estimator_class"] = type(classifier).__name__

    metrics: dict[str, Any] = {
        "model": model_name,
        "dataset": dataset_name,
        "train_rows": int(train_rows),
        "validation_rows": int(len(data.y_val)),
        "test_rows": int(len(data.y_test)),
        "feature_count": int(len(data.feature_names)),
        "fit_seconds": round(float(fit_seconds), 3),
        "predict_seconds": round(float(predict_seconds), 3),
        "selected_threshold": round(float(selected_threshold), 6),
        "threshold_source": "validation_f1",
        "cv_f1_mean": None if cv_f1_mean is None else round(float(cv_f1_mean), 6),
        "cv_f1_std": None if cv_f1_std is None else round(float(cv_f1_std), 6),
        "validation_default_0_5": validation_default,
        "validation_tuned": validation_tuned,
        "test_default_0_5": test_default,
        "test_tuned": test_tuned,
        "classifier_params": classifier_params,
        "best_params": best_params or {},
        "model_path": str(model_path.relative_to(config.PROJECT_ROOT)),
    }
    if extra:
        metrics["extra"] = extra

    save_json(config.RESULTS_DIR / f"metrics_{model_name}.json", metrics)

    figure_dir = config.FIGURES_DIR
    test_predictions = (test_probabilities >= selected_threshold).astype(np.uint8)
    plot_confusion_matrix(
        data.y_test,
        test_predictions,
        f"{model_name.replace('_', ' ')} test confusion matrix",
        figure_dir / f"fig_{model_name}_confusion.png",
    )
    plot_roc_pr(
        data.y_test,
        test_probabilities,
        f"{model_name.replace('_', ' ')} test performance",
        figure_dir / f"fig_{model_name}_curves.png",
    )
    if hasattr(classifier, "feature_importances_"):
        plot_feature_importance(
            classifier.feature_importances_,
            data.feature_names,
            f"{model_name.replace('_', ' ')} feature importance",
            figure_dir / f"fig_{model_name}_importance.png",
        )

    LOGGER.info(
        "%s: validation F1 %.4f (threshold %.3f), test F1 %.4f, PR-AUC %.4f",
        model_name,
        validation_tuned["f1"],
        selected_threshold,
        test_tuned["f1"],
        test_tuned["pr_auc"] if test_tuned["pr_auc"] is not None else -1.0,
    )
    return metrics
