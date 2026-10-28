"""Shared plumbing for the three random-forest hyperparameter searches."""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from src import config
from src.data_loader import BenchmarkData
from src.evaluation import binary_metrics, save_json
from src.experiment_utils import (
    append_comparison,
    benchmark_data,
    build_rf_pipeline,
    evaluate_and_plot,
)

LOGGER = logging.getLogger(__name__)

SEARCH_TRAIN_SIZE = 100_000
SEARCH_TREES = 40
FINAL_TREES = 200
SEARCH_N_JOBS = 1
CLASSIFIER_N_JOBS = 2
TRAIN_AUDIT_SAMPLE = 50_000


def search_data() -> BenchmarkData:
    """Return the fixed 100k stratified training subset used by all searches."""

    data = benchmark_data()
    X_train, _, y_train, _ = train_test_split(
        data.X_train,
        data.y_train,
        train_size=SEARCH_TRAIN_SIZE,
        stratify=data.y_train,
        random_state=config.RANDOM_STATE,
    )
    return replace(data, X_train=X_train, y_train=y_train)


def cv_splitter() -> StratifiedKFold:
    """Return the shared three-fold stratified splitter."""

    return StratifiedKFold(
        n_splits=config.CV_FOLDS,
        shuffle=True,
        random_state=config.RANDOM_STATE,
    )


def search_pipeline(**params: Any):
    """Build the lower-cost pipeline used inside hyperparameter search."""

    return build_rf_pipeline(
        n_estimators=SEARCH_TREES,
        n_jobs=CLASSIFIER_N_JOBS,
        **params,
    )


def clean_params(params: dict[str, Any]) -> dict[str, Any]:
    """Strip the ``classifier__`` prefix used by sklearn searches."""

    cleaned: dict[str, Any] = {}
    for key, value in params.items():
        cleaned[key.replace("classifier__", "")] = value
    return cleaned


def save_cv_results(search: Any, filename: str) -> None:
    """Persist the search table with only useful columns."""

    frame = pd.DataFrame(search.cv_results_)
    keep = [
        column
        for column in frame.columns
        if column.startswith("param_")
        or column
        in {
            "mean_test_score",
            "std_test_score",
            "mean_train_score",
            "std_train_score",
            "rank_test_score",
            "mean_fit_time",
        }
    ]
    frame.loc[:, keep].sort_values("rank_test_score").to_csv(
        config.RESULTS_DIR / filename, index=False
    )


def train_audit_sample(data: BenchmarkData) -> tuple[np.ndarray, np.ndarray]:
    """Return a fixed stratified 50k sample for train-vs-test diagnostics."""

    X_sample, _, y_sample, _ = train_test_split(
        data.X_train,
        data.y_train,
        train_size=TRAIN_AUDIT_SAMPLE,
        stratify=data.y_train,
        random_state=config.RANDOM_STATE,
    )
    return X_sample, y_sample


def refit_and_evaluate(
    data: BenchmarkData,
    model_name: str,
    best_params: dict[str, Any],
    *,
    search_method: str,
    search_seconds: float,
    search_cv_f1: float,
    search_cv_std: float,
    search_train_f1: float | None,
    search_rows: int,
    search_trees: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Refit the best configuration at the baseline capacity and evaluate it."""

    params = clean_params(best_params)
    params.pop("n_estimators", None)
    final_model = build_rf_pipeline(
        n_estimators=FINAL_TREES,
        n_jobs=config.RF_N_JOBS,
        **params,
    )
    LOGGER.info("Refitting %s with %d trees on %d rows", model_name, FINAL_TREES, len(data.y_train))
    started = time.perf_counter()
    final_model.fit(data.X_train, data.y_train)
    fit_seconds = time.perf_counter() - started

    X_audit, y_audit = train_audit_sample(data)
    audit_probabilities = final_model.predict_proba(X_audit)[:, 1]
    train_sample = binary_metrics(y_audit, audit_probabilities, 0.5)

    run_extra = {
        "search_method": search_method,
        "search_rows": int(search_rows),
        "search_trees": int(search_trees),
        "final_trees": int(FINAL_TREES),
        "search_seconds": round(float(search_seconds), 3),
        "search_cv_f1": round(float(search_cv_f1), 6),
        "search_cv_std": round(float(search_cv_std), 6),
        "search_train_f1": None
        if search_train_f1 is None
        else round(float(search_train_f1), 6),
        "train_sample_rows": int(len(y_audit)),
        "train_sample_f1": train_sample["f1"],
        "train_sample_precision": train_sample["precision"],
        "train_sample_recall": train_sample["recall"],
    }
    if extra:
        run_extra.update(extra)
    metrics = evaluate_and_plot(
        final_model,
        data,
        model_name,
        train_rows=len(data.y_train),
        fit_seconds=fit_seconds,
        cv_f1_mean=search_cv_f1,
        cv_f1_std=search_cv_std,
        best_params=params,
        extra=run_extra,
    )
    append_comparison(metrics)
    save_json(
        config.RESULTS_DIR / f"best_params_{model_name}.json",
        {
            "model": model_name,
            "search_method": search_method,
            "best_params": params,
            "search_cv_f1": search_cv_f1,
            "search_cv_std": search_cv_std,
            "search_seconds": search_seconds,
        },
    )
    return metrics
