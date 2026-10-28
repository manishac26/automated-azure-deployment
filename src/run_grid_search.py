"""Exhaustive grid search over a compact random-forest search space."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
from sklearn.model_selection import GridSearchCV

from src import config
from src.hpo_common import (
    CLASSIFIER_N_JOBS,
    FINAL_TREES,
    SEARCH_N_JOBS,
    SEARCH_TREES,
    cv_splitter,
    refit_and_evaluate,
    save_cv_results,
    search_pipeline,
    search_data,
)

LOGGER = logging.getLogger(__name__)

MODEL_NAME = "grid_rf"

PARAM_GRID: dict[str, list[Any]] = {
    "classifier__max_depth": [None, 20, 30],
    "classifier__max_features": ["sqrt", 0.3],
    "classifier__min_samples_leaf": [1, 2],
}


def run_grid_search() -> dict[str, Any]:
    """Run GridSearchCV, then refit and evaluate the selected configuration."""

    data = search_data()
    search = GridSearchCV(
        estimator=search_pipeline(),
        param_grid=PARAM_GRID,
        scoring="f1",
        cv=cv_splitter(),
        n_jobs=SEARCH_N_JOBS,
        refit=True,
        return_train_score=True,
        verbose=1,
        error_score="raise",
    )
    LOGGER.info(
        "Grid search: %d candidates x %d folds, %d trees per candidate",
        int(np.prod([len(values) for values in PARAM_GRID.values()])),
        config.CV_FOLDS,
        SEARCH_TREES,
    )
    started = time.perf_counter()
    search.fit(data.X_train, data.y_train)
    search_seconds = time.perf_counter() - started

    save_cv_results(search, "hpo_grid_cv.csv")
    best_index = int(search.best_index_)
    cv_results = search.cv_results_
    best_params = dict(search.best_params_)
    best_cv = float(cv_results["mean_test_score"][best_index])
    best_std = float(cv_results["std_test_score"][best_index])
    train_score = cv_results.get("mean_train_score")
    best_train = float(train_score[best_index]) if train_score is not None else None

    return refit_and_evaluate(
        data,
        MODEL_NAME,
        best_params,
        search_method="grid",
        search_seconds=search_seconds,
        search_cv_f1=best_cv,
        search_cv_std=best_std,
        search_train_f1=best_train,
        search_rows=len(data.y_train),
        search_trees=SEARCH_TREES,
        extra={
            "n_candidates": int(len(cv_results["params"])),
            "cv_folds": config.CV_FOLDS,
            "classifier_n_jobs": CLASSIFIER_N_JOBS,
            "final_trees": FINAL_TREES,
        },
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    config.ensure_directories()
    run_grid_search()
