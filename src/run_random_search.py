"""Randomized hyperparameter search over the same random-forest family."""

from __future__ import annotations

import logging
import time
from typing import Any

from sklearn.model_selection import RandomizedSearchCV

from src import config
from src.hpo_common import (
    CLASSIFIER_N_JOBS,
    SEARCH_N_JOBS,
    SEARCH_TREES,
    cv_splitter,
    refit_and_evaluate,
    save_cv_results,
    search_pipeline,
    search_data,
)

LOGGER = logging.getLogger(__name__)

MODEL_NAME = "random_rf"
N_ITER = 15

PARAM_DISTRIBUTIONS: dict[str, list[Any]] = {
    "classifier__max_depth": [None, 10, 20, 30, 40],
    "classifier__max_features": ["sqrt", "log2", 0.2, 0.4],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4],
    "classifier__class_weight": [None, "balanced"],
}


def run_random_search() -> dict[str, Any]:
    """Run RandomizedSearchCV, then refit and evaluate the selected model."""

    data = search_data()
    search = RandomizedSearchCV(
        estimator=search_pipeline(),
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=N_ITER,
        scoring="f1",
        cv=cv_splitter(),
        n_jobs=SEARCH_N_JOBS,
        refit=True,
        return_train_score=True,
        random_state=config.RANDOM_STATE,
        verbose=1,
        error_score="raise",
    )
    LOGGER.info(
        "Random search: %d candidates x %d folds, %d trees per candidate",
        N_ITER,
        config.CV_FOLDS,
        SEARCH_TREES,
    )
    started = time.perf_counter()
    search.fit(data.X_train, data.y_train)
    search_seconds = time.perf_counter() - started

    save_cv_results(search, "hpo_random_cv.csv")
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
        search_method="random",
        search_seconds=search_seconds,
        search_cv_f1=best_cv,
        search_cv_std=best_std,
        search_train_f1=best_train,
        search_rows=len(data.y_train),
        search_trees=SEARCH_TREES,
        extra={
            "n_candidates": N_ITER,
            "cv_folds": config.CV_FOLDS,
            "classifier_n_jobs": CLASSIFIER_N_JOBS,
            "random_state": config.RANDOM_STATE,
        },
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    config.ensure_directories()
    run_random_search()
