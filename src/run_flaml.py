"""FLAML AutoML comparison on the shared 100k training subset."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from flaml import AutoML
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src import config
from src.experiment_utils import append_comparison, evaluate_and_plot
from src.hpo_common import search_data

LOGGER = logging.getLogger(__name__)

MODEL_NAME = "flaml_automl"


def run_flaml() -> dict[str, Any]:
    """Search with FLAML, wrap the best estimator, and evaluate it."""

    data = search_data()
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(data.X_train)
    X_val = imputer.transform(data.X_val)
    X_test = imputer.transform(data.X_test)

    automl = AutoML()
    search_started = time.perf_counter()
    try:
        automl.fit(
            X_train=X_train,
            y_train=data.y_train,
            task="classification",
            metric="f1",
            estimator_list=["xgboost"],
            time_budget=180,
            eval_method="cv",
            n_splits=config.CV_FOLDS,
            split_type="stratified",
            n_jobs=1,
            n_concurrent_trials=1,
            seed=config.RANDOM_STATE,
            retrain_full=True,
            early_stop=True,
            verbose=1,
            log_file_name=str(config.RESULTS_DIR / "flaml.log"),
        )
    except Exception as exc:
        LOGGER.warning("FLAML fit failed with %s, retrying with estimator_list None", exc)
        automl = AutoML()
        automl.fit(
            X_train=X_train,
            y_train=data.y_train,
            task="classification",
            metric="f1",
            time_budget=180,
            eval_method="cv",
            n_splits=config.CV_FOLDS,
            split_type="stratified",
            n_jobs=1,
            n_concurrent_trials=1,
            seed=config.RANDOM_STATE,
            retrain_full=True,
            early_stop=True,
            verbose=1,
            log_file_name=str(config.RESULTS_DIR / "flaml.log"),
        )
    search_seconds = time.perf_counter() - search_started
    if automl.model is None:
        raise RuntimeError("FLAML did not produce a fitted model.")

    estimator = automl.model.estimator
    model = Pipeline(
        steps=[
            ("imputer", imputer),
            ("classifier", estimator),
        ]
    )

    best_config = automl.best_config
    try:
        best_config_text = json.dumps(best_config, default=str, sort_keys=True)
    except TypeError:
        best_config_text = str(best_config)
    best_params = {
        "best_estimator": str(automl.best_estimator),
        "best_loss": float(automl.best_loss),
        "best_config": best_config_text,
    }
    extra = {
        "automl_tool": "FLAML",
        "time_budget_seconds": 180,
        "search_seconds": round(float(search_seconds), 3),
        "best_estimator": str(automl.best_estimator),
        "best_loss": float(automl.best_loss),
        "best_config": best_config_text,
        "estimator_list": ["xgboost"],
        "imputation": "median fit on 100k train",
        "eval_method": "3-fold stratified CV",
    }

    metrics = evaluate_and_plot(
        model,
        data,
        MODEL_NAME,
        train_rows=len(data.y_train),
        fit_seconds=search_seconds,
        cv_f1_mean=None,
        cv_f1_std=None,
        best_params=best_params,
        extra=extra,
    )
    append_comparison(metrics)
    return metrics


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    config.ensure_directories()
    run_flaml()
