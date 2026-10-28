"""Optuna TPE search for the random-forest hyperparameters."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import optuna
from sklearn.metrics import f1_score

from src import config
from src.hpo_common import (
    CLASSIFIER_N_JOBS,
    SEARCH_N_JOBS,
    SEARCH_TREES,
    cv_splitter,
    refit_and_evaluate,
    search_pipeline,
    search_data,
)

LOGGER = logging.getLogger(__name__)

MODEL_NAME = "optuna_rf"
N_TRIALS = 15

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _suggest_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "max_depth": trial.suggest_categorical("max_depth", [None, 10, 20, 30, 40]),
        "max_features": trial.suggest_categorical(
            "max_features", ["sqrt", "log2", 0.2, 0.4]
        ),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 6),
        "class_weight": trial.suggest_categorical(
            "class_weight", [None, "balanced"]
        ),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
    }


def run_optuna_search() -> dict[str, Any]:
    """Run TPE search with manual stratified CV and refit the best model."""

    data = search_data()
    splitter = cv_splitter()

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial)
        validation_scores: list[float] = []
        training_scores: list[float] = []
        started = time.perf_counter()
        for fold, (train_index, validation_index) in enumerate(
            splitter.split(data.X_train, data.y_train)
        ):
            model = search_pipeline(**params)
            model.fit(data.X_train[train_index], data.y_train[train_index])
            validation_prob = model.predict_proba(data.X_train[validation_index])[:, 1]
            validation_pred = (validation_prob >= 0.5).astype(np.uint8)
            validation_scores.append(
                float(
                    f1_score(
                        data.y_train[validation_index],
                        validation_pred,
                        zero_division=0,
                    )
                )
            )
            train_prob = model.predict_proba(data.X_train[train_index])[:, 1]
            train_pred = (train_prob >= 0.5).astype(np.uint8)
            training_scores.append(
                float(
                    f1_score(
                        data.y_train[train_index],
                        train_pred,
                        zero_division=0,
                    )
                )
            )
            trial.set_user_attr(f"fold_{fold}_f1", validation_scores[-1])
        trial.set_user_attr("fold_scores", validation_scores)
        trial.set_user_attr("mean_train_f1", float(np.mean(training_scores)))
        trial.set_user_attr("fit_seconds", float(time.perf_counter() - started))
        return float(np.mean(validation_scores))

    sampler = optuna.samplers.TPESampler(
        seed=config.RANDOM_STATE, multivariate=True, group=True
    )
    study = optuna.create_study(direction="maximize", sampler=sampler)
    LOGGER.info(
        "Optuna TPE: %d trials x %d folds, %d trees per candidate",
        N_TRIALS,
        config.CV_FOLDS,
        SEARCH_TREES,
    )
    started = time.perf_counter()
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=SEARCH_N_JOBS)
    search_seconds = time.perf_counter() - started

    trials_frame = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs"))
    trials_frame.to_csv(config.RESULTS_DIR / "hpo_optuna_trials.csv", index=False)
    best_trial = study.best_trial
    best_params = dict(best_trial.params)
    fold_scores = list(best_trial.user_attrs["fold_scores"])
    best_cv = float(best_trial.value)
    best_std = float(np.std(fold_scores))
    best_train = float(best_trial.user_attrs.get("mean_train_f1", 0.0))

    return refit_and_evaluate(
        data,
        MODEL_NAME,
        best_params,
        search_method="optuna_tpe",
        search_seconds=search_seconds,
        search_cv_f1=best_cv,
        search_cv_std=best_std,
        search_train_f1=best_train,
        search_rows=len(data.y_train),
        search_trees=SEARCH_TREES,
        extra={
            "n_trials": N_TRIALS,
            "sampler": "TPESampler(multivariate=True, group=True)",
            "cv_folds": config.CV_FOLDS,
            "classifier_n_jobs": CLASSIFIER_N_JOBS,
            "best_trial_fold_scores": fold_scores,
        },
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    config.ensure_directories()
    run_optuna_search()
