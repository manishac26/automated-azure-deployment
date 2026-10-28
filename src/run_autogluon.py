"""AutoGluon Tabular comparison on the shared 100k training subset."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src import config
from src.evaluation import binary_metrics, plot_confusion_matrix, plot_roc_pr, save_json, tune_threshold
from src.experiment_utils import append_comparison
from src.hpo_common import search_data

LOGGER = logging.getLogger(__name__)

MODEL_NAME = "autogluon_tabular"


class _AutoGluonWrapper:
    def __init__(self, predictor):
        self.predictor = predictor

    def predict_proba(self, X):
        if isinstance(X, np.ndarray):
            frame = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        else:
            frame = pd.DataFrame(X)
        proba = self.predictor.predict_proba(frame)
        if isinstance(proba, pd.DataFrame):
            if 1 in proba.columns:
                column = 1
            else:
                column = proba.columns[1]
            return np.column_stack([1.0 - proba[column].to_numpy(), proba[column].to_numpy()])
        return np.asarray(proba)


def run_autogluon() -> dict[str, Any]:
    """Train AutoGluon Tabular and evaluate it with the same protocol."""

    try:
        from autogluon.tabular import TabularPredictor
    except Exception as exc:
        raise RuntimeError(
            "AutoGluon is not installed or failed to import in this environment. "
            "Install autogluon.tabular if needed."
        ) from exc

    data = search_data()
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(data.X_train)
    X_val = imputer.transform(data.X_val)
    X_test = imputer.transform(data.X_test)

    columns = [str(name) for name in data.feature_names]
    train_frame = pd.DataFrame(X_train, columns=columns)
    train_frame["label"] = data.y_train
    val_frame = pd.DataFrame(X_val, columns=columns)
    val_frame["label"] = data.y_val
    test_frame = pd.DataFrame(X_test, columns=columns)
    test_frame["label"] = data.y_test

    predictor_path = config.MODELS_DIR / "autogluon_tabular_predictor"
    search_started = time.perf_counter()
    predictor = TabularPredictor(
        label="label",
        problem_type="binary",
        eval_metric="f1",
        path=str(predictor_path),
        verbosity=2,
    ).fit(
        train_data=train_frame,
        tuning_data=val_frame,
        time_limit=config.AUTOML_TIME_BUDGET_SECONDS,
        presets="medium_quality",
        ag_args_fit={"num_gpus": 0},
        num_stack_levels=0,
        num_bag_folds=0,
    )
    search_seconds = time.perf_counter() - search_started

    estimator = _AutoGluonWrapper(predictor)
    model = Pipeline(
        steps=[
            ("imputer", imputer),
            ("classifier", estimator),
        ]
    )

    validation_probabilities = estimator.predict_proba(X_val)[:, 1]
    test_probabilities = estimator.predict_proba(X_test)[:, 1]
    predict_seconds = time.perf_counter() - search_started

    validation_default = binary_metrics(data.y_val, validation_probabilities, 0.5)
    selected_threshold, validation_tuned = tune_threshold(
        data.y_val, validation_probabilities
    )
    test_default = binary_metrics(data.y_test, test_probabilities, 0.5)
    test_tuned = binary_metrics(data.y_test, test_probabilities, selected_threshold)

    leaderboard = predictor.leaderboard(val_frame, silent=True)
    try:
        leaderboard_path = config.RESULTS_DIR / "autogluon_leaderboard.csv"
        leaderboard.to_csv(leaderboard_path, index=False)
        leader_text = leaderboard.to_json(orient="records")
    except Exception:
        leader_text = str(leaderboard)

    best_params = {
        "best_model": str(predictor.model_best),
        "leaderboard_head": leader_text[:8000],
    }
    extra = {
        "automl_tool": "AutoGluon",
        "time_budget_seconds": config.AUTOML_TIME_BUDGET_SECONDS,
        "search_seconds": round(float(search_seconds), 3),
        "presets": "medium_quality",
        "best_model": str(predictor.model_best),
        "leaderboard_path": "results/autogluon_leaderboard.csv",
        "imputation": "median fit on 100k train",
    }

    model_path = config.MODELS_DIR / f"{MODEL_NAME}.joblib"
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(model, model_path, compress=3)

    figure_dir = config.FIGURES_DIR
    test_predictions = (test_probabilities >= selected_threshold).astype(np.uint8)
    plot_confusion_matrix(
        data.y_test,
        test_predictions,
        f"{MODEL_NAME.replace('_', ' ')} test confusion matrix",
        figure_dir / f"fig_{MODEL_NAME}_confusion.png",
    )
    plot_roc_pr(
        data.y_test,
        test_probabilities,
        f"{MODEL_NAME.replace('_', ' ')} test performance",
        figure_dir / f"fig_{MODEL_NAME}_curves.png",
    )

    metrics = {
        "model": MODEL_NAME,
        "dataset": "benchmark",
        "train_rows": int(len(data.y_train)),
        "validation_rows": int(len(data.y_val)),
        "test_rows": int(len(data.y_test)),
        "feature_count": int(len(data.feature_names)),
        "fit_seconds": round(float(search_seconds), 3),
        "predict_seconds": round(float(predict_seconds - search_started), 3),
        "selected_threshold": round(float(selected_threshold), 6),
        "threshold_source": "validation_f1",
        "cv_f1_mean": None,
        "cv_f1_std": None,
        "validation_default_0_5": validation_default,
        "validation_tuned": validation_tuned,
        "test_default_0_5": test_default,
        "test_tuned": test_tuned,
        "classifier_params": {"estimator_class": "AutoGluonWrapper"},
        "best_params": best_params,
        "model_path": str(model_path.relative_to(config.PROJECT_ROOT)),
        "extra": extra,
    }
    for key, value in metrics.items():
        if isinstance(value, float):
            metrics[key] = round(value, 6)
    save_json(config.RESULTS_DIR / f"metrics_{MODEL_NAME}.json", metrics)
    append_comparison(metrics)
    LOGGER.info(
        "%s: validation F1 %.4f (threshold %.3f), test F1 %.4f, PR-AUC %.4f",
        MODEL_NAME,
        validation_tuned["f1"],
        selected_threshold,
        test_tuned["f1"],
        test_tuned["pr_auc"] if test_tuned["pr_auc"] is not None else -1.0,
    )
    return metrics


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    config.ensure_directories()
    run_autogluon()
