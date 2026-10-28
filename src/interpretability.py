"""Interpretability for the selected model using permutation importance (robust to env)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src import config
from src.data_loader import load_benchmark_arrays, load_clean_meta
from src.evaluation import save_json

LOGGER = logging.getLogger(__name__)

MODEL_NAME = "flaml_automl"
SAMPLE_SIZE = 10000


def run_interpretability(sample_size: int = SAMPLE_SIZE) -> dict:
    config.ensure_directories()
    model_path = config.MODELS_DIR / f"{MODEL_NAME}.joblib"
    model = joblib.load(model_path)
    data = load_benchmark_arrays()
    meta = load_clean_meta()
    feature_names = [str(x) for x in meta["feature_names"]]

    rng = np.random.RandomState(config.RANDOM_STATE)
    if len(data.X_test) > sample_size:
        idx = rng.choice(len(data.X_test), size=sample_size, replace=False)
        X_sample = np.asarray(data.X_test[idx])
        y_sample = data.y_test[idx]
    else:
        X_sample = np.asarray(data.X_test)
        y_sample = data.y_test

    LOGGER.info("Computing permutation importance on %d samples", len(y_sample))
    result = permutation_importance(
        model,
        X_sample,
        y_sample,
        scoring="f1",
        n_repeats=10,
        random_state=config.RANDOM_STATE,
        n_jobs=2,
    )

    imp = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    imp.to_csv(config.RESULTS_DIR / "feature_importance.csv", index=False)

    top = imp.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.4, max(3.6, 0.28 * len(top))))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color="#3274a1")
    ax.set_xlabel("Permutation importance (mean decrease in F1)")
    ax.set_title(f"{MODEL_NAME} top 20 features (permutation importance)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "feature_importance.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "model": MODEL_NAME,
        "method": "permutation_importance",
        "scoring": "f1",
        "n_repeats": 10,
        "sample_size": int(len(y_sample)),
        "random_state": config.RANDOM_STATE,
        "notes": "Feature importance is not causal. Computed on a stratified/random sample of test set due to compute limits.",
    }
    save_json(config.RESULTS_DIR / "feature_importance_summary.json", summary)
    LOGGER.info("Wrote feature_importance.csv and feature_importance.png")
    return summary


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run_interpretability()
