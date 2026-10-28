"""Train the baseline RF on the same 100k subset used for HPO and AutoML."""

from __future__ import annotations

import logging
import time
from typing import Any

from src import config
from src.experiment_utils import append_comparison, build_rf_pipeline, evaluate_and_plot
from src.hpo_common import search_data

LOGGER = logging.getLogger(__name__)

MODEL_NAME = "baseline_rf_100k"


def run_baseline_100k() -> dict[str, Any]:
    """Fit the default RF pipeline on the fixed 100k train subset."""

    data = search_data()
    model = build_rf_pipeline()
    LOGGER.info(
        "Training %s on %d rows with %d features",
        MODEL_NAME,
        len(data.y_train),
        len(data.feature_names),
    )
    started = time.perf_counter()
    model.fit(data.X_train, data.y_train)
    fit_seconds = time.perf_counter() - started
    metrics = evaluate_and_plot(
        model,
        data,
        MODEL_NAME,
        train_rows=len(data.y_train),
        fit_seconds=fit_seconds,
        extra={"baseline": "untuned default random forest", "subset": "100k"},
    )
    append_comparison(metrics)
    return metrics


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    config.ensure_directories()
    run_baseline_100k()
