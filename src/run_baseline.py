"""Train and evaluate the untuned random-forest baseline."""

from __future__ import annotations

import logging
import time
from typing import Any

from src import config
from src.experiment_utils import (
    append_comparison,
    benchmark_data,
    build_rf_pipeline,
    evaluate_and_plot,
)

LOGGER = logging.getLogger(__name__)

MODEL_NAME = "baseline_rf"


def run_baseline() -> dict[str, Any]:
    """Fit the default RF pipeline on the fixed benchmark training split."""

    data = benchmark_data()
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
        extra={"baseline": "untuned default random forest"},
    )
    append_comparison(metrics)
    return metrics


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    config.ensure_directories()
    run_baseline()
