"""Create and persist the full and benchmark train/validation/test splits."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config
from src.data_loader import load_clean_meta

LOGGER = logging.getLogger(__name__)


def _load_labels() -> np.ndarray:
    meta = load_clean_meta()
    return np.asarray(meta["y_binary"], dtype=np.uint8)


def _stratified_sample(
    indices: np.ndarray,
    labels: np.ndarray,
    size: int,
    seed: int,
) -> np.ndarray:
    if size >= len(indices):
        return np.sort(indices)
    sampled, _ = train_test_split(
        indices,
        train_size=size,
        stratify=labels[indices],
        random_state=seed,
    )
    return np.sort(sampled)


def make_splits(force: bool = False) -> dict[str, np.ndarray]:
    """Create a 70/15/15 stratified split and a fixed 350k benchmark subset."""

    config.ensure_directories()
    if config.SPLIT_PATH.exists() and config.BENCHMARK_PATH.exists() and not force:
        LOGGER.info("Split files already exist; use force=True to rebuild.")
        with np.load(config.BENCHMARK_PATH, allow_pickle=False) as loaded:
            return {key: loaded[key] for key in loaded.files}

    labels = _load_labels()
    all_indices = np.arange(len(labels), dtype=np.int64)
    train_idx, remainder = train_test_split(
        all_indices,
        test_size=config.FULL_VAL_FRACTION + config.FULL_TEST_FRACTION,
        stratify=labels,
        random_state=config.RANDOM_STATE,
    )
    relative_test_size = config.FULL_TEST_FRACTION / (
        config.FULL_VAL_FRACTION + config.FULL_TEST_FRACTION
    )
    val_idx, test_idx = train_test_split(
        remainder,
        test_size=relative_test_size,
        stratify=labels[remainder],
        random_state=config.RANDOM_STATE,
    )
    train_idx = np.sort(train_idx)
    val_idx = np.sort(val_idx)
    test_idx = np.sort(test_idx)

    np.savez_compressed(
        config.SPLIT_PATH,
        train=train_idx,
        val=val_idx,
        test=test_idx,
    )

    benchmark_train = _stratified_sample(
        train_idx, labels, config.BENCHMARK_TRAIN_SIZE, config.RANDOM_STATE
    )
    benchmark_val = _stratified_sample(
        val_idx, labels, config.BENCHMARK_VAL_SIZE, config.RANDOM_STATE
    )
    benchmark_test = _stratified_sample(
        test_idx, labels, config.BENCHMARK_TEST_SIZE, config.RANDOM_STATE
    )
    np.savez_compressed(
        config.BENCHMARK_PATH,
        train=benchmark_train,
        val=benchmark_val,
        test=benchmark_test,
    )

    overlaps = {
        "train_val": np.intersect1d(benchmark_train, benchmark_val).size,
        "train_test": np.intersect1d(benchmark_train, benchmark_test).size,
        "val_test": np.intersect1d(benchmark_val, benchmark_test).size,
    }
    if any(overlaps.values()):
        raise RuntimeError(f"Benchmark splits overlap: {overlaps}")

    rows: list[dict[str, object]] = []
    for name, indices in (
        ("full_train", train_idx),
        ("full_val", val_idx),
        ("full_test", test_idx),
        ("benchmark_train", benchmark_train),
        ("benchmark_val", benchmark_val),
        ("benchmark_test", benchmark_test),
    ):
        y = labels[indices]
        rows.append(
            {
                "split": name,
                "rows": len(indices),
                "benign": int((y == 0).sum()),
                "attack": int((y == 1).sum()),
                "attack_share": round(float((y == 1).mean()), 6),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(config.RESULTS_DIR / "split_summary.csv", index=False)
    LOGGER.info("Split summary:\n%s", summary.to_string(index=False))
    return {key: np.load(config.BENCHMARK_PATH)[key] for key in ("train", "val", "test")}


def load_split(name: str = "benchmark") -> dict[str, np.ndarray]:
    """Load either the full or benchmark split."""

    path = config.BENCHMARK_PATH if name == "benchmark" else config.SPLIT_PATH
    if not path.exists():
        make_splits()
    with np.load(path, allow_pickle=False) as loaded:
        return {key: np.asarray(loaded[key], dtype=np.int64) for key in loaded.files}


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    make_splits()
