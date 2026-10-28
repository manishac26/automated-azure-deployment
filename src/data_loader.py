"""Readers for the raw CIC-IDS2017 CSVs and the prepared clean dataset."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd

from src import config

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BenchmarkData:
    """Arrays and labels for one reproducible experiment split."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]

    def as_frames(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Return train/validation/test feature frames with stable columns."""

        return (
            pd.DataFrame(self.X_train, columns=self.feature_names),
            pd.DataFrame(self.X_val, columns=self.feature_names),
            pd.DataFrame(self.X_test, columns=self.feature_names),
        )


def normalize_columns(columns: Sequence[str]) -> list[str]:
    """Strip whitespace and make duplicate column names explicit."""

    normalized: list[str] = []
    seen: dict[str, int] = {}
    for raw_name in columns:
        name = str(raw_name).strip()
        if not name:
            raise ValueError("Encountered an empty column name.")
        if name in seen:
            seen[name] += 1
            name = f"{name}__dup{seen[name]}"
        else:
            seen[name] = 1
        normalized.append(name)
    return normalized


def get_source_files() -> list[Path]:
    """Return the sorted source CSVs, raising a helpful error if absent."""

    if not config.SOURCE_DIR.exists():
        raise FileNotFoundError(
            "CIC-IDS2017 source directory not found. Expected extracted CSVs at "
            f"{config.SOURCE_DIR}. The expected data/raw/MachineLearningCSV.zip "
            "is not present in this project."
        )
    files = sorted(config.SOURCE_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found under {config.SOURCE_DIR}.")
    return files


def read_header(path: Path) -> list[str]:
    """Read one CSV header as UTF-8 with BOM support."""

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            return next(reader)
        except StopIteration as exc:
            raise ValueError(f"Empty CSV file: {path}") from exc


def validate_source_schema() -> list[str]:
    """Verify that all source files share an identical 79-column schema."""

    files = get_source_files()
    headers = [read_header(path) for path in files]
    first = headers[0]
    for path, header in zip(files, headers, strict=True):
        if header != first:
            raise ValueError(f"Schema mismatch in {path.name}.")
    if len(first) != config.RAW_COLUMN_COUNT:
        raise ValueError(
            f"Expected {config.RAW_COLUMN_COUNT} columns, found {len(first)}."
        )
    normalized = normalize_columns(first)
    if normalized[-1] != config.LABEL_COLUMN:
        raise ValueError(
            f"Expected the final column to be {config.LABEL_COLUMN!r}, "
            f"found {normalized[-1]!r}."
        )
    return normalized


def iter_raw_chunks(
    chunksize: int = 100_000,
    source_files: Sequence[Path] | None = None,
) -> Iterator[pd.DataFrame]:
    """Yield cleaned-header chunks from all source CSVs.

    A ``source_file`` column is added so day/file provenance can be retained
    outside the model feature matrix.
    """

    files = list(source_files) if source_files is not None else get_source_files()
    normalized_columns = validate_source_schema()
    for path in files:
        LOGGER.info("Reading %s", path.name)
        for chunk in pd.read_csv(
            path,
            encoding="utf-8-sig",
            chunksize=chunksize,
            low_memory=False,
        ):
            if chunk.shape[1] != len(normalized_columns):
                raise ValueError(
                    f"Unexpected column count in {path.name}: {chunk.shape[1]}."
                )
            chunk.columns = normalized_columns
            chunk["source_file"] = path.name
            yield chunk


def load_clean_meta(path: Path | None = None) -> dict[str, object]:
    """Load metadata for the prepared clean dataset."""

    meta_path = path or config.CLEAN_META_PATH
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Clean dataset metadata not found at {meta_path}. "
            "Run `python -m src.preprocessing` first."
        )
    with np.load(meta_path, allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


def load_feature_memmap() -> np.ndarray:
    """Memory-map the float32 clean feature matrix."""

    if not config.CLEAN_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Clean feature matrix not found at {config.CLEAN_FEATURES_PATH}. "
            "Run `python -m src.preprocessing` first."
        )
    return np.load(config.CLEAN_FEATURES_PATH, mmap_mode="r")


def load_benchmark_arrays(path: Path | None = None) -> BenchmarkData:
    """Load the fixed benchmark train/validation/test arrays."""

    split_path = path or config.BENCHMARK_PATH
    if not split_path.exists():
        raise FileNotFoundError(
            f"Benchmark split not found at {split_path}. Run `python -m src.split_data`."
        )
    meta = load_clean_meta()
    labels = np.asarray(meta["y_binary"], dtype=np.uint8)
    feature_names = [str(name) for name in np.asarray(meta["feature_names"]).tolist()]
    features = load_feature_memmap()
    with np.load(split_path, allow_pickle=False) as split:
        train_idx = np.asarray(split["train"], dtype=np.int64)
        val_idx = np.asarray(split["val"], dtype=np.int64)
        test_idx = np.asarray(split["test"], dtype=np.int64)
    return BenchmarkData(
        X_train=np.asarray(features[train_idx], dtype=np.float32),
        y_train=labels[train_idx],
        X_val=np.asarray(features[val_idx], dtype=np.float32),
        y_val=labels[val_idx],
        X_test=np.asarray(features[test_idx], dtype=np.float32),
        y_test=labels[test_idx],
        feature_names=feature_names,
    )


def load_json(path: Path) -> dict[str, object]:
    """Load a JSON document with a clear error if it is missing."""

    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
