"""Leakage-aware cleaning and materialisation of the CIC-IDS2017 data."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from src import config
from src.data_loader import (
    BenchmarkData,
    iter_raw_chunks,
    normalize_columns,
    read_header,
    validate_source_schema,
)

LOGGER = logging.getLogger(__name__)


def feature_columns_from_header(header: Sequence[str]) -> list[str]:
    """Derive the model feature list after documented column removals."""

    normalized = normalize_columns(header)
    missing = set(config.DROP_COLUMNS) - set(normalized)
    if missing:
        raise ValueError(f"Expected columns are missing from the schema: {sorted(missing)}")
    features = [
        name
        for name in normalized
        if name not in set(config.DROP_COLUMNS)
        and name not in {config.LABEL_COLUMN, "source_file"}
    ]
    LOGGER.info(
        "Using %d features after dropping %d constant and duplicate columns.",
        len(features),
        len(config.DROP_COLUMNS),
    )
    return features


def clean_features(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    """Replace infinities and impossible negative measurements with NaN.

    The transformation depends only on domain rules, not on the target, so it
    is safe to apply before the train/test split. Learned imputation is fitted
    later on training data only.
    """

    values = frame.loc[:, list(feature_columns)].to_numpy(
        dtype=np.float32, na_value=np.nan
    )
    values[~np.isfinite(values)] = np.nan
    indices = {name: index for index, name in enumerate(feature_columns)}
    for column in config.INVALID_NEGATIVE_COLUMNS:
        if column in indices:
            column_values = values[:, indices[column]]
            column_values[column_values < 0] = np.nan
    return pd.DataFrame(values, columns=list(feature_columns), index=frame.index)


def hash_feature_rows(features: pd.DataFrame) -> np.ndarray:
    """Return a stable 64-bit hash per feature row for deduplication."""

    return pd.util.hash_pandas_object(features, index=False).to_numpy(
        dtype=np.uint64, copy=False
    )


def _read_clean_meta(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


def build_clean_dataset(
    chunksize: int = 100_000,
    force: bool = False,
) -> dict[str, object]:
    """Stream the source CSVs, resolve duplicate labels, and save float32 data.

    Cleaning rules:
    * strip header whitespace and remove the duplicated/constant columns;
    * replace infinities and impossible negative values with NaN;
    * remove every row in a feature-vector group that has conflicting binary
      labels (BENIGN and ATTACK both present);
    * keep one row per remaining exact feature vector.
    """

    config.ensure_directories()
    if config.CLEAN_META_PATH.exists() and not force:
        meta = _read_clean_meta(config.CLEAN_META_PATH)
        LOGGER.info(
            "Clean dataset already exists (%s rows). Use force=True to rebuild.",
            int(meta["n_rows_clean"]),
        )
        return meta

    header = read_header(config.SOURCE_DIR / sorted(config.SOURCE_DIR.glob("*.csv"))[0].name)
    _ = validate_source_schema()
    feature_columns = feature_columns_from_header(header)
    source_files = sorted(path.name for path in config.SOURCE_DIR.glob("*.csv"))
    source_to_code = {name: code for code, name in enumerate(source_files)}
    label_to_code = {label: code for code, label in enumerate(config.LABEL_NAMES)}

    hash_parts: list[np.ndarray] = []
    binary_parts: list[np.ndarray] = []
    raw_label_parts: list[np.ndarray] = []
    source_parts: list[np.ndarray] = []
    total_rows = 0
    unknown_labels: set[str] = set()

    LOGGER.info("Pass 1/2: hashing and label validation")
    for chunk in iter_raw_chunks(chunksize=chunksize):
        labels = chunk[config.LABEL_COLUMN].astype(str)
        unique_labels = set(labels.unique())
        unknown_labels |= unique_labels - set(config.LABEL_NAMES)
        if unknown_labels:
            raise ValueError(f"Unknown label values found: {sorted(unknown_labels)}")
        binary = labels.map(config.BINARY_LABEL_MAP).to_numpy(dtype=np.uint8)
        raw_codes = labels.map(label_to_code).to_numpy(dtype=np.uint8)
        source_codes = (
            chunk["source_file"].map(source_to_code).to_numpy(dtype=np.uint8)
        )
        features = clean_features(chunk, feature_columns)
        hash_parts.append(hash_feature_rows(features))
        binary_parts.append(binary)
        raw_label_parts.append(raw_codes)
        source_parts.append(source_codes)
        total_rows += len(chunk)

    hashes = np.concatenate(hash_parts)
    y_binary = np.concatenate(binary_parts)
    y_raw = np.concatenate(raw_label_parts)
    source_code = np.concatenate(source_parts)
    del hash_parts, binary_parts, raw_label_parts, source_parts

    if total_rows != len(hashes):
        raise RuntimeError("Row count changed unexpectedly during pass 1.")

    benign_hashes = np.unique(hashes[y_binary == 0])
    attack_hashes = np.unique(hashes[y_binary == 1])
    conflict_hashes = np.intersect1d(
        benign_hashes, attack_hashes, assume_unique=True
    )
    conflict_mask = np.isin(hashes, conflict_hashes)
    valid_indices = np.flatnonzero(~conflict_mask)
    _, first_positions = np.unique(hashes[valid_indices], return_index=True)
    keep_indices = np.sort(valid_indices[first_positions])

    rows_after_conflict = len(valid_indices)
    conflict_rows = int(conflict_mask.sum())
    deduplicated_rows = rows_after_conflict - len(keep_indices)
    LOGGER.info(
        "Raw rows=%d; conflict groups=%d; conflict rows=%d; "
        "duplicate rows removed=%d; clean rows=%d",
        total_rows,
        len(conflict_hashes),
        conflict_rows,
        deduplicated_rows,
        len(keep_indices),
    )

    keep_mask = np.zeros(total_rows, dtype=bool)
    keep_mask[keep_indices] = True
    feature_memmap = np.lib.format.open_memmap(
        config.CLEAN_FEATURES_PATH,
        mode="w+",
        dtype=np.float32,
        shape=(len(keep_indices), len(feature_columns)),
    )

    LOGGER.info("Pass 2/2: writing clean float32 feature matrix")
    offset = 0
    write_position = 0
    for chunk in iter_raw_chunks(chunksize=chunksize):
        selection = keep_mask[offset : offset + len(chunk)]
        if selection.any():
            cleaned = clean_features(chunk, feature_columns)
            values = cleaned.to_numpy(dtype=np.float32, na_value=np.nan)
            selected = values[selection]
            feature_memmap[write_position : write_position + len(selected)] = selected
            write_position += len(selected)
        offset += len(chunk)
    feature_memmap.flush()
    del feature_memmap
    if write_position != len(keep_indices):
        raise RuntimeError(
            f"Wrote {write_position} rows but expected {len(keep_indices)}."
        )

    kept_hashes = hashes[keep_indices]
    kept_binary = y_binary[keep_indices]
    kept_raw = y_raw[keep_indices]
    kept_source = source_code[keep_indices]
    if np.unique(kept_hashes).size != len(kept_hashes):
        raise RuntimeError("Duplicate feature vectors remain after deduplication.")
    if len(np.unique(kept_hashes[kept_binary == 0])) + len(
        np.unique(kept_hashes[kept_binary == 1])
    ) != len(np.unique(kept_hashes)):
        raise RuntimeError("Conflicting binary labels remain after cleaning.")

    np.savez_compressed(
        config.CLEAN_META_PATH,
        y_binary=kept_binary,
        y_raw=kept_raw,
        source_code=kept_source,
        hashes=kept_hashes,
        feature_names=np.asarray(feature_columns, dtype=str),
        label_names=np.asarray(config.LABEL_NAMES, dtype=str),
        source_files=np.asarray(source_files, dtype=str),
        n_rows_raw=np.asarray(total_rows),
        n_rows_clean=np.asarray(len(keep_indices)),
        n_conflict_groups=np.asarray(len(conflict_hashes)),
        n_conflict_rows=np.asarray(conflict_rows),
        n_deduplicated_rows=np.asarray(deduplicated_rows),
        n_rows_after_conflict=np.asarray(rows_after_conflict),
    )

    schema = {
        "dataset": "CIC-IDS2017 MachineLearningCSV",
        "source_directory": str(config.SOURCE_DIR),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "random_state": config.RANDOM_STATE,
        "task": "binary network intrusion detection",
        "positive_class": {"name": "ATTACK", "value": 1},
        "negative_class": {"name": "BENIGN", "value": 0},
        "label_map": config.BINARY_LABEL_MAP,
        "label_names": list(config.LABEL_NAMES),
        "feature_names": list(feature_columns),
        "feature_dtype": "float32",
        "raw_column_count": config.RAW_COLUMN_COUNT,
        "raw_feature_count": config.RAW_FEATURE_COUNT,
        "dropped_columns": {
            "constant": list(config.CONSTANT_COLUMNS),
            "duplicate": list(config.DUPLICATE_COLUMNS),
        },
        "cleaning": {
            "header_whitespace": "stripped",
            "infinities": "replaced with NaN",
            "invalid_negatives": "replaced with NaN for "
            + ", ".join(config.INVALID_NEGATIVE_COLUMNS),
            "init_window_negative_values": "retained as -1 sentinel",
            "conflicting_label_groups": "all rows removed",
            "duplicate_feature_vectors": "first occurrence retained",
            "imputation": "median fitted on training data only",
        },
        "counts": {
            "raw_rows": int(total_rows),
            "clean_rows": int(len(keep_indices)),
            "conflict_groups": int(len(conflict_hashes)),
            "conflict_rows": int(conflict_rows),
            "deduplicated_rows": int(deduplicated_rows),
            "rows_after_conflict": int(rows_after_conflict),
        },
        "source_files": source_files,
    }
    with config.SCHEMA_PATH.open("w", encoding="utf-8") as handle:
        json.dump(schema, handle, indent=2)

    LOGGER.info("Saved clean dataset and schema.")
    return _read_clean_meta(config.CLEAN_META_PATH)


def impute_benchmark(data: BenchmarkData) -> BenchmarkData:
    """Median-impute the benchmark matrices using training statistics only."""

    imputer = SimpleImputer(strategy="median", add_indicator=False)
    X_train = imputer.fit_transform(data.X_train).astype(np.float32, copy=False)
    X_val = imputer.transform(data.X_val).astype(np.float32, copy=False)
    X_test = imputer.transform(data.X_test).astype(np.float32, copy=False)
    return BenchmarkData(
        X_train=X_train,
        y_train=data.y_train,
        X_val=X_val,
        y_val=data.y_val,
        X_test=X_test,
        y_test=data.y_test,
        feature_names=data.feature_names,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    metadata = build_clean_dataset()
    print(
        "Clean rows:",
        int(metadata["n_rows_clean"]),
        "| features:",
        len(metadata["feature_names"]),
    )
