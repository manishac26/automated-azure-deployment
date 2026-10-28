"""Reproducible Phase 1 audit of the CIC-IDS2017 source CSVs."""

from __future__ import annotations

import hashlib
import json
import logging
from array import array
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.data_loader import (
    get_source_files,
    iter_raw_chunks,
    normalize_columns,
    read_header,
)

LOGGER = logging.getLogger(__name__)


def _hash_bytes(payload: bytes) -> int:
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def _hash_source_rows(
    files: list[Path],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, np.ndarray, Counter]:
    exact: dict[str, np.ndarray] = {}
    feature: dict[str, np.ndarray] = {}
    labels: list[int] = []
    label_counts: Counter[str] = Counter()
    for path in files:
        exact_values = array("Q")
        feature_values = array("Q")
        with path.open("rb") as handle:
            handle.readline()
            for line in handle:
                line = line.rstrip(b"\r\n")
                if not line:
                    continue
                feature_bytes, raw_label = line.rsplit(b",", 1)
                label = raw_label.decode("utf-8", "replace")
                if label not in config.BINARY_LABEL_MAP:
                    raise ValueError(f"Unknown label in {path.name}: {label!r}")
                label_counts[label] += 1
                labels.append(config.BINARY_LABEL_MAP[label])
                exact_values.append(_hash_bytes(line))
                feature_values.append(_hash_bytes(feature_bytes))
        exact[path.name] = np.frombuffer(exact_values, dtype=np.uint64)
        feature[path.name] = np.frombuffer(feature_values, dtype=np.uint64)
        LOGGER.info("Hashed %s", path.name)
    return exact, feature, np.asarray(labels, dtype=np.uint8), label_counts


def _source_statistics(files: list[Path]) -> tuple[list[dict[str, object]], dict[str, dict[str, object]], dict[str, object]]:
    raw_header = read_header(files[0])
    normalized = normalize_columns(raw_header)
    feature_columns = [
        name
        for name in normalized
        if name not in {config.LABEL_COLUMN, "source_file"}
    ]
    index = {name: position for position, name in enumerate(feature_columns)}
    invalid_positions = [
        index[name] for name in config.INVALID_NEGATIVE_COLUMNS if name in index
    ]
    columns: dict[str, dict[str, object]] = {
        name: {
            "missing_cells": 0,
            "infinite_cells": 0,
            "negative_cells": 0,
            "invalid_negative_cells": 0,
            "zero_cells": 0,
            "min": np.inf,
            "max": -np.inf,
            "dtype": "unknown",
        }
        for name in feature_columns
    }
    per_file: list[dict[str, object]] = []
    invalid_by_binary = {0: 0, 1: 0}
    duplicate_column_mismatches = 0
    fwd_index = index["Fwd Header Length"]
    dup_index = index["Fwd Header Length__dup2"]
    first_chunk = True

    for path in files:
        rows = 0
        missing_rows = 0
        infinite_rows = 0
        missing_or_infinite_rows = 0
        invalid_rows = 0
        file_missing_cells = 0
        file_infinite_cells = 0
        for chunk in pd.read_csv(
            path,
            encoding="utf-8-sig",
            chunksize=100_000,
            low_memory=False,
        ):
            chunk.columns = normalized
            rows += len(chunk)
            labels = chunk[config.LABEL_COLUMN].astype(str)
            binary = labels.map(config.BINARY_LABEL_MAP).to_numpy(dtype=np.uint8)
            values = chunk[feature_columns].to_numpy(
                dtype=np.float64, na_value=np.nan
            )
            missing = np.isnan(values)
            infinite = np.isinf(values)
            if first_chunk:
                for position, name in enumerate(feature_columns):
                    columns[name]["dtype"] = str(chunk[name].dtype)
                first_chunk = False
            missing_rows += int(missing.any(axis=1).sum())
            infinite_rows += int(infinite.any(axis=1).sum())
            missing_or_infinite = missing | infinite
            missing_or_infinite_rows += int(missing_or_infinite.any(axis=1).sum())
            for position, name in enumerate(feature_columns):
                column = columns[name]
                column["missing_cells"] += int(missing[:, position].sum())
                column["infinite_cells"] += int(infinite[:, position].sum())
                file_missing_cells += int(missing[:, position].sum())
                file_infinite_cells += int(infinite[:, position].sum())
                finite = np.where(infinite[:, position], np.nan, values[:, position])
                negative = np.isfinite(finite) & (finite < 0)
                column["negative_cells"] += int(negative.sum())
                if name in config.INVALID_NEGATIVE_COLUMNS:
                    column["invalid_negative_cells"] += int(negative.sum())
                column["zero_cells"] += int((np.isfinite(finite) & (finite == 0)).sum())
                finite_values = finite[np.isfinite(finite)]
                if finite_values.size:
                    column["min"] = min(column["min"], float(finite_values.min()))
                    column["max"] = max(column["max"], float(finite_values.max()))
            invalid = missing_or_infinite.copy()
            for position in invalid_positions:
                invalid[:, position] |= np.isfinite(values[:, position]) & (
                    values[:, position] < 0
                )
            invalid_mask = invalid.any(axis=1)
            invalid_rows += int(invalid_mask.sum())
            for binary_value in (0, 1):
                invalid_by_binary[binary_value] += int(
                    (invalid_mask & (binary == binary_value)).sum()
                )
            duplicate_column_mismatches += int(
                (values[:, fwd_index] != values[:, dup_index]).sum()
            )
        per_file.append(
            {
                "file": path.name,
                "rows": rows,
                "columns": config.RAW_COLUMN_COUNT,
                "size_bytes": path.stat().st_size,
                "missing_cells": file_missing_cells,
                "infinite_cells": file_infinite_cells,
                "rows_any_missing": missing_rows,
                "rows_any_infinite": infinite_rows,
                "rows_missing_or_infinite": missing_or_infinite_rows,
                "invalid_rows": invalid_rows,
            }
        )
        LOGGER.info("Audited columns for %s", path.name)

    duplicate_columns = [
        {
            "kept": "Fwd Header Length",
            "removed": "Fwd Header Length__dup2",
            "mismatching_cells": duplicate_column_mismatches,
        }
    ]
    summary = {
        "invalid_by_binary": invalid_by_binary,
        "duplicate_columns": duplicate_columns,
    }
    return per_file, columns, summary


def _global_duplicate_summary(
    exact: dict[str, np.ndarray],
    feature: dict[str, np.ndarray],
    labels: np.ndarray,
) -> tuple[dict[str, object], dict[str, tuple[int, int]]]:
    all_exact = np.concatenate(list(exact.values()))
    all_feature = np.concatenate(list(feature.values()))
    unique_exact, exact_counts = np.unique(all_exact, return_counts=True)
    unique_feature, feature_counts = np.unique(all_feature, return_counts=True)

    benign_hashes = np.unique(all_feature[labels == 0])
    attack_hashes = np.unique(all_feature[labels == 1])
    conflict_hashes = np.intersect1d(benign_hashes, attack_hashes, assume_unique=True)
    conflict_mask = np.isin(all_feature, conflict_hashes)

    cross_file: dict[str, tuple[int, int]] = {}
    total_cross_exact = 0
    total_cross_feature = 0
    for name in exact:
        unique_file_exact, counts_file_exact = np.unique(
            exact[name], return_counts=True
        )
        unique_file_feature, counts_file_feature = np.unique(
            feature[name], return_counts=True
        )
        exact_global_counts = exact_counts[np.searchsorted(unique_exact, unique_file_exact)]
        feature_global_counts = feature_counts[
            np.searchsorted(unique_feature, unique_file_feature)
        ]
        cross_exact = int((exact_global_counts > counts_file_exact).sum())
        cross_feature = int((feature_global_counts > counts_file_feature).sum())
        cross_file[name] = (cross_exact, cross_feature)
        total_cross_exact += cross_exact
        total_cross_feature += cross_feature

    max_group = int(feature_counts.max())
    summary = {
        "total_rows": int(len(all_feature)),
        "exact_duplicate_rows": int(len(all_exact) - len(unique_exact)),
        "feature_duplicate_rows": int(len(all_feature) - len(unique_feature)),
        "exact_groups_with_duplicates": int((exact_counts > 1).sum()),
        "feature_groups_with_duplicates": int((feature_counts > 1).sum()),
        "rows_in_duplicate_feature_groups": int(
            feature_counts[feature_counts > 1].sum()
        ),
        "max_feature_group_size": max_group,
        "cross_file_exact_rows": total_cross_exact,
        "cross_file_feature_rows": total_cross_feature,
        "conflict_feature_groups": int(len(conflict_hashes)),
        "conflict_rows": int(conflict_mask.sum()),
        "conflict_benign_rows": int((conflict_mask & (labels == 0)).sum()),
        "conflict_attack_rows": int((conflict_mask & (labels == 1)).sum()),
    }
    return summary, cross_file


def _feature_decisions(columns: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    duplicate_of = {"Fwd Header Length__dup2": "Fwd Header Length"}
    rows: list[dict[str, object]] = []
    for name, stats in columns.items():
        is_constant = float(stats["min"]) == float(stats["max"])
        duplicate = duplicate_of.get(name)
        if duplicate:
            decision = "drop_duplicate"
            reason = f"Exact duplicate of '{duplicate}' across all rows."
        elif is_constant:
            decision = "drop_constant"
            reason = "Zero variance across the full dataset."
        elif name in config.INVALID_NEGATIVE_COLUMNS and int(stats["invalid_negative_cells"]):
            decision = "retain_impute"
            reason = "Impossible negative values are set to NaN and median-imputed."
        elif name in {"Init_Win_bytes_forward", "Init_Win_bytes_backward"}:
            decision = "retain_sentinel"
            reason = "Negative values are only -1; treated as a no-window sentinel."
        elif name == "Destination Port":
            decision = "retain_scenario_bias"
            reason = "Legitimate inference-time flow feature; port/scenario correlation noted."
        else:
            decision = "retain"
            reason = "No documented leakage or invalid-value issue."
        rows.append(
            {
                "feature": name,
                "dtype": stats["dtype"],
                "missing_cells": int(stats["missing_cells"]),
                "infinite_cells": int(stats["infinite_cells"]),
                "negative_cells": int(stats["negative_cells"]),
                "invalid_negative_cells": int(stats["invalid_negative_cells"]),
                "zero_cells": int(stats["zero_cells"]),
                "min": stats["min"],
                "max": stats["max"],
                "constant": is_constant,
                "duplicate_of": duplicate or "",
                "model_feature": decision.startswith("retain"),
                "decision": decision,
                "reason": reason,
            }
        )
    return rows


def run_audit() -> dict[str, object]:
    """Compute all audit tables and write them under results/."""

    config.ensure_directories()
    files = get_source_files()
    label_counts_all: Counter[str] = Counter()
    per_file: list[dict[str, object]] = []
    columns: dict[str, dict[str, object]] = {}
    audit_extra: dict[str, object] = {}
    exact_hashes: dict[str, np.ndarray] = {}
    feature_hashes: dict[str, np.ndarray] = {}

    LOGGER.info("Hashing raw rows for duplicate analysis")
    exact_hashes, feature_hashes, labels_array_np, label_counts_all = _hash_source_rows(files)
    labels = np.asarray(labels_array_np, dtype=np.uint8)

    LOGGER.info("Computing per-column quality statistics")
    per_file, columns, audit_extra = _source_statistics(files)
    duplicate_summary, cross_file = _global_duplicate_summary(
        exact_hashes, feature_hashes, labels
    )

    file_offsets: list[tuple[str, int, int]] = []
    position = 0
    for entry in per_file:
        n = int(entry["rows"])
        file_offsets.append((str(entry["file"]), position, position + n))
        position += n
    if position != len(labels):
        raise RuntimeError("Label alignment failed during audit.")

    label_rows: list[dict[str, object]] = []
    for raw_label, count in sorted(
        label_counts_all.items(), key=lambda item: item[1], reverse=True
    ):
        label_rows.append(
            {
                "raw_label": raw_label,
                "binary_label": config.BINARY_LABEL_MAP[raw_label],
                "binary_name": config.BINARY_LABEL_NAMES[config.BINARY_LABEL_MAP[raw_label]],
                "count": int(count),
                "percent": round(100.0 * count / len(labels), 6),
            }
        )
    class_distribution = pd.DataFrame(label_rows)
    class_distribution.to_csv(
        config.RESULTS_DIR / "class_distribution.csv", index=False, encoding="utf-8"
    )

    for entry in per_file:
        name = str(entry["file"])
        start, end = next((start, end) for fname, start, end in file_offsets if fname == name)
        file_labels = labels[start:end]
        entry["benign"] = int((file_labels == 0).sum())
        entry["attack"] = int((file_labels == 1).sum())
        entry["exact_duplicate_rows"] = int(
            len(exact_hashes[name]) - np.unique(exact_hashes[name]).size
        )
        entry["feature_duplicate_rows"] = int(
            len(feature_hashes[name]) - np.unique(feature_hashes[name]).size
        )
        entry["cross_file_exact_rows"] = int(cross_file[name][0])
        entry["cross_file_feature_rows"] = int(cross_file[name][1])
        entry["missing_cells"] = int(entry["missing_cells"])
        entry["infinite_cells"] = int(entry["infinite_cells"])
        entry["invalid_rows"] = int(entry["invalid_rows"])

    dataset_summary = pd.DataFrame(per_file)
    total_row = {
        "file": "TOTAL",
        "rows": int(dataset_summary["rows"].sum()),
        "columns": int(config.RAW_COLUMN_COUNT),
        "size_bytes": int(dataset_summary["size_bytes"].sum()),
        "benign": int(dataset_summary["benign"].sum()),
        "attack": int(dataset_summary["attack"].sum()),
        "missing_cells": int(dataset_summary["missing_cells"].sum()),
        "infinite_cells": int(dataset_summary["infinite_cells"].sum()),
        "rows_any_missing": int(dataset_summary["rows_any_missing"].sum()),
        "rows_any_infinite": int(dataset_summary["rows_any_infinite"].sum()),
        "rows_missing_or_infinite": int(dataset_summary["rows_missing_or_infinite"].sum()),
        "invalid_rows": int(dataset_summary["invalid_rows"].sum()),
        "exact_duplicate_rows": int(duplicate_summary["exact_duplicate_rows"]),
        "feature_duplicate_rows": int(duplicate_summary["feature_duplicate_rows"]),
        "cross_file_exact_rows": int(dataset_summary["cross_file_exact_rows"].sum()),
        "cross_file_feature_rows": int(dataset_summary["cross_file_feature_rows"].sum()),
    }
    dataset_summary = pd.concat(
        [dataset_summary, pd.DataFrame([total_row])], ignore_index=True
    )
    dataset_summary.to_csv(
        config.RESULTS_DIR / "dataset_summary.csv", index=False, encoding="utf-8"
    )

    feature_audit = pd.DataFrame(_feature_decisions(columns))
    feature_audit.to_csv(
        config.RESULTS_DIR / "feature_audit.csv", index=False, encoding="utf-8"
    )
    quality_columns = feature_audit[
        [
            "feature",
            "dtype",
            "missing_cells",
            "infinite_cells",
            "negative_cells",
            "invalid_negative_cells",
            "zero_cells",
            "min",
            "max",
            "constant",
            "duplicate_of",
        ]
    ].copy()
    quality_columns.to_csv(
        config.RESULTS_DIR / "data_quality_report.csv",
        index=False,
        encoding="utf-8",
    )

    duplicate_rows = [
        {"metric": key, "value": value}
        for key, value in duplicate_summary.items()
    ]
    duplicate_rows.append(
        {
            "metric": "duplicate_column_mismatches",
            "value": int(audit_extra["duplicate_columns"][0]["mismatching_cells"]),
        }
    )
    duplicate_rows.append(
        {
            "metric": "invalid_rows_benign",
            "value": int(audit_extra["invalid_by_binary"][0]),
        }
    )
    duplicate_rows.append(
        {
            "metric": "invalid_rows_attack",
            "value": int(audit_extra["invalid_by_binary"][1]),
        }
    )
    pd.DataFrame(duplicate_rows).to_csv(
        config.RESULTS_DIR / "duplicate_summary.csv",
        index=False,
        encoding="utf-8",
    )

    summary = {
        "source_directory": str(config.SOURCE_DIR),
        "files": [path.name for path in files],
        "rows": total_row["rows"],
        "columns": total_row["columns"],
        "benign": total_row["benign"],
        "attack": total_row["attack"],
        "attack_share": round(total_row["attack"] / total_row["rows"], 6),
        "missing_cells": total_row["missing_cells"],
        "infinite_cells": total_row["infinite_cells"],
        "rows_missing_or_infinite": total_row["rows_missing_or_infinite"],
        "invalid_rows": total_row["invalid_rows"],
        "invalid_rows_benign": audit_extra["invalid_by_binary"][0],
        "invalid_rows_attack": audit_extra["invalid_by_binary"][1],
        "duplicate_summary": duplicate_summary,
        "label_counts": dict(label_counts_all),
        "constant_columns": list(config.CONSTANT_COLUMNS),
        "dropped_columns": list(config.DROP_COLUMNS),
        "candidate_feature_count": int(feature_audit["model_feature"].sum()),
    }
    with (config.RESULTS_DIR / "audit_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    LOGGER.info("Wrote audit tables to %s", config.RESULTS_DIR)
    return summary


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    results = run_audit()
    print(json.dumps(results, indent=2)[:4000])
