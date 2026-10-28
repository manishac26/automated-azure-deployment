"""Project-wide configuration for the CIC-IDS2017 AutoML study.

All paths are derived from the repository root so the project can be moved
without editing code. Random behaviour is controlled by ``RANDOM_STATE``.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_DIR = PROJECT_ROOT / "MachineLearningCSV" / "MachineLearningCVE"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

CLEAN_FEATURES_PATH = PROCESSED_DIR / "clean_features.npy"
CLEAN_META_PATH = PROCESSED_DIR / "clean_meta.npz"
SCHEMA_PATH = PROCESSED_DIR / "schema.json"
SPLIT_PATH = PROCESSED_DIR / "split_indices.npz"
BENCHMARK_PATH = PROCESSED_DIR / "benchmark_indices.npz"

RANDOM_STATE = 42

LABEL_COLUMN = "Label"
BENIGN_LABEL = "BENIGN"
ATTACK_LABELS = (
    "Bot",
    "DDoS",
    "DoS GoldenEye",
    "DoS Hulk",
    "DoS Slowhttptest",
    "DoS slowloris",
    "FTP-Patator",
    "Heartbleed",
    "Infiltration",
    "PortScan",
    "SSH-Patator",
    "Web Attack \ufffd Brute Force",
    "Web Attack \ufffd Sql Injection",
    "Web Attack \ufffd XSS",
)
LABEL_NAMES = (BENIGN_LABEL,) + ATTACK_LABELS
BINARY_LABEL_MAP = {BENIGN_LABEL: 0, **{label: 1 for label in ATTACK_LABELS}}
BINARY_LABEL_NAMES = {0: "BENIGN", 1: "ATTACK"}

CONSTANT_COLUMNS = (
    "Bwd PSH Flags",
    "Bwd URG Flags",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
)
DUPLICATE_COLUMNS = ("Fwd Header Length__dup2",)
DROP_COLUMNS = CONSTANT_COLUMNS + DUPLICATE_COLUMNS

INVALID_NEGATIVE_COLUMNS = (
    "Flow Duration",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Min",
    "Fwd Header Length",
    "Bwd Header Length",
    "min_seg_size_forward",
)

RAW_COLUMN_COUNT = 79
RAW_FEATURE_COUNT = 78

FULL_TRAIN_FRACTION = 0.70
FULL_VAL_FRACTION = 0.15
FULL_TEST_FRACTION = 0.15

BENCHMARK_TRAIN_SIZE = 250_000
BENCHMARK_VAL_SIZE = 50_000
BENCHMARK_TEST_SIZE = 50_000

CV_FOLDS = 3
GRID_N_JOBS = 1
RF_N_JOBS = 4
AUTOML_TIME_BUDGET_SECONDS = 900


def ensure_directories() -> None:
    """Create the output directories used by the project."""

    for directory in (
        DATA_DIR,
        RAW_DIR,
        PROCESSED_DIR,
        REPORTS_DIR,
        RESULTS_DIR,
        FIGURES_DIR,
        MODELS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
