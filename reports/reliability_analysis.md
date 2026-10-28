# Reliability and Leakage Analysis

## Summary
- Source files are from different days/files (8 CSVs). No timestamp or explicit timestamp column exists in the feature set (checked for time/date/timestamp). Source file provenance is preserved in metadata (`source_code`).
- Preprocessing (see `src/preprocessing.py`): streaming pass 1 identifies conflict groups by exact feature-vector hash but conflicting labels (698 groups, 7,020 rows) and removes them. Pass 1 also drops exact duplicate feature vectors (325,494 rows removed as reported in metadata). Invalid values are imputed rather than deleted broadly. Constants/duplicate columns dropped as per audit.
- Label distribution changes after cleaning (from audit): original 2,830,743 rows (BENIGN 80.300%, ATTACK 19.700%); cleaned 2,498,105 rows (BENIGN 82.981%, ATTACK 17.019%). Conflict removal affects distribution slightly.
- Duplicate handling: `src/data_loader.py`/preprocessing build hashes; metadata stores `hashes` aligned to clean rows. No exact duplicate feature vectors remain in the cleaned set (verified: 0 duplicate hashes in clean set). No cross-partition duplicates found in benchmark/full splits (overlap 0).
- Group IDs for duplicate groups were not persisted as explicit group identifiers in the saved artifacts; only hashes and aggregate counts are available.

## Split integrity (benchmark/full)
- Benchmark split: no hash overlap across train/val/test (verified). File-level distribution across splits is stratified/random by flow (no chronological split). Per-file hash overlap across splits is 0.
- Random split across mixed days/files risks overestimating generalization if attack patterns are day/file-specific (CIC-IDS2017 is collected over multiple days). No true chronological metadata is available.

## Robustness test (separate, non-destructive)
- Day/file hold-out: train on files 0-5, validation on Tuesday (`Tuesday-WorkingHours.pcap_ISCX.csv`, file 6), test on Wednesday (`Wednesday-workingHours.pcap_ISCX.csv`, file 7). Trained a stratified 150k sample of the training-day set with RF (n_estimators=100, seed 42) and tuned threshold on validation day.
- Results (see `results/reliability_comparison.csv`, `results/reliability_day_split.json`, figures `fig_reliability_dayholdout_*`): test F1 0.9511, precision 0.9282, recall 0.9752, PR-AUC 0.9800, ROC-AUC 0.9859 (with very low threshold tuned on val). This is substantially lower than random-split results, indicating potential generalization drop across days/files - consistent with known CIC-IDS2017 behavior. This evaluation is kept separate from original experiments.

## Recommendations
- Report both random-split and day/file hold-out results with clear qualification.
- If stronger group-aware evaluation is needed, use file/day as groups (available via source provenance). Chronological order not present.
- Do not assume random splits are leak-free; document the above.
