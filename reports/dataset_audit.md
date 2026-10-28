# Dataset Audit (CIC-IDS2017)

Source: MachineLearningCSV/MachineLearningCVE (8 CSVs). Headers normalized, Label stripped.

## Summary (from results/audit_summary.json)

- Rows: 2830743, Cols: 79
- BENIGN: 2273097 (80.300%), ATTACK: 557646 (19.700%)
- Missing cells: 1358, Infinite: 4376, Rows with missing/infinite: 2867
- Invalid rows: 5792 (benign 4508, attack 1284)
- Constant columns dropped: ['Bwd PSH Flags', 'Bwd URG Flags', 'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate']
- Dropped columns: ['Bwd PSH Flags', 'Bwd URG Flags', 'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate', 'Fwd Header Length__dup2']
- Candidate features: 69

## Clean dataset

- Clean rows: 2498105, Features: 69
- BENIGN: 2072961 (82.981%), ATTACK: 425144 (17.019%)
- Conflicts removed: 719 groups / 7144 rows
- Exact duplicates removed: 325494
