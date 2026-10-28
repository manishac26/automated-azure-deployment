# Model Card: flaml_automl (XGBoost)

- Task: Binary intrusion detection (BENIGN vs ATTACK)
- Dataset: CIC-IDS2017 (cleaned subset)
- Features: 69 flow features
- Preprocessing: Median imputation (in pipeline)
- Training: FLAML on 100k benchmark subset (180s)
- Threshold: Tuned on validation (F1-optimal)
- Metrics: See results/verified_results_table.csv
- Limitations: Trained on subset; random split may overestimate; no PCAP processing
- Ethical: For research/eval only
