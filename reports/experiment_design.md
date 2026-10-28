# Experiment Design

- Seed: 42
- Split: stratified 70/15/15 full + fixed 250k/50k/50k benchmark
- Threshold tuning: F1-optimal on validation
- Baseline: RF (sklearn), untuned
- HPO: Grid/Random/Optuna on 100k subset, 3-fold CV, 40 trees search then refit 200 trees
- AutoML: FLAML (XGBoost) 180s budget on 100k; AutoGluon not successfully run in this env
- Metrics: precision/recall/F1, ROC-AUC, PR-AUC, confusion matrix
