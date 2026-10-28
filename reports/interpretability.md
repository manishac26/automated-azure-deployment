# Interpretability Analysis

## Method
- Model: `flaml_automl` (FLAML-selected XGBoost classifier) saved as `models/flaml_automl.joblib` (pipeline includes median imputation).
- Method: Permutation importance on F1 (scikit-learn `permutation_importance`), computed on a representative random sample of the test set due to computational limits.
- Sample size: 10,000 test samples (seed 42), n_repeats=10, n_jobs=2.
- Artifacts: `results/feature_importance.csv`, `results/feature_importance_summary.json`, `results/figures/feature_importance.png`.

## Notes
- Feature importance reflects model dependence/association, not causation.
- Importance was measured in terms of mean decrease in F1 when permuting a feature.
- Results should be interpreted in the context of CIC-IDS2017 flow features and potential artifacts from preprocessing/deduplication.

## Top features (by permutation importance)
See `results/feature_importance.csv` for full ranking.
