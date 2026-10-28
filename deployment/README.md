# FastAPI Deployment (Network Intrusion Detection)

## Model
- Selected model: `flaml_automl` (FLAML-selected XGBoost) saved at `models/flaml_automl.joblib`
- Pipeline: `SimpleImputer(strategy='median')` + XGBoost classifier (preprocessing embedded in pipeline)
- Features: 69 features from cleaned CIC-IDS2017 schema (exact order from `data/processed/clean_meta.npz`)

## Endpoints
- `GET /health` - health status and model path
- `GET /metadata` - feature schema and model info
- `POST /predict` - predict on pre-extracted flow features

## Running
```bash
# from project root
export NID_MODEL_PATH=models/flaml_automl.joblib  # optional
uvicorn deployment.app:app --host 0.0.0.0 --port 8000
```

## Notes
- Only accepts pre-extracted CIC-IDS2017 flow features (69 features in exact schema). Does **not** process raw PCAP files.
- All features required; unknown/extra features rejected. NaN/Inf rejected.
- Probability returned if model supports `predict_proba`.
- Model loaded at startup (no retraining).
- Use `deployment/example_request.json` as a template.
