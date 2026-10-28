"""FastAPI deployment for the intrusion detection model."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from deployment.schemas import (
    HealthResponse,
    MetadataResponse,
    PredictionRequest,
    PredictionResponse,
)
from src import config
from src.data_loader import load_clean_meta

app = FastAPI(
    title="Network Intrusion Detection API",
    description="CIC-IDS2017 binary intrusion detection using the saved model",
    version="1.0.0",
)

# CORS for local frontend development
_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_feature_names = None
_model_name = "flaml_automl"
_model_path = None


def _get_model_path() -> Path:
    env_path = os.environ.get("NID_MODEL_PATH")
    if env_path:
        return Path(env_path)
    return config.MODELS_DIR / f"{_model_name}.joblib"


def load_model() -> None:
    global _model, _feature_names, _model_path
    try:
        meta = load_clean_meta()
        _feature_names = [str(x) for x in meta["feature_names"]]
        _model_path = _get_model_path()
        if not _model_path.exists():
            raise FileNotFoundError(f"Model file not found: {_model_path}")
        _model = joblib.load(_model_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to load model: {exc}") from exc


@app.on_event("startup")
def _startup() -> None:
    load_model()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if _model is not None else "error",
        model_loaded=_model is not None,
        model_path=str(_model_path) if _model_path else str(_get_model_path()),
    )


@app.get("/metadata", response_model=MetadataResponse)
def metadata() -> MetadataResponse:
    if _feature_names is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return MetadataResponse(
        model_name=_model_name,
        model_path=str(_model_path) if _model_path else str(_get_model_path()),
        feature_names=_feature_names,
        feature_count=len(_feature_names),
        label_map={"BENIGN": 0, "ATTACK": 1},
        preprocessing="Pipeline with sklearn.impute.SimpleImputer(strategy='median') as saved in artifact",
        notes="Does not process raw PCAP files. Expects pre-extracted CIC-IDS2017 flow features in the exact schema.",
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    if _model is None or _feature_names is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    features = request.features
    missing = [f for f in _feature_names if f not in features]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing required features: {missing[:10]}")

    extra = [f for f in features if f not in _feature_names]
    if extra:
        raise HTTPException(status_code=422, detail=f"Unexpected features not in schema: {extra[:10]}")

    # build in order with strict checks
    row = []
    for name in _feature_names:
        if name not in features:
            raise HTTPException(status_code=422, detail=f"Missing required feature: {name}")
        val = features[name]
        if isinstance(val, bool):
            raise HTTPException(status_code=422, detail=f"Invalid numeric value for feature '{name}'")
        if isinstance(val, str):
            s = val.strip().lower()
            if s in ("nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"):
                raise HTTPException(status_code=422, detail=f"Invalid numeric value for feature '{name}'")
        try:
            fval = float(val)
        except Exception:
            raise HTTPException(status_code=422, detail=f"Invalid numeric value for feature '{name}'")
        if math.isnan(fval) or math.isinf(fval):
            raise HTTPException(status_code=422, detail=f"Invalid numeric value for feature '{name}'")
        row.append(fval)

    X = np.array([row], dtype=float)
    try:
        pred = int(_model.predict(X)[0])
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Prediction failed") from exc

    prob = None
    try:
        if hasattr(_model, "predict_proba"):
            proba = _model.predict_proba(X)[0]
            if len(proba) == 2:
                prob = float(proba[1])
            elif len(proba) > 0:
                prob = float(proba[-1])
            else:
                prob = None
    except Exception:
        prob = None

    return PredictionResponse(
        predicted_label=pred,
        predicted_label_name="ATTACK" if pred == 1 else "BENIGN",
        probability=prob,
        model_name=_model_name,
        feature_count=len(_feature_names),
    )
