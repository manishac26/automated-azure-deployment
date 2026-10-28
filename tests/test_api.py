"""API tests for the FastAPI deployment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from deployment import app as app_module


@pytest.fixture(autouse=True)
def reset_model():
    app_module._model = None
    app_module._feature_names = None
    app_module._model_path = None
    yield
    app_module._model = None
    app_module._feature_names = None
    app_module._model_path = None


def test_health_no_model():
    client = TestClient(app_module.app)
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("ok", "error")
    assert "model_loaded" in data


def test_metadata_no_model():
    client = TestClient(app_module.app)
    resp = client.get("/metadata")
    assert resp.status_code == 503


def test_predict_no_model():
    client = TestClient(app_module.app)
    resp = client.post("/predict", json={"features": {}})
    assert resp.status_code == 503


def test_predict_missing_feature_with_mock():
    # mock model and features
    app_module._feature_names = ["f1", "f2", "f3"]
    class Dummy:
        def predict(self, X):
            return np.array([0])
        def predict_proba(self, X):
            return np.array([[0.9, 0.1]])
    app_module._model = Dummy()
    app_module._model_path = Path("dummy")
    client = TestClient(app_module.app)
    resp = client.post("/predict", json={"features": {"f1": 1.0, "f2": 2.0}})
    assert resp.status_code == 422
    assert "Missing" in resp.text


def test_predict_extra_feature_with_mock():
    app_module._feature_names = ["f1", "f2"]
    class Dummy:
        def predict(self, X): return np.array([0])
        def predict_proba(self, X): return np.array([[1.0, 0.0]])
    app_module._model = Dummy()
    app_module._model_path = Path("dummy")
    client = TestClient(app_module.app)
    resp = client.post("/predict", json={"features": {"f1": 1.0, "f2": 2.0, "f3": 3.0}})
    assert resp.status_code == 422
    assert "Extra" in resp.text or "unknown" in resp.text.lower() or "Unexpected" in resp.text


def test_predict_nan_with_mock():
    app_module._feature_names = ["f1", "f2"]
    class Dummy:
        def predict(self, X): return np.array([0])
        def predict_proba(self, X): return np.array([[1.0, 0.0]])
    app_module._model = Dummy()
    app_module._model_path = Path("dummy")
    client = TestClient(app_module.app)
    resp = client.post("/predict", json={"features": {"f1": "nan", "f2": 2.0}})
    assert resp.status_code in (400,422)


def test_predict_inf_with_mock():
    app_module._feature_names = ["f1", "f2"]
    class Dummy:
        def predict(self, X): return np.array([0])
        def predict_proba(self, X): return np.array([[1.0, 0.0]])
    app_module._model = Dummy()
    app_module._model_path = Path("dummy")
    client = TestClient(app_module.app)
    resp = client.post("/predict", json={"features": {"f1": "inf", "f2": 2.0}})
    assert resp.status_code in (400,422)


def test_predict_valid_with_mock():
    app_module._feature_names = ["f1", "f2"]
    class Dummy:
        def predict(self, X): return np.array([1])
        def predict_proba(self, X): return np.array([[0.2, 0.8]])
    app_module._model = Dummy()
    app_module._model_path = Path("dummy")
    client = TestClient(app_module.app)
    resp = client.post("/predict", json={"features": {"f1": 1.0, "f2": 2.0}})
    assert resp.status_code == 200
    data = resp.json()
    assert data["predicted_label"] == 1
    assert data["predicted_label_name"] == "ATTACK"
    assert "probability" in data
