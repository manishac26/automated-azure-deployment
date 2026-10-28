"""Integration test for the real saved model if available."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from deployment import app as app_module
from src import config


def test_integration_real_model_if_exists():
    model_path = config.MODELS_DIR / "flaml_automl.joblib"
    if not model_path.exists():
        pytest.skip("Real model not present")
    app_module._model = None
    app_module._feature_names = None
    app_module._model_path = None
    # load
    from deployment.app import load_model

    load_model()
    client = TestClient(app_module.app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["model_loaded"] is True
    resp = client.get("/metadata")
    assert resp.status_code == 200
    meta = resp.json()
    assert meta["feature_count"] == len(meta["feature_names"])
    # example
    ex_path = Path("deployment/example_request.json")
    if ex_path.exists():
        ex = json.loads(ex_path.read_text())
        resp = client.post("/predict", json=ex)
        assert resp.status_code == 200
        data = resp.json()
        assert data["predicted_label"] in (0, 1)
    # cleanup
    app_module._model = None
    app_module._feature_names = None
    app_module._model_path = None
