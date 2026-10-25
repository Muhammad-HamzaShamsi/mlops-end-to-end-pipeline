import importlib
import os

import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import load_breast_cancer


@pytest.fixture(scope="session")
def client(tmp_path_factory):
    model_dir = tmp_path_factory.mktemp("models")
    os.environ["MODEL_DIR"] = str(model_dir)
    os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{model_dir}/mlflow.db"

    from src import train

    importlib.reload(train)  # pick up MODEL_DIR set above
    meta = train.main()

    from src import serve  # imported once: Prometheus metrics must only register once

    with TestClient(serve.app) as c:
        c.meta = meta
        yield c


def test_training_passes_quality_gate(client):
    assert client.meta["metrics"]["f1"] >= 0.90


def test_predict_endpoint(client):
    assert client.get("/health").json()["model_loaded"] is True
    row = load_breast_cancer(as_frame=True).data.iloc[0].to_dict()
    r = client.post("/predict", json={"instances": [row]})
    assert r.status_code == 200
    body = r.json()
    assert body["predictions"][0] in (0, 1)
    assert 0.0 <= body["probabilities"][0] <= 1.0
    assert "predict_requests_total" in client.get("/metrics").text


def test_missing_features_rejected(client):
    r = client.post("/predict", json={"instances": [{"mean radius": 1.0}]})
    assert r.status_code == 422
