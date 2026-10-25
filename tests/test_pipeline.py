import os

import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import load_breast_cancer


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    model_dir = tmp_path_factory.mktemp("models")
    os.environ["MODEL_DIR"] = str(model_dir)
    os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{model_dir}/mlflow.db"
    import importlib

    from src import serve, train

    importlib.reload(train)
    meta = train.main()
    importlib.reload(serve)
    return meta, serve


def test_training_passes_quality_gate(trained):
    meta, _ = trained
    assert meta["metrics"]["f1"] >= 0.90


def test_predict_endpoint(trained):
    _, serve = trained
    with TestClient(serve.app) as client:
        assert client.get("/health").json()["model_loaded"] is True
        row = load_breast_cancer(as_frame=True).data.iloc[0].to_dict()
        r = client.post("/predict", json={"instances": [row]})
        assert r.status_code == 200
        body = r.json()
        assert body["predictions"][0] in (0, 1)
        assert 0.0 <= body["probabilities"][0] <= 1.0
        assert "predict_requests_total" in client.get("/metrics").text


def test_missing_features_rejected(trained):
    _, serve = trained
    with TestClient(serve.app) as client:
        r = client.post("/predict", json={"instances": [{"mean radius": 1.0}]})
        assert r.status_code == 422
