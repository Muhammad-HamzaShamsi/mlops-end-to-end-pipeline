"""FastAPI inference service with health checks and Prometheus metrics."""
from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field


def model_dir() -> Path:
    """Resolved at call time so tests and containers can override MODEL_DIR."""
    return Path(os.getenv("MODEL_DIR", "models"))


REQUESTS = Counter("predict_requests_total", "Prediction requests", ["outcome"])
LATENCY = Histogram("predict_latency_seconds", "Prediction latency in seconds")
PREDICTED = Counter("predicted_class_total", "Predicted class counts", ["label"])

_state: dict = {}


def load_model() -> None:
    _state["model"] = joblib.load(model_dir() / "model.joblib")
    _state["meta"] = json.loads((model_dir() / "metadata.json").read_text())


@asynccontextmanager
async def lifespan(_: FastAPI):
    if (model_dir() / "model.joblib").exists():
        load_model()
    yield


app = FastAPI(title="Breast Cancer Classifier API", version="1.0.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    instances: list[dict[str, float]] = Field(..., min_length=1, description="Feature dicts")


class PredictResponse(BaseModel):
    model: str
    predictions: list[int]
    probabilities: list[float]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in _state}


@app.get("/model")
def model_info():
    if "meta" not in _state:
        raise HTTPException(503, "Model not loaded")
    return _state["meta"]


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if "model" not in _state:
        REQUESTS.labels("error").inc()
        raise HTTPException(503, "Model not loaded. Run `python -m src.train` first.")
    features = _state["meta"]["features"]
    df = pd.DataFrame(req.instances)
    missing = set(features) - set(df.columns)
    if missing:
        REQUESTS.labels("bad_request").inc()
        raise HTTPException(422, f"Missing features: {sorted(missing)[:5]}...")
    start = time.perf_counter()
    model = _state["model"]
    preds = model.predict(df[features]).tolist()
    proba = model.predict_proba(df[features])[:, 1].round(4).tolist()
    LATENCY.observe(time.perf_counter() - start)
    REQUESTS.labels("success").inc()
    for p in preds:
        PREDICTED.labels(str(p)).inc()
    return PredictResponse(
        model=_state["meta"]["model_name"], predictions=preds, probabilities=proba
    )


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
