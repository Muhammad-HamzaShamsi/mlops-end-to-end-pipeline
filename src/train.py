"""Train a breast-cancer classifier, track it with MLflow, and export the best model.

Usage:
    python -m src.train                       # default hyper-parameter sweep
    MLFLOW_TRACKING_URI=http://localhost:5000 python -m src.train
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "breast-cancer-classifier")
MIN_F1 = float(os.getenv("MIN_F1", "0.90"))  # quality gate used in CI

CANDIDATES = {
    "logreg_c0.1": lambda: LogisticRegression(C=0.1, max_iter=2000),
    "logreg_c1": lambda: LogisticRegression(C=1.0, max_iter=2000),
    "rf_100": lambda: RandomForestClassifier(n_estimators=100, random_state=42),
    "rf_300_d8": lambda: RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42),
}


def load_data():
    data = load_breast_cancer(as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )
    return X_train, X_test, y_train, y_test, list(data.feature_names)


def evaluate(model, X_test, y_test) -> dict[str, float]:
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, proba),
    }


def main() -> dict:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(EXPERIMENT)
    X_train, X_test, y_train, y_test, features = load_data()

    best = {"name": None, "f1": -1.0, "model": None, "metrics": None}
    for name, build in CANDIDATES.items():
        with mlflow.start_run(run_name=name):
            pipe = Pipeline([("scaler", StandardScaler()), ("clf", build())])
            pipe.fit(X_train, y_train)
            metrics = evaluate(pipe, X_test, y_test)
            mlflow.log_params({"candidate": name, **pipe.named_steps["clf"].get_params()})
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(pipe, artifact_path="model")
            print(f"{name:12s} " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
            if metrics["f1"] > best["f1"]:
                best = {"name": name, "f1": metrics["f1"], "model": pipe, "metrics": metrics}

    if best["f1"] < MIN_F1:
        raise SystemExit(f"Quality gate failed: best F1 {best['f1']:.3f} < {MIN_F1}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["model"], MODEL_DIR / "model.joblib")
    meta = {"model_name": best["name"], "metrics": best["metrics"], "features": features}
    (MODEL_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"\nBest model: {best['name']} (F1={best['f1']:.4f}) -> {MODEL_DIR / 'model.joblib'}")
    return meta


if __name__ == "__main__":
    main()
