"""
Model registry. Each backend exposes:

    train_and_evaluate(X_train, y_train, X_test, y_test, num_classes, cfg) -> dict

returning the 5 macro metrics (see evaluation.metrics.METRIC_KEYS).
"""
from __future__ import annotations

from . import mlp, logreg, xgb, knn, svm

MODELS = {
    "mlp": mlp.train_and_evaluate,
    "logreg": logreg.train_and_evaluate,
    "xgboost": xgb.train_and_evaluate,
    "knn": knn.train_and_evaluate,
    "svm": svm.train_and_evaluate,
}


def get_model(name: str):
    key = name.strip().lower()
    if key not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Choices: {list(MODELS)}")
    return MODELS[key]


__all__ = ["MODELS", "get_model", "mlp", "logreg", "xgb", "knn", "svm"]
