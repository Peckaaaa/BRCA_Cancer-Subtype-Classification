"""
Support Vector Machine classifier (RBF kernel).

Extracted from the SVM pipeline `Code/Code/train.py`:
    SVC(kernel="rbf", probability=True, random_state=42)

Integrated into the unified benchmark so it consumes the SAME shared feature
selection, data loader and (macro) evaluation as every other model.
"""
from __future__ import annotations

import numpy as np
from sklearn.svm import SVC

from ..evaluation.metrics import compute_metrics


def train_and_evaluate(X_train, y_train, X_test, y_test, num_classes, cfg: dict) -> dict:
    cfg = cfg or {}
    clf = SVC(
        kernel=cfg.get("kernel", "rbf"),
        C=float(cfg.get("C", 1.0)),
        gamma=cfg.get("gamma", "scale"),
        probability=bool(cfg.get("probability", True)),
        class_weight=cfg.get("class_weight", None),
        random_state=int(cfg.get("seed", 42)),
    )
    clf.fit(np.asarray(X_train), np.asarray(y_train))
    y_pred = clf.predict(X_test)
    try:
        y_prob = clf.predict_proba(X_test)
    except Exception:
        y_prob = None
    return compute_metrics(y_test, y_pred, y_prob=y_prob)
