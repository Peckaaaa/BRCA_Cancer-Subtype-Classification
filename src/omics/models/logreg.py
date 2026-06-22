"""
Logistic Regression (ElasticNet) classifier.

Extracted from the original `Code/RF/ML/model.py`. Handles the scikit-learn
>= 1.8 API change for the elasticnet penalty transparently.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning

from ..evaluation.metrics import compute_metrics

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def _make_elasticnet_lr(C=1.0, l1_ratio=0.5, max_iter=5000, random_state=42):
    import sklearn
    from packaging.version import Version

    kwargs = dict(
        solver="saga", l1_ratio=l1_ratio, C=C, max_iter=max_iter,
        random_state=random_state, class_weight="balanced",
    )
    if Version(sklearn.__version__) < Version("1.8"):
        kwargs["penalty"] = "elasticnet"
        kwargs["n_jobs"] = -1
    return LogisticRegression(**kwargs)


def train_and_evaluate(X_train, y_train, X_test, y_test, num_classes, cfg: dict) -> dict:
    cfg = cfg or {}
    clf = _make_elasticnet_lr(
        C=float(cfg.get("C", 1.0)),
        l1_ratio=float(cfg.get("l1_ratio", 0.5)),
        max_iter=int(cfg.get("max_iter", 5000)),
        random_state=int(cfg.get("seed", 42)),
    )
    clf.fit(np.asarray(X_train), np.asarray(y_train))
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)
    return compute_metrics(y_test, y_pred, y_prob=y_prob)
