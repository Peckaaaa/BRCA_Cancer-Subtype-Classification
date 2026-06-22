"""
XGBoost classifier with GridSearchCV hyper-parameter tuning.

Extracted from the original `Code/XGBoost/run_brca_feature_selection_xgboost.py`.
Falls back to a RandomForest classifier when the `xgboost` package is not
installed (same behaviour as the original code).
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import GridSearchCV

from ..evaluation.metrics import compute_metrics

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    from sklearn.ensemble import RandomForestClassifier as XGBClassifier
    XGB_AVAILABLE = False


def train_and_evaluate(X_train, y_train, X_test, y_test, num_classes, cfg: dict) -> dict:
    cfg = cfg or {}
    seed = int(cfg.get("seed", 42))
    n_estimators = int(cfg.get("n_estimators", 400))
    param_grid = cfg.get("param_grid", {"max_depth": [3, 5, 7], "learning_rate": [0.01, 0.1, 0.2]})
    scoring = cfg.get("scoring", "f1_weighted")
    cv = int(cfg.get("cv", 3))

    if XGB_AVAILABLE:
        base_model = XGBClassifier(
            n_estimators=n_estimators, random_state=seed, eval_metric="mlogloss"
        )
    else:
        # RandomForest fallback ignores learning_rate; restrict the grid.
        base_model = XGBClassifier(n_estimators=n_estimators, random_state=seed)
        param_grid = {k: v for k, v in param_grid.items() if k == "max_depth"} or {
            "max_depth": [None, 5, 7]
        }

    search = GridSearchCV(
        estimator=base_model, param_grid=param_grid, scoring=scoring, cv=cv, n_jobs=-1
    )
    search.fit(np.asarray(X_train), np.asarray(y_train))
    best = search.best_estimator_

    y_pred = best.predict(X_test)
    try:
        y_prob = best.predict_proba(X_test)
    except Exception:
        y_prob = None
    return compute_metrics(y_test, y_pred, y_prob=y_prob)
