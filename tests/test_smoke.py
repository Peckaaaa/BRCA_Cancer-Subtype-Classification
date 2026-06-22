"""
Lightweight smoke tests — run every feature-selection method and every model
on a tiny synthetic dataset, so imports and the full FS->model->metrics path are
exercised without touching the large BRCA matrices.

Run:
    PYTHONPATH=src python -m pytest tests/test_smoke.py      # if pytest installed
    PYTHONPATH=src python tests/test_smoke.py                # plain run
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from omics.feature_selection import select_features, METHODS as FS_METHODS
from omics.models import get_model, MODELS
from omics.evaluation.metrics import METRIC_KEYS


def _toy_data(n=80, d=40, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(float)
    y = rng.integers(0, 3, size=n)
    # make a few features informative
    for c in range(3):
        X[y == c, c] += 3.0
    omics = ["mRNA", "CNV", "miRNA", "Methy"]
    names = [f"{omics[i % 4]}__G{i}" for i in range(d)]
    # ensure a couple of PAM50 gene symbols exist for the pam50 method
    names[0] = "mRNA__ESR1"
    names[1] = "mRNA__ERBB2"
    return X, y, names


FS_CFG = {
    "seed": 0, "k": 8,
    "fisher": {"k": 8},
    "l1": {"k": 8, "C": 0.1, "solver": "liblinear", "max_iter": 500},
    "snr": {"k_total": 8},
    "mrmr": {"k": 8, "redundancy_weight": 0.3},
    "relieff": {"k": 8, "n_iter": 30, "k_neighbors": 5},
    "elmo": {"l1_c": 0.1, "lr_c": 1.0, "ifs_max_k": 10, "n_inner": 2, "rfe_step": 0.5},
    "pam50": {"omics": None},
}

MODEL_CFG = {
    "mlp": {"epochs": 5, "batch_size": 16, "patience": 3, "hidden_dims": [16], "verbose": False, "seed": 0},
    "logreg": {"C": 1.0, "max_iter": 500, "seed": 0},
    "xgboost": {"n_estimators": 20, "cv": 2, "param_grid": {"max_depth": [2, 3]}, "seed": 0},
    "knn": {"tune": True, "k_grid": [3, 5], "nca_components_grid": [4], "tune_n_inner": 2,
            "nca_max_iter": 20, "tune_max_iter": 20, "seed": 0},
    "svm": {"kernel": "rbf", "C": 1.0, "gamma": "scale", "probability": True, "seed": 0},
}


def test_all_combinations():
    X, y, names = _toy_data()
    split = 60
    Xtr, Xte, ytr, yte = X[:split], X[split:], y[:split], y[split:]
    for method in FS_METHODS:
        Xs_tr, ys_tr, Xs_te, ys_te = select_features(
            method, Xtr, ytr, Xte, yte, names, FS_CFG, out_dir="outputs/_smoke", fold=1
        )
        assert Xs_tr.shape[0] == len(ys_tr) and Xs_te.shape[0] == len(ys_te)
        assert Xs_tr.shape[1] >= 1
        for model_name in MODELS:
            metrics = get_model(model_name)(Xs_tr, ys_tr, Xs_te, ys_te, 3, MODEL_CFG[model_name])
            assert set(METRIC_KEYS).issubset(metrics.keys())
            assert 0.0 <= metrics["accuracy"] <= 1.0
    print("OK: all FS x model combinations ran.")


if __name__ == "__main__":
    test_all_combinations()
