"""
ReliefF feature selection (multiclass, weighted).

Canonical version from `Code/MLP_benchmark.ipynb`. Input is expected to be
already imputed + scaled (ReliefF uses Euclidean distances). At each iteration
a random instance updates feature weights by its nearest hits/misses; the
top-k features by weight are kept.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from .common import save_metadata

_RELIEF_ITER = 200
_RELIEF_K = 10
_RANDOM_STATE = 42


def _relieff_weights(
    X: np.ndarray,
    y: np.ndarray,
    n_iter: int = _RELIEF_ITER,
    k: int = _RELIEF_K,
    seed: int = _RANDOM_STATE,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n, f = X.shape
    W = np.zeros(f, dtype=np.float64)
    classes = np.unique(y)
    class_p = {c: (y == c).mean() for c in classes}
    class_idx = {c: np.where(y == c)[0] for c in classes}

    dist_mat = cdist(X, X, metric="euclidean")
    np.fill_diagonal(dist_mat, np.inf)

    for _ in range(n_iter):
        i = rng.integers(n)
        xi, yi = X[i], y[i]

        same = class_idx[yi]
        same = same[same != i]
        if len(same) > 0:
            d_same = dist_mat[i, same]
            hits = same[np.argpartition(d_same, min(k, len(same) - 1))[:k]]
            W -= np.abs(X[hits] - xi).mean(axis=0) / n_iter

        for c in classes:
            if c == yi:
                continue
            other = class_idx[c]
            if len(other) > 0:
                d_oth = dist_mat[i, other]
                misses = other[np.argpartition(d_oth, min(k, len(other) - 1))[:k]]
                weight = class_p[c] / (1.0 - class_p[yi] + 1e-10)
                W += weight * np.abs(X[misses] - xi).mean(axis=0) / n_iter

    return W


def relieff_extract_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    k: int,
    out_dir: str = ".",
    fold: int = 1,
    n_iter: int = _RELIEF_ITER,
    k_neighbors: int = _RELIEF_K,
    seed: int = _RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    k_actual = min(k, X_train.shape[1])
    weights = _relieff_weights(X_train, y_train, n_iter=n_iter, k=k_neighbors, seed=seed)
    top_idx = np.argsort(weights)[::-1][:k_actual]

    selected_names = [feature_names[i] for i in top_idx]
    print(f"  ReliefF selection   : {X_train.shape[1]:,} -> {k_actual} features")
    save_metadata(
        selected_names, f"{out_dir}/relieff_selected_features_fold{fold}.csv", weights[top_idx]
    )
    return X_train[:, top_idx], y_train, X_test[:, top_idx], y_test
