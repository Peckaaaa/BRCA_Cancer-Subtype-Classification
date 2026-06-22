"""
mRMR (minimum Redundancy Maximum Relevance) feature selection.

Canonical version from `Code/MLP_benchmark.ipynb`:
  * relevance = multiclass mutual information, normalised to [0, 1]
  * redundancy = mean |Pearson corr| with already-selected features,
    accumulated INCREMENTALLY (memory O(F) instead of O(F^2))
  * score(f) = relevance(f) - redundancy_weight * redundancy(f)

`redundancy_weight` (alpha): 1.0 = classic mRMR, 0.0 = pure MaxRel.
Input is expected to be already imputed + scaled (numpy arrays).
"""
from __future__ import annotations

import numpy as np

from .common import save_metadata

_MRMR_K = 50
_RANDOM_STATE = 42


def _mrmr_select(
    X: np.ndarray,
    y: np.ndarray,
    k_select: int = _MRMR_K,
    seed: int = _RANDOM_STATE,
    redundancy_weight: float = 1.0,
) -> np.ndarray:
    from sklearn.feature_selection import mutual_info_classif

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)

    mi = mutual_info_classif(X, y, random_state=seed)
    mi_norm = mi / (mi.max() + 1e-12)

    # Standardise columns (centre + unit L2) so dot product = Pearson corr
    Xc = X - X.mean(axis=0, keepdims=True)
    norm = np.sqrt((Xc ** 2).sum(axis=0, keepdims=True))
    norm[norm == 0] = 1.0
    Xs = Xc / norm

    F = X.shape[1]
    k_actual = min(k_select, F)
    redund_sum = np.zeros(F)
    selected_mask = np.zeros(F, dtype=bool)
    selected: list[int] = []

    for _ in range(k_actual):
        if not selected:
            best = int(np.argmax(mi_norm))
        else:
            score = mi_norm - redundancy_weight * (redund_sum / len(selected))
            score[selected_mask] = -np.inf
            best = int(np.argmax(score))
        selected.append(best)
        selected_mask[best] = True

        corr_best = np.abs(Xs[:, best] @ Xs)
        corr_best[best] = 0.0
        redund_sum += corr_best

    return np.array(selected)


def mrmr_extract_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    k: int = _MRMR_K,
    out_dir: str = ".",
    fold: int = 1,
    seed: int = _RANDOM_STATE,
    redundancy_weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    k_actual = min(k, X_train.shape[1])
    top_idx = _mrmr_select(
        X_train, y_train, k_select=k_actual, seed=seed, redundancy_weight=redundancy_weight
    )
    selected_names = [feature_names[i] for i in top_idx]
    print(f"  mRMR selection      : {X_train.shape[1]:,} -> {k_actual} features")
    save_metadata(selected_names, f"{out_dir}/mrmr_selected_features_fold{fold}.csv")
    return X_train[:, top_idx], y_train, X_test[:, top_idx], y_test
