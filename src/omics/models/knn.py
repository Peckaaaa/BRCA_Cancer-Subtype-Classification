"""
NCA-KNN classifier with per-fold k tuning.

Model core extracted from the original `Code/final_code_kNN.py`: L2-normalise the
(already feature-selected) inputs, learn a NeighborhoodComponentsAnalysis
embedding, and classify with KNeighborsClassifier. `k` (and optionally NCA
n_components) are tuned by inner-CV macro-AUC on the training fold only.

Note: the original kNN script also performed its own per-omics preprocessing
(VarianceThreshold / QuantileTransformer / stratified KNN imputation /
F-score prefilter). In this unified repo, feature selection is shared across all
models, so that bespoke preprocessing is no longer applied here.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize, label_binarize
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.metrics import roc_auc_score

from ..evaluation.metrics import compute_metrics


def _fit_nca_knn(X_tr, y_tr, X_te, k, n_components, max_iter, tol, random_state):
    n_feats = X_tr.shape[1]
    n_comp = min(n_components, n_feats) if n_components else n_feats
    nca = NeighborhoodComponentsAnalysis(
        n_components=n_comp, max_iter=max_iter, tol=tol, random_state=random_state
    )
    nca.fit(X_tr, y_tr)
    X_tr_nca = nca.transform(X_tr)
    X_te_nca = nca.transform(X_te)
    knn = KNeighborsClassifier(n_neighbors=k, weights="uniform", algorithm="brute")
    knn.fit(X_tr_nca, y_tr)
    return knn, X_te_nca


def _inner_cv_auc(X, y, k, n_components, max_iter, n_inner, random_state, tol):
    min_class = int(np.bincount(y.astype(int)).min())
    n_splits = max(min(n_inner, min_class), 2)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    classes = np.unique(y)
    aucs = []
    for tr, va in cv.split(X, y):
        try:
            knn, X_va = _fit_nca_knn(
                X[tr], y[tr], X[va], k, n_components, max_iter, tol, random_state
            )
            y_prob = knn.predict_proba(X_va)
            y_bin = label_binarize(y[va], classes=classes)
            if y_bin.shape[1] == 1:
                y_bin = np.hstack([1 - y_bin, y_bin])
            aucs.append(roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr"))
        except Exception:
            continue
    return float(np.nanmean(aucs)) if aucs else float("-inf")


def train_and_evaluate(X_train, y_train, X_test, y_test, num_classes, cfg: dict) -> dict:
    cfg = cfg or {}
    seed = int(cfg.get("seed", 42))
    tol = float(cfg.get("nca_tol", 1e-5))
    max_iter = int(cfg.get("nca_max_iter", 100))
    n_components = int(cfg.get("nca_components", 50))
    tune = bool(cfg.get("tune", True))
    k_grid = list(cfg.get("k_grid", [5, 7, 9, 11]))
    comp_grid = list(cfg.get("nca_components_grid", [50]))
    n_inner = int(cfg.get("tune_n_inner", 3))

    X_tr = normalize(np.asarray(X_train, dtype=np.float32), norm="l2")
    X_te = normalize(np.asarray(X_test, dtype=np.float32), norm="l2")
    y_train = np.asarray(y_train)

    if tune:
        best_k, best_comp, best_score = k_grid[0], comp_grid[0], float("-inf")
        for k in k_grid:
            for n_comp in comp_grid:
                n_comp = min(n_comp, X_tr.shape[1])
                score = _inner_cv_auc(
                    X_tr, y_train, k, n_comp, max_iter=int(cfg.get("tune_max_iter", 100)),
                    n_inner=n_inner, random_state=seed, tol=tol,
                )
                if score > best_score:
                    best_score, best_k, best_comp = score, k, n_comp
    else:
        best_k, best_comp = int(cfg.get("k", 7)), n_components

    knn, X_te_nca = _fit_nca_knn(
        X_tr, y_train, X_te, best_k, best_comp, max_iter, tol, seed
    )
    y_pred = knn.predict(X_te_nca)
    y_prob = knn.predict_proba(X_te_nca)
    return compute_metrics(y_test, y_pred, y_prob=y_prob)
