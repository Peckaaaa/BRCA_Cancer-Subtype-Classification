"""
ELMO feature selection — leakage-free 4-step embedded pipeline.

Canonical version from `Code/MLP_benchmark.ipynb` (the `ELMO.py` writefile cell
plus its `prep_elmo` adapter):

    Step 1 - L1 Logistic Regression  -> drop zero-weight features
    Step 2 - LR-RFE                   -> rank surviving features
    Step 3 - IFS (inner CV)           -> pick optimal feature count
    Step 4 - SBS (inner CV)           -> refine the feature set

All steps run on the TRAIN fold only (no leakage). Input arrays are imputed +
scaled here using train statistics. `select()` returns numpy arrays of the
selected, scaled train/test features (ELMO adapts the feature count itself).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


def impute_mean(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean-impute NaNs using train statistics only."""
    col_means = np.nanmean(X_train, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    for X in (X_train, X_test):
        nans = np.where(np.isnan(X))
        X[nans] = np.take(col_means, nans[1])
    return X_train, X_test


def scale(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sc = StandardScaler()
    return sc.fit_transform(X_train), sc.transform(X_test)


def _make_lr(C: float, solver: str, max_iter: int, random_state: int) -> LogisticRegression:
    return LogisticRegression(
        C=C, penalty="l2", solver=solver, max_iter=max_iter, random_state=random_state
    )


def inner_cv_accuracy(
    X: np.ndarray, y: np.ndarray, *, lr_c: float, lr_solver: str, lr_max_iter: int,
    random_state: int, n_inner: int,
) -> float:
    n_splits = min(n_inner, int(np.min(np.bincount(y.astype(int)))))
    n_splits = max(n_splits, 2)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    accs = []
    for tr, va in cv.split(X, y):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[tr])
        X_va = sc.transform(X[va])
        lr = _make_lr(lr_c, lr_solver, lr_max_iter, random_state)
        lr.fit(X_tr, y[tr])
        accs.append(accuracy_score(y[va], lr.predict(X_va)))
    return float(np.mean(accs))


def step1_l1_selection(
    X_train_s: np.ndarray, y_train: np.ndarray, feature_names: Sequence[str],
    *, l1_c: float, l1_solver: str, l1_max_iter: int, random_state: int,
) -> tuple[np.ndarray, list[str]]:
    base = LogisticRegression(
        penalty="l1", C=l1_c, solver=l1_solver, max_iter=l1_max_iter, random_state=random_state
    )
    if l1_solver == "liblinear" and len(np.unique(y_train)) > 2:
        clf = OneVsRestClassifier(base)
        clf.fit(X_train_s, y_train)
        importance = np.vstack([np.abs(e.coef_).ravel() for e in clf.estimators_]).max(axis=0)
    else:
        base.fit(X_train_s, y_train)
        importance = np.abs(base.coef_).max(axis=0)
    mask = importance > 0
    selected = [fn for fn, m in zip(feature_names, mask) if m]
    return mask, selected


def step2_rfe_order(
    X_train_s: np.ndarray, y_train: np.ndarray,
    *, lr_c: float, lr_solver: str, lr_max_iter: int, random_state: int, rfe_step: float,
) -> np.ndarray:
    rfe = RFE(
        estimator=_make_lr(lr_c, lr_solver, lr_max_iter, random_state),
        n_features_to_select=1,
        step=rfe_step,
    )
    rfe.fit(X_train_s, y_train)
    return np.argsort(rfe.ranking_)


def step3_ifs(
    X_train_ranked: np.ndarray, y_train: np.ndarray, ranked_features: Sequence[str],
    *, lr_c: float, lr_solver: str, lr_max_iter: int, random_state: int, n_inner: int,
    ifs_max_k: int,
) -> tuple[int, list[str]]:
    best_score, best_k = -1.0, 1
    k_max = min(X_train_ranked.shape[1], ifs_max_k)
    for k in range(1, k_max + 1):
        score = inner_cv_accuracy(
            X_train_ranked[:, :k], y_train, lr_c=lr_c, lr_solver=lr_solver,
            lr_max_iter=lr_max_iter, random_state=random_state, n_inner=n_inner,
        )
        if score > best_score:
            best_score, best_k = score, k
    return best_k, list(ranked_features[:best_k])


def step4_sbs(
    X_train_opt: np.ndarray, y_train: np.ndarray, optimal_features: Sequence[str],
    *, lr_c: float, lr_solver: str, lr_max_iter: int, random_state: int, n_inner: int,
) -> tuple[list[int], list[str]]:
    current_cols = list(range(X_train_opt.shape[1]))
    current_features = list(optimal_features)
    baseline = inner_cv_accuracy(
        X_train_opt[:, current_cols], y_train, lr_c=lr_c, lr_solver=lr_solver,
        lr_max_iter=lr_max_iter, random_state=random_state, n_inner=n_inner,
    )
    improved = True
    while improved and len(current_cols) > 1:
        improved = False
        best_score = baseline
        best_drop = -1
        for i in range(len(current_cols)):
            trial = current_cols[:i] + current_cols[i + 1:]
            score = inner_cv_accuracy(
                X_train_opt[:, trial], y_train, lr_c=lr_c, lr_solver=lr_solver,
                lr_max_iter=lr_max_iter, random_state=random_state, n_inner=n_inner,
            )
            if score >= best_score:
                best_score = score
                best_drop = i
        if best_drop >= 0:
            current_cols.pop(best_drop)
            current_features.pop(best_drop)
            baseline = best_score
            improved = True
    return current_cols, current_features


def select(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: Sequence[str],
    *,
    l1_c: float = 0.1,
    l1_solver: str = "liblinear",
    l1_max_iter: int = 5000,
    lr_c: float = 1.0,
    lr_solver: str = "lbfgs",
    lr_max_iter: int = 5000,
    rfe_step: float = 0.1,
    ifs_max_k: int = 150,
    n_inner: int = 4,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run the full 4-step ELMO selection. Returns scaled, reduced numpy arrays."""
    X_tr_i, X_te_i = impute_mean(np.asarray(X_train, float).copy(), np.asarray(X_test, float).copy())
    X_tr_s, X_te_s = scale(X_tr_i, X_te_i)

    mask, sel_features = step1_l1_selection(
        X_tr_s, y_train, feature_names,
        l1_c=l1_c, l1_solver=l1_solver, l1_max_iter=l1_max_iter, random_state=random_state,
    )
    if mask.sum() == 0:
        raise RuntimeError("ELMO: no features survived L1 screening.")
    X_tr_sel, X_te_sel = X_tr_s[:, mask], X_te_s[:, mask]

    order = step2_rfe_order(
        X_tr_sel, y_train,
        lr_c=lr_c, lr_solver=lr_solver, lr_max_iter=lr_max_iter,
        random_state=random_state, rfe_step=rfe_step,
    )
    ranked_features = [sel_features[i] for i in order]
    X_tr_r, X_te_r = X_tr_sel[:, order], X_te_sel[:, order]

    best_k, ifs_features = step3_ifs(
        X_tr_r, y_train, ranked_features,
        lr_c=lr_c, lr_solver=lr_solver, lr_max_iter=lr_max_iter,
        random_state=random_state, n_inner=n_inner, ifs_max_k=ifs_max_k,
    )
    X_tr_ifs, X_te_ifs = X_tr_r[:, :best_k], X_te_r[:, :best_k]

    kept_cols, _ = step4_sbs(
        X_tr_ifs, y_train, ifs_features,
        lr_c=lr_c, lr_solver=lr_solver, lr_max_iter=lr_max_iter,
        random_state=random_state, n_inner=n_inner,
    )
    X_tr_f, X_te_f = X_tr_ifs[:, kept_cols], X_te_ifs[:, kept_cols]
    print(f"  ELMO selection      : L1 -> RFE -> IFS -> SBS -> {X_tr_f.shape[1]} features (adaptive)")
    return X_tr_f.astype(np.float32), y_train, X_te_f.astype(np.float32), y_test
