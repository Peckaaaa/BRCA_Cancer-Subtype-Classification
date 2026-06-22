"""
Unified feature-selection dispatcher.

A single entry point — `select_features(method, ...)` — used by every model
pipeline (MLP, Logistic Regression, XGBoost, kNN). It mirrors the per-method
adapters (`prep_*`) of the canonical MLP benchmark in
`Code/MLP_benchmark.ipynb`, so all classifiers consume the SAME feature
selection implementations.

Input
-----
X_train, X_test : raw (un-scaled) numpy arrays, shape (n_samples, n_features)
y_train, y_test : label arrays
feature_names   : column names aligned with axis-1 of X_train ("omics__gene")
cfg             : feature-selection config dict (see configs/feature_selection.yaml)

Output
------
(X_train_sel, y_train, X_test_sel, y_test) as numpy arrays (selected features).
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

from . import fisher, l1, snr, mrmr, relieff, elmo, pam50

METHODS = ["fisher", "l1", "snr", "mrmr", "relieff", "elmo", "pam50"]


def _impute_scale(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean-impute (train stats) then StandardScaler — fit on train only."""
    col_mean = np.nanmean(X_train, axis=0)
    col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
    X_train = np.where(np.isnan(X_train), col_mean, X_train)
    X_test = np.where(np.isnan(X_test), col_mean, X_test)
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test)


def select_features(
    method: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    cfg: dict[str, Any],
    *,
    out_dir: str = "outputs",
    fold: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    name = method.strip().lower()
    seed = int(cfg.get("seed", 42))
    k = int(cfg.get("k", 50))
    import pandas as pd

    if name == "fisher":
        c = cfg.get("fisher", {})
        a, b, d, e = fisher.extract_features(
            pd.DataFrame(X_train, columns=feature_names),
            y_train,
            pd.DataFrame(X_test, columns=feature_names),
            y_test,
            k=int(c.get("k", k)),
            output_csv=f"{out_dir}/fisher_selected_features.csv",
        )
        return a.to_numpy(np.float32), b.to_numpy(), d.to_numpy(np.float32), e.to_numpy()

    if name == "l1":
        c = cfg.get("l1", {})
        a, b, d, e = l1.extract_features(
            pd.DataFrame(X_train, columns=feature_names),
            y_train,
            pd.DataFrame(X_test, columns=feature_names),
            y_test,
            k=int(c.get("k", k)),
            C=float(c.get("C", 0.1)),
            solver=c.get("solver", "liblinear"),
            max_iter=int(c.get("max_iter", 5000)),
            random_state=seed,
            output_csv=f"{out_dir}/l1_selected_features.csv",
        )
        return a.to_numpy(np.float32), b.to_numpy(), d.to_numpy(np.float32), e.to_numpy()

    if name == "snr":
        c = cfg.get("snr", {})
        a, b, d, e = snr.extract_features(
            pd.DataFrame(X_train, columns=feature_names),
            y_train,
            pd.DataFrame(X_test, columns=feature_names),
            y_test,
            k_total=int(c.get("k_total", k)),
            output_csv=f"{out_dir}/snr_selected_features.csv",
        )
        return a.to_numpy(np.float32), b.to_numpy(), d.to_numpy(np.float32), e.to_numpy()

    if name == "relieff":
        c = cfg.get("relieff", {})
        Xtr_s, Xte_s = _impute_scale(X_train, X_test)
        a, b, d, e = relieff.relieff_extract_features(
            X_train=Xtr_s, y_train=y_train, X_test=Xte_s, y_test=y_test,
            feature_names=feature_names, k=int(c.get("k", k)),
            out_dir=out_dir, fold=fold, seed=seed,
            n_iter=int(c.get("n_iter", 200)), k_neighbors=int(c.get("k_neighbors", 10)),
        )
        return np.asarray(a, np.float32), b, np.asarray(d, np.float32), e

    if name == "mrmr":
        c = cfg.get("mrmr", {})
        Xtr_s, Xte_s = _impute_scale(X_train, X_test)
        a, b, d, e = mrmr.mrmr_extract_features(
            X_train=Xtr_s, y_train=y_train, X_test=Xte_s, y_test=y_test,
            feature_names=feature_names, k=int(c.get("k", k)),
            out_dir=out_dir, fold=fold, seed=seed,
            redundancy_weight=float(c.get("redundancy_weight", 1.0)),
        )
        return np.asarray(a, np.float32), b, np.asarray(d, np.float32), e

    if name == "elmo":
        c = cfg.get("elmo", {})
        return elmo.select(
            X_train, y_train, X_test, y_test, feature_names,
            l1_c=float(c.get("l1_c", 0.1)),
            l1_solver=c.get("l1_solver", "liblinear"),
            l1_max_iter=int(c.get("l1_max_iter", 5000)),
            lr_c=float(c.get("lr_c", 1.0)),
            lr_solver=c.get("lr_solver", "lbfgs"),
            lr_max_iter=int(c.get("lr_max_iter", 5000)),
            rfe_step=float(c.get("rfe_step", 0.1)),
            ifs_max_k=int(c.get("ifs_max_k", 150)),
            n_inner=int(c.get("n_inner", 4)),
            random_state=seed,
        )

    if name == "pam50":
        c = cfg.get("pam50", {})
        omics = c.get("omics", list(pam50.DEFAULT_PAM50_OMICS))
        omics = tuple(omics) if omics is not None else None
        idx = pam50.select_indices(feature_names, omics=omics)
        Xtr_s, Xte_s = _impute_scale(X_train[:, idx], X_test[:, idx])
        return Xtr_s.astype(np.float32), y_train, Xte_s.astype(np.float32), y_test

    raise ValueError(f"Unknown feature-selection method '{method}'. Choices: {METHODS}")
