"""
L1-regularised Logistic Regression feature selection (top-k by |coef|).

Canonical version from `Code/MLP_benchmark.ipynb`. For multiclass problems with
the `liblinear` solver it uses One-vs-Rest and takes the per-feature max
absolute coefficient across the binary problems.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from .common import as_dataframe, as_series, impute_from_train, save_metadata


def l1_scores(
    X: np.ndarray, y: np.ndarray, *, C: float, solver: str, max_iter: int, random_state: int
) -> np.ndarray:
    base = LogisticRegression(
        C=C, penalty="l1", solver=solver, max_iter=max_iter, random_state=random_state
    )
    n_classes = len(np.unique(y))
    if solver == "liblinear" and n_classes > 2:
        # liblinear does not do multinomial -> One-vs-Rest (matches multi_class='auto')
        clf = OneVsRestClassifier(base)
        clf.fit(X, y)
        return np.vstack([np.abs(e.coef_).ravel() for e in clf.estimators_]).max(axis=0)
    base.fit(X, y)
    coefficients = np.abs(base.coef_)
    return coefficients if coefficients.ndim == 1 else coefficients.max(axis=0)


def extract_features(
    X_train: pd.DataFrame | np.ndarray | Sequence[Sequence[float]],
    y_train: pd.Series | np.ndarray | Sequence[int],
    X_test: pd.DataFrame | np.ndarray | Sequence[Sequence[float]],
    y_test: pd.Series | np.ndarray | Sequence[int],
    *,
    k: int = 500,
    output_csv: str | Path = "l1_selected_features.csv",
    random_state: int = 42,
    C: float = 0.1,
    solver: str = "liblinear",
    max_iter: int = 5000,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    X_train_df = as_dataframe(X_train)
    X_test_df = as_dataframe(X_test, feature_names=X_train_df.columns)
    y_train_series = as_series(y_train, name="y_train")
    y_test_series = as_series(y_test, name="y_test")

    X_train_df, X_test_df = impute_from_train(X_train_df, X_test_df)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df)
    X_test_scaled = scaler.transform(X_test_df)

    scores = l1_scores(
        X_train_scaled,
        y_train_series.to_numpy(),
        C=C,
        solver=solver,
        max_iter=max_iter,
        random_state=random_state,
    )

    selected_count = min(int(k), X_train_df.shape[1])
    selected_indices = np.argsort(scores)[-selected_count:]
    selected_indices = np.sort(selected_indices)
    selected_columns = X_train_df.columns[selected_indices].tolist()

    X_train_selected = pd.DataFrame(
        X_train_scaled[:, selected_indices], columns=selected_columns, index=X_train_df.index
    )
    X_test_selected = pd.DataFrame(
        X_test_scaled[:, selected_indices], columns=selected_columns, index=X_test_df.index
    )

    save_metadata(selected_columns, output_csv, scores[selected_indices])
    return X_train_selected, y_train_series, X_test_selected, y_test_series
