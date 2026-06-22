"""
Fisher score feature selection (ANOVA F-value, top-k).

Canonical version from `Code/MLP_benchmark.ipynb`. Operates on a DataFrame,
imputes + standard-scales on the train split only, ranks features by ANOVA
F-value and keeps the top-k. Returns scaled, reduced train/test frames.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif

from .common import as_dataframe, as_series, impute_from_train, save_metadata


def fisher_scores(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    scores, _ = f_classif(X, y)
    return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)


def extract_features(
    X_train: pd.DataFrame | np.ndarray | Sequence[Sequence[float]],
    y_train: pd.Series | np.ndarray | Sequence[int],
    X_test: pd.DataFrame | np.ndarray | Sequence[Sequence[float]],
    y_test: pd.Series | np.ndarray | Sequence[int],
    *,
    k: int = 500,
    output_csv: str | Path = "fisher_selected_features.csv",
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    X_train_df = as_dataframe(X_train)
    X_test_df = as_dataframe(X_test, feature_names=X_train_df.columns)
    y_train_series = as_series(y_train, name="y_train")
    y_test_series = as_series(y_test, name="y_test")

    X_train_df, X_test_df = impute_from_train(X_train_df, X_test_df)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df)
    X_test_scaled = scaler.transform(X_test_df)

    scores = fisher_scores(X_train_scaled, y_train_series.to_numpy())
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
