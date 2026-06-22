"""
Signal-to-Noise Ratio (SNR) feature selection.

Canonical version from `Code/MLP_benchmark.ipynb`:
  * multiclass one-vs-rest SNR score = |mean_c - mean_rest| / (std_c + std_rest)
  * `k_total` mode: global top-k by multiclass SNR (used by the MLP benchmark
    for a fair comparison against the other top-k rankers).
  * `k_per_class` mode: top-k highest-SNR features per class (the old "bottom-k"
    branch that picked the *least* discriminative features has been removed).
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .common import as_dataframe, as_series, impute_from_train, save_metadata


def snr_scores(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    classes = np.unique(y)
    scores = np.zeros(X.shape[1], dtype=float)
    eps = 1e-12
    for class_label in classes:
        class_mask = y == class_label
        rest_mask = ~class_mask
        class_mean = np.nanmean(X[class_mask], axis=0)
        rest_mean = np.nanmean(X[rest_mask], axis=0)
        class_std = np.nanstd(X[class_mask], axis=0)
        rest_std = np.nanstd(X[rest_mask], axis=0)
        scores = np.maximum(
            scores, np.abs(class_mean - rest_mean) / (class_std + rest_std + eps)
        )
    return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)


def extract_features(
    X_train: pd.DataFrame | np.ndarray | Sequence[Sequence[float]],
    y_train: pd.Series | np.ndarray | Sequence[int],
    X_test: pd.DataFrame | np.ndarray | Sequence[Sequence[float]],
    y_test: pd.Series | np.ndarray | Sequence[int],
    *,
    k_per_class: int = 10,
    k_total: int | None = None,
    output_csv: str | Path = "snr_selected_features.csv",
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """SNR-based feature selection with one-vs-all ranking for multiclass data."""
    X_train_df = as_dataframe(X_train)
    X_test_df = as_dataframe(X_test, feature_names=X_train_df.columns)
    y_train_series = as_series(y_train, name="y_train")
    y_test_series = as_series(y_test, name="y_test")

    X_train_df, X_test_df = impute_from_train(X_train_df, X_test_df)

    X_train_array = X_train_df.to_numpy(dtype=float, copy=True)
    X_test_array = X_test_df.to_numpy(dtype=float, copy=True)
    y_train_array = y_train_series.to_numpy()

    # --- Global top-k mode: rank ALL features by multiclass SNR score ---
    if k_total is not None:
        scores_all = snr_scores(X_train_array, y_train_array)
        k_actual = min(int(k_total), X_train_array.shape[1])
        sel = np.sort(np.argsort(scores_all)[::-1][:k_actual])
        cols = X_train_df.columns[sel].tolist()
        X_train_selected = pd.DataFrame(X_train_array[:, sel], columns=cols, index=X_train_df.index)
        X_test_selected = pd.DataFrame(X_test_array[:, sel], columns=cols, index=X_test_df.index)
        save_metadata(cols, output_csv, scores_all[sel])
        return X_train_selected, y_train_series, X_test_selected, y_test_series

    classes = np.unique(y_train_array)
    selected_features: set[int] = set()
    selected_scores: dict[int, float] = {}

    for class_label in classes:
        y_binary = (y_train_array == class_label).astype(int)
        scores = snr_scores(X_train_array, y_binary)
        pos_idx = np.argsort(scores)[::-1][:k_per_class]
        for index in pos_idx:
            selected_features.add(int(index))
            selected_scores[int(index)] = float(scores[int(index)])

    selected_idx = np.array(sorted(selected_features))
    selected_columns = X_train_df.columns[selected_idx].tolist()

    X_train_selected = pd.DataFrame(
        X_train_array[:, selected_idx], columns=selected_columns, index=X_train_df.index
    )
    X_test_selected = pd.DataFrame(
        X_test_array[:, selected_idx], columns=selected_columns, index=X_test_df.index
    )

    ordered_scores = np.array([selected_scores[int(i)] for i in selected_idx], dtype=float)
    save_metadata(selected_columns, output_csv, ordered_scores)
    return X_train_selected, y_train_series, X_test_selected, y_test_series
