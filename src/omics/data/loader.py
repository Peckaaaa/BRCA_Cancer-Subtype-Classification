"""
Multi-omics data loading & merging.

Canonical loader from `Code/MLP_benchmark.ipynb` (`load_omics`/`load_labels`/
`load_dataset`). Each omics CSV is stored as (features x samples); it is
transposed to (samples x features) and column-prefixed with its omics block,
e.g. ``mRNA__ESR1``. The four blocks are inner-joined on samples, labels are
LabelEncoded, and the merged matrix is returned.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_omics(path: str | Path, sep: str = ",") -> pd.DataFrame:
    """Read one omics CSV (features x samples) -> DataFrame (samples x features)."""
    df = pd.read_csv(path, sep=sep, index_col=0)
    return df.T


def load_labels(path: str | Path, sep: str = ",") -> np.ndarray:
    """Read the single-column label file in sample order."""
    df = pd.read_csv(path, sep=sep)
    return df.iloc[:, 0].to_numpy()


def load_dataset(
    data_files: dict[str, str | Path],
    labels_file: str | Path,
    sep: str = ",",
) -> tuple[np.ndarray, np.ndarray, list[str], int]:
    """
    Load + merge omics blocks and labels.

    Returns (X, y, feature_names, num_classes) where X is float (samples x
    features) and y is LabelEncoded integers.
    """
    print("Loading omics data ...")
    dfs: dict[str, pd.DataFrame] = {}
    for name, path in data_files.items():
        if not Path(path).exists():
            print(f"  [WARN] '{path}' not found - skipping '{name}'")
            continue
        dfs[name] = load_omics(path, sep)
        print(f"  {name:<8}: {dfs[name].shape[0]} samples, {dfs[name].shape[1]} features")

    if not dfs:
        raise FileNotFoundError(
            "No omics data files found. Check configs/data.yaml paths."
        )

    merged = pd.concat(list(dfs.values()), axis=1, join="inner")
    merged.columns = [f"{omic}__{feat}" for omic, df in dfs.items() for feat in df.columns]
    feature_names = list(merged.columns)
    print(f"\nMerged matrix : {merged.shape[0]} samples x {merged.shape[1]} features")

    if not Path(labels_file).exists():
        raise FileNotFoundError(f"Labels file '{labels_file}' not found.")

    y_raw = load_labels(labels_file, sep)
    X = merged.values.astype(float)
    if len(y_raw) != X.shape[0]:
        raise ValueError(
            f"Label count ({len(y_raw)}) != sample count ({X.shape[0]})."
        )

    le = LabelEncoder()
    y = le.fit_transform(np.asarray(y_raw))
    return X, y, feature_names, len(le.classes_)
