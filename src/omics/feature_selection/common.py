"""
Shared helpers for all feature-selection modules.

These utilities were previously duplicated verbatim inside every FS file of
every pipeline (root, RF/ML, XGBoost). They are now defined once here and
imported by each method module — the deduplication requested for the refactor.

The numerical behaviour is identical to the canonical versions found in
`Code/MLP_benchmark.ipynb` (the agreed source of truth).
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

DEFAULT_OMICS_TYPES = ("CNV", "Methy", "miRNA", "mRNA")


def as_dataframe(
    data: pd.DataFrame | np.ndarray | Sequence[Sequence[float]],
    feature_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    array = np.asarray(data)
    if feature_names is None:
        feature_names = [f"feature_{index}" for index in range(array.shape[1])]
    return pd.DataFrame(array, columns=list(feature_names))


def as_series(data: pd.Series | np.ndarray | Sequence[int], name: str) -> pd.Series:
    if isinstance(data, pd.Series):
        series = data.copy()
    else:
        series = pd.Series(np.asarray(data))
    series = series.reset_index(drop=True)
    series.name = name
    return series


def impute_from_train(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mean-impute NaNs using train statistics only (no leakage)."""
    train_means = X_train.mean(numeric_only=True).fillna(0.0)
    return X_train.fillna(train_means), X_test.fillna(train_means)


def infer_omics_type(feature_name: str) -> str:
    if "__" in feature_name:
        prefix = feature_name.split("__", 1)[0].strip().lower()
        mapping = {
            "cnv": "CNV",
            "methylation": "Methy",
            "methy": "Methy",
            "mirna": "miRNA",
            "mrna": "mRNA",
        }
        if prefix in mapping:
            return mapping[prefix]
    for omics_type in DEFAULT_OMICS_TYPES:
        if feature_name.endswith(f"_{omics_type}") or feature_name.endswith(f"__{omics_type}"):
            return omics_type
    return "Unknown"


def split_gene_and_omics(feature_name: str) -> tuple[str, str]:
    omics_type = infer_omics_type(feature_name)
    if "__" in feature_name:
        prefix, suffix = feature_name.split("__", 1)
        if prefix.strip().lower() in {"cnv", "methylation", "methy", "mirna", "mrna"}:
            return suffix, omics_type
    for suffix in (f"_{omics_type}", f"__{omics_type}"):
        if feature_name.endswith(suffix):
            return feature_name[: -len(suffix)], omics_type
    return feature_name, omics_type


def feature_metadata(
    feature_names: Sequence[str], scores: Sequence[float] | None = None
) -> pd.DataFrame:
    rows = []
    for index, feature_name in enumerate(feature_names):
        gene_name, omics_type = split_gene_and_omics(feature_name)
        row = {"feature": feature_name, "gene_name": gene_name, "omics_type": omics_type}
        if scores is not None:
            row["score"] = float(scores[index])
        rows.append(row)
    return pd.DataFrame(rows)


def save_metadata(
    feature_names: Sequence[str],
    output_csv: str | Path,
    scores: Sequence[float] | None = None,
) -> pd.DataFrame:
    metadata = feature_metadata(feature_names, scores=scores)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(output_path, index=False)
    return metadata
