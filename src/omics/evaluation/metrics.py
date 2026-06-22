from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)

METRIC_KEYS = ["accuracy", "f1_macro", "precision_macro", "recall_macro", "auc_macro"]


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None
) -> dict:
    """Compute the 5 macro metrics for one fold. AUC is macro one-vs-rest."""
    y_true = np.asarray(y_true)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    auc = float("nan")
    if y_prob is not None:
        try:
            y_prob = np.asarray(y_prob)
            n_classes = y_prob.shape[1] if y_prob.ndim == 2 else 2
            if n_classes == 2:
                prob = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
                auc = float(roc_auc_score(y_true, prob))
            else:
                auc = float(
                    roc_auc_score(
                        y_true, y_prob, multi_class="ovr", average="macro",
                        labels=np.arange(n_classes),
                    )
                )
        except Exception:
            auc = float("nan")
    out["auc_macro"] = auc
    return out


def summarize_folds(fold_records: list[dict]) -> pd.DataFrame:
    """mean ± std across folds for each metric. Returns a tidy DataFrame."""
    df = pd.DataFrame(fold_records)
    rows = []
    for col in METRIC_KEYS:
        if col in df.columns:
            vals = df[col].to_numpy(dtype=float)
            rows.append(
                {
                    "Metric": col,
                    "Mean": float(np.nanmean(vals)),
                    "Std": float(np.nanstd(vals)),
                    "Min": float(np.nanmin(vals)),
                    "Max": float(np.nanmax(vals)),
                }
            )
    return pd.DataFrame(rows).set_index("Metric")
