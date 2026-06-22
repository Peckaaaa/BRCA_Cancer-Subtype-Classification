"""
Unified benchmark pipeline.

For a chosen classifier (mlp | logreg | xgboost | knn), run Stratified K-Fold CV
where each fold applies one of the shared feature-selection methods and then
trains/evaluates the classifier. All methods use the SAME folds and the SAME
feature-selection implementations (src/omics/feature_selection), mirroring the
benchmark in Code/MLP_benchmark.ipynb but with the model as a parameter.

Run:
    python -m omics.pipelines.benchmark --model mlp
    python -m omics.pipelines.benchmark --model logreg --methods fisher l1 snr
    python -m omics.pipelines.benchmark --model knn --folds 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from ..config import load_config, resolve_data_config, PROJECT_ROOT
from ..data import load_dataset
from ..feature_selection import select_features, METHODS as FS_METHODS
from ..models import get_model
from ..evaluation import compute_metrics, summarize_folds, plot_radar
from ..evaluation.metrics import METRIC_KEYS


def run_benchmark(
    model_name: str,
    methods: list[str] | None = None,
    n_folds: int | None = None,
    seed: int | None = None,
    out_dir: str | Path | None = None,
) -> pd.DataFrame:
    data_cfg = load_config("data")
    fs_cfg = load_config("feature_selection")
    models_cfg = load_config("models")

    seed = int(seed if seed is not None else fs_cfg.get("seed", 42))
    n_folds = int(n_folds if n_folds is not None else fs_cfg.get("n_folds", 5))
    methods = methods or list(FS_METHODS)
    fs_cfg["seed"] = seed

    model_cfg = dict(models_cfg.get(model_name, {}))
    model_cfg.setdefault("seed", seed)

    out_dir = Path(out_dir) if out_dir else PROJECT_ROOT / "outputs" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    omics_files, labels_file, sep = resolve_data_config(data_cfg)
    X, y, feature_names, num_classes = load_dataset(omics_files, labels_file, sep)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = list(skf.split(X, y))
    print(f"\n[CV] model={model_name} | {n_folds}-fold | {len(y)} samples | "
          f"{num_classes} classes | methods={methods}")

    model_fn = get_model(model_name)
    per_method: dict[str, list[dict]] = {m: [] for m in methods}

    for fold, (tr_idx, te_idx) in enumerate(folds, start=1):
        print("\n" + "#" * 70)
        print(f"FOLD {fold}/{n_folds} | train={len(tr_idx)} test={len(te_idx)}")
        print("#" * 70)
        for method in methods:
            print(f"\n[Fold {fold}] METHOD: {method.upper()}")
            try:
                Xtr_sel, ytr, Xte_sel, yte = select_features(
                    method, X[tr_idx], y[tr_idx], X[te_idx], y[te_idx],
                    feature_names, fs_cfg, out_dir=str(out_dir), fold=fold,
                )
                metrics = model_fn(Xtr_sel, ytr, Xte_sel, yte, num_classes, model_cfg)
                rec = {"fold": fold, "n_feat": int(np.asarray(Xtr_sel).shape[1]), **metrics}
                print(f"  feat={rec['n_feat']:<5} | "
                      + " | ".join(f"{k}={rec.get(k, float('nan')):.4f}" for k in METRIC_KEYS))
            except Exception as exc:
                import traceback
                traceback.print_exc()
                print(f"  [ERROR] {method} failed at fold {fold}: {exc}")
                rec = {"fold": fold, "n_feat": np.nan, **{k: np.nan for k in METRIC_KEYS}}
            per_method[method].append(rec)

    # ── Aggregate, save, radar ───────────────────────────────────────────────
    summary_rows = []
    mean_by_method: dict[str, dict] = {}
    for method in methods:
        df = pd.DataFrame(per_method[method])
        df.to_csv(out_dir / f"fold_results_{method}.csv", index=False)
        summ = summarize_folds(per_method[method])
        row = {"method": method, "n_feat": float(np.nanmean(df["n_feat"]))}
        for k in METRIC_KEYS:
            row[f"{k}_mean"] = float(summ.loc[k, "Mean"]) if k in summ.index else np.nan
            row[f"{k}_std"] = float(summ.loc[k, "Std"]) if k in summ.index else np.nan
        summary_rows.append(row)
        mean_by_method[method] = {k: row[f"{k}_mean"] for k in METRIC_KEYS}

    summary = pd.DataFrame(summary_rows).set_index("method")
    summary_path = out_dir / "benchmark_summary.csv"
    summary.round(4).to_csv(summary_path)
    print(f"\nSummary saved -> {summary_path}")
    print(summary.round(4).to_string())

    plot_radar(mean_by_method, f"Radar Chart Comparison ({model_name})",
               out_dir / "radar_comparison.png")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified multi-omics FS + classification benchmark")
    p.add_argument("--model", default="mlp", choices=["mlp", "logreg", "xgboost", "knn", "svm"])
    p.add_argument("--methods", nargs="+", default=None, choices=FS_METHODS, metavar="METHOD",
                   help=f"Feature-selection methods (default: all {FS_METHODS})")
    p.add_argument("--folds", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out-dir", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_benchmark(
        model_name=args.model, methods=args.methods,
        n_folds=args.folds, seed=args.seed, out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
