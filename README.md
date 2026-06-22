# BRCA Multi-Omics — Feature Selection & Classification

Reproducible benchmark of **7 feature-selection methods** × **4 classifiers** on the
TCGA **BRCA** multi-omics dataset (mRNA, miRNA, methylation, CNV), evaluated with
Stratified K-Fold cross-validation.

- **Feature selection:** Fisher · L1 · SNR · mRMR · ReliefF · ELMO · PAM50
- **Classifiers:** MLP (PyTorch) · Logistic Regression (ElasticNet) · XGBoost · NCA-KNN · SVM (RBF)
- **Metrics:** Accuracy · F1-macro · Precision-macro · Recall-macro · AUC-macro

All classifiers share **one** feature-selection implementation and **one** data
loader / evaluation stack, so methods are compared on identical folds.

## Project layout

```text
.
├── README.md
├── requirements.txt
├── configs/                      # all tunables (paths + hyper-parameters)
│   ├── data.yaml
│   ├── feature_selection.yaml
│   └── models.yaml
├── data/                         # BRCA_*_aligned.csv + BRCA_label_num.csv
├── DataPreprocess/               # raw TCGA -> aligned data (R + produce_aligned_dataset.py)
├── src/omics/
│   ├── config.py                 # YAML loading / path resolution
│   ├── data/loader.py            # load + merge omics, encode labels
│   ├── feature_selection/        # ONE canonical impl per method + shared common.py
│   │   ├── common.py  fisher.py  l1.py  snr.py
│   │   ├── mrmr.py    relieff.py elmo.py pam50.py
│   │   └── __init__.py           # select_features(method, ...) dispatcher
│   ├── models/                   # mlp.py  logreg.py  xgb.py  knn.py  svm.py
│   ├── evaluation/               # metrics.py  radar.py
│   └── pipelines/benchmark.py    # unified CV benchmark (model as a parameter)
├── scripts/run_benchmark.py      # CLI entry point
├── outputs/                      # results (CSV summaries + radar charts)
└── tests/test_smoke.py          # FS × model smoke test on synthetic data
```

## Setup

```bash
pip install -r requirements.txt
# Place the data files in ./data/ (already present in this repo):
#   BRCA_CNV_aligned.csv  BRCA_Methy_aligned.csv
#   BRCA_miRNA_aligned.csv  BRCA_mRNA_aligned.csv  BRCA_label_num.csv
```

`xgboost` is optional — if it is not installed, the XGBoost backend automatically
falls back to a scikit-learn RandomForest (same as the original code).

## Usage

Run a full benchmark (all 7 FS methods) for a given classifier:

```bash
python scripts/run_benchmark.py --model mlp
python scripts/run_benchmark.py --model logreg
python scripts/run_benchmark.py --model xgboost
python scripts/run_benchmark.py --model knn
python scripts/run_benchmark.py --model svm
```

Restrict to specific methods / change folds:

```bash
python scripts/run_benchmark.py --model logreg --methods fisher l1 snr --folds 5
```

Equivalent module form:

```bash
PYTHONPATH=src python -m omics.pipelines.benchmark --model mlp
```

Outputs are written to `outputs/<model>/`:
- `benchmark_summary.csv` — mean ± std of the 5 metrics per FS method
- `fold_results_<method>.csv` — per-fold metrics
- `radar_comparison.png` — radar chart across methods
- `<method>_selected_features*.csv` — selected feature lists

## Configuration

Edit the YAML files in `configs/` — no code changes needed:
- `data.yaml` — data directory, file names, CSV separator
- `feature_selection.yaml` — `k`, seeds, per-method parameters
- `models.yaml` — per-classifier hyper-parameters

## Smoke test

```bash
PYTHONPATH=src python tests/test_smoke.py
```

Runs every FS × model combination on a tiny synthetic dataset (seconds).
