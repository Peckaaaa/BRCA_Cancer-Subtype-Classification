#!/usr/bin/env python
"""
Thin entry point so the package can be run without installation.

Usage:
    python scripts/run_benchmark.py --model mlp
    python scripts/run_benchmark.py --model logreg --methods fisher l1 snr
    python scripts/run_benchmark.py --model knn --folds 5
    python scripts/run_benchmark.py --model xgboost
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omics.pipelines.benchmark import main  # noqa: E402

if __name__ == "__main__":
    main()
