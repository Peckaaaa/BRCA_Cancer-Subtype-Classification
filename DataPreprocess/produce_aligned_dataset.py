"""
produce_aligned_dataset.py
==========================
Final data-preparation stage: turn the per-omics preprocessed CSVs (output of the
R scripts in this folder) into the *aligned* matrices consumed by the ML pipeline.

Output format — MUST match what `src/omics/data/loader.py` expects:
  * orientation : features x samples  (rows = features, first column = feature id)
    The loader reads each file with `index_col=0` and transposes it, so the file on
    disk has FEATURES as rows and SAMPLES as columns.
  * file names  : the keys in `configs/data.yaml` -> `BRCA_<omics>_aligned.csv`
    (e.g. BRCA_CNV_aligned.csv, BRCA_Methy_aligned.csv, ...).
  * default out : the project's `data/` directory.

Steps:
  1. Load each preprocessed omics CSV (features x samples).
  2. Intersect sample IDs across all modalities (inner join on samples).
  3. Per-feature z-score normalisation (fit across samples).
  4. Save as features x samples with the pipeline's naming, into `data/`.

NOTE: the aligned BRCA dataset is already provided in `data/`; this script
documents how it was produced and lets you regenerate it from raw inputs.

Usage:
  python DataPreprocess/produce_aligned_dataset.py \
      --input-dir processed_omics --output-dir data
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

# Map an input file (matched case-insensitively by this token in its name) to the
# omics key used by configs/data.yaml. Output file = BRCA_<key>_aligned.csv.
OMICS_KEYS = {
    "cnv": "CNV",
    "methy": "Methy",
    "mirna": "miRNA",
    "mrna": "mRNA",
}


def _omics_key_from_name(file_name: str) -> str:
    """Infer the canonical omics key from an input file name."""
    low = file_name.lower()
    # check miRNA before mRNA so 'mirna' is not captured by 'mrna'
    for token in ("cnv", "methy", "mirna", "mrna"):
        if token in low:
            return OMICS_KEYS[token]
    # fallback: use the stem unchanged
    return Path(file_name).stem


def produce_aligned_dataset(input_dir: str | Path, output_dir: str | Path) -> None:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {input_dir}. Run the R preprocessing scripts first.")
        return
    print(f"Found {len(csv_files)} omics files: {[f.name for f in csv_files]}")

    # 1. Load each omics CSV as (features x samples).
    frames: dict[str, pd.DataFrame] = {}
    sample_sets = []
    for f in csv_files:
        print(f"Loading {f.name} ...")
        df = pd.read_csv(f, index_col=0)          # rows = features, cols = samples
        frames[f.name] = df
        sample_sets.append(set(df.columns))

    # 2. Intersect samples across all modalities.
    common = sorted(set.intersection(*sample_sets))
    print(f"Identified {len(common)} common samples across all modalities.")
    if not common:
        print("Error: no common samples found. Check sample ID formatting.")
        return

    # 3 + 4. Align, z-score per feature, save as features x samples.
    for name, df in frames.items():
        aligned = df[common]                       # features x samples (common cols)

        # z-score each feature across samples: StandardScaler works column-wise,
        # so transpose to (samples x features), scale, then transpose back.
        scaler = StandardScaler()
        scaled = scaler.fit_transform(aligned.T)   # (samples x features)
        out_df = pd.DataFrame(scaled, index=aligned.columns, columns=aligned.index).T

        key = _omics_key_from_name(name)
        out_path = output_dir / f"BRCA_{key}_aligned.csv"
        out_df.to_csv(out_path)
        print(f"Saved {out_path}  (features x samples = {out_df.shape})")

    print("\nDone. Ensure the file names match configs/data.yaml -> omics_files.")
    print("The label file (BRCA_label_num.csv) is produced upstream alongside the omics.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Align + normalise preprocessed omics for the pipeline.")
    p.add_argument("--input-dir", default="processed_omics",
                   help="Folder with per-omics preprocessed CSVs (features x samples).")
    p.add_argument("--output-dir", default="data",
                   help="Where to write BRCA_<omics>_aligned.csv (default: project data/).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("Starting alignment and normalization process...")
    produce_aligned_dataset(args.input_dir, args.output_dir)
    print("Process complete!")
