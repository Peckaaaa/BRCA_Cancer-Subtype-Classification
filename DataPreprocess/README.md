# DataPreprocess — raw TCGA → aligned dataset

Upstream data-preparation stage that produces the aligned matrices the ML
pipeline consumes (`data/BRCA_*_aligned.csv`). Following the **MLOmics** paper
methodology. The aligned dataset is **already provided** in `data/`; these scripts
document how it was built and let you regenerate it from raw TCGA data.

## Pipeline-compatible output

`produce_aligned_dataset.py` writes files in the exact format
`src/omics/data/loader.py` expects:

- **orientation:** features × samples (rows = features, first column = feature id;
  the loader reads with `index_col=0` and transposes).
- **file names:** `BRCA_<omics>_aligned.csv` matching `configs/data.yaml`
  (`CNV`, `Methy`, `miRNA`, `mRNA`).
- **default output dir:** the project `data/` folder.

## Order of execution

```text
collect_tcga_data.R            # download raw TCGA-BRCA omics from GDC (TCGAbiolinks)
        │
        ├── preprocess_transcriptomics.R   # mRNA & miRNA: FPKM, filter, log2(x+1)
        ├── preprocess_cnv.R               # CNV: somatic filter, GAIA, biomaRt annotate
        └── preprocess_methylation.R       # methylation: median-normalise, promoter select
        │
integrate_and_filter.R         # align samples, multi-class ANOVA + BH (FDR<0.05), z-score
        │
produce_aligned_dataset.py     # -> data/BRCA_<omics>_aligned.csv  (features × samples)
        │
        ▼
   src/omics pipeline  (python scripts/run_benchmark.py --model ...)
```

```bash
python DataPreprocess/produce_aligned_dataset.py --input-dir processed_omics --output-dir data
```

> Note: the R scripts are reference implementations of the methodology. The CNV
> and methylation steps contain conceptual/placeholder sections (e.g. GAIA
> recurrent regions, probe→gene selection) and require appropriately formatted
> raw inputs to run end-to-end.
