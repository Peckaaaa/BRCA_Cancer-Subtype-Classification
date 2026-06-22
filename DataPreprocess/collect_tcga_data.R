# collect_tcga_data.R
# This script uses TCGAbiolinks to download TCGA data from the GDC portal.
# Based on the MLOmics paper data collection methodology.

# Install required packages if not present
if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
if (!requireNamespace("TCGAbiolinks", quietly = TRUE))
    BiocManager::install("TCGAbiolinks")

library(TCGAbiolinks)

# Function to download TCGA data for a given project and data category
download_tcga_data <- function(project_id = "TCGA-BRCA", data_category, data_type, out_dir = "raw_data") {
  
  if (!dir.exists(out_dir)) {
    dir.create(out_dir, recursive = TRUE)
  }
  
  print(paste("Querying data for", project_id, "-", data_category))
  
  query <- GDCquery(
    project = project_id,
    data.category = data_category,
    data.type = data_type,
    access = "open"
  )
  
  print("Downloading data...")
  GDCdownload(query, directory = out_dir)
  
  print("Preparing data...")
  data_prep <- GDCprepare(query, directory = out_dir)
  
  return(data_prep)
}

# Example Usage:
# 1. Transcriptomics (mRNA)
# mrna_data <- download_tcga_data("TCGA-BRCA", "Transcriptome Profiling", "Gene Expression Quantification")

# 2. Transcriptomics (miRNA)
# mirna_data <- download_tcga_data("TCGA-BRCA", "Transcriptome Profiling", "miRNA Expression Quantification")

# 3. Genomic (CNV)
# cnv_data <- download_tcga_data("TCGA-BRCA", "Copy Number Variation", "Copy Number Segment")

# 4. Epigenomic (Methylation)
# methy_data <- download_tcga_data("TCGA-BRCA", "DNA Methylation", "Methylation Beta Value")

print("TCGAbiolinks data collection script loaded. Uncomment the function calls to download specific omics data.")
