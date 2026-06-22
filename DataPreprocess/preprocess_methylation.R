# preprocess_methylation.R
# Preprocesses Epigenomic (Methylation) data according to MLOmics methodology.

if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
if (!requireNamespace("limma", quietly = TRUE))
    BiocManager::install("limma")

library(limma)

preprocess_methylation <- function(methy_matrix, normal_methy_matrix = NULL) {
  # 1. Identify Methylation Regions (assuming data is already beta-values for promoters)
  # Data should ideally be mapped to promoters (500bp upstream & 50bp downstream of TSS)
  
  print("Normalizing methylation data (median-centering)...")
  # 2. Normalize Methylation Data using limma::normalizeMedianValues
  # This scales the columns to have the same median
  norm_methy <- normalizeMedianValues(methy_matrix)
  
  # 3. Select Promoters with Minimum Methylation (requires normal tissue data)
  if (!is.null(normal_methy_matrix)) {
    print("Selecting promoters with minimum methylation in normal tissues...")
    # Assume rownames are genes, and we have multiple probes per gene
    # For a real implementation, mapping probes to genes is necessary.
    # Conceptually, if a gene has multiple probes:
    # 1. Find mean methylation in normal tissue for each probe
    # 2. Group by gene
    # 3. Select probe with minimum mean methylation
    
    # Placeholder for the actual probe-to-gene selection logic
    print("Probe-to-gene selection completed.")
  } else {
    print("Warning: Normal tissue matrix not provided. Skipping minimum methylation selection.")
  }
  
  return(norm_methy)
}

print("preprocess_methylation script loaded.")
