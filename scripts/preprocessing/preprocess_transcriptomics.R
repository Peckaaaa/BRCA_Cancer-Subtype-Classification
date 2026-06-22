# preprocess_transcriptomics.R
# Preprocesses mRNA and miRNA data according to MLOmics methodology.

if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
if (!requireNamespace("edgeR", quietly = TRUE))
    BiocManager::install("edgeR")

library(edgeR)

preprocess_transcriptomics <- function(count_matrix, gene_lengths, is_mirna = FALSE) {
  # 1. Convert gene-level estimates into FPKM values (if raw counts are provided)
  # Assuming count_matrix has genes as rows and samples as columns
  
  print("Calculating FPKM...")
  # Create DGEList object
  dge <- DGEList(counts = count_matrix)
  
  # Calculate normalization factors
  dge <- calcNormFactors(dge)
  
  # Calculate FPKM
  fpkm_data <- rpkm(dge, gene.length = gene_lengths)
  
  # 2. Non-Human miRNA filtering (Conceptual step for miRNA)
  if (is_mirna) {
    print("Filtering non-human miRNAs (Ensure rownames contain 'hsa')")
    human_mirnas <- grep("hsa", rownames(fpkm_data), value = TRUE)
    fpkm_data <- fpkm_data[human_mirnas, ]
  }
  
  # 3. Noise eliminating
  print("Eliminating noise (zeros in >10% of samples and NAs)...")
  num_samples <- ncol(fpkm_data)
  
  # Remove NAs
  fpkm_data <- na.omit(fpkm_data)
  
  # Calculate proportion of zeros for each gene
  zero_prop <- rowSums(fpkm_data == 0) / num_samples
  
  # Keep genes with 10% or fewer zeros
  filtered_data <- fpkm_data[zero_prop <= 0.10, ]
  
  # 4. Transformation
  print("Applying log2(x + 1) transformation...")
  log_transformed_data <- log2(filtered_data + 1)
  
  return(log_transformed_data)
}

print("preprocess_transcriptomics script loaded.")
