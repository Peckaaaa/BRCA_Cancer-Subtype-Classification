# integrate_and_filter.R
# Aligns omics data, performs multi-class ANOVA and BH correction.

integrate_and_filter <- function(omics_list, labels, feature_scale = "Top") {
  
  print("Aligning samples across omics datasets...")
  # Find intersection of sample IDs
  sample_ids <- lapply(omics_list, colnames)
  common_samples <- Reduce(intersect, sample_ids)
  
  aligned_omics <- lapply(omics_list, function(df) df[, common_samples])
  aligned_labels <- labels[common_samples]
  
  if (feature_scale == "Aligned") {
    print("Returning Aligned scale (z-score normalized)...")
    # Z-score normalization for each feature across samples
    scaled_omics <- lapply(aligned_omics, function(df) {
      t(scale(t(df)))
    })
    return(scaled_omics)
  }
  
  if (feature_scale == "Top") {
    print("Generating Top scale...")
    top_omics <- list()
    
    for (i in seq_along(aligned_omics)) {
      df <- aligned_omics[[i]]
      
      print(paste("Performing multi-class ANOVA for dataset", i))
      p_values <- apply(df, 1, function(row) {
        # Check if feature has zero variance across all samples
        if(var(row) == 0) return(1.0)
        
        fit <- aov(row ~ aligned_labels)
        summary(fit)[[1]][["Pr(>F)"]][1]
      })
      
      print("Applying Benjamini-Hochberg (FDR) correction...")
      adj_p_values <- p.adjust(p_values, method = "BH")
      
      # Select features with adjusted p-value < 0.05
      significant_features <- names(adj_p_values[adj_p_values < 0.05])
      
      if (length(significant_features) == 0) {
        warning(paste("No significant features found for dataset", i, "under p < 0.05"))
        top_omics[[i]] <- df # Fallback
      } else {
        filtered_df <- df[significant_features, ]
        
        print("Performing z-score normalization...")
        scaled_df <- t(scale(t(filtered_df)))
        top_omics[[i]] <- scaled_df
      }
    }
    return(top_omics)
  }
}

print("integrate_and_filter script loaded.")
