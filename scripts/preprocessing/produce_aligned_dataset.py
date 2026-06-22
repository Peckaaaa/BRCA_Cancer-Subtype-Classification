import pandas as pd
import numpy as np
import os
import glob
from sklearn.preprocessing import StandardScaler

def produce_aligned_dataset(input_dir, output_dir):
    """
    Produces aligned datasets from preprocessed omics CSV files.
    - Intersects samples across all modalities.
    - Applies z-score feature normalization.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {input_dir}. Please ensure R scripts save output to CSV.")
        return

    print(f"Found {len(csv_files)} omics files: {[os.path.basename(f) for f in csv_files]}")

    dataframes = {}
    sample_sets = []

    for file in csv_files:
        print(f"Loading {os.path.basename(file)}...")
        df = pd.read_csv(file, index_col=0)
        
        samples = set(df.columns)
        sample_sets.append(samples)
        dataframes[os.path.basename(file)] = df

    common_samples = set.intersection(*sample_sets)
    common_samples = sorted(list(common_samples))
    print(f"Identified {len(common_samples)} common samples across all modalities.")

    if len(common_samples) == 0:
        print("Error: No common samples found across datasets. Check sample ID formatting.")
        return

    for file_name, df in dataframes.items():
        aligned_df = df[common_samples]
        
        aligned_df = aligned_df.T
        
        print(f"Normalizing {file_name} (z-score)...")
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(aligned_df)
        
        normalized_df = pd.DataFrame(
            normalized_data, 
            index=aligned_df.index, 
            columns=aligned_df.columns
        )
        
        
        out_name = f"Aligned_{file_name}"
        out_path = os.path.join(output_dir, out_name)
        normalized_df.to_csv(out_path)
        print(f"Saved aligned and normalized dataset to {out_path}")
        
        print(f"Final shape of {out_name}: {normalized_df.shape} (Samples, Features)")

if __name__ == "__main__":
    INPUT_DIRECTORY = "processed_omics" 
    OUTPUT_DIRECTORY = "aligned_omics"
    
    print("Starting alignment and normalization process...")
    produce_aligned_dataset(INPUT_DIRECTORY, OUTPUT_DIRECTORY)
    print("Process complete!")
