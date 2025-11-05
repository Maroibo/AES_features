from clause_features import ClauseAnalyzer
import pandas as pd
import os
from tqdm import tqdm
import torch

# Constants
INPUT_OUTPUT_FILE_PATH='./output_features/example_full_feature_set.csv'

def main():
    # Force GPU usage
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("CUDA not available, falling back to CPU")
        device = torch.device("cpu")
    
    # Check if the input file exists
    if not os.path.exists(INPUT_OUTPUT_FILE_PATH):
        print(f"Error: Input file {INPUT_OUTPUT_FILE_PATH} does not exist!")
        print("Please run the main extractor_script.py first.")
        return
    
    # Initialize clause analyzer ONCE before processing all essays
    print("Initializing clause analyzer (this may take a moment)...")
    clause_analyzer = ClauseAnalyzer(device=device)
    print("Clause analyzer initialized successfully!")
    
    # Read the existing features file
    print(f"Loading existing features from: {INPUT_OUTPUT_FILE_PATH}")
    df = pd.read_csv(INPUT_OUTPUT_FILE_PATH)
    
    print(f"Processing {len(df)} essays for clause features...")
    
    # Add progress bar to the loop
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Extracting clause features", unit="essay"):
        essay = row['essay']
        
        # Calculate clause features using the pre-initialized analyzer
        clause_features = clause_analyzer.calculate_features(essay)
        
        # Add clause features to the dataframe
        for feature_name, feature_value in clause_features.items():
            df.at[index, feature_name] = feature_value
    
    print("\nClause feature extraction completed!")
    
    # Save the updated dataframe back to the same file
    df.to_csv(INPUT_OUTPUT_FILE_PATH, index=False)
    print(f"Updated features saved to: {INPUT_OUTPUT_FILE_PATH}")
    
    # Print stats of the updated features
    print("\nUpdated dataframe info:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {len(df.columns)}")
    
    # Show clause feature columns that were added
    clause_columns = [col for col in df.columns if 'clause' in col.lower()]
    if clause_columns:
        print(f"Clause features added: {clause_columns}")
    else:
        print("No clause features found in column names")

if __name__ == "__main__":
    main() 