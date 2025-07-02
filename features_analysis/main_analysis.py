import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import os
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load the Arabic feature dataset"""
    print("Loading data...")
    # Load the CSV file
    df = pd.read_csv('../output_features/full_arabic_feature_set_prompts[1,2,3,4].csv')
    print(f"Data loaded successfully. Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    return df

def identify_feature_columns(df, target_columns, metadata_columns):
    """Identify feature columns by excluding target and metadata columns"""
    all_columns = set(df.columns)
    target_set = set(target_columns)
    metadata_set = set(metadata_columns)
    
    # Remove target and metadata columns to get feature columns
    feature_columns = list(all_columns - target_set - metadata_set)
    
    print(f"\nTarget columns ({len(target_columns)}): {target_columns}")
    print(f"Metadata columns ({len(metadata_columns)}): {metadata_columns}")
    print(f"Feature columns ({len(feature_columns)}): {feature_columns[:10]}...")  # Show first 10
    
    return feature_columns

def calculate_correlations_for_target(df, target_col, feature_columns):
    """Calculate correlations for a single target column and return maximum correlation"""
    print(f"\nProcessing target column: {target_col}")
    
    if target_col not in df.columns:
        print(f"Warning: Target column '{target_col}' not found in dataset")
        return None
    
    results = []
    
    for feature_col in feature_columns:
        if feature_col not in df.columns:
            continue
            
        # Get non-null values for both columns
        mask = df[target_col].notna() & df[feature_col].notna()
        
        if mask.sum() < 50:  # Need at least 2 points for correlation
            continue
            
        target_values = df[target_col][mask]
        feature_values = df[feature_col][mask]
        
        # Skip if no variance in either variable
        if target_values.var() == 0 or feature_values.var() == 0:
            continue
        
        try:
            # Calculate Pearson correlation
            pearson_corr, pearson_p = pearsonr(target_values, feature_values)
            
            # Calculate Spearman correlation
            spearman_corr, spearman_p = spearmanr(target_values, feature_values)
            
            # Take maximum of absolute correlations (conservative strength measure)
            max_abs_corr = max(abs(pearson_corr), abs(spearman_corr))
            
            results.append({
                'feature': feature_col,
                'max_correlation': max_abs_corr
            })
            
        except Exception as e:
            print(f"Error calculating correlation for {target_col} vs {feature_col}: {e}")
            continue
    
    # Convert to DataFrame and sort by absolute correlation
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values(by='max_correlation', key=abs, ascending=False)
    
    print(f"  Completed correlations for {len(results)} features")
    return results_df

def analyze_single_target(df, target_col, feature_columns):
    """Analyze correlations for a single target column"""
    print(f"\nProcessing target: {target_col}")
    
    # Calculate correlations
    results_df = calculate_correlations_for_target(df, target_col, feature_columns)
    
    if results_df is None or results_df.empty:
        print(f"No valid correlations found for {target_col}")
        return None
    
    # Save results for this target
    save_target_results(target_col, results_df)
    
    return {
        'target': target_col,
        'results': results_df
    }

def create_output_directories():
    """Create necessary output directories"""
    base_output_dir = '../output'
    whole_dataset_dir = os.path.join(base_output_dir, 'whole_dataset')
    
    # Create directories if they don't exist
    os.makedirs(base_output_dir, exist_ok=True)
    os.makedirs(whole_dataset_dir, exist_ok=True)
    
    return base_output_dir, whole_dataset_dir

def save_target_results(target_col, results_df):
    """Save results for a single target column"""
    base_output_dir, whole_dataset_dir = create_output_directories()
    
    filename = f'{target_col}_whole_data_set_correlations.csv'
    filepath = os.path.join(whole_dataset_dir, filename)
    results_df.to_csv(filepath, index=False)
    print(f"  Saved {len(results_df)} feature correlations to {filepath}")

def save_target_results_by_essay_set(target_col, results_df, essay_set_value):
    """Save results for a single target column by essay_set"""
    base_output_dir, _ = create_output_directories()
    
    # Create prompt-specific directory
    prompt_dir = os.path.join(base_output_dir, f'prompt_{essay_set_value}')
    os.makedirs(prompt_dir, exist_ok=True)
    
    filename = f'{target_col}_prompt_{essay_set_value}_correlations.csv'
    filepath = os.path.join(prompt_dir, filename)
    results_df.to_csv(filepath, index=False)
    print(f"    Saved {len(results_df)} feature correlations to {filepath}")

def analyze_single_target_by_essay_set(df_subset, target_col, feature_columns, essay_set_value):
    """Analyze correlations for a single target column within a specific essay_set"""
    
    # Calculate correlations
    results_df = calculate_correlations_for_target(df_subset, target_col, feature_columns)
    
    if results_df is None or results_df.empty:
        print(f"    No valid correlations found for {target_col} in essay_set {essay_set_value}")
        return None
    
    # Save results for this target and essay_set
    save_target_results_by_essay_set(target_col, results_df, essay_set_value)
    
    return {
        'target': target_col,
        'essay_set': essay_set_value,
        'results': results_df
    }

def analyze_by_essay_set(df, target_columns, feature_columns, metadata_columns):
    """Analyze correlations grouped by essay_set"""
    print(f"\n{'='*80}")
    print("STARTING ANALYSIS BY ESSAY_SET")
    print(f"{'='*80}")
    
    # Check if essay_set column exists
    if 'essay_set' not in df.columns:
        print("Warning: 'essay_set' column not found in dataset. Skipping essay_set analysis.")
        return
    
    # Get unique essay_set values
    essay_sets = sorted(df['essay_set'].unique())
    print(f"Found essay_set values: {essay_sets}")
    
    # Analyze each essay_set
    for essay_set_value in essay_sets:
        print(f"\nProcessing essay_set: {essay_set_value}")
        
        # Filter data for this essay_set
        df_subset = df[df['essay_set'] == essay_set_value].copy()
        print(f"  Dataset size for essay_set {essay_set_value}: {len(df_subset)} rows")
        
        # Skip if subset is too small
        if len(df_subset) < 10:
            print(f"  Skipping essay_set {essay_set_value} - insufficient data (< 10 rows)")
            continue
        
        # Analyze each target column for this essay_set
        for target_col in target_columns:
            analyze_single_target_by_essay_set(df_subset, target_col, feature_columns, essay_set_value)



def main():
    """Main function to run the correlation analysis"""
    print("Starting Arabic NLP Features Correlation Analysis")
    print("Using MINIMUM correlation approach (Conservative)")
    print("="*80)
    
    # Define target and metadata columns
    target_columns = ['holistic', 'relevance', 'vocabulary', 'style', 
                     'development', 'mechanics', 'grammar', 'organization']
    metadata_columns = ['essay', 'essay_set', 'prompt', 'essay_id']
    
    # Load data
    df = load_data()
    
    # Identify feature columns
    feature_columns = identify_feature_columns(df, target_columns, metadata_columns)
    
    # PART 1: Analyze whole dataset
    print(f"\n{'='*80}")
    print("PART 1: WHOLE DATASET ANALYSIS")
    print(f"{'='*80}")
    
    for target_col in target_columns:
        analyze_single_target(df, target_col, feature_columns)
    
    # PART 2: Analyze by essay_set
    analyze_by_essay_set(df, target_columns, feature_columns, metadata_columns)
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETED SUCCESSFULLY!")
    print(f"{'='*80}")
    
    # Count and display generated files
    essay_sets = sorted(df['essay_set'].unique()) if 'essay_set' in df.columns else []
    
    print("\nGenerated directory structure:")
    print("\n../output_correlations/")
    print("├── whole_dataset/")
    for target_col in target_columns:
        prefix = "├──" if target_col != target_columns[-1] or essay_sets else "└──"
        print(f"│   {prefix} {target_col}_whole_data_set_correlations.csv")
    
    if essay_sets:
        for i, essay_set_value in enumerate(essay_sets):
            is_last_essay_set = (i == len(essay_sets) - 1)
            prefix = "└──" if is_last_essay_set else "├──"
            print(f"{prefix} prompt_{essay_set_value}/")
            
            for j, target_col in enumerate(target_columns):
                is_last_target = (j == len(target_columns) - 1)
                if is_last_essay_set:
                    file_prefix = "    └──" if is_last_target else "    ├──"
                else:
                    file_prefix = "│   └──" if is_last_target else "│   ├──"
                print(f"{file_prefix} {target_col}_prompt_{essay_set_value}_correlations.csv")
    
    total_files = len(target_columns) + (len(target_columns) * len(essay_sets) if essay_sets else 0)
    print(f"\nTotal of {total_files} CSV files created")
    print("Each file contains ALL features with their maximum correlation values")
    print("(maximum of Pearson and Spearman correlations for conservative estimates)")

if __name__ == "__main__":
    main()
