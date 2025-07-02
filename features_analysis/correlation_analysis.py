import pandas as pd
import numpy as np
import os
import json
from collections import defaultdict
from datetime import datetime

def load_categorization():
    """Load feature categorization from JSON file"""
    json_path = "../output/feature_categorization.json"
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            categorization = json.load(f)
            return categorization
    else:
        print(f"Categorization file not found: {json_path}")
        print("Please run cat_analysis.py first to generate the categorization.")
        return None

def get_csv_filename(target_col, dataset_name):
    """Generate appropriate CSV filename based on dataset"""
    if dataset_name == "whole_dataset":
        return f"{target_col}_whole_data_set_correlations.csv"
    else:
        return f"{target_col}_{dataset_name}_correlations.csv"

def analyze_correlations_by_category(df, categorization):
    """Analyze correlations by main categories and subcategories"""
    # Create feature mappings
    feature_to_subcategory = {}
    feature_to_category = {}
    
    for main_category, subcategories in categorization['categories'].items():
        for subcategory, features in subcategories.items():
            for feature in features:
                feature_to_subcategory[feature] = subcategory
                feature_to_category[feature] = main_category
    
    # Organize data by categories and subcategories
    category_data = defaultdict(list)
    subcategory_data = defaultdict(list)
    global_data = []
    
    for _, row in df.iterrows():
        feature = row['feature']
        correlation = abs(row['max_correlation'])  # Use absolute correlation
        
        # Add to global data
        global_data.append({
            'feature': feature,
            'correlation': correlation,
            'category': feature_to_category.get(feature, 'Unknown'),
            'subcategory': feature_to_subcategory.get(feature, 'Unknown')
        })
        
        # Add to category data
        if feature in feature_to_category:
            main_category = feature_to_category[feature]
            category_data[main_category].append({
                'feature': feature,
                'correlation': correlation,
                'subcategory': feature_to_subcategory[feature]
            })
        
        # Add to subcategory data
        if feature in feature_to_subcategory:
            subcategory = feature_to_subcategory[feature]
            subcategory_data[subcategory].append({
                'feature': feature,
                'correlation': correlation,
                'category': feature_to_category[feature]
            })
    
    return category_data, subcategory_data, global_data

def get_top_features(data_list, n=5, reverse=True):
    """Get top N features from a list of feature dictionaries"""
    if not data_list:
        return []
    
    # Sort by correlation
    sorted_data = sorted(data_list, key=lambda x: x['correlation'], reverse=reverse)
    return sorted_data[:n]

def generate_correlation_report(target_col, dataset_name, categorization, output_dir):
    """Generate a comprehensive correlation report for a target column"""
    
    # Load correlation data
    csv_filename = get_csv_filename(target_col, dataset_name)
    csv_path = f"../output/{dataset_name}/{csv_filename}"
    
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return False
    
    # Read correlation data
    df = pd.read_csv(csv_path)
    
    # Analyze correlations
    category_data, subcategory_data, global_data = analyze_correlations_by_category(df, categorization)
    
    # Generate report
    report_lines = []
    report_lines.append("="*80)
    report_lines.append(f"CORRELATION ANALYSIS REPORT")
    report_lines.append(f"Target: {target_col.upper()}")
    report_lines.append(f"Dataset: {dataset_name.replace('_', ' ').title()}")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("="*80)
    report_lines.append("")
    
    # Global analysis
    report_lines.append("GLOBAL FEATURE ANALYSIS")
    report_lines.append("-" * 40)
    report_lines.append("")
    
    # Top 5 global features
    top_global = get_top_features(global_data, 5, reverse=True)
    report_lines.append("TOP 5 FEATURES (Highest Correlation):")
    for i, feature_data in enumerate(top_global, 1):
        report_lines.append(f"  {i}. {feature_data['feature']}")
        report_lines.append(f"     Correlation: {feature_data['correlation']:.4f}")
        report_lines.append(f"     Category: {feature_data['category']}")
        report_lines.append(f"     Subcategory: {feature_data['subcategory']}")
        report_lines.append("")
    
    # Bottom 5 global features
    bottom_global = get_top_features(global_data, 5, reverse=False)
    report_lines.append("BOTTOM 5 FEATURES (Lowest Correlation):")
    for i, feature_data in enumerate(bottom_global, 1):
        report_lines.append(f"  {i}. {feature_data['feature']}")
        report_lines.append(f"     Correlation: {feature_data['correlation']:.4f}")
        report_lines.append(f"     Category: {feature_data['category']}")
        report_lines.append(f"     Subcategory: {feature_data['subcategory']}")
        report_lines.append("")
    
    # Category analysis
    report_lines.append("="*80)
    report_lines.append("CATEGORY ANALYSIS")
    report_lines.append("="*80)
    report_lines.append("")
    
    # Define category order for consistency
    category_order = [
        'Surface features',
        'Lexical features', 
        'Readability measures',
        'Semantic features',
        'Syntactic features'
    ]
    
    for category in category_order:
        if category in category_data:
            features = category_data[category]
            top_features = get_top_features(features, 5, reverse=True)
            
            report_lines.append(f"{category.upper()}")
            report_lines.append("-" * len(category))
            report_lines.append(f"Total features in category: {len(features)}")
            if features:
                avg_correlation = np.mean([f['correlation'] for f in features])
                report_lines.append(f"Average correlation: {avg_correlation:.4f}")
            report_lines.append("")
            report_lines.append("Top 5 features in this category:")
            
            for i, feature_data in enumerate(top_features, 1):
                report_lines.append(f"  {i}. {feature_data['feature']}")
                report_lines.append(f"     Correlation: {feature_data['correlation']:.4f}")
                report_lines.append(f"     Subcategory: {feature_data['subcategory']}")
                report_lines.append("")
            
            report_lines.append("")
    
    # Subcategory analysis
    report_lines.append("="*80)
    report_lines.append("SUBCATEGORY ANALYSIS")
    report_lines.append("="*80)
    report_lines.append("")
    
    # Sort subcategories by their average correlation
    subcategory_averages = {}
    for subcategory, features in subcategory_data.items():
        if features:
            avg_correlation = np.mean([f['correlation'] for f in features])
            subcategory_averages[subcategory] = avg_correlation
    
    sorted_subcategories = sorted(subcategory_averages.items(), key=lambda x: x[1], reverse=True)
    
    for subcategory, avg_correlation in sorted_subcategories:
        features = subcategory_data[subcategory]
        top_features = get_top_features(features, 5, reverse=True)
        
        # Get the main category for this subcategory
        main_category = features[0]['category'] if features else 'Unknown'
        
        report_lines.append(f"{subcategory.upper()} ({main_category})")
        report_lines.append("-" * (len(subcategory) + len(main_category) + 3))
        report_lines.append(f"Total features in subcategory: {len(features)}")
        report_lines.append(f"Average correlation: {avg_correlation:.4f}")
        report_lines.append("")
        report_lines.append("Top 5 features in this subcategory:")
        
        for i, feature_data in enumerate(top_features, 1):
            report_lines.append(f"  {i}. {feature_data['feature']}")
            report_lines.append(f"     Correlation: {feature_data['correlation']:.4f}")
            report_lines.append("")
        
        report_lines.append("")
    
    # Summary statistics
    report_lines.append("="*80)
    report_lines.append("SUMMARY STATISTICS")
    report_lines.append("="*80)
    report_lines.append("")
    
    if global_data:
        correlations = [f['correlation'] for f in global_data]
        report_lines.append(f"Total features analyzed: {len(global_data)}")
        report_lines.append(f"Mean correlation: {np.mean(correlations):.4f}")
        report_lines.append(f"Median correlation: {np.median(correlations):.4f}")
        report_lines.append(f"Standard deviation: {np.std(correlations):.4f}")
        report_lines.append(f"Min correlation: {np.min(correlations):.4f}")
        report_lines.append(f"Max correlation: {np.max(correlations):.4f}")
        report_lines.append("")
        
        # Category summary
        report_lines.append("Category Performance Summary (by average correlation):")
        category_averages = {}
        for category, features in category_data.items():
            if features:
                avg_correlation = np.mean([f['correlation'] for f in features])
                category_averages[category] = (avg_correlation, len(features))
        
        sorted_categories = sorted(category_averages.items(), key=lambda x: x[1][0], reverse=True)
        for i, (category, (avg_corr, count)) in enumerate(sorted_categories, 1):
            report_lines.append(f"  {i}. {category}: {avg_corr:.4f} (from {count} features)")
        
        report_lines.append("")
        report_lines.append("Top 3 Subcategories (by average correlation):")
        top_subcategories = sorted_subcategories[:3]
        for i, (subcategory, avg_corr) in enumerate(top_subcategories, 1):
            feature_count = len(subcategory_data[subcategory])
            main_cat = subcategory_data[subcategory][0]['category'] if subcategory_data[subcategory] else 'Unknown'
            report_lines.append(f"  {i}. {subcategory} ({main_cat}): {avg_corr:.4f} (from {feature_count} features)")
    
    # Save report
    report_filename = f"{target_col}_{dataset_name}_correlation_report.txt"
    report_path = os.path.join(output_dir, report_filename)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Generated report: {report_path}")
    return True

def process_dataset(dataset_name, categorization, target_columns):
    """Process a single dataset and generate all reports for it"""
    print(f"\n{'='*60}")
    print(f"Processing dataset: {dataset_name}")
    print(f"{'='*60}")
    
    # Create output directory for this dataset's reports
    output_dir = f"../output/reports/{dataset_name}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # Generate reports for each target column
    successful_reports = 0
    
    for target_col in target_columns:
        print(f"\nGenerating report for {target_col}...")
        success = generate_correlation_report(target_col, dataset_name, categorization, output_dir)
        if success:
            successful_reports += 1
    
    print(f"\nCompleted {dataset_name}: {successful_reports}/{len(target_columns)} reports generated")
    return successful_reports

def main():
    """Main function to generate all correlation analysis reports"""
    
    # Load categorization
    categorization = load_categorization()
    if not categorization:
        return
    
    # Define target columns
    target_columns = ['holistic', 'relevance', 'vocabulary', 'style', 
                     'development', 'mechanics', 'grammar', 'organization']
    
    # Define datasets to process
    datasets = ['whole_dataset', 'prompt_1', 'prompt_2', 'prompt_3', 'prompt_4']
    
    # Create main output directory for reports
    main_output_dir = "../output/reports"
    if not os.path.exists(main_output_dir):
        os.makedirs(main_output_dir)
    
    print("Starting correlation analysis report generation for all datasets...")
    print("="*80)
    
    total_reports = 0
    processed_datasets = 0
    
    # Process each dataset
    for dataset_name in datasets:
        # Check if dataset directory exists
        dataset_dir = f"../output/{dataset_name}"
        if not os.path.exists(dataset_dir):
            print(f"Dataset directory not found: {dataset_dir}")
            continue
        
        # Process this dataset
        reports_generated = process_dataset(dataset_name, categorization, target_columns)
        total_reports += reports_generated
        processed_datasets += 1
    
    # Print final summary
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY - CORRELATION ANALYSIS REPORTS")
    print(f"{'='*80}")
    print(f"Processed datasets: {processed_datasets}")
    print(f"Total reports generated: {total_reports}")
    print(f"Target columns per dataset: {len(target_columns)}")
    print(f"Reports saved to subdirectories in: {main_output_dir}")
    print(f"\nReport Features:")
    print(f"- Top 5 and bottom 5 global features across all categories")
    print(f"- Top 5 features from each main category")
    print(f"- Top 5 features from each subcategory")
    print(f"- Categories ranked by average correlation performance")
    print(f"- Subcategories ranked by average correlation performance")
    print(f"- Comprehensive summary statistics for each target")
    print(f"- Detailed feature information including correlations and categorization")
    print(f"- Reports organized by dataset mirroring the output directory structure")
    print(f"- Text format for easy reading and analysis")

if __name__ == "__main__":
    main() 