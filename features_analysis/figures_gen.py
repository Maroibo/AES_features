import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from collections import defaultdict
import seaborn as sns
import textwrap

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

def get_category_colors():
    """Define color palette for main categories"""
    return {
        'Surface features': '#1f77b4',      # Blue
        'Lexical features': '#ff7f0e',      # Orange  
        'Readability measures': '#2ca02c',   # Green
        'Semantic features': '#d62728',      # Red
        'Syntactic features': '#9467bd'      # Purple
    }

def get_fixed_category_order():
    """Define fixed order for main categories (consistent across all charts)"""
    return [
        'Surface features',
        'Lexical features', 
        'Readability measures',
        'Semantic features',
        'Syntactic features'
    ]

def calculate_global_subcategory_rankings(categorization, target_columns):
    """Calculate global ranking of all subcategories across all target columns"""
    all_subcategory_scores = {}
    
    for target_col in target_columns:
        csv_path = f"../output/whole_dataset/{target_col}_whole_data_set_correlations.csv"
        
        if not os.path.exists(csv_path):
            continue
        
        # Read correlation data
        df = pd.read_csv(csv_path)
        
        # Aggregate correlations by subcategory
        subcategory_df = aggregate_correlations_by_subcategory(df, categorization, 'mean')
        
        for _, row in subcategory_df.iterrows():
            subcategory = row['subcategory']
            correlation = row['correlation']
            
            if subcategory not in all_subcategory_scores:
                all_subcategory_scores[subcategory] = []
            all_subcategory_scores[subcategory].append(correlation)
    
    # Calculate mean correlation for each subcategory across all target columns
    subcategory_means = {}
    for subcategory, scores in all_subcategory_scores.items():
        subcategory_means[subcategory] = np.mean(scores)
    
    # Sort subcategories by mean correlation (highest to lowest) and create ranking
    sorted_subcategories = sorted(subcategory_means.items(), key=lambda x: x[1], reverse=True)
    
    # Create ranking dictionary (1-based ranking)
    subcategory_rankings = {}
    for rank, (subcategory, _) in enumerate(sorted_subcategories, 1):
        subcategory_rankings[subcategory] = rank
    
    return subcategory_rankings

def wrap_category_text(category_name, max_chars_per_line=12):
    """Wrap category text to fit in rectangles with word hyphenation"""
    # Remove 'features' and 'measures' suffixes for processing
    clean_name = category_name.replace(' features', '').replace(' measures', '')
    
    # Check if the final result (clean_name + " Features") would be too long
    final_text = f"{clean_name} Features"
    if len(final_text) <= max_chars_per_line:
        return final_text
    
    # If clean_name itself is too long, try to hyphenate it
    if len(clean_name) > max_chars_per_line:
        # Try to find a good break point for hyphenation
        if len(clean_name) >= 8:  # Only hyphenate reasonably long words
            # Look for natural break points (vowels followed by consonants)
            for i in range(4, len(clean_name) - 3):  # Don't break too early or late
                char = clean_name[i].lower()
                next_char = clean_name[i+1].lower() if i+1 < len(clean_name) else ''
                
                # Break after vowels before consonants, or after common prefixes
                if ((char in 'aeiou' and next_char not in 'aeiou') or 
                    clean_name[:i+1].lower() in ['read', 'synt', 'lex', 'sem']):
                    if len(clean_name[:i+1]) <= max_chars_per_line - 1:  # Leave space for hyphen
                        return f"{clean_name[:i+1]}-\n{clean_name[i+1:]} Features"
            
            # If no good break point found, break at middle
            mid_point = len(clean_name) // 2
            # Adjust break point to avoid breaking at vowels when possible
            for offset in range(-2, 3):
                break_point = mid_point + offset
                if (0 < break_point < len(clean_name) - 1 and 
                    len(clean_name[:break_point]) <= max_chars_per_line - 1):
                    return f"{clean_name[:break_point]}-\n{clean_name[break_point:]} Features"
    
    # If clean_name fits but final text doesn't, put "Features" on second line
    return f"{clean_name}\nFeatures"

def aggregate_correlations_by_subcategory(df, categorization, aggregation='mean'):
    """Aggregate correlation values by subcategory"""
    subcategory_data = []
    
    # Create feature to subcategory mapping
    feature_to_subcategory = {}
    feature_to_category = {}
    
    for main_category, subcategories in categorization['categories'].items():
        for subcategory, features in subcategories.items():
            for feature in features:
                feature_to_subcategory[feature] = subcategory
                feature_to_category[feature] = main_category
    
    # Group features by subcategory and calculate aggregation
    subcategory_correlations = defaultdict(list)
    subcategory_to_category = {}
    
    for _, row in df.iterrows():
        feature = row['feature']
        correlation = abs(row['max_correlation'])  # Use absolute correlation
        
        if feature in feature_to_subcategory:
            subcategory = feature_to_subcategory[feature]
            main_category = feature_to_category[feature]
            subcategory_correlations[subcategory].append(correlation)
            subcategory_to_category[subcategory] = main_category
    
    # Calculate aggregated values
    for subcategory, correlations in subcategory_correlations.items():
        if aggregation == 'mean':
            agg_value = np.mean(correlations)
        elif aggregation == 'max':
            agg_value = np.max(correlations)
        elif aggregation == 'median':
            agg_value = np.median(correlations)
        else:
            agg_value = np.mean(correlations)  # Default to mean
        
        subcategory_data.append({
            'subcategory': subcategory,
            'main_category': subcategory_to_category[subcategory],
            'correlation': agg_value,
            'feature_count': len(correlations)
        })
    
    return pd.DataFrame(subcategory_data)

def create_bar_chart(target_col, subcategory_df, category_colors, output_dir, global_max_y, subcategory_rankings, aggregation='mean'):
    """Create a bar chart for a target column with grouped categories"""
    if subcategory_df.empty:
        print(f"No data for {target_col}")
        return
    
    # Calculate category means (mean of subcategory means)
    category_means = {}
    for main_cat in subcategory_df['main_category'].unique():
        cat_data = subcategory_df[subcategory_df['main_category'] == main_cat]
        category_means[main_cat] = cat_data['correlation'].mean()
    
    # Use fixed category order (consistent across all charts)
    fixed_order = get_fixed_category_order()
    category_order = [cat for cat in fixed_order if cat in subcategory_df['main_category'].values]
    
    # Build the grouped structure
    x_positions = []
    x_labels = []
    bar_colors = []
    bar_values = []
    bar_counts = []
    category_positions = {}  # Track where each category starts
    category_ranges = {}     # Track start and end positions for each category
    
    current_pos = 0
    
    for main_cat in category_order:
        if main_cat not in subcategory_df['main_category'].values:
            continue
            
        category_positions[main_cat] = current_pos
        start_pos = current_pos
        
        # Add subcategory bars for this main category
        cat_data = subcategory_df[subcategory_df['main_category'] == main_cat].sort_values('correlation', ascending=False)
        
        for _, row in cat_data.iterrows():
            x_positions.append(current_pos)
            x_labels.append(row['subcategory'])
            bar_colors.append(category_colors[main_cat])
            bar_values.append(row['correlation'])
            bar_counts.append(row['feature_count'])
            current_pos += 1
        
        # Store category range for rectangle placement
        end_pos = current_pos - 1
        category_ranges[main_cat] = (start_pos, end_pos)
        
        # Add spacing between categories
        current_pos += 0.5
    
    # Create figure
    plt.figure(figsize=(16, 10))
    
    # Create bars
    bars = plt.bar(x_positions, bar_values, color=bar_colors, alpha=0.8)
    
    # Customize the plot with target name in uppercase
    plt.title(target_col.upper(), fontsize=16, fontweight='bold')
    plt.ylabel('Absolute Correlation', fontsize=12)
    
    # Set x-axis labels
    plt.xticks(x_positions, x_labels, rotation=45, ha='right')
    
    # Add value labels on bars and ranking circles
    for i, (bar, value, count) in enumerate(zip(bars, bar_values, bar_counts)):
        # Add correlation value label above bar
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{value:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add ranking rectangle at the upper portion of the bar
        subcategory_name = x_labels[i]
        if subcategory_name in subcategory_rankings:
            rank = subcategory_rankings[subcategory_name]
            
            # Calculate rectangle dimensions (smaller width than bar)
            rect_width = bar.get_width() * 0.7  # 70% of bar width
            rect_height = 0.018  # Fixed small height
            rect_x = bar.get_x() + (bar.get_width() - rect_width) / 2  # Center horizontally
            rect_y = bar.get_height() - rect_height - 0.003  # Just under the top
            
            # Draw white rectangle with black border
            rectangle = plt.Rectangle((rect_x, rect_y), rect_width, rect_height,
                                    facecolor='white', edgecolor='white', linewidth=1, zorder=15)
            plt.gca().add_patch(rectangle)
            
            # Add rank number inside rectangle (readable font size)
            text_x = rect_x + rect_width/2
            text_y = rect_y + rect_height/2
            plt.text(text_x, text_y, str(rank), ha='center', va='center', 
                    fontsize=8, fontweight='bold', color='black', zorder=16)
    
    # Add category mean rectangles with consistent y-axis range
    # Set y-axis limit to be consistent across all charts
    plt.ylim(0, global_max_y * 1.25)
    
    for main_cat in category_order:
        if main_cat in category_ranges:
            start_pos, end_pos = category_ranges[main_cat]
            mean_val = category_means[main_cat]
            
            # Create rectangle with consistent positioning using global scale
            rect_x = start_pos - 0.4
            rect_width = (end_pos - start_pos) + 0.8
            rect_y = global_max_y * 1.10  # Position higher up
            rect_height = global_max_y * 0.08  # Slightly larger height for category name
            
            # Add rectangle
            rect = plt.Rectangle((rect_x, rect_y), rect_width, rect_height,
                               facecolor=category_colors[main_cat], alpha=0.7,
                               edgecolor='black', linewidth=1)
            plt.gca().add_patch(rect)
            
            # Add text label with category name and mean
            wrapped_cat_name = wrap_category_text(main_cat)
            plt.text(rect_x + rect_width/2, rect_y + rect_height/2,
                    f'{wrapped_cat_name}\n{mean_val:.3f}', ha='center', va='center', 
                    fontsize=8, fontweight='bold', color='white')
    
    # Add vertical lines to separate main categories  
    for i, main_cat in enumerate(category_order[:-1]):  # Don't add line after last category
        if main_cat in category_ranges:
            _, end_pos = category_ranges[main_cat]
            plt.axvline(x=end_pos + 0.75, color='gray', linestyle='--', alpha=0.5)
    
    # No legend - category information is shown in rectangles above bars
    
    # Adjust layout
    plt.tight_layout()
    plt.grid(axis='y', alpha=0.3)
    
    # Save the figure
    filename = f"{target_col}_complete_feature_set_correlation.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {filepath}")
    
    # Print category summary (ordered by correlation)
    print(f"  Category means for {target_col} (ordered by correlation):")
    for cat in category_order:
        if cat in category_means:
            mean_val = category_means[cat]
            num_subcat = len(subcategory_df[subcategory_df['main_category'] == cat])
            print(f"    {cat}: {mean_val:.3f} (from {num_subcat} subcategories)")

def create_combined_chart(categorization, category_colors, output_dir, global_max_y, subcategory_rankings, aggregation='mean'):
    """Create a combined chart with all 8 target columns in 2x4 layout"""
    
    target_columns = ['holistic', 'relevance', 'vocabulary', 'style', 
                     'development', 'mechanics', 'grammar', 'organization']
    
    # Create figure with subplots (3 rows, 3 columns) - larger individual subplots
    fig, axes = plt.subplots(3, 3, figsize=(30, 24))
    fig.suptitle('Complete Feature Set Correlations - All Target Columns', fontsize=18, fontweight='bold', y=0.96)
    
    # Load all data for combined chart (global_max_y already calculated in main function)
    all_subplot_data = []
    
    for idx, target_col in enumerate(target_columns):
        # Load correlation data
        csv_path = f"../output/whole_dataset/{target_col}_whole_data_set_correlations.csv"
        
        if not os.path.exists(csv_path):
            all_subplot_data.append(None)
            continue
        
        # Read correlation data
        df = pd.read_csv(csv_path)
        
        # Aggregate correlations by subcategory
        subcategory_df = aggregate_correlations_by_subcategory(df, categorization, aggregation)
        
        if not subcategory_df.empty:
            all_subplot_data.append(subcategory_df)
        else:
            all_subplot_data.append(None)
    
    # Second pass: create the actual plots with consistent y-axis
    for idx, target_col in enumerate(target_columns):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        # Use pre-computed data
        subcategory_df = all_subplot_data[idx]
        
        if subcategory_df is None or subcategory_df.empty:
            ax.text(0.5, 0.5, f"No data for\n{target_col}", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(target_col.upper(), fontsize=16, fontweight='bold', pad=20)
            ax.set_ylim(0, global_max_y * 1.35)  # Set consistent y-axis even for empty plots
            continue
        
        # Calculate category means and use fixed order
        category_means = {}
        for main_cat in subcategory_df['main_category'].unique():
            cat_data = subcategory_df[subcategory_df['main_category'] == main_cat]
            category_means[main_cat] = cat_data['correlation'].mean()
        
        # Use fixed category order (consistent across all charts)
        fixed_order = get_fixed_category_order()
        category_order = [cat for cat in fixed_order if cat in subcategory_df['main_category'].values]
        
        # Build the grouped structure with wider spacing
        x_positions = []
        x_labels = []
        bar_colors = []
        bar_values = []
        category_ranges = {}
        
        current_pos = 0
        bar_width = 1.2  # Increase bar width for combined view
        
        for main_cat in category_order:
            if main_cat not in subcategory_df['main_category'].values:
                continue
                
            start_pos = current_pos
            
            # Add subcategory bars for this main category
            cat_data = subcategory_df[subcategory_df['main_category'] == main_cat].sort_values('correlation', ascending=False)
            
            for _, row in cat_data.iterrows():
                x_positions.append(current_pos)
                x_labels.append(row['subcategory'])
                bar_colors.append(category_colors[main_cat])
                bar_values.append(row['correlation'])
                current_pos += bar_width + 0.2  # Add space between individual bars
            
            # Store category range for rectangle placement
            end_pos = current_pos - (bar_width + 0.2)
            category_ranges[main_cat] = (start_pos, end_pos)
            
            # Add larger spacing between categories
            current_pos += 1.0
        
        # Create bars with increased width
        bars = ax.bar(x_positions, bar_values, width=bar_width, color=bar_colors, alpha=0.8)
        
        # Set title with trait name in uppercase - larger fonts for 3x3 layout
        ax.set_title(target_col.upper(), fontsize=16, fontweight='bold', pad=20)
        
        # Set x-axis labels with better readability - larger fonts
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=11)
        
        # Add value labels on bars and ranking circles (larger fonts for 3x3 layout)
        for i, (bar, value) in enumerate(zip(bars, bar_values)):
            # Add correlation value label above bar
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            # Add ranking rectangle at the upper portion of the bar
            subcategory_name = x_labels[i]
            if subcategory_name in subcategory_rankings:
                rank = subcategory_rankings[subcategory_name]
                
                # Calculate rectangle dimensions (smaller width than bar)
                rect_width = bar.get_width() * 0.7  # 70% of bar width
                rect_height = 0.015  # Fixed small height for combined view
                rect_x = bar.get_x() + (bar.get_width() - rect_width) / 2  # Center horizontally
                rect_y = bar.get_height() - rect_height - 0.003  # Just under the top
                
                # Draw white rectangle with black border
                rectangle = plt.Rectangle((rect_x, rect_y), rect_width, rect_height,
                                        facecolor='white', edgecolor='black', linewidth=1, zorder=15)
                ax.add_patch(rectangle)
                
                # Add rank number inside rectangle (readable font size)
                text_x = rect_x + rect_width/2
                text_y = rect_y + rect_height/2
                ax.text(text_x, text_y, str(rank), ha='center', va='center', 
                        fontsize=7, fontweight='bold', color='black', zorder=16)
        
        # Add category mean rectangles with consistent y-axis range
        ax.set_ylim(0, global_max_y * 1.35)
        
        for main_cat in category_order:
            if main_cat in category_ranges:
                start_pos, end_pos = category_ranges[main_cat]
                mean_val = category_means[main_cat]
                
                # Create rectangle with adjusted proportions for wider bars
                rect_x = start_pos - (bar_width * 0.5)
                rect_width = (end_pos - start_pos) + bar_width
                rect_y = global_max_y * 1.15
                rect_height = global_max_y * 0.12
                
                # Add rectangle
                rect = plt.Rectangle((rect_x, rect_y), rect_width, rect_height,
                                   facecolor=category_colors[main_cat], alpha=0.8,
                                   edgecolor='black', linewidth=1.5)
                ax.add_patch(rect)
                
                # Add text label with larger fonts for 3x3 layout
                wrapped_cat_name = wrap_category_text(main_cat, max_chars_per_line=14)
                ax.text(rect_x + rect_width/2, rect_y + rect_height/2,
                        f'{wrapped_cat_name}\n{mean_val:.3f}', ha='center', va='center', 
                        fontsize=9, fontweight='bold', color='white')
        
        # Add vertical lines to separate main categories
        for i, main_cat in enumerate(category_order[:-1]):
            if main_cat in category_ranges:
                _, end_pos = category_ranges[main_cat]
                ax.axvline(x=end_pos + (bar_width * 0.75), color='gray', linestyle='--', alpha=0.5)
        
        ax.grid(axis='y', alpha=0.3)
    
    # Hide the empty subplot (position 8 in 3x3 grid)
    if len(target_columns) < 9:
        axes[2, 2].set_visible(False)
    
    # Add single y-axis label for the entire figure
    fig.text(0.03, 0.5, 'Absolute Correlation', va='center', rotation='vertical', 
             fontsize=16, fontweight='bold')
    
    # Adjust layout with better spacing for the larger 3x3 figure
    plt.tight_layout()
    plt.subplots_adjust(left=0.07, top=0.93, hspace=0.3, wspace=0.25)
    
    # Save the combined figure as PDF to preserve all formatting
    filename = f"all_targets_combined_feature_correlations.pdf"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, format='pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Saved combined chart: {filepath}")

def main():
    """Main function to generate all bar charts"""
    
    # Load categorization
    categorization = load_categorization()
    if not categorization:
        return
    
    # Define target columns
    target_columns = ['holistic', 'relevance', 'vocabulary', 'style', 
                     'development', 'mechanics', 'grammar', 'organization']
    
    # Get color palette
    category_colors = get_category_colors()
    
    # Create output directory for figures
    output_dir = "../output/figures"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # First pass: collect all data and find global max for consistent y-axis across all charts
    print("Calculating global maximum for consistent y-axis...")
    all_max_values = []
    
    for target_col in target_columns:
        csv_path = f"../output/whole_dataset/{target_col}_whole_data_set_correlations.csv"
        
        if not os.path.exists(csv_path):
            continue
        
        # Read correlation data
        df = pd.read_csv(csv_path)
        
        # Aggregate correlations by subcategory
        subcategory_df = aggregate_correlations_by_subcategory(df, categorization, 'mean')
        
        if not subcategory_df.empty:
            max_val = subcategory_df['correlation'].max()
            all_max_values.append(max_val)
    
    # Calculate global max for consistent y-axis across all charts
    global_max_y = max(all_max_values) if all_max_values else 1.0
    print(f"Global maximum correlation: {global_max_y:.3f}")
    
    # Calculate global subcategory rankings across all target columns
    print("Calculating global subcategory rankings...")
    subcategory_rankings = calculate_global_subcategory_rankings(categorization, target_columns)
    print(f"Total subcategories ranked: {len(subcategory_rankings)}")
    
    # Generate charts for different aggregation methods
    aggregation_methods = ['mean']
    
    print("Generating bar charts...")
    print("="*50)
    
    for aggregation in aggregation_methods:
        print(f"\nGenerating charts with {aggregation} aggregation...")
        
        for target_col in target_columns:
            # Load correlation data
            csv_path = f"../output/whole_dataset/{target_col}_whole_data_set_correlations.csv"
            
            if not os.path.exists(csv_path):
                print(f"CSV file not found: {csv_path}")
                continue
            
            # Read correlation data
            df = pd.read_csv(csv_path)
            
            # Aggregate correlations by subcategory
            subcategory_df = aggregate_correlations_by_subcategory(df, categorization, aggregation)
            
            # Create bar chart with consistent y-axis and rankings
            create_bar_chart(target_col, subcategory_df, category_colors, output_dir, global_max_y, subcategory_rankings, aggregation)
    
    # Create combined chart with all targets
    print(f"\nGenerating combined chart with all targets...")
    create_combined_chart(categorization, category_colors, output_dir, global_max_y, subcategory_rankings, 'mean')
    
    print(f"\nAll charts saved to: {output_dir}")
    print(f"Generated {len(aggregation_methods) * len(target_columns)} individual charts + 1 combined chart")
    
    # Print summary statistics
    print(f"\nSummary:")
    print(f"- Target columns: {len(target_columns)}")
    print(f"- Aggregation methods: {aggregation_methods}")
    print(f"- Main categories: {len(category_colors)}")
    print(f"- Categories are ordered by correlation strength (highest to lowest)")
    print(f"- Category names and means shown as rectangles at the top of each section")
    print(f"- Correlation values displayed on bars (feature counts removed for cleaner look)")
    print(f"- No legend - category information shown in rectangles with 'Features' label")
    print(f"- Text wrapping applied to long category names to prevent overflow")
    print(f"- Individual charts saved as PNG, combined chart saved as PDF")
    print(f"- Combined 3x3 chart uses larger dimensions (30x24) for maximum readability")
    print(f"- Larger individual subplots with enhanced font sizes and spacing")
    print(f"- Wider bars (1.2x width) with increased spacing between categories")
    print(f"- Consistent y-axis range across all subplots for better comparison")
    print(f"- Fixed category order across all charts (Surface, Lexical, Readability, Semantic, Syntactic)")
    print(f"- Ranking rectangles at top of bars showing global subcategory rankings")
    print(f"- Subplot titles show trait names in uppercase (e.g., HOLISTIC, RELEVANCE)")
    print(f"- Single y-axis label on left side of combined chart (no duplication)")
    print(f"- Empty subplot hidden for clean 8-target layout in 3x3 grid")

if __name__ == "__main__":
    main()
