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

def get_category_short_names():
    """Map full category names to shortened versions"""
    return {
        'Surface features': 'Surface',
        'Lexical features': 'Lexical',
        'Readability measures': 'Readability',
        'Semantic features': 'Semantic',
        'Syntactic features': 'Syntactic'
    }

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

def get_fixed_subcategory_order():
    """Define fixed order for subcategories (consistent across all charts)"""
    return {
        'Surface features': [
            'Character-based',
            'Word-based',
            'Sentence-based',
            'Paragraph-based'
        ],
        'Lexical features': [
            'Dialect',
            'Paragraph keywords',
            'N-gram',
            'Punctuation'
        ],
        'Readability measures': [
            'Arabic',
            'English'
        ],
        'Semantic features': [
            'Prompt adherence',
            'Sentiment',
            'Text similarity'
        ],
        'Syntactic features': [
            'Arabic grammetical',
            'Discourse connectives',
            'POS bigrams',
            'POS tags',
            'Pronoun',
            'Sentence structure'
        ]
    }

def get_csv_filename(target_col, dataset_name):
    """Generate appropriate CSV filename based on dataset"""
    if dataset_name == "whole_dataset":
        return f"{target_col}_whole_data_set_correlations.csv"
    else:
        return f"{target_col}_{dataset_name}_correlations.csv"

def calculate_target_specific_subcategory_rankings(target_col, categorization, dataset_name):
    """Calculate ranking of subcategories for a specific target column"""
    csv_filename = get_csv_filename(target_col, dataset_name)
    csv_path = f"../output/{dataset_name}/{csv_filename}"
    
    if not os.path.exists(csv_path):
        return {}
    
    # Read correlation data
    df = pd.read_csv(csv_path)
    
    # Aggregate correlations by subcategory
    subcategory_df = aggregate_correlations_by_subcategory(df, categorization, 'mean')
    
    if subcategory_df.empty:
        return {}
    
    # Sort subcategories by correlation for this specific target (highest to lowest)
    subcategory_df_sorted = subcategory_df.sort_values('correlation', ascending=False)
    
    # Create ranking dictionary (1-based ranking)
    subcategory_rankings = {}
    for rank, (_, row) in enumerate(subcategory_df_sorted.iterrows(), 1):
        subcategory_rankings[row['subcategory']] = rank
    
    return subcategory_rankings

def wrap_category_text(category_name, max_chars_per_line=12):
    """Return short category name instead of wrapping text"""
    short_names = get_category_short_names()
    return short_names.get(category_name, category_name)

def calculate_simple_text_positions(bars, bar_values):
    """Calculate simple text positions above bars"""
    if not bars:
        return []
    
    # Calculate simple positions for each text label
    positions = []
    for i, (bar, value) in enumerate(zip(bars, bar_values)):
        x = bar.get_x() + bar.get_width()/2
        y = bar.get_height() + 0.003  # Small vertical offset
        positions.append({
            'x': x, 'y': y, 'value': value, 'index': i
        })
    
    return positions

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

def create_bar_chart(target_col, subcategory_df, category_colors, output_dir, global_max_y, categorization, dataset_name, aggregation='mean'):
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
        
        # Add subcategory bars for this main category using fixed order
        cat_data = subcategory_df[subcategory_df['main_category'] == main_cat]
        # Sort by fixed subcategory order instead of correlation
        fixed_subcategory_order = get_fixed_subcategory_order()
        if main_cat in fixed_subcategory_order:
            # Create a mapping for sorting
            order_mapping = {subcat: idx for idx, subcat in enumerate(fixed_subcategory_order[main_cat])}
            cat_data = cat_data.sort_values('subcategory', key=lambda x: x.map(order_mapping))
        
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
    
    # Create bars (set zorder to appear above grid)
    bars = plt.bar(x_positions, bar_values, color=bar_colors, alpha=0.8, zorder=3)
    
    # Customize the plot with target name in uppercase and dataset name
    title = f"{target_col.upper()} - {dataset_name.replace('_', ' ').title()}"
    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylabel('Absolute Correlation', fontsize=12)
    
    # Set x-axis labels
    plt.xticks(x_positions, x_labels, rotation=45, ha='right')
    
    # Calculate target-specific rankings
    subcategory_rankings = calculate_target_specific_subcategory_rankings(target_col, categorization, dataset_name)
    
    # Add value labels on bars and ranking circles
    # Calculate simple text positions for correlation values
    text_positions = calculate_simple_text_positions(bars, bar_values)
    
    for i, (bar, value, count) in enumerate(zip(bars, bar_values, bar_counts)):
        # Add correlation value label above bar using simple position
        if i < len(text_positions):
            pos = text_positions[i]
            plt.text(pos['x'], pos['y'], f'{value:.3f}', ha='center', va='bottom', 
                    fontsize=10, fontweight='bold')
        else:
            # Fallback to original positioning if something goes wrong
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add ranking number directly on the bar
        subcategory_name = x_labels[i]
        if subcategory_name in subcategory_rankings:
            rank = subcategory_rankings[subcategory_name]
            
            # Add rank number at the top of the bar with increased white font
            text_x = bar.get_x() + bar.get_width()/2
            text_y = bar.get_height() - 0.02  # Position at the top of the bar
            plt.text(text_x, text_y, str(rank), ha='center', va='bottom', 
                    fontsize=22, fontweight='bold', color='white', zorder=16)
    
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
                    fontsize=10, fontweight='bold', color='white')
    
    # Add vertical lines to separate main categories  
    for i, main_cat in enumerate(category_order[:-1]):  # Don't add line after last category
        if main_cat in category_ranges:
            _, end_pos = category_ranges[main_cat]
            plt.axvline(x=end_pos + 0.75, color='gray', linestyle='--', alpha=0.5)
    
    # No legend - category information is shown in rectangles above bars
    
    # Adjust layout
    plt.tight_layout()
    plt.grid(axis='y', alpha=0.3)
    
    # Save the figure with dataset-specific naming
    filename = f"{target_col}_{dataset_name}_feature_correlations.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {filepath}")
    
    # Print category summary (ordered by correlation)
    print(f"  Category means for {target_col} - {dataset_name} (ordered by correlation):")
    for cat in category_order:
        if cat in category_means:
            mean_val = category_means[cat]
            num_subcat = len(subcategory_df[subcategory_df['main_category'] == cat])
            print(f"    {cat}: {mean_val:.3f} (from {num_subcat} subcategories)")



def create_combined_chart_from_pdfs(categorization, category_colors, output_dir, dataset_name, aggregation='mean'):
    """
    Create a combined chart with all 8 target columns in 3x3 layout for a specific dataset
    This version uses much larger dimensions to avoid squishing
    """
    target_columns = ['holistic', 'relevance', 'vocabulary', 'style', 
                     'development', 'mechanics', 'grammar', 'organization']
    
    # Create figure with subplots (3 rows, 3 columns) - Wider to ensure right padding
    fig, axes = plt.subplots(3, 3, figsize=(60, 34))  # Increased width to ensure right padding
    dataset_title = dataset_name.replace('_', ' ').title()
    fig.suptitle(f'Complete Feature Set Correlations - {dataset_title}', fontsize=28, fontweight='bold', y=0.98)
    
    # Load all data for combined chart
    all_subplot_data = []
    
    for idx, target_col in enumerate(target_columns):
        # Load correlation data
        csv_filename = get_csv_filename(target_col, dataset_name)
        csv_path = f"../output/{dataset_name}/{csv_filename}"
        
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
    
    # Calculate global max for consistent y-axis
    all_max_values = []
    for subcategory_df in all_subplot_data:
        if subcategory_df is not None and not subcategory_df.empty:
            all_max_values.append(subcategory_df['correlation'].max())
    
    global_max_y = max(all_max_values) if all_max_values else 1.0
    
    # Create the actual plots with consistent y-axis
    for idx, target_col in enumerate(target_columns):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        # Use pre-computed data
        subcategory_df = all_subplot_data[idx]
        
        if subcategory_df is None or subcategory_df.empty:
            ax.text(0.5, 0.5, f"No data for\n{target_col}", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(target_col.upper(), fontsize=20, fontweight='bold', pad=20)
            ax.set_ylim(0, global_max_y * 1.35)
            continue
        
        # Calculate category means and use fixed order
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
        category_ranges = {}
        
        current_pos = 0
        
        for main_cat in category_order:
            if main_cat not in subcategory_df['main_category'].values:
                continue
                
            start_pos = current_pos
            
            # Add subcategory bars for this main category using fixed order
            cat_data = subcategory_df[subcategory_df['main_category'] == main_cat]
            # Sort by fixed subcategory order instead of correlation
            fixed_subcategory_order = get_fixed_subcategory_order()
            if main_cat in fixed_subcategory_order:
                # Create a mapping for sorting
                order_mapping = {subcat: idx for idx, subcat in enumerate(fixed_subcategory_order[main_cat])}
                cat_data = cat_data.sort_values('subcategory', key=lambda x: x.map(order_mapping))
            
            for _, row in cat_data.iterrows():
                x_positions.append(current_pos)
                x_labels.append(row['subcategory'])
                bar_colors.append(category_colors[main_cat])
                bar_values.append(row['correlation'])
                current_pos += 1
            
            # Store category range for rectangle placement
            end_pos = current_pos - 1
            category_ranges[main_cat] = (start_pos, end_pos)
            
            # Add spacing between categories
            current_pos += 0.5
        
        # Create bars
        bars = ax.bar(x_positions, bar_values, color=bar_colors, alpha=0.8, zorder=3)
        
        # Set title with trait name in uppercase
        ax.set_title(target_col.upper(), fontsize=20, fontweight='bold', pad=20)
        
        # Set x-axis labels
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=14)
        
        # Set y-axis tick label font size
        ax.tick_params(axis='y', labelsize=16)
        
        # Calculate target-specific rankings for this subplot
        target_specific_rankings = calculate_target_specific_subcategory_rankings(target_col, categorization, dataset_name)
        
        # Add value labels on bars and ranking rectangles
        text_positions = calculate_simple_text_positions(bars, bar_values)
        
        for i, (bar, value) in enumerate(zip(bars, bar_values)):
            # Add correlation value label above bar
            if i < len(text_positions):
                pos = text_positions[i]
                ax.text(pos['x'], pos['y'], f'{value:.3f}', ha='center', va='bottom', 
                        fontsize=12, fontweight='bold')
            else:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
            
            # Add ranking number directly on the bar
            subcategory_name = x_labels[i]
            if subcategory_name in target_specific_rankings:
                rank = target_specific_rankings[subcategory_name]
                
                # Add rank number directly on the bar with increased white font
                text_x = bar.get_x() + bar.get_width()/2
                text_y = bar.get_height() - 0.029  # Position at the top of the bar
                ax.text(text_x, text_y, str(rank), ha='center', va='bottom', 
                        fontsize=22, fontweight='bold', color='white', zorder=16)
        
        # Add category mean rectangles with consistent y-axis range
        ax.set_ylim(0, global_max_y * 1.35)
        
        for main_cat in category_order:
            if main_cat in category_ranges:
                start_pos, end_pos = category_ranges[main_cat]
                mean_val = category_means[main_cat]
                
                rect_x = start_pos - 0.4
                rect_width = (end_pos - start_pos) + 0.8
                rect_y = global_max_y * 1.15
                rect_height = global_max_y * 0.12
                
                rect = plt.Rectangle((rect_x, rect_y), rect_width, rect_height,
                                   facecolor=category_colors[main_cat], alpha=0.8,
                                   edgecolor='black', linewidth=1.5)
                ax.add_patch(rect)
                
                wrapped_cat_name = wrap_category_text(main_cat)
                ax.text(rect_x + rect_width/2, rect_y + rect_height/2,
                        f'{wrapped_cat_name}\n{mean_val:.3f}', ha='center', va='center', 
                        fontsize=14, fontweight='bold', color='white')
        
        # Add vertical lines to separate main categories
        for i, main_cat in enumerate(category_order[:-1]):
            if main_cat in category_ranges:
                _, end_pos = category_ranges[main_cat]
                ax.axvline(x=end_pos + 0.75, color='gray', linestyle='--', alpha=0.5)
        
        ax.grid(axis='y', alpha=0.3)
    
    # Hide the empty subplot (position 8 in 3x3 grid)
    if len(target_columns) < 9:
        axes[2, 2].set_visible(False)
    
    # Add single y-axis label for the entire figure
    fig.text(0.03, 0.5, 'Absolute Correlation', va='center', rotation='vertical', 
             fontsize=60, fontweight='bold')
    
    # Adjust layout with reduced spacing between columns and better title positioning
    # Use tight_layout with padding to ensure proper margins
    plt.tight_layout(pad=4.0)
    plt.subplots_adjust(left=0.08, right=0.95, top=0.93, hspace=0.4, wspace=0.08)
    
    # Save the combined figure as PDF with additional padding
    filename = f"all_targets_combined_{dataset_name}_correlations.pdf"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, format='pdf', bbox_inches='tight', dpi=300, pad_inches=0.5)
    plt.close()
    
    print(f"Saved combined chart: {filepath}")


def create_three_target_chart(categorization, category_colors, output_dir, aggregation='mean'):
    """
    Create a 1-row chart with 3 target columns (holistic, relevance, organization) for whole dataset only
    This version uses individual subplot rendering like the combined figure to avoid squishing
    """
    target_columns = ['holistic', 'relevance', 'organization']
    dataset_name = 'whole_dataset'
    
    # Create figure with subplots (3 rows, 1 column) - Vertical layout
    fig, axes = plt.subplots(3, 1, figsize=(15, 45))  # Vertical, tall figure
    
    # Load all data for the chart
    all_subplot_data = []
    
    for idx, target_col in enumerate(target_columns):
        # Load correlation data
        csv_filename = get_csv_filename(target_col, dataset_name)
        csv_path = f"../output/{dataset_name}/{csv_filename}"
        
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
    
    # Calculate global max for consistent y-axis
    all_max_values = []
    for subcategory_df in all_subplot_data:
        if subcategory_df is not None and not subcategory_df.empty:
            all_max_values.append(subcategory_df['correlation'].max())
    
    global_max_y = max(all_max_values) if all_max_values else 1.0
    
    # Create the actual plots with consistent y-axis
    for idx, target_col in enumerate(target_columns):
        ax = axes[idx]
        
        # Use pre-computed data
        subcategory_df = all_subplot_data[idx]
        
        if subcategory_df is None or subcategory_df.empty:
            ax.text(0.5, 0.5, f"No data for\n{target_col}", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(target_col.upper(), fontsize=20, fontweight='bold', pad=20)
            ax.set_ylim(0, 0.4)  # Set chart range to end at exactly 0.4
            continue
        
        # Calculate category means and use fixed order
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
        category_ranges = {}
        
        current_pos = 0
        
        for main_cat in category_order:
            if main_cat not in subcategory_df['main_category'].values:
                continue
                
            start_pos = current_pos
            
            # Add subcategory bars for this main category using fixed order
            cat_data = subcategory_df[subcategory_df['main_category'] == main_cat]
            # Sort by fixed subcategory order instead of correlation
            fixed_subcategory_order = get_fixed_subcategory_order()
            if main_cat in fixed_subcategory_order:
                # Create a mapping for sorting
                order_mapping = {subcat: idx for idx, subcat in enumerate(fixed_subcategory_order[main_cat])}
                cat_data = cat_data.sort_values('subcategory', key=lambda x: x.map(order_mapping))
            
            for _, row in cat_data.iterrows():
                x_positions.append(current_pos)
                x_labels.append(row['subcategory'])
                bar_colors.append(category_colors[main_cat])
                bar_values.append(row['correlation'])
                current_pos += 1
            
            # Store category range for rectangle placement
            end_pos = current_pos - 1
            category_ranges[main_cat] = (start_pos, end_pos)
            
            # Add spacing between categories
            current_pos += 0.5
        
        # Create bars
        bars = ax.bar(x_positions, bar_values, color=bar_colors, alpha=0.8, zorder=3)
        
        # Set title with trait name in uppercase
        ax.set_title(target_col.upper(), fontsize=20, fontweight='bold', pad=20)
        
        # Set x-axis labels - only show labels for the bottom subplot (last one)
        ax.set_xticks(x_positions)
        if idx == len(target_columns) - 1:  # Only show labels for the last subplot
            ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=14)
        else:
            ax.set_xticklabels([])  # Hide labels for other subplots
        
        # Set y-axis tick label font size (normal size like combined figure)
        ax.tick_params(axis='y', labelsize=16)
        
        # Calculate target-specific rankings for this subplot
        target_specific_rankings = calculate_target_specific_subcategory_rankings(target_col, categorization, dataset_name)
        
        # Add value labels on bars and ranking rectangles
        text_positions = calculate_simple_text_positions(bars, bar_values)
        
        for i, (bar, value) in enumerate(zip(bars, bar_values)):
            # Add correlation value label above bar - make it vertical with increased font size
            if i < len(text_positions):
                pos = text_positions[i]
                ax.text(pos['x'], pos['y'], f'{value:.3f}', ha='center', va='bottom', 
                        fontsize=20, fontweight='bold', rotation=90)
            else:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=16, fontweight='bold', rotation=90)
            
            # Add ranking number directly on the bar
            subcategory_name = x_labels[i]
            if subcategory_name in target_specific_rankings:
                rank = target_specific_rankings[subcategory_name]
                
                # Add rank number directly on the bar with increased white font
                text_x = bar.get_x() + bar.get_width()/2
                text_y = bar.get_height() - 0.018  # Position at the top of the bar
                ax.text(text_x, text_y, str(rank), ha='center', va='bottom', 
                        fontsize=22, fontweight='bold', color='white', zorder=16)
        
        # Add category mean rectangles with consistent y-axis range
        ax.set_ylim(0, 0.4)  # Set chart range to end at exactly 0.4
        
        for main_cat in category_order:
            if main_cat in category_ranges:
                start_pos, end_pos = category_ranges[main_cat]
                mean_val = category_means[main_cat]
                rect_x = start_pos - 0.4
                rect_width = (end_pos - start_pos) + 0.8
                # Move rectangles to the very top (95% of 0.4 range)
                rect_y = 0.4 * 0.9
                rect_height = 0.4 * 0.06  # Smaller height (8% of the 0.4 range)
                rect = plt.Rectangle((rect_x, rect_y), rect_width, rect_height,
                                   facecolor=category_colors[main_cat], alpha=0.8,
                                   edgecolor='black', linewidth=1.5)
                ax.add_patch(rect)
                wrapped_cat_name = wrap_category_text(main_cat)
                ax.text(rect_x + rect_width/2, rect_y + rect_height/2,
                        f'{wrapped_cat_name}\n{mean_val:.3f}', ha='center', va='center', 
                        fontsize=11, fontweight='bold', color='white')
        
        # Add vertical lines to separate main categories
        for i, main_cat in enumerate(category_order[:-1]):
            if main_cat in category_ranges:
                _, end_pos = category_ranges[main_cat]
                ax.axvline(x=end_pos + 0.75, color='gray', linestyle='--', alpha=0.5)
        
        ax.grid(axis='y', alpha=0.3)
    
    # Add single y-axis label for the entire figure (same font size as combined figure)
    fig.text(0.03, 0.5, 'Absolute Correlation', va='center', rotation='vertical', 
             fontsize=25, fontweight='bold')
    
    # Adjust layout with more spacing to prevent squishing
    plt.tight_layout(pad=4.0)
    plt.subplots_adjust(left=0.1, right=0.95, top=0.93, hspace=0.08, wspace=0.15)  # Increased wspace
    
    # Save the three-target figure as PDF with additional padding
    filename = f"three_targets_whole_dataset_correlations.pdf"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, format='pdf', bbox_inches='tight', dpi=300, pad_inches=0.5)
    plt.close()
    
    print(f"Saved three-target chart: {filepath}")





def process_dataset(dataset_name, categorization, target_columns, category_colors, aggregation_methods):
    """Process a single dataset and generate all charts for it"""
    print(f"\n{'='*60}")
    print(f"Processing dataset: {dataset_name}")
    print(f"{'='*60}")
    
    # Create output directory for this dataset's figures
    output_dir = f"../output/figures/{dataset_name}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # First pass: collect all data and find global max for consistent y-axis across all charts
    print(f"Calculating global maximum for consistent y-axis for {dataset_name}...")
    all_max_values = []
    
    for target_col in target_columns:
        csv_filename = get_csv_filename(target_col, dataset_name)
        csv_path = f"../output/{dataset_name}/{csv_filename}"
        
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
    print(f"Global maximum correlation for {dataset_name}: {global_max_y:.3f}")
    
    # Generate charts for different aggregation methods (rankings will be calculated per target)
    print(f"Generating bar charts for {dataset_name}...")
    print("-" * 50)
    
    for aggregation in aggregation_methods:
        print(f"\nGenerating charts with {aggregation} aggregation for {dataset_name}...")
        
        for target_col in target_columns:
            # Load correlation data
            csv_filename = get_csv_filename(target_col, dataset_name)
            csv_path = f"../output/{dataset_name}/{csv_filename}"
            
            if not os.path.exists(csv_path):
                print(f"CSV file not found: {csv_path}")
                continue
            
            # Read correlation data
            df = pd.read_csv(csv_path)
            
            # Aggregate correlations by subcategory
            subcategory_df = aggregate_correlations_by_subcategory(df, categorization, aggregation)
            
            # Create bar chart with consistent y-axis and target-specific rankings
            create_bar_chart(target_col, subcategory_df, category_colors, output_dir, global_max_y, categorization, dataset_name, aggregation)
    
    # Create combined chart by stitching individual PDF files
    print(f"\nGenerating combined chart by stitching PDFs for {dataset_name}...")
    create_combined_chart_from_pdfs(categorization, category_colors, output_dir, dataset_name, 'mean')
    
    print(f"\nAll charts for {dataset_name} saved to: {output_dir}")
    print(f"Generated {len(aggregation_methods) * len(target_columns)} individual charts + 1 combined chart for {dataset_name}")

def main():
    """Main function to generate all bar charts for all datasets"""
    
    # Load categorization
    categorization = load_categorization()
    if not categorization:
        return
    
    # Define target columns
    target_columns = ['holistic', 'relevance', 'vocabulary', 'style', 
                     'development', 'mechanics', 'grammar', 'organization']
    
    # Define datasets to process
    datasets = ['whole_dataset', 'prompt_1', 'prompt_2', 'prompt_3', 'prompt_4']
    
    # Get color palette
    category_colors = get_category_colors()
    
    # Create main output directory for figures
    main_output_dir = "../output/figures"
    if not os.path.exists(main_output_dir):
        os.makedirs(main_output_dir)
    
    # Generate charts for different aggregation methods
    aggregation_methods = ['mean']
    
    print("Starting figure generation for all datasets...")
    print("="*80)
    
    # Process each dataset
    for dataset_name in datasets:
        # Check if dataset directory exists
        dataset_dir = f"../output/{dataset_name}"
        if not os.path.exists(dataset_dir):
            print(f"Dataset directory not found: {dataset_dir}")
            continue
        
        # Process this dataset
        process_dataset(dataset_name, categorization, target_columns, category_colors, aggregation_methods)
    
    # Create three-target chart for whole dataset separately
    print(f"\n{'='*60}")
    print(f"Generating three-target chart for whole dataset")
    print(f"{'='*60}")
    whole_dataset_output_dir = f"../output/figures/whole_dataset"
    if os.path.exists(whole_dataset_output_dir):
        create_three_target_chart(categorization, category_colors, whole_dataset_output_dir, 'mean')
        print(f"Three-target chart saved to: {whole_dataset_output_dir}")
    else:
        print(f"Whole dataset output directory not found: {whole_dataset_output_dir}")
    
    # Print final summary
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Processed datasets: {len([d for d in datasets if os.path.exists(f'../output/{d}')])}")
    print(f"Target columns per dataset: {len(target_columns)}")
    print(f"Aggregation methods: {aggregation_methods}")
    print(f"Main categories: {len(category_colors)}")
    print(f"Figures saved to subdirectories in: {main_output_dir}")
    print(f"\nFeatures:")
    print(f"- Categories are ordered by correlation strength (highest to lowest)")
    print(f"- Category names and means shown as rectangles at the top of each section")
    print(f"- Correlation values displayed on bars")
    print(f"- Individual charts saved as PNG, combined charts saved as PDF")
    print(f"- Combined 3x3 charts use larger dimensions (30x24) for maximum readability")
    print(f"- Consistent y-axis range across all charts for better comparison")
    print(f"- Fixed category order across all charts (Surface, Lexical, Readability, Semantic, Syntactic)")
    print(f"- Ranking rectangles at top of bars showing target-specific subcategory rankings (rank 1 = highest correlation for that target)")
    print(f"- Chart titles include dataset names for clear identification")
    print(f"- Separate subdirectories for each dataset mirroring the output structure")

if __name__ == "__main__":
    main()
