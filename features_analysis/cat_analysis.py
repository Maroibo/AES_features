import pandas as pd
import os
import json
from collections import defaultdict

def define_feature_categories():
    """Define all feature categories and subcategories"""
    
    surface_features = {
        'Paragraph-based': [
            'first_paragraph_has_intro_words',
            'last_paragraph_has_conclusion_words',
            'paragraphs',
            'longest_paragraph_length',
            'shortest_paragraph_length',
            'more_than_1_paragraph',
            'paragraphs_count',
            'is_first_paragraph_less_than_or_equal_to_10',
            'average_length_paragraph',
            'max_length_paragraph',
            'min_length_paragraph',
            'shortest_paragraph_length_ratio',
            'longest_paragraph_length_ratio'
        ],
        'Character-based': [
            'hmpz_count',
            'misspelled_count',
            'intro_hmpz_count',
            'body_hmpz_count',
            'conclusion_hmpz_count',
            'chars_count'
        ],
        'Sentence-based': [
            'average_length_sentence',
            'sentences_count',
            'sent_var',
            'max_length_sentence',
            'min_length_sentence',
            'standard_deviation_sentence',
            'intro_sentences_count',
            'intro_average_length_sentence',
            'intro_max_length_sentence',
            'intro_min_length_sentence',
            'intro_standard_deviation_sentence',
            'body_sentences_count',
            'body_average_length_sentence',
            'body_max_length_sentence',
            'body_min_length_sentence',
            'body_standard_deviation_sentence',
            'conclusion_sentences_count',
            'conclusion_average_length_sentence',
            'conclusion_max_length_sentence',
            'conclusion_min_length_sentence',
            'conclusion_standard_deviation_sentence'
        ],
        'Word-based': [
            'no_of_words_in_first',
            'no_of_words_in_body',
            'no_of_words_in_conclusion',
            'words_count',
            'syllables',
            'syll_per_word',
            'long_words',
            'long_words_count',
            'complex_words',
            'nominalization',
            'unique_words_count',
            'avg_lemma_length',
            'total_lemmas',
            'stop_words_count',
            'words_count_without_stopwords',
            'word_var',
            'stop_prop',
            'type_token_ratio',
            'log_words_count',
            'log_unique_words_count',
            'average_word_length',
            'max_length_word',
            'min_length_word',
            'standard_deviation_words',
            'intro_words_count',
            'intro_log_words_count',
            'intro_unique_words_count',
            'intro_log_unique_words_count',
            'intro_average_word_length',
            'intro_max_length_word',
            'intro_min_length_word',
            'intro_standard_deviation_words',
            'intro_chars_count',
            'body_words_count',
            'body_log_words_count',
            'body_unique_words_count',
            'body_log_unique_words_count',
            'body_average_word_length',
            'body_max_length_word',
            'body_min_length_word',
            'body_standard_deviation_words',
            'body_chars_count',
            'conclusion_words_count',
            'conclusion_log_words_count',
            'conclusion_unique_words_count',
            'conclusion_log_unique_words_count',
            'conclusion_average_word_length',
            'conclusion_max_length_word',
            'conclusion_min_length_word',
            'conclusion_standard_deviation_words',
            'conclusion_chars_count'
        ]
    }
    
    lexical_features = {
        'Dialect': [
            'dialect_counts',
            'msa_percentage', 
            'dialect_percentage'
        ],
        'Paragraph keywords': [
            'basmallah_use',
            'hamd_use',
            'amma_baad_use',
            'salla_allah_use',
            'sallam_use',
            'salla_alla_mohammed_use'
        ],
        'N-gram': [
            'top n words',
            'top_n_num_sent',
            'top_n_percentage_sent'
        ],
        'Punctuation': [
            'period_count',
            'comma_count',
            'question_mark_correct',
            'question_mark_missing',
            'question_mark_incorrect',
            'exclamation_mark_correct',
            'exclamation_mark_missing',
            'exclamation_mark_incorrect',
            'semicolon_correct',
            'semicolon_missing',
            'semicolon_incorrect',
            'comma_missing',
            'comma_incorrect',
            'period_correct',
            'period_incorrect',
            'colon_correct',
            'colon_missing',
            'colon_incorrect',
            'quotation_mark_correct',
            'quotation_mark_missing',
            'quotation_mark_incorrect',
            'prep_comma',
            'colon_exists',
            'parantheses_exists',
            'paranthesis_exists',
            'question_mark_exits',
            'question_mark_exists',
            'punc_count',
            'dup_punctuation_count',
            'exclamation_count',
            'semicolon_count',
            'quotation_mark_count',
            'dash_count',
            'colon_count',
            'question_count',
            'parenthesis_count',
            'intro_exclamation_count',
            'intro_semicolon_count',
            'intro_dash_count',
            'intro_colon_count',
            'intro_question_count',
            'intro_period_count',
            'intro_comma_count',
            'intro_quotation_mark_count',
            'intro_parenthesis_count',
            'intro_punc_count',
            'body_exclamation_count',
            'body_semicolon_count',
            'body_dash_count',
            'body_colon_count',
            'body_question_count',
            'body_period_count',
            'body_comma_count',
            'body_quotation_mark_count',
            'body_parenthesis_count',
            'body_punc_count',
            'conclusion_exclamation_count',
            'conclusion_semicolon_count',
            'conclusion_dash_count',
            'conclusion_colon_count',
            'conclusion_question_count',
            'conclusion_period_count',
            'conclusion_comma_count',
            'conclusion_quotation_mark_count',
            'conclusion_parenthesis_count',
            'conclusion_punc_count'
        ]
    }
    
    readability_measures = {
        'Arabic': [
            'OSMAN',
            'Heeti', 
            'AARI',
            'AARIBase'
        ],
        'English': [
            'linsear_write',
            'LinsearWrite',
            'Kincaid',
            'ARI',
            'Coleman-Liau',
            'FleschReadingEase',
            'GunningFogIndex',
            'LIX',
            'RIX',
            'SMOGIndex'
        ]
    }
    
    semantic_features = {
        'Prompt adherence': [
            'max_sentence_dot_score',
            'mean_sentence_dot_score',
            'min_sentence_dot_scor',
            'min_sentence_dot_score',
            'dot_score'
        ],
        'Sentiment': [
            'overall_positivity_score',
            'overall_negativity_score',
            'overall_positivity',
            'overall_negativity',
            'overall_neutrality',
            'positive_sentence_prop',
            'neutral_sentence_prop',
            'negative_sentence_prop'
        ],
        'Text similarity': [
            'max_matched_words',
            'avg_matched_words',
            'max_paragraph_sim',
            'max_sent_sim',
            'avg_paragraph_sim',
            'avg_sent_sim'
        ]
    }
    
    syntactic_features = {
        'Arabic grammetical': [
            'auxverb',
            'num_of_jazm',
            'num_of_jazm_followed_plural_verb',
            'jazm_with_plural_verb',
            'total_jazm',
            'kanna_count',
            'kana_count',
            'inna_count'
        ],
        'Discourse connectives': [
            'conjunction',
            'discourse_conn_count',
            'unique_connective_ratio',
            'average_connective_distance',
            'connective_ratio',
            'total_conjunctions',
            # Arabic discourse connectives from conjunctions_dict
            'الا ان',
            'بيد ان', 
            'غير ان',
            'على الرغم',
            'رغمان',
            'بالرغم من',
            'برغم',
            'بالمقابل',
            'في المقابل',
            'بيد',
            'بعدما',
            'اذ',
            'بينما',
            'عقب',
            'قبيل',
            'وقبل',
            'من ثم',
            'قبل ان',
            'جراء',
            'نظرا ل',
            'بفضل',
            'لأن',
            'بحيث',
            'الا اذا',
            'حتى لو',
            'لولا',
            'طالما',
            'كلما',
            'بغية',
            'كأن',
            'خلافا ل',
            'بمعنى اخر',
            'في ظل',
            'حال'
        ],
        'POS bigrams': [
            'Bigram features'  # Features starting with 'pos_bigram' will be detected dynamically
        ],
        'POS tags': [
            'pos tags features',  # Features starting with 'pos' will be detected dynamically
            'verb_count'
        ],
        'Pronoun': [
            'pronoun',
            'Pronoun Count',
            'Pronoun Group Count',
            'Sent Pronoun',
            'Sent Pronoun Group',
            'Sent Pronoun Portion',
            'Sent Pronoun Group Portion'
            # Features containing 'pron' will be detected dynamically
        ],
        'Syntactic Pronoun': [
            # Dynamic detection for syntactic pronoun features:
            # - Features starting with 'group_' (group_demonstrative, group_first_person, etc.)
            # - Features starting with 'sent_count_' (sent_count_demonstrative, etc.)
            # - Features starting with 'sent_percent_' (sent_percent_demonstrative, etc.)
        ],
        'Sentence structure': [
            'clause_per_s',
            'sent_ave_depth',
            'ave_leaf_depth',
            'max_clause_in_s',
            'mean_clause',
            'begin_w_preposition',
            'begin_w_conjunction',
            'begin_w_subordination',
            'begin_w_article',
            'begin_w_interrogative',
            'begin_w_pronoun',
            'num_of_nominal',
            'num_of_verbal',
            'nominal_sentences',
            'verbal_sentences'
        ]
    }
    
    # Create feature set for quick lookup
    categorized_features = set()
    for subcategory, features in surface_features.items():
        categorized_features.update(features)
    for subcategory, features in lexical_features.items():
        categorized_features.update(features)
    for subcategory, features in readability_measures.items():
        categorized_features.update(features)
    for subcategory, features in semantic_features.items():
        categorized_features.update(features)
    for subcategory, features in syntactic_features.items():
        categorized_features.update(features)
    
    return surface_features, lexical_features, readability_measures, semantic_features, syntactic_features, categorized_features

def analyze_features_for_target(target_col, dataset_type, categorized_features, surface_features, lexical_features, readability_measures, semantic_features, syntactic_features):
    """Analyze features for a single target column and dataset type"""
    
    # Determine filepath based on dataset type
    if dataset_type == 'whole_dataset':
        filepath = f'../output/whole_dataset/{target_col}_whole_data_set_correlations.csv'
    else:
        # Extract prompt number from dataset_type (e.g., 'prompt_1' -> '1')
        prompt_num = dataset_type.split('_')[1]
        filepath = f'../output/{dataset_type}/{target_col}_prompt_{prompt_num}_correlations.csv'
    
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None
    
    df = pd.read_csv(filepath)
    all_features = set(df['feature'].tolist())
    
    print(f"\n{target_col.upper()} - {dataset_type.upper()}:")
    print(f"  Total features: {len(all_features)}")
    
    # Track feature assignments to detect overlaps
    feature_assignments = defaultdict(list)
    
    # Count Surface features by subcategory
    surface_total = 0
    surface_assigned_features = set()
    print(f"  Surface features:")
    for subcategory, feature_list in surface_features.items():
        matched_features = [f for f in feature_list if f in all_features]
        count = len(matched_features)
        print(f"    {subcategory}: {count}")
        surface_total += count
        surface_assigned_features.update(matched_features)
        for f in matched_features:
            feature_assignments[f].append(f"Surface-{subcategory}")
    print(f"    Total Surface features: {surface_total}")
    
    # Count Lexical features by subcategory  
    lexical_total = 0
    lexical_assigned_features = set()
    print(f"  Lexical features:")
    for subcategory, feature_list in lexical_features.items():
        if subcategory == 'N-gram':
            # Explicit features
            explicit_matched = [f for f in feature_list if f in all_features]
            # Pattern-based features
            pattern_matched = [f for f in all_features if f.startswith('top_n') and f not in feature_list]
            all_matched = explicit_matched + pattern_matched
            count = len(all_matched)
            lexical_assigned_features.update(all_matched)
            for f in all_matched:
                feature_assignments[f].append(f"Lexical-{subcategory}")
        else:
            matched_features = [f for f in feature_list if f in all_features]
            count = len(matched_features)
            lexical_assigned_features.update(matched_features)
            for f in matched_features:
                feature_assignments[f].append(f"Lexical-{subcategory}")
        print(f"    {subcategory}: {count}")
        lexical_total += count
    print(f"    Total Lexical features: {lexical_total}")
    
    # Count Readability measures by subcategory
    readability_total = 0
    readability_assigned_features = set()
    print(f"  Readability measures:")
    for subcategory, feature_list in readability_measures.items():
        matched_features = [f for f in feature_list if f in all_features]
        count = len(matched_features)
        print(f"    {subcategory}: {count}")
        readability_total += count
        readability_assigned_features.update(matched_features)
        for f in matched_features:
            feature_assignments[f].append(f"Readability-{subcategory}")
    print(f"    Total Readability measures: {readability_total}")
    
    # Count Semantic features by subcategory
    semantic_total = 0
    semantic_assigned_features = set()
    print(f"  Semantic features:")
    for subcategory, feature_list in semantic_features.items():
        matched_features = [f for f in feature_list if f in all_features]
        count = len(matched_features)
        print(f"    {subcategory}: {count}")
        semantic_total += count
        semantic_assigned_features.update(matched_features)
        for f in matched_features:
            feature_assignments[f].append(f"Semantic-{subcategory}")
    print(f"    Total Semantic features: {semantic_total}")
    
    # Count Syntactic features by subcategory
    syntactic_total = 0
    syntactic_assigned_features = set()
    print(f"  Syntactic features:")
    for subcategory, feature_list in syntactic_features.items():
        if subcategory == 'POS bigrams':
            # Explicit features
            explicit_matched = [f for f in feature_list if f in all_features]
            # Pattern-based features
            pattern_matched = [f for f in all_features if f.startswith('pos_bigram')]
            all_matched = list(set(explicit_matched + pattern_matched))  # Remove duplicates
            count = len(all_matched)
            syntactic_assigned_features.update(all_matched)
            for f in all_matched:
                feature_assignments[f].append(f"Syntactic-{subcategory}")
        elif subcategory == 'POS tags':
            # Explicit features
            explicit_matched = [f for f in feature_list if f in all_features]
            # Pattern-based features (exclude sentiment features that start with "pos")
            sentiment_pos_features = ['positive_sentence_prop']
            pattern_matched = [f for f in all_features 
                             if f.startswith('pos') 
                             and not f.startswith('pos_bigram')
                             and f not in sentiment_pos_features]
            all_matched = list(set(explicit_matched + pattern_matched))  # Remove duplicates
            count = len(all_matched)
            syntactic_assigned_features.update(all_matched)
            for f in all_matched:
                feature_assignments[f].append(f"Syntactic-{subcategory}")
        elif subcategory == 'Pronoun':
            # Explicit features
            explicit_matched = [f for f in feature_list if f in all_features]
            # Pattern-based features (excluding various syntactic patterns)
            exclude_patterns = [
                'group_', 'sent_count_', 'sent_percent_',  # Syntactic Pronoun patterns
                'pos_bigram_', 'pos_',  # POS patterns
                'begin_w_'  # Sentence structure patterns
            ]
            pattern_matched = [f for f in all_features 
                             if 'pron' in f.lower() 
                             and f not in feature_list
                             and not any(f.startswith(pattern) for pattern in exclude_patterns)]
            all_matched = list(set(explicit_matched + pattern_matched))  # Remove duplicates
            count = len(all_matched)
            syntactic_assigned_features.update(all_matched)
            for f in all_matched:
                feature_assignments[f].append(f"Syntactic-{subcategory}")
        elif subcategory == 'Syntactic Pronoun':
            # Explicit features
            explicit_matched = [f for f in feature_list if f in all_features]
            # Pattern-based features
            group_matched = [f for f in all_features if f.startswith('group_')]
            sent_count_matched = [f for f in all_features if f.startswith('sent_count_')]
            sent_percent_matched = [f for f in all_features if f.startswith('sent_percent_')]
            all_matched = list(set(explicit_matched + group_matched + sent_count_matched + sent_percent_matched))
            count = len(all_matched)
            syntactic_assigned_features.update(all_matched)
            for f in all_matched:
                feature_assignments[f].append(f"Syntactic-{subcategory}")
        else:
            matched_features = [f for f in feature_list if f in all_features]
            count = len(matched_features)
            syntactic_assigned_features.update(matched_features)
            for f in matched_features:
                feature_assignments[f].append(f"Syntactic-{subcategory}")
        print(f"    {subcategory}: {count}")
        syntactic_total += count
    print(f"    Total Syntactic features: {syntactic_total}")
    
    # Find overlapping features
    overlapping_features = {f: assignments for f, assignments in feature_assignments.items() if len(assignments) > 1}
    
    if overlapping_features:
        print(f"\n  OVERLAPPING FEATURES ({len(overlapping_features)}):")
        for feature, assignments in sorted(overlapping_features.items()):
            print(f"    {feature}: {', '.join(assignments)}")
    
    # Calculate unique categorized features (no double counting)
    all_categorized_features = set()
    all_categorized_features.update(surface_assigned_features)
    all_categorized_features.update(lexical_assigned_features)
    all_categorized_features.update(readability_assigned_features)
    all_categorized_features.update(semantic_assigned_features)
    all_categorized_features.update(syntactic_assigned_features)
    
    # Count uncategorized
    uncategorized = all_features - all_categorized_features
    uncategorized_count = len(uncategorized)
    print(f"  Uncategorized features: {uncategorized_count}")
    print(f"  Unique categorized features: {len(all_categorized_features)}")
    
    return {
        'total': len(all_features),
        'surface_total': surface_total,
        'lexical_total': lexical_total,
        'readability_total': readability_total,
        'semantic_total': semantic_total,
        'syntactic_total': syntactic_total,
        'unique_categorized': len(all_categorized_features),
        'overlapping_count': len(overlapping_features),
        'overlapping_features': overlapping_features,
        'uncategorized': uncategorized_count,
        'uncategorized_features': uncategorized,
        'all_features': all_features,
        'surface_assigned_features': surface_assigned_features,
        'lexical_assigned_features': lexical_assigned_features,
        'readability_assigned_features': readability_assigned_features,
        'semantic_assigned_features': semantic_assigned_features,
        'syntactic_assigned_features': syntactic_assigned_features,
        'feature_assignments': feature_assignments
    }

def main():
    """Main function to run simple feature categorization"""
    
    print("Feature Categorization Analysis")
    print("="*40)
    
    # Define target columns and dataset types
    target_columns = ['holistic', 'relevance', 'vocabulary', 'style', 
                     'development', 'mechanics', 'grammar', 'organization']
    dataset_types = ['whole_dataset', 'prompt_1', 'prompt_2', 'prompt_3', 'prompt_4']
    
    # Load feature definitions
    surface_features, lexical_features, readability_measures, semantic_features, syntactic_features, categorized_features = define_feature_categories()
    
    print(f"Defined categories:")
    print(f"  Surface features subcategories: {len(surface_features)}")
    print(f"  Lexical features subcategories: {len(lexical_features)}")
    print(f"  Readability measures subcategories: {len(readability_measures)}")
    print(f"  Semantic features subcategories: {len(semantic_features)}")
    print(f"  Syntactic features subcategories: {len(syntactic_features)}")
    print(f"  Total categorized features defined: {len(categorized_features)}")
    
    # Initialize comprehensive categorization tracking
    comprehensive_categorization = {
        "categories": {
            "Surface features": {},
            "Lexical features": {},
            "Readability measures": {},
            "Semantic features": {},
            "Syntactic features": {}
        },
        "datasets": {},
        "metadata": {
            "total_features_defined": len(categorized_features),
            "target_columns": target_columns,
            "dataset_types": dataset_types
        }
    }
    
    # Analyze each target for each dataset type
    results = {}
    all_uncategorized_features = set()  # To collect unique uncategorized features
    all_found_features = set()  # To collect all features found across datasets
    
    for dataset_type in dataset_types:
        print(f"\n{'='*60}")
        print(f"ANALYZING DATASET: {dataset_type.upper()}")
        print(f"{'='*60}")
        
        results[dataset_type] = {}
        
        for target_col in target_columns:
            result = analyze_features_for_target(target_col, dataset_type, categorized_features, surface_features, lexical_features, readability_measures, semantic_features, syntactic_features)
            if result:
                results[dataset_type][target_col] = result
                # Collect uncategorized features from this target
                if 'uncategorized_features' in result:
                    all_uncategorized_features.update(result['uncategorized_features'])
                
                # Collect all features from this target
                if 'all_features' in result:
                    all_found_features.update(result['all_features'])
                
                # Initialize dataset tracking if not exists
                if dataset_type not in comprehensive_categorization["datasets"]:
                    comprehensive_categorization["datasets"][dataset_type] = {
                        "features_found": set(),
                        "categorized_features": set(),
                        "uncategorized_features": set()
                    }
                
                # Update dataset tracking
                comprehensive_categorization["datasets"][dataset_type]["features_found"].update(result.get('all_features', set()))
                comprehensive_categorization["datasets"][dataset_type]["categorized_features"].update(
                    result.get('surface_assigned_features', set()) |
                    result.get('lexical_assigned_features', set()) |
                    result.get('readability_assigned_features', set()) |
                    result.get('semantic_assigned_features', set()) |
                    result.get('syntactic_assigned_features', set())
                )
                comprehensive_categorization["datasets"][dataset_type]["uncategorized_features"].update(result.get('uncategorized_features', set()))
    
    # Summary for each dataset type
    for dataset_type in dataset_types:
        if dataset_type in results and results[dataset_type]:
            print(f"\n{'='*60}")
            print(f"SUMMARY FOR {dataset_type.upper()}")
            print(f"{'='*60}")
            
            dataset_results = results[dataset_type]
            total_all = sum(r['total'] for r in dataset_results.values())
            total_surface = sum(r['surface_total'] for r in dataset_results.values())
            total_lexical = sum(r['lexical_total'] for r in dataset_results.values())
            total_readability = sum(r['readability_total'] for r in dataset_results.values())
            total_semantic = sum(r['semantic_total'] for r in dataset_results.values())
            total_syntactic = sum(r['syntactic_total'] for r in dataset_results.values())
            total_categorized_with_overlaps = total_surface + total_lexical + total_readability + total_semantic + total_syntactic
            total_unique_categorized = sum(r['unique_categorized'] for r in dataset_results.values())
            total_overlapping = sum(r['overlapping_count'] for r in dataset_results.values())
            total_uncategorized = sum(r['uncategorized'] for r in dataset_results.values())
            
            print(f"Total features across all targets: {total_all}")
            print(f"Total Surface features: {total_surface}")
            print(f"Total Lexical features: {total_lexical}")
            print(f"Total Readability measures: {total_readability}")
            print(f"Total Semantic features: {total_semantic}")
            print(f"Total Syntactic features: {total_syntactic}")
            print(f"Total categorized (with overlaps): {total_categorized_with_overlaps}")
            print(f"Total overlapping features: {total_overlapping}")
            print(f"Total unique categorized: {total_unique_categorized}")
            print(f"Total uncategorized: {total_uncategorized}")
            if total_all > 0:
                print(f"Coverage (with overlaps): {(total_categorized_with_overlaps/total_all)*100:.1f}%")
                print(f"Correct Coverage: {(total_unique_categorized/total_all)*100:.1f}%")
            
            # Collect overlapping features for this dataset
            dataset_overlapping_features = set()
            for result in dataset_results.values():
                if 'overlapping_features' in result:
                    dataset_overlapping_features.update(result['overlapping_features'].keys())
            
            # Print overlapping features for this dataset
            if dataset_overlapping_features:
                print(f"\n  OVERLAPPING FEATURES FOR {dataset_type.upper()} ({len(dataset_overlapping_features)}):")
                sorted_overlapping = sorted(list(dataset_overlapping_features))
                for i, feature in enumerate(sorted_overlapping, 1):
                    # Show which categories this feature appears in
                    categories = set()
                    for result in dataset_results.values():
                        if 'overlapping_features' in result and feature in result['overlapping_features']:
                            categories.update(result['overlapping_features'][feature])
                    print(f"    {i:3d}. {feature} -> {', '.join(sorted(categories))}")
    
    # Overall Summary
    if results:
        print(f"\n{'='*80}")
        print("OVERALL SUMMARY ACROSS ALL DATASETS")
        print(f"{'='*80}")
        
        # Collect all unique overlapping features across all datasets
        all_overlapping_features = set()
        for dataset_type in dataset_types:
            if dataset_type in results:
                for result in results[dataset_type].values():
                    if 'overlapping_features' in result:
                        all_overlapping_features.update(result['overlapping_features'].keys())
        
        # Print overall overlapping features
        if all_overlapping_features:
            print(f"UNIQUE OVERLAPPING FEATURES ACROSS ALL DATASETS ({len(all_overlapping_features)}):")
            sorted_overlapping = sorted(list(all_overlapping_features))
            for i, feature in enumerate(sorted_overlapping, 1):
                # Show which categories this feature appears in across all datasets
                categories = set()
                for dataset_type in dataset_types:
                    if dataset_type in results:
                        for result in results[dataset_type].values():
                            if 'overlapping_features' in result and feature in result['overlapping_features']:
                                categories.update(result['overlapping_features'][feature])
                print(f"  {i:3d}. {feature} -> {', '.join(sorted(categories))}")
        
        # Print unique uncategorized features
        if all_uncategorized_features:
            print(f"\nUNIQUE UNCATEGORIZED FEATURES ACROSS ALL DATASETS ({len(all_uncategorized_features)}):")
            sorted_uncategorized = sorted(list(all_uncategorized_features))
            for i, feature in enumerate(sorted_uncategorized, 1):
                print(f"  {i:3d}. {feature}")
        else:
            print("\nAll features are categorized across all datasets!")
    
    # Build comprehensive categorization structure
    print(f"\n{'='*80}")
    print("BUILDING COMPREHENSIVE CATEGORIZATION...")
    print(f"{'='*80}")
    
    # Map features to their categories and subcategories across all datasets
    feature_to_category_mapping = {}
    
    # Process all results to build the mapping
    for dataset_type in dataset_types:
        if dataset_type in results:
            for target_col, result in results[dataset_type].items():
                if 'feature_assignments' in result:
                    for feature, assignments in result['feature_assignments'].items():
                        if feature not in feature_to_category_mapping:
                            feature_to_category_mapping[feature] = set()
                        feature_to_category_mapping[feature].update(assignments)
    
    # Organize features by category and subcategory
    # Surface Features
    for subcategory, feature_list in surface_features.items():
        found_features = [f for f in feature_list if f in all_found_features]
        comprehensive_categorization["categories"]["Surface features"][subcategory] = found_features
    
    # Lexical Features  
    for subcategory, feature_list in lexical_features.items():
        if subcategory == 'N-gram':
            # Include both explicit and pattern-matched features
            explicit_features = [f for f in feature_list if f in all_found_features]
            pattern_features = [f for f in all_found_features if f.startswith('top_n') and f not in feature_list]
            found_features = explicit_features + pattern_features
        else:
            found_features = [f for f in feature_list if f in all_found_features]
        comprehensive_categorization["categories"]["Lexical features"][subcategory] = found_features
    
    # Readability Measures
    for subcategory, feature_list in readability_measures.items():
        found_features = [f for f in feature_list if f in all_found_features]
        comprehensive_categorization["categories"]["Readability measures"][subcategory] = found_features
    
    # Semantic Features
    for subcategory, feature_list in semantic_features.items():
        found_features = [f for f in feature_list if f in all_found_features]
        comprehensive_categorization["categories"]["Semantic features"][subcategory] = found_features
    
    # Syntactic Features
    for subcategory, feature_list in syntactic_features.items():
        if subcategory == 'POS bigrams':
            explicit_features = [f for f in feature_list if f in all_found_features]
            pattern_features = [f for f in all_found_features if f.startswith('pos_bigram')]
            found_features = list(set(explicit_features + pattern_features))
        elif subcategory == 'POS tags':
            explicit_features = [f for f in feature_list if f in all_found_features]
            sentiment_pos_features = ['positive_sentence_prop']
            pattern_features = [f for f in all_found_features 
                              if f.startswith('pos') 
                              and not f.startswith('pos_bigram')
                              and f not in sentiment_pos_features]
            found_features = list(set(explicit_features + pattern_features))
        elif subcategory == 'Pronoun':
            explicit_features = [f for f in feature_list if f in all_found_features]
            exclude_patterns = ['group_', 'sent_count_', 'sent_percent_', 'pos_bigram_', 'pos_', 'begin_w_']
            pattern_features = [f for f in all_found_features 
                              if 'pron' in f.lower() 
                              and f not in feature_list
                              and not any(f.startswith(pattern) for pattern in exclude_patterns)]
            found_features = list(set(explicit_features + pattern_features))
        elif subcategory == 'Syntactic Pronoun':
            explicit_features = [f for f in feature_list if f in all_found_features]
            group_features = [f for f in all_found_features if f.startswith('group_')]
            sent_count_features = [f for f in all_found_features if f.startswith('sent_count_')]
            sent_percent_features = [f for f in all_found_features if f.startswith('sent_percent_')]
            found_features = list(set(explicit_features + group_features + sent_count_features + sent_percent_features))
        else:
            found_features = [f for f in feature_list if f in all_found_features]
        comprehensive_categorization["categories"]["Syntactic features"][subcategory] = found_features
    
    # Convert sets to lists for JSON serialization
    for dataset_type in comprehensive_categorization["datasets"]:
        for key in ["features_found", "categorized_features", "uncategorized_features"]:
            if isinstance(comprehensive_categorization["datasets"][dataset_type][key], set):
                comprehensive_categorization["datasets"][dataset_type][key] = sorted(list(comprehensive_categorization["datasets"][dataset_type][key]))
    
    # Add overall statistics
    comprehensive_categorization["statistics"] = {
        "total_unique_features_found": len(all_found_features),
        "total_categorized_features": len(all_found_features - all_uncategorized_features),
        "total_uncategorized_features": len(all_uncategorized_features),
        "uncategorized_features_list": sorted(list(all_uncategorized_features))
    }
    
    # Save to JSON file
    output_dir = "../output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    json_filepath = os.path.join(output_dir, "feature_categorization.json")
    
    try:
        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_categorization, f, indent=2, ensure_ascii=False)
        print(f"\nComprehensive categorization saved to: {json_filepath}")
        
        # Print summary of what was saved
        total_categorized_in_json = 0
        for category, subcategories in comprehensive_categorization["categories"].items():
            for subcategory, features in subcategories.items():
                total_categorized_in_json += len(features)
        
        print(f"JSON file contains:")
        print(f"  - {len(comprehensive_categorization['categories'])} main categories")
        print(f"  - {sum(len(subcat) for subcat in comprehensive_categorization['categories'].values())} subcategories")
        print(f"  - {total_categorized_in_json} total feature assignments")
        print(f"  - Analysis for {len(comprehensive_categorization['datasets'])} datasets")
        print(f"  - {len(all_uncategorized_features)} uncategorized features")
        
    except Exception as e:
        print(f"Error saving JSON file: {e}")
        
    print(f"\nCategorization analysis complete!")

if __name__ == "__main__":
    main()