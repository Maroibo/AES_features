from camel_tools_init import (get_disambiguator, get_analyzer, get_sentiment_analyzer, 
                              get_bert_model, get_tagger, _mle_disambiguator, 
                              _morph_analyzer, _sentiment_analyzer, _bert_tokenizer, 
                              _bert_model, _default_tagger,get_dialect_id,_dialect_id)
from essay_proccessing import split_into_sentences, split_into_paragraphs
from pos_features import get_top_n_pos_tags,get_top_n_pos_bigrams,get_essay_pos_features
from readability_measures import calculate_readability_scores
from semantic_features import calculate_prompt_adherence_features,calculate_sentiment_scores,calculate_semantic_similarities,calculate_sent_match_words
from surface_level_features import calculate_religious_phrases,calculate_advanced_punctuation_features,extract_surface_features,calculate_lemma_features,calculate_variance_features,long_words_count,calculate_punctuation_counts,calculate_dup_punctuation_count
from syntactic_features import (count_jazm_particles,analyze_dialect_usage,
    calculate_syllable_features,calculate_pronoun_features,
    calculate_possessive_features,calculate_grammar_features,
    calculate_nominal_verbal_sentences,count_conjunctions_and_transitions,
    extract_syntactic_features,extract_lexical_features,
    get_top_n_words_from_essays,calculate_top_n_word_features)
# from clause_features import ClauseAnalyzer
import pandas as pd
from camel_tools.utils.normalize import normalize_unicode
import numpy as np
import os
from tqdm import tqdm
# contansts
INPUT_FILE_PATH='../../../../shared/Arabic_Dataset/cleaned_cqc.csv'
INPUT_PARAGRAPHS_FILE_PATH='../../../../shared/Arabic_Dataset/cq_essay_paragraphs.csv'
OUTPUT_FILE_PATH='./output_features/full_arabic_feature_set_prompts[1,2,3,4].csv'
OUTPUT_DIR='./output_features'
PROMPTS_FILE_PATH='../../../../shared/Arabic_Dataset/arabic_prompts'

def load_prompts():
    prompts={}
    for essay_set in [1,2,3,4]:
        prompt_file=os.path.join(PROMPTS_FILE_PATH,f'prompt{essay_set}_text.txt')
        with open(prompt_file, 'r', encoding='utf-8') as file:
            prompts[essay_set]=file.read()
    return prompts

def main():
    # Initialize models
    _mle_disambiguator = get_disambiguator()
    _morph_analyzer = get_analyzer()
    _sentiment_analyzer = get_sentiment_analyzer()
    _bert_tokenizer, _bert_model = get_bert_model()
    _default_tagger = get_tagger()
    _dialect_id=get_dialect_id()
    df=pd.read_csv(INPUT_FILE_PATH)
    df=df[df.essay_set.isin([1,2,3,4])];
    paragraphs_df=pd.read_csv(INPUT_PARAGRAPHS_FILE_PATH)
    top_n_pos_tags=get_top_n_pos_tags(df['essay'],_mle_disambiguator,100)
    top_n_pos_bigrams=get_top_n_pos_bigrams(df['essay'],_mle_disambiguator,100)
    total_word_counts=get_top_n_words_from_essays(df['essay'],100)
    # clause_analyzer=ClauseAnalyzer()
    prompts=load_prompts()
    # Read the input file
    features_df=pd.DataFrame()
    
    # Add progress bar to the loop
    print(f"Processing {len(df)} essays...")
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features", unit="essay"):
        essay=row['essay']
        id=row['essay_id']
        prompt=prompts[row['essay_set']]
        intro_paragraph=paragraphs_df[paragraphs_df['essay_id']==id]['introduction'].values[0]
        body_paragraph=paragraphs_df[paragraphs_df['essay_id']==id]['body'].values[0]
        conclusion_paragraph=paragraphs_df[paragraphs_df['essay_id']==id]['conclusion'].values[0]
        if intro_paragraph is np.nan:
            intro_paragraph=''
        if body_paragraph is np.nan:
            body_paragraph=''
        if conclusion_paragraph is np.nan:
            conclusion_paragraph=''
        longest_paragaph_length=max(len(intro_paragraph),len(body_paragraph),len(conclusion_paragraph))
        shortest_paragaph_length=min(len(intro_paragraph),len(body_paragraph),len(conclusion_paragraph))
        # Surface level features
        surface_features=extract_surface_features(essay,intro_paragraph,body_paragraph,conclusion_paragraph)
        religious_features=calculate_religious_phrases(intro_paragraph,body_paragraph,conclusion_paragraph)
        punctuation_features=calculate_advanced_punctuation_features(essay,_mle_disambiguator)
        lemma_features=calculate_lemma_features(essay,_morph_analyzer)
        variance_features=calculate_variance_features(essay)
        long_words_features={"long_words_count": long_words_count(essay)}
        punctuation_counts_features=calculate_punctuation_counts(essay,_mle_disambiguator)
        dup_punctuation_features={"dup_punctuation_count": calculate_dup_punctuation_count(essay,_mle_disambiguator)}
        advanced_punctuation_features=calculate_advanced_punctuation_features(essay,_mle_disambiguator)

        # POS features
        pos_features=get_essay_pos_features(essay,top_n_pos_tags,top_n_pos_bigrams,_mle_disambiguator,100)
        # Readability features
        readability_features=calculate_readability_scores(essay,_mle_disambiguator)
        # Semantic features
        semantic_features=calculate_semantic_similarities(intro_paragraph,body_paragraph,conclusion_paragraph,_bert_tokenizer, _bert_model)
        sentiment_features=calculate_sentiment_scores(essay)
        sent_match_words_features=calculate_sent_match_words(essay)
        prompt_adherence_features=calculate_prompt_adherence_features(essay,prompt,_bert_tokenizer, _bert_model)
        # syntax features
        dialect_features=analyze_dialect_usage(essay,_dialect_id)
        syllable_features=calculate_syllable_features(essay,_mle_disambiguator)
        pronoun_features=calculate_pronoun_features(essay,_mle_disambiguator)
        possessive_features=calculate_possessive_features(essay,_morph_analyzer)
        # clause_features=clause_analyzer.calculate_features(essay)
        grammar_features=calculate_grammar_features(essay,_mle_disambiguator)
        nominal_verbal_sentences=calculate_nominal_verbal_sentences(essay,_mle_disambiguator)
        conjunctions_and_transitions=count_conjunctions_and_transitions(essay)
        syntactic_features=extract_syntactic_features(essay,_mle_disambiguator)
        lexical_features=extract_lexical_features(essay,intro_paragraph,body_paragraph,conclusion_paragraph,_morph_analyzer)
        top_n_word_features=calculate_top_n_word_features(essay,total_word_counts,100)
        jazm_features=count_jazm_particles(essay,_morph_analyzer)

        features_df=pd.concat([features_df,pd.DataFrame({
            'essay_id':id,
            'longest_paragaph_length':longest_paragaph_length,
            'shortest_paragaph_length':shortest_paragaph_length,
            'longest_paragaph_length_ratio':longest_paragaph_length/shortest_paragaph_length if shortest_paragaph_length!=0 else 0,
            'shortest_paragaph_length_ratio':shortest_paragaph_length/longest_paragaph_length if longest_paragaph_length!=0 else 0,
            'no_of _words_in_first':len(normalize_unicode(intro_paragraph).split()),
            'no_of _words_in_body':len(normalize_unicode(body_paragraph).split()),
            'no_of _words_in_conclusion':len(normalize_unicode(conclusion_paragraph).split()),
            'more_than_1_paragraph': 1 if essay.count('\n')>1 else 0,
            'colon_exists': 1 if ':' in essay else 0,
            'paranthesis_exists': 1 if '(' in essay else 0,
            'question_mark_exists': 1 if '?' in essay else 0,
            **pos_features,
            **readability_features,
            **semantic_features,
            **sentiment_features,
            **sent_match_words_features,
            **prompt_adherence_features,
            **dialect_features,
            **surface_features,
            **religious_features,
            **punctuation_features,
            **lemma_features,
            **variance_features,
            **long_words_features,
            **punctuation_counts_features,
            **dup_punctuation_features,
            **advanced_punctuation_features,
            **syllable_features,
            **pronoun_features,
            **possessive_features,
            # **clause_features,
            **grammar_features,
            **nominal_verbal_sentences,
            **conjunctions_and_transitions,
            **syntactic_features,
            **lexical_features,
            **top_n_word_features,
            **jazm_features,
        },index=[0])], ignore_index=True)
    
    print("\nFeature extraction completed!")
    # print stats of features
    print(features_df.describe())
    # create the dir if not exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    # save features to csv
    features_df.to_csv(OUTPUT_FILE_PATH,index=False)
    print(f"Features saved to: {OUTPUT_FILE_PATH}")
    
main()

    
