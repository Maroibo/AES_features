from camel_tools_init import (get_disambiguator, get_analyzer, get_sentiment_analyzer, 
                              get_bert_model, get_tagger, _mle_disambiguator, 
                              _morph_analyzer, _sentiment_analyzer, _bert_tokenizer, 
                              _bert_model, _default_tagger)
from essay_proccessing import split_into_sentences, split_into_paragraphs
from syntactic_features import count_jazm_particles
import pandas as pd
from camel_tools.utils.normalize import normalize_unicode
# contansts
INPUT_FILE_PATH='../../../../shared/Arabic_Dataset/cleaned_cqc.csv'
INPUT_PARAGRAPHS_FILE_PATH='../../../../shared/Arabic_Dataset/cq_essay_paragraphs.csv'
OUTPUT_FILE_PATH='./output_features/full_arabic_feature_set.csv'

def process_essay(essay, prompt=None):
    """Process an Arabic essay to compute all available features."""
    return

def main():
    # Initialize models
    _mle_disambiguator = get_disambiguator()
    _morph_analyzer = get_analyzer()
    _sentiment_analyzer = get_sentiment_analyzer()
    _bert_tokenizer, _bert_model = get_bert_model()
    _default_tagger = get_tagger()
    df=pd.read_csv(INPUT_FILE_PATH)
    df=df[df.essay_set.isin([1,2,3,4])];
    paragraphs_df=pd.read_csv(INPUT_PARAGRAPHS_FILE_PATH)
    # Read the input file
    features_df=pd.DataFrame()
    for index, row in df.iterrows():
        essay=row['essay']
        id=row['essay_id']
        intro_paragraph=paragraphs_df[paragraphs_df['essay_id']==id]['introduction'].values[0]
        body_paragraph=paragraphs_df[paragraphs_df['essay_id']==id]['body'].values[0]
        conclusion_paragraph=paragraphs_df[paragraphs_df['essay_id']==id]['conclusion'].values[0]
        longest_paragaph_length=max(len(intro_paragraph),len(body_paragraph),len(conclusion_paragraph))
        shortest_paragaph_length=min(len(intro_paragraph),len(body_paragraph),len(conclusion_paragraph))
        features_df=pd.concat([features_df,pd.DataFrame({
            'essay_id':id,
            'longest_paragaph_length':longest_paragaph_length,
            'shortest_paragaph_length':shortest_paragaph_length,
            'longest_paragaph_length_ratio':longest_paragaph_length/shortest_paragaph_length,
            'shortest_paragaph_length_ratio':shortest_paragaph_length/longest_paragaph_length,
            'no_of _words_in_first':len(normalize_unicode(intro_paragraph).split()),
            'no_of _words_in_body':len(normalize_unicode(body_paragraph).split()),
            'no_of _words_in_conclusion':len(normalize_unicode(conclusion_paragraph).split()),
            'more_than_1_paragraph': 1 if essay.count('\n')>1 else 0,
            'colon_exists': 1 if ':' in essay else 0,
            'paranthesis_exists': 1 if '(' in essay else 0,
            'question_mark_exists': 1 if '?' in essay else 0,
        },ignore_index=True)])
    
main()

    
