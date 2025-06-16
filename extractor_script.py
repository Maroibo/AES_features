from camel_tools_init import (get_disambiguator, get_analyzer, get_sentiment_analyzer, 
                              get_bert_model, get_tagger, _mle_disambiguator, 
                              _morph_analyzer, _sentiment_analyzer, _bert_tokenizer, 
                              _bert_model, _default_tagger)
from essay_proccessing import split_into_sentences, split_into_paragraphs
import pandas as pd
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
    paragraphs_df=pd.read_csv(INPUT_PARAGRAPHS_FILE_PATH)
    # Read the input file
    for index, row in df.iterrows():
        essay=row['essay']
        id=row['essay_id']
        intro_paragraph=paragraphs_df[paragraphs_df['essay_id']==id]['introduction'].values[0]
        body_paragraph=paragraphs_df[paragraphs_df['essay_id']==id]['body'].values[0]
        conclusion_paragraph=paragraphs_df[paragraphs_df['essay_id']==id]['conclusion'].values[0]

    
