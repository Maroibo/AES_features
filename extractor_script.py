from camel_tools_init import (get_disambiguator, get_analyzer, get_sentiment_analyzer, 
                              get_bert_model, get_tagger, _mle_disambiguator, 
                              _morph_analyzer, _sentiment_analyzer, _bert_tokenizer, 
                              _bert_model, _default_tagger)
from essay_proccessing import split_into_sentences, split_into_paragraphs

# contansts
INPUT_FILE_PATH='../../../../shared/Arabic_Dataset/cleaned_cqc.csv'
OUTPUT_FILE_PATH='./output_features/full_arabic_feature_set.csv'

def process_essay(essay, prompt=None):
    """Process an Arabic essay to compute all available features."""
    return

def main():
    # Initialize models
    get_disambiguator()
    get_analyzer()
    get_sentiment_analyzer()
    get_bert_model()
    get_tagger()
    # Read the input file
    
    
