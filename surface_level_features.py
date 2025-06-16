from essay_proccessing import split_into_words, split_into_sentences
from camel_tools_init import _morph_analyzer
import numpy as np
from nltk.corpus import stopwords

ARABIC_STOPWORDS = set(stopwords.words('arabic'))



def calculate_lemma_features(essay):
    """
    Calculate lemma-related features for Arabic text using CAMeL Tools morphological analyzer.
    Returns only total unique lemmas and average lemma length.
    """
    # Initialize feature counts
    features = {
        "total_lemmas": 0,        # Total number of unique lemmas
        "avg_lemma_length": 0,    # Average length of lemmas
    }
    
    # Set to store unique lemmas
    lemma_types = set()
    
    # Split text into words
    words = split_into_words(essay)
    
    # Process each word
    for word in words:
        # Get morphological analysis
        analyses = _morph_analyzer.analyze(word)
        
        if analyses:
            # Get the first analysis (most likely)
            analysis = analyses[0]
            
            # Extract lemma if available
            if 'lex' in analysis:
                lemma = analysis['lex']
                lemma_types.add(lemma)
    
    # Calculate final metrics
    features["total_lemmas"] = len(lemma_types)
    
    # Calculate average lemma length
    if features["total_lemmas"] > 0:
        total_length = sum(len(lemma) for lemma in lemma_types)
        features["avg_lemma_length"] = total_length / features["total_lemmas"]
    
def long_words_count(essay):
    """
    Counts the number of words with 7 or more characters in the essay.
    """
    words = split_into_words(essay)
    long_words_count = sum(1 for word in words if len(word) >= 7)
    return long_words_count


def calculate_comma_period_count(essay):
    """
    Counts the number of commas and periods in the essay.
    """
    comma_count = essay.count('،')
    period_count = essay.count('.')
    return {
        "comma_count": comma_count,
        "period_count": period_count
    }

def calculate_variance_features(essay):
    """
    Calculates variance-related features for Arabic text including:
    - sent_var: variance of sentence lengths
    - word_var: variance of word lengths
    - stop_prop: proportion of stopwords
    - unique_word: total number of unique words
    - type_token_ratio: ratio of unique words to total words
    """
    # Process words
    words = split_into_words(essay)
    words_count = len(words)
    
    # Calculate word length variance
    word_lengths = [len(word) for word in words]
    word_var = np.var(word_lengths)
    
    # Calculate sentence length variance
    sentences = split_into_sentences(essay)
    sentence_lengths = [len(sentence) for sentence in sentences]
    sent_var = np.var(sentence_lengths)
    
    # Calculate stopword proportion using module-level Arabic stopwords
    stop_words_count = sum(1 for word in words if word in ARABIC_STOPWORDS)
    stop_prop = stop_words_count / words_count if words_count > 0 else 0
    
    # Calculate unique words and type-token ratio
    unique_words = set(words)
    unique_word_count = len(unique_words)
    type_token_ratio = unique_word_count / words_count if words_count > 0 else 0
    
    return {
        "sent_var": sent_var,
        "word_var": word_var,
        "stop_prop": stop_prop,
        "type_token_ratio": type_token_ratio
    }




        
