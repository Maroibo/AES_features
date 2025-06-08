from essay_proccessing import split_into_words, split_into_sentences
from collections import Counter
import pandas as pd
from camel_tools_init import (get_disambiguator, get_analyzer, get_sentiment_analyzer, 
                              get_bert_model, get_tagger, _mle_disambiguator, 
                              _morph_analyzer, _sentiment_analyzer, _bert_tokenizer, 
                              _bert_model, _default_tagger)
def get_top_n_pos_tags(essays, n=100):
    """
    Get the top n POS tags from a list of essays using CAMeL Tools.
    """
    pos_tags = []
    for essay in essays:
        sentences = split_into_sentences(essay)
        for sentence in sentences:
            words = split_into_words(sentence)          
            disambiguated = _mle_disambiguator.disambiguate(words)
            for disambiguated_word in disambiguated:
                if disambiguated_word and len(disambiguated_word) > 0 and disambiguated_word.analyses:
                    analysis =disambiguated_word.analyses[0].analysis
                    if 'pos' in analysis:
                        pos_tags.append(analysis['pos'])
    # Count POS tag frequencies and return top n
    pos_counter = Counter(pos_tags)
    return [tag for tag, count in pos_counter.most_common(n)]
