from essay_proccessing import split_into_words, split_into_sentences
from collections import Counter
from camel_tools_init import _mle_disambiguator
import numpy as np

def get_all_pos_tags(essays, n=100):
    """
    Get the top n POS tags from a list of essays using CAMeL Tools.
    """
    pos_tags = []
    pos_bigrams = []
    for essay in essays:
        sentences = split_into_sentences(essay)
        for sentence in sentences:
            words = split_into_words(sentence)          
            disambiguated = _mle_disambiguator.disambiguate(words)
            sentence_pos_tags = []

            for disambiguated_word in disambiguated:
                if disambiguated_word and len(disambiguated_word) > 0 and disambiguated_word.analyses:
                    analysis =disambiguated_word.analyses[0].analysis
                    if 'pos' in analysis:
                        pos_tags.append(analysis['pos'])
                        sentence_pos_tags.append(analysis['pos'])

            for i in range(len(sentence_pos_tags) - 1):
                bigram = (sentence_pos_tags[i], sentence_pos_tags[i + 1])
                pos_bigrams.append(bigram)

    bigram_counter = Counter(pos_bigrams)
    bigrams_tags = [bigram for bigram, count in bigram_counter.most_common(n)]

    return np.unique(pos_tags), np.unique(bigrams_tags)

def get_top_n_pos_tags(essays,_mle_disambiguator,n=100):
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


def get_top_n_pos_bigrams(essays,_mle_disambiguator,n=100):
    """
    Get the top n POS tag bigrams from a list of essays using CAMeL Tools.
    """
    pos_bigrams = []
    for essay in essays:
        sentences = split_into_sentences(essay)
        for sentence in sentences:
            words = split_into_words(sentence)          
            disambiguated = _mle_disambiguator.disambiguate(words)
            sentence_pos_tags = []
            
            # Extract POS tags for this sentence
            for disambiguated_word in disambiguated:
                if disambiguated_word and len(disambiguated_word) > 0 and disambiguated_word.analyses:
                    analysis = disambiguated_word.analyses[0].analysis
                    if 'pos' in analysis:
                        sentence_pos_tags.append(analysis['pos'])
            
            # Create bigrams from consecutive POS tags in this sentence
            for i in range(len(sentence_pos_tags) - 1):
                bigram = (sentence_pos_tags[i], sentence_pos_tags[i + 1])
                pos_bigrams.append(bigram)
    
    # Count bigram frequencies and return top n
    bigram_counter = Counter(pos_bigrams)
    return [bigram for bigram, count in bigram_counter.most_common(n)]

def get_essay_pos_features(essay, pos_tags_list, pos_bigrams_list,_mle_disambiguator,n=100):
    """
    Get the POS features for an essay using a list of POS tags and POS bigrams.
    Returns counts of tags/bigrams that exist in the provided lists.
    """
    features = {}
    
    # Initialize all possible features to 0
    features = {f'pos_{tag}': 0 for tag in pos_tags_list}
    features.update({f'pos_bigram_{b0}_{b1}': 0 for (b0, b1) in pos_bigrams_list})

    all_pos_tags = []
    all_pos_bigrams = []
    
    sentences = split_into_sentences(essay)
    for sentence in sentences:
        words = split_into_words(sentence)
        disambiguated = _mle_disambiguator.disambiguate(words)
        sentence_pos_tags = []
        
        # Extract POS tags for this sentence
        for disambiguated_word in disambiguated:
            if disambiguated_word and len(disambiguated_word) > 0 and disambiguated_word.analyses:
                analysis = disambiguated_word.analyses[0].analysis
                if 'pos' in analysis:
                    sentence_pos_tags.append(analysis['pos'])
        
        # Add sentence POS tags to overall list
        all_pos_tags.extend(sentence_pos_tags)
        
        # Create bigrams from this sentence
        all_pos_bigrams.extend(zip(sentence_pos_tags, sentence_pos_tags[1:]))
    
    # Convert lists to sets for efficiency 
    pos_tags_list = set(pos_tags_list)
    pos_bigrams_list = set(pos_bigrams_list)

    tag_counts = Counter(tag for tag in all_pos_tags if tag in pos_tags_list)
    bigram_counts = Counter(b for b in all_pos_bigrams if b in pos_bigrams_list)

    # Count occurrences of mapped POS tags
    for tag, count in tag_counts.items():
        features[f'pos_{tag}'] = count

    # Count occurrences of mapped POS bigrams
    for (b0, b1), count in bigram_counts.items():
        features[f'pos_bigram_{b0}_{b1}'] = count

    
    return features

