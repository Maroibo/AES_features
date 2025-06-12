import re
from camel_tools.utils.normalize import normalize_unicode
from essay_proccessing import split_into_words, count_chars, split_into_sentences
import math
import re
import pandas as pd
from semantic_features import calculate_syllable_features


def calculate_readability_scores(essay):
    """
    Calculate readability scores for Arabic text including:
    - FleschReadingEase: adapted for Arabic using custom syllable count
    - SMOGIndex: adapted for Arabic
    - ARI: adapted for Arabic
    - LinsearWrite: adapted for Arabic
    - Kincaid: adapted for Arabic
    - Coleman-Liau: adapted for Arabic
    - LIX: adapted for Arabic
    - RIX: adapted for Arabic
    - GunningFogIndex: adapted for Arabic
    - DaleChallIndex: adapted for Arabic using common word list
    - MADAD: Arabic Specific Readability Measure
    - OSMAN: Arabic Specific Readability Measure
    """
    syllable_features = calculate_syllable_features(essay)
    syllable_count = syllable_features['syllables']
    complex_words_count = syllable_features['complex_words_dc']
    # Count sentences
    sentences = list(filter(str.strip, re.split(r'[.،!؛:؟]', essay)))
    sentence_count = len(sentences)
    words = split_into_words(essay)
    word_count = len(words)
    char_count = count_chars(essay)
    sentences=split_into_sentences(essay)
    sentence_count = len(sentences)

    long_words_count = 0
    for word in words:
        word_normalized = normalize_unicode(word.strip())
        if len(word_normalized) >= 6:
            long_words_count += 1
    
    # Calculate percentages and ratios
    words_per_sentence = word_count / sentence_count
    chars_per_word = char_count / word_count
    percent_complex_words = (complex_words_count / word_count) * 100
    percent_long_words = (long_words_count / word_count) * 100
    percent_difficult_words = (complex_words_count / word_count) * 100
    characters_per_word = char_count / word_count

    # Calculate Flesch Reading Ease (adapted for Arabic)
    # Original formula: 206.835 - 1.015 * (words/sentences) - 84.6 * (syllables/words)
    # Adapted weights for Arabic based on research
    flesch_score = 206.835 - (1.015 * (word_count / sentence_count)) - (84.6 * (syllable_count / word_count))
    # Calculate SMOG Index (adapted for Arabic)
    # Original formula: 1.0430 * sqrt(complex_words * (30 / sentence_count)) + 3.1291
    # We'll use the original formula but with our Arabic syllable counting method
    smog_score = 1.0430 * math.sqrt(complex_words_count * (30 / sentence_count)) + 3.1291
    # Calculate Automated Readability Index (ARI)
    # Original: 4.71 * (chars_per_word) + 0.5 * (words_per_sentence) - 21.43
    # Adapted slightly for Arabic
    ari_score = 4.71 * chars_per_word + 0.5 * words_per_sentence - 21.43
    
    # Calculate Linsear Write Formula
    # Original: (easy_words*1 + difficult_words*3)/sentence_count
    # where easy_words have 1-2 syllables, difficult have 3+ syllables
    easy_words_count = word_count - complex_words_count
    linsear_score = (easy_words_count * 1 + complex_words_count * 3) / sentence_count
    linsear_score = (linsear_score - 2) / 2 if linsear_score > 10 else linsear_score / 2
    
    # Calculate Flesch-Kincaid Grade Level (Kincaid)
    # Original: 0.39 * (words_per_sentence) + 11.8 * (syllables_per_word) - 15.59
    # Adapted for Arabic
    syllables_per_word = syllable_count / word_count
    kincaid_score = 0.39 * words_per_sentence + 11.8 * syllables_per_word - 15.59
    
    # Calculate Coleman-Liau Index
    # Original: 0.0588 * L - 0.296 * S - 15.8
    # where L = letters per 100 words, S = sentences per 100 words
    letters_per_100_words = chars_per_word * 100
    sentences_per_100_words = (sentence_count / word_count) * 100
    coleman_liau_score = 0.0588 * letters_per_100_words - 0.296 * sentences_per_100_words - 15.8
    
    # Calculate LIX (Läsbarhetsindex)
    # Original: words_per_sentence + (long_words_count / word_count) * 100
    lix_score = words_per_sentence + percent_long_words
    
    # Calculate RIX (Reading Index)
    # Original: long_words_count / sentence_count
    rix_score = long_words_count / sentence_count if sentence_count > 0 else 0
    
    # Calculate Gunning Fog Index
    # Original: 0.4 * ((words/sentences) + 100 * (complex_words/words))
    gunning_fog_score = 0.4 * (words_per_sentence + percent_complex_words)
    
    # Dale-Chall calculation
    raw_dale_chall = 0.1579 * (percent_difficult_words / 100) + 0.0496 * words_per_sentence
    dale_chall_score = raw_dale_chall + 3.6365 if percent_difficult_words > 5 else raw_dale_chall


    # Calculate MADAD
    madad = 4.414 * characters_per_word + 1.498 * words_per_sentence + 3.436
    
    # Calculate OSMAN
    osman = 200 - (characters_per_word * 10) - (words_per_sentence * 2)

    return {
        "FleschReadingEase": flesch_score,
        "SMOGIndex": smog_score,
        "ARI": ari_score,
        "LinsearWrite": linsear_score,
        "Kincaid": kincaid_score,
        "Coleman-Liau": coleman_liau_score,
        "LIX": lix_score,
        "RIX": rix_score,
        "GunningFogIndex": gunning_fog_score,
        "DaleChallIndex": dale_chall_score,
        "MADAD": madad,
        "OSMAN": osman
    }




# df=pd.read_csv('../../../../shared/Arabic_Dataset/cleaned_cqc.csv')
# print(calculate_readability_scores(df['essay'][0]))