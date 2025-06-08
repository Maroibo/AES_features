import re
from nltk.tokenize import word_tokenize

def calculate_madad_osman(essay):
    """
    Calculates MADAD and OSMAN readability measures for Arabic text.
    """
    # Prepare the text
    words = word_tokenize(re.sub(r'[^\w\s]', '', essay))
    words_count = len(words)
    
    # For sentence analysis (using Arabic punctuation)
    sentences = list(filter(str.strip, re.split(r'[.،!؛:؟]', essay)))
    sentences_count = len(sentences)
    
    # Calculate characters per word
    total_word_length = sum(len(word) for word in words)
    characters_per_word = total_word_length / words_count if words_count > 0 else 0
    
    # Calculate words per sentence
    sentence_lengths = [len(word_tokenize(sentence)) for sentence in sentences]
    words_per_sentence = sum(sentence_lengths) / sentences_count if sentences_count > 0 else 0
    
    # Calculate MADAD
    madad = 4.414 * characters_per_word + 1.498 * words_per_sentence + 3.436
    
    # Calculate OSMAN
    osman = 200 - (characters_per_word * 10) - (words_per_sentence * 2)
    
    return madad, osman