from essay_proccessing import split_into_words
def long_words_count(essay):
    """
    Counts the number of words with 7 or more characters in the essay.
    """
    words = split_into_words(essay)
    long_words_count = sum(1 for word in words if len(word) >= 7)
    return long_words_count

    
