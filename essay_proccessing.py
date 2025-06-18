import re
from nltk.tokenize import word_tokenize
def split_into_sentences(essay):
    _SENTENCE_SPLIT_REGEX = re.compile(r'[.!?;؟،:]\s+')
    return _SENTENCE_SPLIT_REGEX.split(essay)

def split_into_paragraphs(essay):
    # this method is just a naive implementation
    # this should be reimplemented
    if not essay or len(essay.strip()) == 0:
        return ["", "", ""]
    essay = essay.strip()
    total_length = len(essay)
    target_length = total_length // 3
    paragraphs = []
    start = 0
    for i in range(2):  # First two paragraphs
        end = start + target_length  
        # Try to find a word boundary near the target position
        # Look for space within a reasonable range
        search_range = min(50, target_length // 4)  # Search within 25% of target length 
        best_end = end
        for j in range(max(0, end - search_range), min(total_length, end + search_range)):
            if essay[j] == ' ':
                # Prefer positions closer to the target
                if abs(j - end) < abs(best_end - end):
                    best_end = j    
        paragraphs.append(essay[start:best_end].strip())
        start = best_end + 1 if best_end < total_length else best_end 
    # Add the remaining text as the third paragraph
    paragraphs.append(essay[start:].strip()) 
    return paragraphs

def split_into_words(essay):
    return word_tokenize(re.sub(r'[^\w\s]', '', essay))


def count_chars(essay):
    char_count = len(essay.replace(" ", ""))
    return char_count


def remove_diatrics_normalizer(essay):
    """
    Remove all Arabic diacritics from the text.
    
    Args:
        essay (str): Arabic text with diacritics
        
    Returns:
        str: Text with all diacritics removed
    """
    # Define all Arabic diacritics with their Unicode codes
    diacritics = {
        # Basic short vowels
        '\u064E',  # Fatha (َ)
        '\u064F',  # Damma (ُ)
        '\u0650',  # Kasra (ِ)
        
        # Tanwin (nunation) - double vowels
        '\u064B',  # Fathatan (ً)
        '\u064C',  # Dammatan (ٌ)
        '\u064D',  # Kasratan (ٍ)
        
        # Other diacritical marks
        '\u0652',  # Sukoon (ْ)
        '\u0651',  # Shadda (ّ)
        
        # Additional diacritics
        '\u0653',  # Madda (ٓ)
        '\u0654',  # Hamza above (ٔ)
        '\u0655',  # Hamza below (ٕ)
        '\u0670',  # Superscript alef (ٰ)
    }
    
    # Remove all diacritics from the text
    cleaned_text = essay
    for diacritic in diacritics:
        cleaned_text = cleaned_text.replace(diacritic, '')
    
    return cleaned_text