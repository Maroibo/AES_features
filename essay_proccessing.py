import re
from nltk.tokenize import word_tokenize
from camel_tools.utils.dediac import dediac_ar
from difflib import SequenceMatcher
from camel_tools.utils.normalize import normalize_unicode

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

def get_lemmas_and_roots(word_list,_morph_analyzer):
    """Extract lemmas and roots from keywords for robust matching"""
    lemmas = set()
    roots = set()
    
    for word in word_list:
        # For multi-word phrases, analyze each word
        if ' ' in word:
            words = split_into_words(word)
            for word in words:
                analyses = _morph_analyzer.analyze(word)
                if analyses:
                    # Get lemma and add variations
                    lemma = analyses[0].get('lex', '')
                    if lemma:
                        lemmas.add(dediac_ar(lemma))
                        lemmas.add(lemma)  # Keep original too
                    
                    # Get root if available
                    root = analyses[0].get('root', '')
                    if root:
                        roots.add(dediac_ar(root))
        else:
            # Single word analysis
            analyses = _morph_analyzer.analyze(word)
            if analyses:
                lemma = analyses[0].get('lex', '')
                if lemma:
                    lemmas.add(dediac_ar(lemma))
                    lemmas.add(lemma)
                
                root = analyses[0].get('root', '')
                if root:
                    roots.add(dediac_ar(root))
            
            # Also add dediacritized version of original
            lemmas.add(dediac_ar(word))
    
    return lemmas, roots

def fuzzy_match(text, target_phrase, threshold=0.90):
    """Check if target_phrase appears in text with fuzzy matching"""
    # Normalize both strings
    text = normalize_unicode(text.strip())
    target_phrase = normalize_unicode(target_phrase.strip())
    
    # Check if target phrase length is reasonable for fuzzy matching
    if len(target_phrase) < 3:
        return target_phrase in text
        
    # Split text into overlapping windows of similar length to target phrase
    words = text.split()
    target_words = target_phrase.split()
    target_len = len(target_words)
    
    if target_len == 1:
        # For single word phrases, check each word
        for word in words:
            similarity = SequenceMatcher(None, word, target_phrase).ratio()
            if similarity >= threshold:
                return True
    else:
        # For multi-word phrases, check sliding windows
        for i in range(len(words) - target_len + 1):
            window = ' '.join(words[i:i + target_len])
            similarity = SequenceMatcher(None, window, target_phrase).ratio()
            if similarity >= threshold:
                return True
                
        # Also check the entire text for longer phrases
        similarity = SequenceMatcher(None, text, target_phrase).ratio()
        if similarity >= threshold:
            return True
            
    return False