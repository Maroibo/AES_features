# Standard library imports
import os
import re
import math
from collections import Counter, defaultdict

# Third-party imports
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Initialize Arabic stopwords
ARABIC_STOPWORDS = set(stopwords.words('arabic'))

# Import camel-tools
try:
    from camel_tools.disambig.mle import MLEDisambiguator
    from camel_tools.utils.normalize import normalize_unicode
except ImportError as e:
    print(f"Error importing camel-tools: {e}")
    print("Please ensure camel-tools is installed correctly with: pip install --upgrade camel-tools")
    raise

# Cache for MLE disambiguator
_mle_disambiguator = None

def get_disambiguator():
    """Get or initialize the MLE disambiguator."""
    global _mle_disambiguator
    if _mle_disambiguator is None:
        _mle_disambiguator = MLEDisambiguator.pretrained()
    return _mle_disambiguator

def syllabify_arabic_word(word):
    """
    Syllabify an Arabic word based on the rules from Zeki et al.
    
    Rules:
    - Every syllable begins with a consonant followed by a vowel
    - Syllable types: CV, CVV, CVC, CVVC, CVCC
    """
    # Define Arabic character sets
    consonants = "ءأؤإئابتثجحخدذرزسشصضطظعغفقكلمنهوي"
    long_vowels = "اوي"  # alif, waw, yaa
    short_vowels = "َُِ"  # fatha, damma, kasra
    sukoon = "ْ"  # sukoon marks absence of vowel
    shadda = "ّ"  # gemination mark (doubles consonant)
    
    # Normalize word: handle shadda by duplicating the consonant
    normalized = ""
    i = 0
    while i < len(word):
        normalized += word[i]
        if i < len(word) - 1 and word[i+1] == shadda:
            normalized += word[i]  # Duplicate the consonant
            i += 2
        else:
            i += 1
    
    word = normalized
    
    # Syllabify
    syllables = []
    i = 0
    current_syllable = ""
    
    while i < len(word):
        # Skip non-Arabic characters
        if not (word[i] in consonants or word[i] in short_vowels or 
                word[i] in long_vowels or word[i] in sukoon):
            i += 1
            continue
            
        # 1. Each syllable must start with a consonant
        if word[i] in consonants:
            current_syllable = word[i]
            i += 1
            
            # 2. Followed by a vowel (short or long)
            if i < len(word):
                # Case: short vowel (diacritic)
                if word[i] in short_vowels:
                    current_syllable += word[i]
                    i += 1
                    
                    # Check for CV pattern (end of syllable)
                    if i >= len(word) or word[i] in consonants:
                        if i < len(word) and word[i] in consonants:
                            # Check for CVC pattern
                            if i + 1 < len(word) and word[i+1] == sukoon:
                                current_syllable += word[i] + word[i+1]
                                i += 2
                            # Check for CVCC pattern (when followed by two consonants with sukoon)
                            elif i + 3 < len(word) and word[i+1] in consonants and word[i+2] == sukoon:
                                current_syllable += word[i] + word[i+1] + word[i+2]
                                i += 3
                        
                        syllables.append(current_syllable)
                        current_syllable = ""
                    
                # Case: long vowel
                elif word[i] in long_vowels:
                    current_syllable += word[i]
                    i += 1
                    
                    # Check for CVV pattern (end of syllable)
                    if i >= len(word) or word[i] in consonants:
                        if i < len(word) and word[i] in consonants:
                            # Check for CVVC pattern
                            if i + 1 < len(word) and word[i+1] == sukoon:
                                current_syllable += word[i] + word[i+1]
                                i += 2
                        
                        syllables.append(current_syllable)
                        current_syllable = ""
                else:
                    # Consonant without vowel - assume implicit vowel
                    syllables.append(current_syllable)
                    current_syllable = ""
            else:
                # End of word with just a consonant - assume implicit vowel
                syllables.append(current_syllable)
                current_syllable = ""
        else:
            # Skip anything that doesn't start a valid syllable
            i += 1
    
    # Add any remaining syllable
    if current_syllable:
        syllables.append(current_syllable)
    
    return syllables

def syllabify_arabic_text(text):
    """Syllabify Arabic text with or without diacritics."""
    try:
        # Get the disambiguator
        mle = get_disambiguator()
        
        # Normalize and diacritize
        normalized = normalize_unicode(text)
        analyses = mle.disambiguate([normalized])
        diacritized = analyses[0].analyses[0].analysis['diac']
    except Exception as e:
        # If diacritization fails, use the original text
        diacritized = text
    
    # Now apply syllabification rules
    words = diacritized.split()
    all_syllables = []
    
    for word in words:
        word_syllables = syllabify_arabic_word(word)
        all_syllables.extend(word_syllables)
    
    return all_syllables, words

def count_syllables_in_text(text):
    """Count syllables in an Arabic text."""
    syllables, _ = syllabify_arabic_text(text)
    return len(syllables)

def calculate_syllable_features(essay):
    """
    Calculate syllable-related features for Arabic text including:
    - syllables: total number of syllables
    - syll_per_word: average syllables per word
    - complex_words: words with 3+ syllables
    - complex_words_dc: words that would be considered difficult per Dale-Chall criteria
    """
    try:
        # Get syllables and words
        syllables, words = syllabify_arabic_text(essay)
        
        # Count total syllables
        syllable_count = len(syllables)
        
        # Get word count
        word_count = len(words)
        
        # Calculate syllables per word
        syll_per_word = syllable_count / word_count if word_count > 0 else 0
        
        # Count complex words (words with 3+ syllables)
        complex_words_count = 0
        # Count words that are not in our Arabic adaptation of Dale-Chall list
        complex_words_dc_count = 0
        
        for i, word in enumerate(words):
            word_normalized = normalize_unicode(word.strip())
            word_syllables = syllabify_arabic_word(word)
            syllable_count_per_word = len(word_syllables)
            
            # Count words with 3+ syllables
            if syllable_count_per_word >= 3:
                complex_words_count += 1
            
            # Count words that would be considered difficult per Dale-Chall criteria
            # For Arabic: Words that are not stopwords AND (have 3+ syllables OR are 6+ characters)
            is_difficult = (
                word_normalized not in ARABIC_STOPWORDS and
                (syllable_count_per_word >= 3 or len(word_normalized) >= 6)
            )
            if is_difficult:
                complex_words_dc_count += 1
        
        return {
            "syllables": syllable_count,
            "syll_per_word": syll_per_word,
            "complex_words": complex_words_count,
            "complex_words_dc": complex_words_dc_count
        }
    except Exception as e:
        # Return default values if calculation fails
        return {
            "syllables": 0,
            "syll_per_word": 0,
            "complex_words": 0,
            "complex_words_dc": 0
        }

def calculate_readability_scores(essay):
    """
    Calculate readability scores for Arabic text including:
    - FleschReadingEase: adapted for Arabic using custom syllable count
    - SMOGIndex: adapted for Arabic
    """
    try:
        # Get necessary counts first
        syllables, words = syllabify_arabic_text(essay)
        syllable_count = len(syllables)
        word_count = len(words)
        
        # Count complex words (words with 3+ syllables)
        complex_words_count = 0
        for word in words:
            word_syllables = syllabify_arabic_word(word)
            if len(word_syllables) >= 3:
                complex_words_count += 1
        
        # Count sentences
        sentences = list(filter(str.strip, re.split(r'[.،!؛:؟]', essay)))
        sentence_count = len(sentences)
        
        # Ensure we have enough data to calculate scores
        if word_count == 0 or sentence_count == 0:
            return {
                "FleschReadingEase": 0,
                "SMOGIndex": 0
            }
        
        # Calculate Flesch Reading Ease (adapted for Arabic)
        # Original formula: 206.835 - 1.015 * (words/sentences) - 84.6 * (syllables/words)
        # Adapted weights for Arabic based on research
        flesch_score = 206.835 - (1.015 * (word_count / sentence_count)) - (84.6 * (syllable_count / word_count))
        
        # Calculate SMOG Index (adapted for Arabic)
        # Original formula: 1.0430 * sqrt(complex_words * (30 / sentence_count)) + 3.1291
        # We'll use the original formula but with our Arabic syllable counting method
        if sentence_count >= 30:
            smog_score = 1.0430 * math.sqrt(complex_words_count * (30 / sentence_count)) + 3.1291
        else:
            # Adjust for texts with fewer than 30 sentences
            smog_score = 1.0430 * math.sqrt(complex_words_count * (30 / max(1, sentence_count))) + 3.1291
        
        return {
            "FleschReadingEase": max(0, min(100, flesch_score)),  # Clamp to 0-100 range
            "SMOGIndex": max(0, smog_score)
        }
    except Exception as e:
        # Return default values if calculation fails
        return {
            "FleschReadingEase": 0,
            "SMOGIndex": 0
        }

def calculate_extended_readability_scores(essay):
    """
    Calculate additional readability scores for Arabic text including:
    - AutomatedReadability: ARI adapted for Arabic
    - LinsearWrite: adapted for Arabic
    - Kincaid: Flesch-Kincaid Grade Level adapted for Arabic
    - ARI: Automated Readability Index
    - Coleman-Liau: adapted for Arabic
    - LIX: Läsbarhetsindex adapted for Arabic
    - RIX: Readability Index adapted for Arabic
    - GunningFogIndex: adapted for Arabic
    - DaleChallIndex: adapted for Arabic using common word list
    """
    try:
        # Get necessary counts first
        syllables, words = syllabify_arabic_text(essay)
        syllable_count = len(syllables)
        word_count = len(words)
        
        # Count characters (excluding spaces)
        char_count = len(essay.replace(" ", ""))
        
        # Count complex words (words with 3+ syllables) and long words (6+ characters)
        complex_words_count = 0
        long_words_count = 0
        difficult_words_count = 0  # For Dale-Chall
        
        for word in words:
            word_normalized = normalize_unicode(word.strip())
            word_syllables = syllabify_arabic_word(word)
            syllable_count_per_word = len(word_syllables)
            
            # Count for various metrics
            if syllable_count_per_word >= 3:
                complex_words_count += 1
            if len(word_normalized) >= 6:
                long_words_count += 1
                
            # For Dale-Chall - define difficult word criteria for Arabic
            is_difficult = (
                word_normalized not in ARABIC_STOPWORDS and
                (syllable_count_per_word >= 3 or len(word_normalized) >= 6)
            )
            if is_difficult:
                difficult_words_count += 1
        
        # Count sentences
        sentences = list(filter(str.strip, re.split(r'[.،!؛:؟]', essay)))
        sentence_count = len(sentences)
        
        # Ensure we have enough data to calculate scores
        if word_count == 0 or sentence_count == 0:
            return {
                "AutomatedReadability": 0,
                "LinsearWrite": 0,
                "Kincaid": 0,
                "ARI": 0,
                "Coleman-Liau": 0,
                "LIX": 0,
                "RIX": 0,
                "GunningFogIndex": 0,
                "DaleChallIndex": 0
            }
        
        # Calculate percentages and ratios
        words_per_sentence = word_count / sentence_count
        chars_per_word = char_count / word_count
        percent_complex_words = (complex_words_count / word_count) * 100
        percent_long_words = (long_words_count / word_count) * 100
        percent_difficult_words = (difficult_words_count / word_count) * 100
        
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
        
        # Return all scores
        return {
            "AutomatedReadability": max(0, ari_score),
            "LinsearWrite": max(0, linsear_score),
            "Kincaid": max(0, kincaid_score),
            "ARI": max(0, ari_score),
            "Coleman-Liau": max(0, coleman_liau_score),
            "LIX": max(0, lix_score),
            "RIX": max(0, rix_score),
            "GunningFogIndex": max(0, gunning_fog_score),
            "DaleChallIndex": max(0, dale_chall_score)
        }
    except Exception as e:
        print(f"Error calculating extended readability scores: {e}")
        return {
            "AutomatedReadability": 0,
            "LinsearWrite": 0,
            "Kincaid": 0,
            "ARI": 0,
            "Coleman-Liau": 0,
            "LIX": 0,
            "RIX": 0,
            "GunningFogIndex": 0,
            "DaleChallIndex": 0
        }

def calculate_grammar_features(essay):
    """Calculate grammar-related features for Arabic text."""
    try:
        # Initialize feature counts
        features = {
            "tobeverb": 0,          # كان وأخواتها - Arabic "to be" equivalents
            "auxverb": 0,           # Auxiliary verbs
            "conjunction": 0,        # Coordinating conjunctions
            "pronoun": 0,           # All pronouns
            "preposition": 0,       # Prepositions
            "nominalization": 0,    # Masdar forms (verbal nouns)
            "begin_w_pronoun": 0,    # Sentences starting with pronouns
            "begin_w_interrogative": 0, # Sentences starting with question words
            "begin_w_article": 0,    # Sentences starting with definite article
            "begin_w_subordination": 0, # Sentences starting with subordinating conjunctions
            "begin_w_conjunction": 0, # Sentences starting with coordinating conjunctions
            "begin_w_preposition": 0, # Sentences starting with prepositions
            "spelling_err": 0,       # Misspelled words
            "prep_comma": 0,         # Prepositions and commas
        }
        
        # Get MLE disambiguator for POS tagging
        mle = get_disambiguator()
        
        # Lists of Arabic POS patterns to match
        kana_sisters = ["كان", "أصبح", "أضحى", "أمسى", "ظل", "بات", "صار", "ليس", "مازال", "مادام", "مابرح", "مافتئ", "ماانفك"]
        aux_verbs = ["قد", "سوف", "س", "سـ"]
        conjunctions = ["و", "أو", "ثم", "ف", "لكن", "بل", "أم", "حتى", "إذ", "إذا", "لو", "كي", "لأن"]
        subordinating_conj = ["أن", "كي", "لكي", "عندما", "بعدما", "قبلما", "حيثما", "كلما", "طالما", "لو", "إذا", "حتى"]
        interrogatives = ["من", "ما", "متى", "أين", "كيف", "لماذا", "هل", "أ"]
        
        # Split text into sentences
        sentences = list(filter(str.strip, re.split(r'[.،!؛:؟]', essay)))
        
        # Count commas
        comma_count = essay.count('،')  # Arabic comma
        
        # Process each sentence for sentence-beginning features
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            # Normalize and get the first word
            normalized = normalize_unicode(sentence.strip())
            words = normalized.split()
            if not words:
                continue
                
            first_word = words[0]
            
            # Get POS tag for first word
            first_word_analysis = mle.disambiguate([first_word])
            if first_word_analysis and first_word_analysis[0].analyses:
                pos = first_word_analysis[0].analyses[0].analysis['pos']
                
                # Check beginning patterns - FIXED to handle lowercase tags
                if pos == 'pron' or pos == 'pron_dem':
                    features["begin_w_pronoun"] += 1
                elif first_word in interrogatives:
                    features["begin_w_interrogative"] += 1
                elif 'prc0' in first_word_analysis[0].analyses[0].analysis and first_word_analysis[0].analyses[0].analysis['prc0'] == 'Al_det':
                    features["begin_w_article"] += 1
                elif first_word in subordinating_conj:
                    features["begin_w_subordination"] += 1
                elif first_word in conjunctions or pos == 'conj' or pos == 'conj_sub':
                    features["begin_w_conjunction"] += 1
                elif pos == 'prep':
                    features["begin_w_preposition"] += 1
        
        # Process whole text for POS counts
        normalized_text = normalize_unicode(essay)
        text_analysis = mle.disambiguate(normalized_text.split())
        
        for word_analysis in text_analysis:
            if not word_analysis.analyses:
                # Potentially a spelling error if no analysis found
                features["spelling_err"] += 1
                continue
                
            analysis = word_analysis.analyses[0].analysis
            pos = analysis.get('pos', '')
            lemma = analysis.get('lex', '')
            
            # Count POS tags - FIXED to handle lowercase tags
            if pos == 'verb':
                if lemma in kana_sisters:
                    features["tobeverb"] += 1
            if pos == 'pron' or pos == 'pron_dem':
                features["pronoun"] += 1
            elif pos == 'prep':
                features["preposition"] += 1
                # Also count for prep_comma
                features["prep_comma"] += 1
            elif pos == 'conj' or pos == 'conj_sub' or lemma in conjunctions:
                features["conjunction"] += 1
            elif lemma in aux_verbs:
                features["auxverb"] += 1
            
            # Check for nominalization (masdar/verbal noun)
            # Look for verbal nouns indicated by specific patterns
            if pos == 'noun' and analysis.get('stemcat', '').startswith('Nap'):
                # Many masdar forms are indicated by Nap in stemcat
                features["nominalization"] += 1
        
        # Add commas to prep_comma count
        features["prep_comma"] += comma_count
        
        return features
        
    except Exception as e:
        print(f"Error calculating grammar features: {e}")
        # Return zero counts if calculation fails
        return {
            "tobeverb": 0, "auxverb": 0, "conjunction": 0, "pronoun": 0,
            "preposition": 0, "nominalization": 0, "begin_w_pronoun": 0,
            "begin_w_interrogative": 0, "begin_w_article": 0, 
            "begin_w_subordination": 0, "begin_w_conjunction": 0,
            "begin_w_preposition": 0, "spelling_err": 0, "prep_comma": 0
        }

def calculate_specific_pos_features(essay):
    """
    Calculate counts of specific POS tags in Arabic text using CAMeL Tools.
    Returns counts for tags like MD, DT, TO, PRP$, etc.
    """
    try:
        # Initialize feature counts with zeros
        pos_features = {
            "MD": 0, "DT": 0, "TO": 0, "PRP$": 0, "JJR": 0, "WDT": 0,
            "VBD": 0, "WP": 0, "VBG": 0, "RBR": 0, "CC": 0, "VBP": 0,
            "JJS": 0, "VBN": 0, "POS": 0, "NNS": 0, "WRB": 0, "JJ": 0,
            "CD": 0, "NNP": 0, "RP": 0, "RB": 0, "IN": 0, "VB": 0, 
            "VBZ": 0, "NN": 0, "PRP": 0
        }
        
        # Get MLE disambiguator for POS tagging
        mle = get_disambiguator()
        
        # Process the text
        normalized_text = normalize_unicode(essay)
        words = normalized_text.split()
        
        # Only process if we have words
        if not words:
            return pos_features
        
        # Get analyses for all words
        analyses = mle.disambiguate(words)
        
        # Mapping from CAMeL Tools lowercase tags to Penn Treebank tags
        camel_to_penn = {
            # Mapping lowercase tags as seen in logs
            'verb': 'VB',        # Base form verb
            'noun': 'NN',         # Singular noun
            'adj': 'JJ',          # Adjective
            'adv': 'RB',          # Adverb
            'pron': 'PRP',        # Personal pronoun
            'pron_dem': 'WP',     # Demonstrative pronoun
            'prep': 'IN',         # Preposition
            'conj': 'CC',         # Conjunction
            'conj_sub': 'IN',     # Subordinating conjunction
            'noun_prop': 'NNP',   # Proper noun
            'num': 'CD'           # Number
        }
        
        # Process each analysis
        for word_analysis in analyses:
            if not word_analysis.analyses:
                continue
                
            # Get the top analysis
            analysis = word_analysis.analyses[0].analysis
            
            # Extract POS tag
            pos_tag = analysis.get('pos', '')
            ud_tag = analysis.get('ud', '')
            
            # Map CAMeL Tools tag to Penn Treebank tag
            if pos_tag in camel_to_penn:
                penn_tag = camel_to_penn[pos_tag]
                if penn_tag in pos_features:
                    pos_features[penn_tag] += 1
            
            # Also use the UD tag if available (which is usually uppercase)
            if ud_tag in pos_features:
                pos_features[ud_tag] += 1
            
            # Get more specific tags based on morphological features
            if pos_tag == 'verb':
                aspect = analysis.get('asp', '')
                if aspect == 'p':  # Past tense
                    pos_features["VBD"] += 1
                elif aspect == 'i':  # Imperfect/present
                    # Check for 3rd person singular
                    if analysis.get('per', '') == '3' and analysis.get('num', '') == 's':
                        pos_features["VBZ"] += 1
                    else:
                        pos_features["VBP"] += 1
            
            # For nouns, check plurality
            elif pos_tag == 'noun':
                num = analysis.get('num', '')
                if num == 'p':  # Plural
                    pos_features["NNS"] += 1
            
            # Look for determiners (typically 'ال' prefix)
            if 'prc0' in analysis and analysis['prc0'] == 'Al_det':
                pos_features["DT"] += 1
        
        return pos_features
        
    except Exception as e:
        print(f"Error calculating specific POS features: {e}")
        # Return zero counts if calculation fails
        return {
            "MD": 0, "DT": 0, "TO": 0, "PRP$": 0, "JJR": 0, "WDT": 0,
            "VBD": 0, "WP": 0, "VBG": 0, "RBR": 0, "CC": 0, "VBP": 0,
            "JJS": 0, "VBN": 0, "POS": 0, "NNS": 0, "WRB": 0, "JJ": 0,
            "CD": 0, "NNP": 0, "RP": 0, "RB": 0, "IN": 0, "VB": 0, 
            "VBZ": 0, "NN": 0, "PRP": 0
        }

def process_essay(essay, prompt=None):
    """Process an Arabic essay to compute all available features."""
    # Initialize results dictionary
    features = {}
    
    try:
        # Calculate syllable features
        syllable_features = calculate_syllable_features(essay)
        features.update(syllable_features)
        
        # Calculate basic readability scores
        readability_scores = calculate_readability_scores(essay)
        features.update(readability_scores)
        
        # Calculate extended readability scores
        extended_readability_scores = calculate_extended_readability_scores(essay)
        features.update(extended_readability_scores)
        
        # Calculate grammar features
        grammar_features = calculate_grammar_features(essay)
        features.update(grammar_features)
        
        # Calculate specific POS tag features
        pos_features = calculate_specific_pos_features(essay)
        features.update(pos_features)
        
        return features
        
    except Exception as e:
        print(f"Error in essay processing: {e}")
        # Return at least an empty dict on error
        return features
