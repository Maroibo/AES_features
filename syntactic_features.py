from nltk.corpus import stopwords
from camel_tools_init import _mle_disambiguator, _morph_analyzer, _dialect_id
from camel_tools.utils.normalize import normalize_unicode
from essay_proccessing import split_into_words, split_into_sentences
import re
from collections import defaultdict
import torch
from collections import Counter
import subprocess
import os
from camel_tools.utils.dediac import dediac_ar
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Initialize Arabic stopwords
ARABIC_STOPWORDS = set(stopwords.words('arabic'))
from camel_tools_init import _morph_analyzer



def syllabify_arabic_word(word):
    """
    Syllabify an Arabic word based on the rules from Zeki et al.
    
    Rules:
    - Every syllable begins with a consonant followed by a vowel
    - Syllable types: CV, CVV, CVC, CVVC, CVCC
    """
    # Define Arabic character sets
    consonants = "ةءأؤإئابتثجحخدذرزسشصضطظعغفقكلمنهويى"
    long_vowels = "اوي"  # alif, waw, yaa
    short_vowels = "َُِ"  # fatha, damma, kasra
    sukoon = "ْ"  # sukoon marks absence of vowel
    shadda = "ّ"  # gemination mark (doubles consonant)
    
    diacritized_word = _mle_disambiguator.disambiguate([word])[0].analyses[0].analysis['diac']
    word = diacritized_word

    # If last character is a dicaritic, remove it and replace with sukoon
    if word and word[-1] not in consonants:
        word = word[:-1] + sukoon
    elif word and word[-1] != sukoon:
        word += sukoon
    
    # Normalize word: handle shadda by duplicating the consonant
    normalized = ""
    i = 0
    while i < len(word):
        normalized += word[i]
        if i < len(word) - 1 and word[i+1] == shadda:
            normalized += sukoon 
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
                            # Check for CVCC pattern 
                            if i + 3 < len(word) and word[i+1] == sukoon and word[i+2] in consonants and word[i+3] == sukoon:
                                current_syllable += word[i] + word[i+1] + word[i+2] + word[i+3]
                                i += 4
                            # Check for CVC pattern (when followed by two consonants with sukoon)
                            elif i + 1 < len(word) and word[i+1] == sukoon:
                                current_syllable += word[i] + word[i+1]
                                i += 2
                            # elif i + 4 < len(word) and word[i+1] == sukoon and word[i+2] in consonants and word[i+3] == sukoon:
                            #     current_syllable += word[i] + word[i+1] + word[i+2] + word[i+3]
                            #     i += 4
                                # i += 3
                        
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


def syllabify_arabic_text(essay):
    """Syllabify Arabic text with or without diacritics."""
    words = split_into_words(essay)
    all_syllables = []
    for word in words:
        all_syllables.extend(syllabify_arabic_word(word))
    return all_syllables, words

def calculate_syllable_features(essay):
    """
    Calculate syllable-related features for Arabic text including:
    - syllables: total number of syllables
    - syll_per_word: average syllables per word
    - complex_words: words with 3+ syllables
    - complex_words_dc: words that would be considered difficult per Dale-Chall criteria
    """
    # Get syllables and words
    syllables, words = syllabify_arabic_text(essay)
    # Count total syllables
    syllable_count = len(syllables)
    # Get word count
    word_count = len(words)
    # Calculate syllables per word
    syll_per_word = syllable_count / word_count if word_count > 0 else 0
    # Count words that are not in our Arabic adaptation of Dale-Chall list
    complex_words_count = 0
    
    for i, word in enumerate(words):
        word_normalized = normalize_unicode(word.strip())
        word_syllables = syllabify_arabic_word(word)
        syllable_count_per_word = len(word_syllables)
        # Count words that would be considered difficult per Dale-Chall criteria
        # For Arabic: Words that are not stopwords AND (have 3+ syllables OR are 6+ characters)
        is_difficult = (syllable_count_per_word >= 3)
        if is_difficult:
            complex_words_count += 1
    
    return {
        "syllables": syllable_count,
        "syll_per_word": syll_per_word,
        "complex_words": complex_words_count
    }


def calculate_pronoun_features(essay):
    """Extract pronoun features using CAMeL Tools with optimized processing."""
    features = {}
    pronoun_counts = defaultdict(int)
    group_counts = defaultdict(int)  
    # Get cached sentences
    sentences = split_into_sentences(essay)
    
    # Track sentence-level statistics
    sentences_with_pronoun = defaultdict(set)
    
    # Process each sentence
    for sent_idx, sentence in enumerate(sentences):
        
        # Get POS tags using CAMeL Tools
        words = split_into_words(sentence)
        analyses = _mle_disambiguator.disambiguate(words)
        
        # Process each word analysis
        for word_idx, analysis in enumerate(analyses):
            # Get the top analysis
            top_analysis = analysis.analyses[0].analysis
            pos = top_analysis.get('pos', '')
            # Check for pronouns using CAMeL Tools POS tags
            # if pos == 'pron' or pos == 'pron_dem':
            if 'pron' in pos:
                # Get more specific features
                pron_type = pos
                # gen = top_analysis.get('gen', '')
                # num = top_analysis.get('num', '')
                per = top_analysis.get('per', '')
                
                # # Create pronoun feature name
                # feature_key = f"{pron_type}"
                if per and per != 'na':
                    feature_key += f"_{per}"
                # if gen and gen != 'na':
                #     feature_key += f"_{gen}"
                # if num and num != 'na':
                #     feature_key += f"_{num}"

                feature_key = f"{pron_type}_" + dediac_ar(top_analysis.get('stem', ''))
                
                # Increment counts
                pronoun_counts[feature_key] += 1
                sentences_with_pronoun[feature_key].add(sent_idx)
                # sentences_with_pronoun[pron_type].add(sent_idx)
                
                # Count pronoun groups efficiently
                if pos == 'pron_dem':
                    group_counts['demonstrative'] += 1
                    sentences_with_pronoun['demonstrative'].add(sent_idx)
                # elif pos == 'pron' and 'rat' in top_analysis and top_analysis['rat'] == 'r':
                elif pos == 'pron_rel':
                    group_counts['relative'] += 1
                    sentences_with_pronoun['relative'].add(sent_idx)
                elif pos == 'pron_interrog':
                    group_counts['interrogative'] += 1
                    sentences_with_pronoun['interrogative'].add(sent_idx)
                elif pos == 'pron_exclam':
                    group_counts['exclamation'] += 1
                    sentences_with_pronoun['exclamation'].add(sent_idx)
                elif per == '1':
                    group_counts['first_person'] += 1
                    sentences_with_pronoun['first_person'].add(sent_idx)
                elif per == '2':
                    group_counts['second_person'] += 1
                    sentences_with_pronoun['second_person'].add(sent_idx)
                elif per == '3':
                    group_counts['third_person'] += 1
                    sentences_with_pronoun['third_person'].add(sent_idx)

    # Add counts to features (only for pronouns that actually appear)
    for pron_type, count in pronoun_counts.items():
        features[f"pron_{pron_type.lower()}"] = count
        # The number of sentences that contain pron_type
        features[f"sent_count_{pron_type.lower()}"] = len(sentences_with_pronoun[pron_type])
        # The percentage of sentences that contain pronoun pron_type
        features[f"sent_percent_{pron_type.lower()}"] = (len(sentences_with_pronoun[pron_type]) / len(sentences) * 100) if len(sentences) > 0 else 0

    
    for group, count in group_counts.items():
        features[f"group_{group}"] = count
        # The number of sentences that contain group
        features[f"sent_count_{group}"] = len(sentences_with_pronoun[group])
        # The percentage of sentences that contain group
        features[f"sent_percent_{group}"] = (len(sentences_with_pronoun[group]) / len(sentences) * 100) if len(sentences) > 0 else 0
    
    # Add sentence-level statistics
    # total_sentences = len(sentences)
    # if total_sentences > 0:
    #     for pron_type, sent_set in sentences_with_pronoun.items():
    #         features[f"sent_count_{pron_type.lower()}"] = len(sent_set)
    #         features[f"sent_percent_{pron_type.lower()}"] = (len(sent_set) / total_sentences) * 100
    
    return features

def calculate_possessive_features(essay):
    """Extract possessive features efficiently using CAMeL Tools."""
    features = {}
    poss_counts = defaultdict(int)
    
    sentences = split_into_sentences(essay)
    sentences_with_poss = set()
    
    for sent_idx, sentence in enumerate(sentences):
        has_poss_in_sentence = False
        words = split_into_words(sentence)
        
        for word in words:
            analyses = _morph_analyzer.analyze(word)
            
            for analysis in analyses:
                enc0 = analysis.get('enc0', '0')
                
                # Look for possessive pronouns (end with _poss)
                if enc0 != '0' and enc0.endswith('_poss'):
                    has_poss_in_sentence = True
                    poss_counts['general_possessive'] += 1
                    
                    # Extract person from the enc0 value
                    if enc0.startswith('1'):  # 1s_poss, 1p_poss
                        poss_counts['first_person_poss'] += 1
                    elif enc0.startswith('2'):  # 2ms_poss, 2fs_poss, 2p_poss
                        poss_counts['second_person_poss'] += 1
                    elif enc0.startswith('3'):  # 3ms_poss, 3fs_poss, 3p_poss
                        poss_counts['third_person_poss'] += 1
                    
                    break  # Found possessive in this analysis
        
        if has_poss_in_sentence:
            sentences_with_poss.add(sent_idx)
    
    # Add counts to features
    for poss_type, count in poss_counts.items():
        features[f"group_{poss_type}"] = count
    
    # Add sentence-level statistics
    total_sentences = len(sentences)
    if total_sentences > 0:
        features["sent_count_general_possessive"] = len(sentences_with_poss)
        features["sent_percent_general_possessive"] = (len(sentences_with_poss) / total_sentences) * 100
    
    return features


def get_top_n_words_from_essays(essays, n=100):
    """
    Get the top N most frequent words across all essays.
    
    Args:
        essays (list): List of essay texts
        n (int): Number of top words to consider
    Returns:
        set: Set of top N words
    """        
    # Count words across all essays
    total_word_counts = Counter()
    
    for essay in essays:
        # Normalize and tokenize text
        normalized_text = normalize_unicode(essay)
        words = split_into_words(normalized_text)
        
        # Remove stop words and normalize words
        words = [normalize_unicode(word) for word in words if word not in ARABIC_STOPWORDS]
        
        # Update counts
        total_word_counts.update(words)
    # Get the top N words
    top_n_words = set(word for word, count in total_word_counts.most_common(n))
    return top_n_words

def calculate_top_n_word_features(essay, total_word_counts, n=100):
    """
    Calculates features related to top N words in the essay.
    Only includes features for the pre-determined top N words across all essays.
    
    Args:
        essay (str): The essay text
        n (int): Number of top words to consider (default 300)
    """
    # Tokenize text
    words = split_into_words(essay)
    # Remove stop words and normalize words
    words = [normalize_unicode(word) for word in words if word not in ARABIC_STOPWORDS]
    # Count word frequencies in this essay
    word_counts = Counter(words)
    
    # Get sentences for sentence-level features
    sentences = split_into_sentences(essay)
    total_sentences = len(sentences)
    
    # Count sentences containing each word
    word_sentence_counts = defaultdict(int)
    word_sentence_percentages = defaultdict(float)
    total_word_set = set(total_word_counts)
    
    for sentence in sentences:
        # Normalize and tokenize sentence
        sent_words = set(normalize_unicode(w) for w in split_into_words(sentence)
                        if w not in ARABIC_STOPWORDS)
        
        # Count sentences containing each word
        for word in sent_words & total_word_set:
            # if word in total_word_counts:  # Only count if word is in top N
                word_sentence_counts[word] += 1
    
    # Calculate sentence percentages
    for word, count in word_sentence_counts.items():
        word_sentence_percentages[word] = (count / total_sentences if total_sentences > 0 else 0)
    
    features = {}
    
    # Add features only for the pre-determined top N words
    for word in total_word_counts:
        # Word count features
        features[f"top_n_word_count_{word}"] = word_counts[word]
        # Sentence count features
        features[f"top_n_num_sent_have_{word}"] = word_sentence_counts[word]
        # Sentence percentage features
        features[f"top_n_percentage_sent_have_{word}"] = word_sentence_percentages[word]
    
    return features

def calculate_clause_features(essay):
    """
    Calculates various clause-related features including:
    - mean_clause: average number of words in each clause
    - clause_per_s: average number of clauses per sentence
    - sent_ave_depth: average parse tree depth per sentence
    - ave_leaf_depth: average parse depth of leaf nodes
    - max_clause_in_s: maximum number of clauses in any sentence
    """
    # Common Arabic coordinating conjunctions
    all_arabic_conjunctions = [
    'و', 'أو', 'أم', 'ف', 'ثم', 'لكن', 'لكنَّ', 'بل', 'حتى', 'لا', 
    'إما', 'كلا', 'إلا', 'غير', 'سوى', 'عدا', 'خلا', 'حاشا', 'ليس',
    'ما عدا', 'ما خلا', 'ما حاشا',
    # Subordinating  
    'أن', 'إن', 'أنَّ', 'إنَّ', 'كأن', 'كأنَّ', 'لأن', 'كي', 'لكي',
    'إذ', 'إذا', 'لو', 'لولا', 'لوما', 'لمّا', 'مذ', 'منذ', 'ريثما',
    'بينما', 'طالما', 'كلما', 'أينما', 'حيثما', 'مهما', 'كيفما',
    'أنَّى', 'حيث', 'بحيث', 'كون', 'ولو', 'وإن',
    # Multi-word conjunctions
    'بعد أن', 'قبل أن', 'منذ أن', 'في حين', 'ما دام', 'أيًّا ما',
    'متى ما', 'إذ أن', 'على أن', 'شريطة أن', 'غير أن', 'سوى أن',
    'إلا أن', 'بيد أن', 'على الرغم من أن', 'رغم أن', 'مع أن',
    'برغم أن', 'ولو أن', 'حتى لو', 'حتى وإن'
    ]
    
    # Split into sentences first
    sentences = split_into_sentences(essay)
    
    # Initialize lists to store metrics
    clauses_per_sentence = []
    sentence_depths = []
    leaf_depths = []
    clause_lengths = []
    
    for sentence in sentences:
       
        # Create a regex pattern for splitting on conjunctions
        pattern = '|'.join(r'\s+{}\s+'.format(conj) for conj in all_arabic_conjunctions)
        
        # Split sentence into clauses
        clauses = re.split(pattern, sentence)
        clauses = [clause.strip() for clause in clauses if clause.strip()]
        
        # Store number of clauses in this sentence
        clauses_per_sentence.append(len(clauses))
        
        # Calculate clause lengths
        for clause in clauses:
            words = split_into_words(clause)
            clause_lengths.append(len(words))
        
        normalized = normalize_unicode(sentence)
        tokens = split_into_words(normalized)
        tokens = [token.strip() for token in tokens]
        # Analyze each token individually
        all_analyses = []
        for token in tokens:
            analyses = _morph_analyzer.analyze(token)
            all_analyses.extend(analyses)
        
        # Calculate parse tree depths
        max_depth = 0
        leaf_depth_sum = 0
        leaf_count = 0
        
        for analysis in all_analyses:
            # analysis is a dictionary, not a string
            if analysis and 'bw' in analysis:  # 'bw' contains morphological breakdown
                bw_analysis = analysis['bw']  # e.g., 'أُ/IV1S+وافِق/IV'
                
                # Count morphological complexity levels
                if bw_analysis:
                    # Count slashes and plus signs as complexity indicators
                    complexity = bw_analysis.count('/') + bw_analysis.count('+')
                    max_depth = max(max_depth, complexity)
                    
                    # Simple leaf detection - if no subdivisions
                    if '/' not in bw_analysis and '+' not in bw_analysis:
                        leaf_depth_sum += 1
                        leaf_count += 1
                    else:
                        # Count segments
                        segments = len(bw_analysis.replace('+', '/').split('/'))
                        leaf_depth_sum += segments
                        leaf_count += 1
        
        sentence_depths.append(max_depth if max_depth > 0 else 1)
        if leaf_count > 0:
            leaf_depths.append(leaf_depth_sum / leaf_count)
        else:
            leaf_depths.append(1)
                
    # Calculate final metrics with error handling
    mean_clause = sum(clause_lengths) / len(clause_lengths) 
    clause_per_s = sum(clauses_per_sentence) / len(sentences) 
    sent_ave_depth = sum(sentence_depths) / len(sentences) 
    ave_leaf_depth = sum(leaf_depths) / len(leaf_depths)
    max_clause_in_s = max(clauses_per_sentence) 
    
    return {
        "mean_clause": mean_clause,
        "clause_per_s": clause_per_s,
        "sent_ave_depth": sent_ave_depth,
        "ave_leaf_depth": ave_leaf_depth,
        "max_clause_in_s": max_clause_in_s
    }

def calculate_grammar_features(essay):
    """Calculate grammar-related features for Arabic text."""

    # Initialize feature counts
    features = {
        "auxverb": 0,           # Auxiliary verbs
        "nominalization": 0,    # Masdar forms (verbal nouns)
        "begin_w_pronoun": 0,    # Sentences starting with pronouns
        "begin_w_interrogative": 0, # Sentences starting with question words
        "begin_w_article": 0,    # Sentences starting with definite article
        "begin_w_subordination": 0, # Sentences starting with subordinating conjunctions
        "begin_w_conjunction": 0, # Sentences starting with coordinating conjunctions
        "begin_w_preposition": 0, # Sentences starting with prepositions
        "prep_comma": 0,         # Prepositions and commas
        "pronoun": 0,            # Pronouns
        "conjunction": 0,        # Conjunctions
    }
    
    # Lists of Arabic POS patterns to match
    aux_verbs = [
        # Future tense markers
        "سوف", "س", "سـ", "سَ",
        # Past tense markers
        "قد", "لقد", "ما زال", "ما يزال", "ما برح", "ما يبرح", 
        "ما انفك", "ما ينفك", "ما فتئ", "ما يفتئ",
        # Modal auxiliaries
        "كان", "أصبح", "أضحى", "أمسى", "ظل", "بات", "صار", "ليس",
        # Aspectual auxiliaries
        "بدأ", "شرع", "أخذ", "جعل", "عاد", "رجع", "انبرى", "هب"
        ]
    conjunctions = [ 
        'و', 'أو', 'أم', 'ف', 'ثم', 'لكن', 'لكنَّ', 'بل', 'حتى', 'لا', 
        'إما', 'كلا', 'إلا', 'غير', 'سوى', 'عدا', 'خلا', 'حاشا', 'ليس',
        'ما عدا', 'ما خلا', 'ما حاشا',
        # Subordinating  
        'أن', 'إن', 'أنَّ', 'إنَّ', 'كأن', 'كأنَّ', 'لأن', 'كي', 'لكي',
        'إذ', 'إذا', 'لو', 'لولا', 'لوما', 'لمّا', 'مذ', 'منذ', 'ريثما',
        'بينما', 'طالما', 'كلما', 'أينما', 'حيثما', 'مهما', 'كيفما',
        'أنَّى', 'حيث', 'بحيث', 'كون', 'ولو', 'وإن',
        # Multi-word conjunctions
        'بعد أن', 'قبل أن', 'منذ أن', 'في حين', 'ما دام', 'أيًّا ما',
        'متى ما', 'إذ أن', 'على أن', 'شريطة أن', 'غير أن', 'سوى أن',
        'إلا أن', 'بيد أن', 'على الرغم من أن', 'رغم أن', 'مع أن',
        'برغم أن', 'ولو أن', 'حتى لو', 'حتى وإن'
        ]
    subordinating_conj = [
        # Basic subordinating conjunctions
        "أن", "إن", "أنَّ", "إنَّ", "كأن", "كأنَّ", "لأن", "كي", "لكي",
        # Time-related conjunctions
        "عندما", "بعدما", "قبلما", "حيثما", "كلما", "طالما", "متى", "إذ", "إذا",
        "حين", "حينما", "لمّا", "مذ", "منذ", "ريثما", "بينما",
        # Conditional conjunctions
        "لو", "لولا", "لوما", "إذا", "إن", "أما", "أينما", "حيثما", "مهما",
        "كيفما", "أنَّى", "حيث", "بحيث",
        # Purpose and result conjunctions
        "كي", "لكي", "حتى", "لئلا", "كي لا", "لكي لا",
        # Multi-word subordinating conjunctions
        "بعد أن", "قبل أن", "منذ أن", "في حين", "ما دام", "أيًّا ما",
        "متى ما", "إذ أن", "على أن", "شريطة أن", "غير أن", "سوى أن",
        "إلا أن", "بيد أن", "على الرغم من أن", "رغم أن", "مع أن",
        "برغم أن", "ولو أن", "حتى لو", "حتى وإن", "كون", "ولو", "وإن"
        ]
    interrogatives = [
        # Basic interrogatives
        "من", "ما", "متى", "أين", "كيف", "لماذا", "هل", "أ",
        # Additional interrogatives
        "أي", "أيان", "أنى", "كم", "كيفما", "أيما", "أينما",
        # Compound interrogatives
        "بماذا", "فيم", "علام", "إلام", "إلى متى", "من أين", "إلى أين",
        "كيف حال", "ما هو", "ما هي", "ما هم", "ما هن", "ما أنت", "ما أنتم",
        # Interrogative particles
        "أليس", "ألا", "أما", "أم", "أو", "هل", "أ",
        # Rhetorical questions
        "أترى", "أتعلم", "أتدرى", "أتعرف", "أتذكر", "أتخيل", "أتوقع",
        # Negative interrogatives
        "أليس", "ألم", "ألا", "أما", "أوما", "أفلا", "أوليس"
        ]
    
    # Split text into sentences
    sentences = split_into_sentences(essay)
    
    # Count commas
    comma_count = essay.count('،')  # Arabic comma
    
    # Process each sentence for sentence-beginning features
    for sentence in sentences:     
        # Normalize and get all words
        normalized = normalize_unicode(sentence.strip())
        words = split_into_words(normalized)            
        # Check each word in the sentence
        for i, word in enumerate(words):
            # Get POS tag for word
            word_analysis = _mle_disambiguator.disambiguate([word])
            if word_analysis and word_analysis[0].analyses:
                pos = word_analysis[0].analyses[0].analysis['pos']
                
                # Check patterns for each word
                if pos == 'pron' or pos == 'pron_dem':
                    if i == 0:  # Only count as beginning if it's the first word
                        features["begin_w_pronoun"] += 1
                    features["pronoun"] += 1
                elif word in interrogatives:
                    if i == 0:  # Only count as beginning if it's the first word
                        features["begin_w_interrogative"] += 1
                elif 'prc0' in word_analysis[0].analyses[0].analysis and word_analysis[0].analyses[0].analysis['prc0'] == 'Al_det':
                    if i == 0:  # Only count as beginning if it's the first word
                        features["begin_w_article"] += 1
                elif word in subordinating_conj:
                    if i == 0:  # Only count as beginning if it's the first word
                        features["begin_w_subordination"] += 1
                elif word in conjunctions or pos == 'conj' or pos == 'conj_sub':
                    if i == 0:  # Only count as beginning if it's the first word
                        features["begin_w_conjunction"] += 1
                    features["conjunction"] += 1
                elif pos == 'prep':
                    if i == 0:  # Only count as beginning if it's the first word
                        features["begin_w_preposition"] += 1
                    features["prep_comma"] += 1
                elif pos == 'verb':
                    lemma = word_analysis[0].analyses[0].analysis.get('lex', '')
                    if lemma in aux_verbs:
                        features["auxverb"] += 1
                
                # Check for nominalization (masdar/verbal noun)
                if pos == 'noun' and word_analysis[0].analyses[0].analysis.get('stemcat', '').startswith('Nap'):
                    features["nominalization"] += 1
    
    # Add commas to prep_comma count
    features["prep_comma"] += comma_count
    
    return features

def analyze_dialect_usage(text):
    """
    Analyze dialect usage in Arabic text using CAMeL Tools dialect identification models.
    
    Args:
        text (str): The Arabic text to analyze
        
    Returns:
        dict: A dictionary containing dialect usage statistics including:
            - dialect_percentages: Dictionary of dialect percentages
            - dialect_counts: Dictionary of dialect counts
            - msa_percentage: Percentage of MSA text
            - dialect_percentage: Percentage of dialectal text
            - most_common_dialect: The most frequently used dialect
    """
    
    # Normalize the text
    normalized_text = normalize_unicode(text)
    
    # Split text into sentences for better analysis
    sentences = split_into_sentences(normalized_text)
    
    # Initialize counters
    dialect_counts = Counter()
    total_sentences = len(sentences)
    
    # Analyze each sentence
    for sentence in sentences:
        # Get dialect prediction for the sentence
        prediction = _dialect_id.predict(sentence)
        dialect = prediction['dialect']
        dialect_counts[dialect] += 1
    # Calculate MSA vs dialect percentages
    msa_count = dialect_counts.get('MSA', 0)
    msa_percentage = (msa_count / total_sentences) * 100
    dialect_percentage = 100 - msa_percentage
    
    
    return {
        'dialect_counts': dict(dialect_counts),
        'msa_percentage': msa_percentage,
        'dialect_percentage': dialect_percentage,
    }

def calculate_nominal_verbal_sentences(essay):
    sentences=split_into_sentences(essay)
    nominal_sentences=0
    verbal_sentences=0
    for sentence in sentences:
        words=split_into_words(sentence)
        # check the first three words of it contains a verb then the sentence is verbal if not then the sentence is nominal
        words_to_check=words[:3]
        for word in words_to_check:
            word_analysis = _mle_disambiguator.disambiguate([word])
            if word_analysis and word_analysis[0].analyses:
                pos = word_analysis[0].analyses[0].analysis['pos']
                if pos == 'verb':
                    verbal_sentences+=1
                    break
        else:
            nominal_sentences+=1
    return {
        'nominal_sentences': nominal_sentences,
        'verbal_sentences': verbal_sentences
    }

def count_jazm_particles(essay):
    """
    Count jazm particles and track when they're followed by plural verbs ending with ن.
    Uses both morphological tags and particle list for validation.
    """
    jazm_stats = {
        "total_jazm": 0,
        "jazm_with_plural_verb": 0,
    }
    
    # Correct list of jazm particles
    jazm_particles = {
        "لم", "لما", "لام الأمر", "لا", "إن", "من", "ما", "مهما", "متى",
        "كيفما", "أنى", "أيان", "أي", "أينما", "حيثما", "إذما", "إذا",'لن'
    }
    
    words = split_into_words(essay)
    
    for i, word in enumerate(words):
        analyses = _morph_analyzer.analyze(word)
        if analyses:
            analysis = analyses[0]
            is_jazm = False
            pos_type = analysis.get('pos')
            # Check for لام using prc1 tag, and it should be followed by a verb
            # if 'prc1' in analysis and analysis['prc1'] == 'li_prep':
            if 'prc1' in analysis and pos_type == 'verb' and 'li' in analysis['prc1']:
                jazm_stats["total_jazm"] += 1 # This is for sure a jazm particle
                is_jazm = True
            
            # Check for negative particles using pos tag and bw field
            elif pos_type == 'part_neg':
                bw = analysis.get('bw', '')
                if '/NEG_PART' in bw:  # This will catch both لم and لن
                    is_jazm = True
            
            # Check for conditional and relative particles
            elif pos_type in ['part', 'conj']:
                bw = analysis.get('bw', '')
                # Check if the word contains any of the particles in the bw field
                if any(particle in bw for particle in jazm_particles):
                    is_jazm = True
            
            if is_jazm:
                # print(f"Found jazm particle: {word}")
                # jazm_stats["total_jazm"] += 1
                
                # Check if followed by a plural verb ending with ن
                if i + 1 < len(words):
                    next_word = words[i + 1]
                    next_analysis = _morph_analyzer.analyze(next_word)
                    
                    if next_analysis and next_analysis[0].get('pos') == 'verb':
                        # jazm particles must be followed by a verb
                        if pos_type != 'verb': # This condition is to execlude لام الأمر و لام الناهية because they are already considered
                            jazm_stats["total_jazm"] += 1
                        # Check for plural verb with ن ending
                        if (next_analysis[0].get('num') == 'p' and 
                            next_word.endswith('ن')):
                            jazm_stats["jazm_with_plural_verb"] += 1
    
    return jazm_stats

def count_conjunctions_and_transitions(essay):
    """
    Count occurrences of Arabic conjunctions and transitional phrases in the text.
    
    Args:
        essay (str): The Arabic text to analyze
        
    Returns:
        dict: A dictionary containing:
            - Counts of each conjunction and transition phrase
            - Ratio of connectives to total words
            - Ratio of unique connectives to paragraph words
            - Maximum and minimum distances between connectives
    """
    # Dictionary of conjunctions and transitions with their variations
    conjunctions_dict = {
        # Contrast and Opposition
        'الا ان': ['الا ان', 'الا أن'],
        'بيد ان': ['بيد ان', 'بيد أن'],
        'غير ان': ['غير ان', 'غير أن'],
        'على الرغم': ['على الرغم'],
        'رغمان': ['رغمان'],
        'بالرغم من': ['بالرغم من'],
        'برغم': ['برغم'],
        'بالمقابل': ['بالمقابل'],
        'في المقابل': ['في المقابل'],
        'بيد': ['بيد'],
        
        # Time and Sequence
        'بعدما': ['بعدما'],
        'اذ': ['اذ', 'إذ'],
        'بينما': ['بينما'],
        'عقب': ['عقب'],
        'قبيل': ['قبيل'],
        'وقبل': ['وقبل'],
        'من ثم': ['من ثم'],
        'قبل ان': ['قبل ان', 'قبل أن'],
        
        # Cause and Effect
        'جراء': ['جراء'],
        'نظرا ل': ['نظرا ل'],
        'بفضل': ['بفضل'],
        'لأن': ['لأن'],
        'بحيث': ['بحيث'],
        
        # Condition and Exception
        'الا اذا': ['الا اذا', 'الا إذا'],
        'حتى لو': ['حتى لو'],
        'لولا': ['لولا'],
        'طالما': ['طالما'],
        'كلما': ['كلما'],
        
        # Purpose and Comparison
        'بغية': ['بغية'],
        'كأن': ['كأن'],
        'خلافا ل': ['خلافا ل'],
        'بمعنى اخر': ['بمعنى اخر', 'بمعنى آخر'],
        
        # Context and Situation
        'في ظل': ['في ظل'],
        'حال': ['حال']
    }
    
    # Initialize results dictionary with all conjunctions set to 0
    results = {key: 0 for key in conjunctions_dict.keys()}
    
    # Normalize the text
    normalized_text = normalize_unicode(essay)
    
    # Get all words in the text
    all_words = split_into_words(normalized_text)
    total_words = len(all_words)
    
    # Track positions of connectives for distance calculations
    connective_positions = []
    unique_connectives = set()
    
    # Count occurrences of each conjunction and its variations
    for conjunction, variations in conjunctions_dict.items():
        for variation in variations:
            # Count exact matches
            count = normalized_text.count(variation)
            results[conjunction] += count
            
            # Count with spaces around the conjunction
            spaced_variation = f" {variation} "
            count = normalized_text.count(spaced_variation)
            results[conjunction] += count
            
            # Count at the beginning of sentences
            sentence_start = f"{variation} "
            count = normalized_text.count(sentence_start)
            results[conjunction] += count
            
            # Count at the end of sentences
            sentence_end = f" {variation}"
            count = normalized_text.count(sentence_end)
            results[conjunction] += count
            
            # Track positions and unique connectives
            if count > 0:
                unique_connectives.add(conjunction)
                # Find all positions of this variation
                start = 0
                while True:
                    pos = normalized_text.find(variation, start)
                    if pos == -1:
                        break
                    # Convert character position to word position
                    word_pos = len(split_into_words(normalized_text[:pos]))
                    connective_positions.append(word_pos)
                    start = pos + 1
    
    # Calculate distances between connectives
    connective_positions.sort()
    distances = []
    if len(connective_positions) > 1:
        for i in range(len(connective_positions) - 1):
            distance = connective_positions[i + 1] - connective_positions[i]
            distances.append(distance)
    
    # Calculate ratios and add to results
    if total_words > 0:
        results['connective_ratio'] = sum(results.values()) / total_words
        results['unique_connective_ratio'] = len(unique_connectives) / total_words
    else:
        results['connective_ratio'] = 0
        results['unique_connective_ratio'] = 0
    
    # Add distance metrics
    if distances:
        results["average_connective_distance"] = sum(distances) / len(distances)
    else:
        results["average_connective_distance"] = 0
    # Calculate total count
    total_count = sum(results.values())
    
    # Add total to results
    results['total_conjunctions'] = total_count
    
    # Ensure all 34 conjunctions are in the results with at least 0
    for conjunction in conjunctions_dict.keys():
        if conjunction not in results:
            results[conjunction] = 0
    
    return results

