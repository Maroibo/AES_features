from essay_proccessing import split_into_words, split_into_sentences, fuzzy_match
from camel_tools_init import _morph_analyzer
import numpy as np
from nltk.corpus import stopwords
from camel_tools.utils.normalize import normalize_unicode
import math
import re


ARABIC_STOPWORDS = set(stopwords.words('arabic'))



def calculate_lemma_features(essay,_morph_analyzer):
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
    return features
    
def long_words_count(essay):
    """
    Counts the number of words with 7 or more characters in the essay.
    """
    words = split_into_words(essay)
    long_words_count = sum(1 for word in words if len(word) >= 7)
    return long_words_count


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

def calculate_punctuation_counts(essay,_mle_disambiguator):
    """
    Counts various punctuation marks in an Arabic essay:
    - Exclamation mark count (!)
    - Semicolon count (;)
    - Double quotes count ("")
    - Dash count (-)
    - Colon count (:)
    - Question mark count (?)
    - Period count (.)
    - Comma count (,)
    - Apostrophe count (')
    - Quotation mark count (")
    - Parenthesis count (()
    - Total punctuation count from morphological analysis
    
    Args:
        essay (str): The Arabic essay text
        _morph_analyzer: The CAMeL Tools morphological analyzer
        
    Returns:
        dict: Dictionary containing counts of each punctuation mark
    """
    #how many times the punc tag appears in the essay
    punc_count = 0
    normalized_essay=normalize_unicode(essay)
    disambiguated = _mle_disambiguator.disambiguate(normalized_essay)   
    if disambiguated:
        for disambiguated_word in disambiguated:
            if disambiguated_word and len(disambiguated_word) > 0 and disambiguated_word.analyses:
                analysis = disambiguated_word.analyses[0].analysis
                if 'pos' in analysis:
                    if analysis['pos'] == 'punc':
                        punc_count += 1

    return {
        "exclamation_count": essay.count('!'),
        "semicolon_count": essay.count(';'),
        "dash_count": essay.count('-'),
        "colon_count": essay.count(':'),
        "question_count": essay.count('?'),
        "period_count": essay.count('.'),
        "comma_count": essay.count(','),
        "quotation_mark_count": essay.count('"'),
        "parenthesis_count": essay.count('('),
        "punc_count": punc_count
    }

def calculate_dup_punctuation_count(essay,_mle_disambiguator):
    """"
    check the number of times there is a two consecutive punctuation marks
    """
    normalized_essay=normalize_unicode(essay)
    disambiguated = _mle_disambiguator.disambiguate(normalized_essay)   
    dup_punc_count = 0
    prev_was_punc = False
    
    if disambiguated:
        for disambiguated_word in disambiguated:
            if disambiguated_word and len(disambiguated_word) > 0 and disambiguated_word.analyses:
                analysis = disambiguated_word.analyses[0].analysis
                if 'pos' in analysis:
                    is_punc = analysis['pos'] == 'punc'
                    if is_punc and prev_was_punc:
                        dup_punc_count += 1
                    prev_was_punc = is_punc
                else:
                    prev_was_punc = False
            else:
                prev_was_punc = False
    
    return dup_punc_count

def calculate_religious_phrases(intro_paragraph,body_paragraph,conclusion_paragraph):
    """
    Checks for religious phrases in specific paragraphs of the essay using fuzzy matching (93% similarity):
    - basmallah_use: First or second paragraph contains بسم الله الرحمن الرحيم
    - hamd_use: First or second paragraph contains الحمد لله رب العالمين
    - amma_baad_use: First or second paragraph contains أما بعد
    - salla_allah_use: First or second paragraph contains صلى الله وبارك
    - sallam_use: Last paragraph contains والسلام عليكم ورحمة الله وبركاته
    - salla_alla_mohammed_use: Last paragraph contains وصلى الله وسلم على نبينا محمد
    
    Args:
        intro_paragraph (str): The introduction paragraph
        body_paragraph (str): The body paragraph  
        conclusion_paragraph (str): The conclusion paragraph
        
    Returns:
        dict: Dictionary containing boolean values for each religious phrase check
    """

    
    # Split essay into paragraphs
    paragraphs = [intro_paragraph, body_paragraph, conclusion_paragraph]
    
    # Initialize result dictionary
    result = {
        "basmallah_use": 0,
        "hamd_use": 0,
        "amma_baad_use": 0,
        "salla_allah_use": 0,
        "sallam_use": 0,
        "salla_alla_mohammed_use": False
    }
    
    # Religious phrases to search for
    opening_phrases = {
        "basmallah_use": "بسم الله الرحمن الرحيم",
        "hamd_use": "الحمد لله رب العالمين", 
        "amma_baad_use": "أما بعد",
        "salla_allah_use": "صلى الله وبارك"
    }
    
    closing_phrases = {
        "sallam_use": "والسلام عليكم ورحمة الله وبركاته",
        "salla_alla_mohammed_use": "وصلى الله وسلم على نبينا محمد"
    }
    
    # Check first and second paragraphs for opening phrases
    for i in range(min(2, len(paragraphs))):
        paragraph = paragraphs[i]
        for key, phrase in opening_phrases.items():
            if fuzzy_match(paragraph, phrase,0.93):
                result[key] = 1
    
    # Check last paragraph for closing phrases
    if paragraphs:
        last_paragraph = paragraphs[-1]
        for key, phrase in closing_phrases.items():
            if fuzzy_match(last_paragraph, phrase,0.93):
                result[key] = 1
    
    return result

def calculate_advanced_punctuation_features(essay, _mle_disambiguator):
    """
    Analyzes advanced punctuation usage in Arabic text according to specific rules.
    
    Args:
        essay (str): The Arabic essay text
        _mle_disambiguator: The CAMeL Tools morphological analyzer
        
    Returns:
        dict: Dictionary containing counts of correct, missing, and incorrect uses of various punctuation marks
    """
    # Initialize counters
    features = {
        # Question mark features
        "question_mark_correct": 0,  # Correct question mark usage
        "question_mark_missing": 0,  # Missing question mark
        "question_mark_incorrect": 0,  # Incorrect question mark usage
        
        # Exclamation mark features
        "exclamation_mark_correct": 0,  # Correct exclamation mark usage
        "exclamation_mark_missing": 0,  # Missing exclamation mark
        "exclamation_mark_incorrect": 0,  # Incorrect exclamation mark usage
        
        # Semicolon features
        "semicolon_correct": 0,  # Correct semicolon usage
        "semicolon_missing": 0,  # Missing semicolon
        "semicolon_incorrect": 0,  # Incorrect semicolon usage
        
        # Comma features
        "comma_incorrect": 0,  # Incorrect comma usage with discourse connectives
        "comma_missing": 0,  # Missing comma with discourse connectives
        
        # Period features
        "period_correct": 0,  # Correct period usage
        "period_incorrect": 0,  # Incorrect period usage
        
        # Colon features
        "colon_correct": 0,  # Correct colon usage
        "colon_missing": 0,  # Missing colon
        "colon_incorrect": 0,  # Incorrect colon usage
        
        # Quotation mark features
        "quotation_mark_correct": 0,  # Correct quotation mark usage
        "quotation_mark_missing": 0,  # Missing quotation mark
        "quotation_mark_incorrect": 0,  # Incorrect quotation mark usage
    }
    
    # Question tools
    question_tools = ['هل', 'كيف', 'ماذا', 'لماذا', 'لم', 'كم', 'متى', 'أين']
    
    # Exaggerating styles
    exaggerating_styles = ['ياليت', 'بئس', 'رائع', 'لله در']
    
    # Causative indicators
    causative_indicators = ['لأن', 'بسبب', 'لكي']
    causative_prefixes = ['ل', 'ف']
    
    # Colon indicators
    colon_indicators = ['مثال', 'التالية', 'الآتية', 'مايلي']
    
    # Split into sentences and paragraphs
    sentences = split_into_sentences(essay)
    paragraphs = essay.split('\n\n')
    
    # Process each sentence
    for sentence in sentences:
        words = split_into_words(sentence)
        
        # Question mark analysis
        has_question_tool = any(fuzzy_match(sentence, tool, 0.95) for tool in question_tools)
        has_question_mark = '?' in sentence
        
        if has_question_tool and has_question_mark:
            features["question_mark_correct"] += 1
        elif has_question_tool and not has_question_mark:
            features["question_mark_missing"] += 1
        elif not has_question_tool and has_question_mark:
            features["question_mark_incorrect"] += 1
            
        # Exclamation mark analysis
        has_exaggerating_style = any(fuzzy_match(sentence, style, 0.95) for style in exaggerating_styles)
        has_exclamation_mark = '!' in sentence
        
        if has_exaggerating_style and has_exclamation_mark:
            features["exclamation_mark_correct"] += 1
        elif has_exaggerating_style and not has_exclamation_mark:
            features["exclamation_mark_missing"] += 1
        elif not has_exaggerating_style and has_exclamation_mark:
            features["exclamation_mark_incorrect"] += 1
            
        # Semicolon analysis
        if ';' in sentence:
            next_word = None
            for i, word in enumerate(words):
                if word == ';' and i + 1 < len(words):
                    next_word = words[i + 1]
                    break
                    
            if next_word:
                has_causative = (any(fuzzy_match(next_word, indicator, 0.95) for indicator in causative_indicators) or 
                               any(next_word.startswith(prefix) for prefix in causative_prefixes))
                if has_causative:
                    features["semicolon_correct"] += 1
                else:
                    features["semicolon_incorrect"] += 1
        else:
            # Check for missing semicolon
            for i, word in enumerate(words):
                if any(fuzzy_match(word, indicator, 0.95) for indicator in causative_indicators) or any(word.startswith(prefix) for prefix in causative_prefixes):
                    if i > 0 and words[i-1] != ';':
                        features["semicolon_missing"] += 1
                        
        # Colon analysis
        has_colon_indicator = any(fuzzy_match(sentence, indicator, 0.95) for indicator in colon_indicators)
        has_colon = ':' in sentence
        
        if has_colon_indicator and has_colon:
            features["colon_correct"] += 1
        elif has_colon_indicator and not has_colon:
            features["colon_missing"] += 1
        elif not has_colon_indicator and has_colon:
            features["colon_incorrect"] += 1
            
    # Process paragraphs for period and comma analysis
    for paragraph in paragraphs:
        # Period analysis
        if paragraph.strip().endswith('.'):
            features["period_correct"] += 1
        elif '.' in paragraph[:-1]:  # Period before end
            features["period_incorrect"] += 1
            
        # Comma analysis with discourse connectives
        # Note: This is a simplified version. You may want to add more discourse connectives
        discourse_connectives = {
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
        
        for connective in discourse_connectives:
            if fuzzy_match(paragraph, connective, 0.95):
                if ',' not in paragraph:
                    features["comma_incorrect"] += 1
                elif ',' in paragraph and not any(fuzzy_match(word, indicator, 0.95) for word in split_into_words(paragraph) for indicator in causative_indicators):
                    features["comma_missing"] += 1
                    
    # Quotation mark analysis
    # Comprehensive list of attribution cues categorized by type
    attribution_cues = {
        "assertion": [  # Declarative/assertive reporting
            "قال",      # said
            "ذكر",      # mentioned
            "صرح",      # declared
            "أعلن",     # announced
            "أفاد",     # stated
            "أكد",      # confirmed
            "جزم"       # asserted
        ],
        "directive": [  # Requesting/questioning
            "سأل",      # asked
            "طلب",      # requested
            "أمر",      # ordered
            "استفهم"    # questioned/inquired
        ],
        "expression": [  # Emotions, social acts
            "اعتذر",    # apologized
            "شكر",      # thanked
            "هنأ"       # congratulated
        ],
        "commissive": [  # Commitments/promises
            "وعد",      # promised
            "أقسم",     # swore
            "راهن"      # bet
        ],
        "declarative": [  # State-changing speech acts
            "أبلغ",     # informed
            "اعترف"     # admitted
        ],
        "adverbs_adjectives": [  # Used as modifiers in implicit attributions
            "مضيفاً",   # adding
            "معلقاً",   # commenting
            "مؤكداً",   # emphasizing
            "واصفاً"    # describing
        ],
        "prepositional_phrases": [  # Used with indirect attributions
            "بحسب",     # according to
            "وفقاً لـ",  # according to
            "على حد قوله"  # as he said / according to his words
        ]
    }
    
    # Flatten the attribution cues dictionary into a single list for easier checking
    all_attribution_cues = []
    for category in attribution_cues.values():
        all_attribution_cues.extend(category)
    
    for sentence in sentences:
        # Check for explicit attribution cues
        has_attribution = any(fuzzy_match(sentence, cue, 0.95) for cue in all_attribution_cues)
        
        # Check for implicit patterns (name followed by colon)
        words = split_into_words(sentence)
        for i, word in enumerate(words):
            if i + 1 < len(words) and words[i + 1] == ':':
                has_attribution = True
                break
        
        has_quotes = '"' in sentence
        
        if has_attribution and has_quotes:
            features["quotation_mark_correct"] += 1
        elif has_attribution and not has_quotes:
            features["quotation_mark_missing"] += 1
        elif not has_attribution and has_quotes:
            features["quotation_mark_incorrect"] += 1
            
    return features


def extract_surface_features(essay,intro_paragraph,body_paragraph,conclusion_paragraph):
    #words
    words = split_into_words(essay)
    words_count = len(words)
    log_words_count = math.log10(words_count)
    unique_words = set(words)
    unique_words_count = len(unique_words) #the set() removes duplicates
    log_unique_words_count = math.log10(unique_words_count)
    total_word_length = sum(len(word) for word in unique_words)#for unique words
    average_word_length = total_word_length / unique_words_count if unique_words_count > 0 else 0
    max_length_word = max(len(word) for word in unique_words)
    min_length_word = min(len(word) for word in unique_words)
    squared_diffs_words = [(len(word) - average_word_length) ** 2 for word in unique_words]
    mean_squared_diffs_words = sum(squared_diffs_words) / len(squared_diffs_words)
    standard_deviation_words= math.sqrt(mean_squared_diffs_words) #the standard deviation as a way to understand how much individual values within a group differ from the average value of that group

    #General counts
    chars_count = len(essay.replace(" ", "")) #not counting spaces
    hmpz_count = len(re.findall(r'[أإءؤئ]', essay))# Number of <hmzp> (F22)
    
    #Paragraphs
    paragraphs = [intro_paragraph,body_paragraph,conclusion_paragraph]
    paragraphs_count =len(paragraphs)  #num_paragraphs = len(essay.split('\n')) #Number of paragraphs (F3)
    is_first_paragraph_less_than_or_equal_to_10 = int(len(split_into_words(paragraphs[0])) <= 10 )#(F16)
    paragraphs_lengths = [len(split_into_words(paragraph)) for paragraph in paragraphs] #length of each paragraph interms of words
    average_length_paragraph = sum(paragraphs_lengths)/ paragraphs_count# Average length of paragraph (F11)
    max_length_paragraph = max(paragraphs_lengths) # Maximum length of paragraph (F12)
    min_length_paragraph = min(paragraphs_lengths) # Minimum length of paragraph (F13)
    #Sentences
    sentences = split_into_sentences(essay)
    sentences_count = len(sentences) # Number of sentences (F5)
    sentence_lengths = [len(split_into_words(sentence)) for sentence in sentences]
    average_length_sentence = sum(sentence_lengths) / sentences_count    # Average length of sentence (F10)
    max_length_sentence = max(sentence_lengths) # Maximum length of sentence 
    min_length_sentence = min(sentence_lengths)# Minimum length of sentence 
    squared_diff_sentence = [(length - average_length_sentence) ** 2 for length in sentence_lengths]
    mean_squared_diff_sentence = np.mean(squared_diff_sentence)
    standard_deviation_sentence = np.sqrt(mean_squared_diff_sentence)   
    
    #Grouping the features into a list
    extracted_surface_features= [words_count,log_words_count,unique_words_count,log_unique_words_count,
    average_word_length,max_length_word,min_length_word,standard_deviation_words,chars_count,hmpz_count, 
    paragraphs_count,is_first_paragraph_less_than_or_equal_to_10,average_length_paragraph,
    max_length_paragraph, min_length_paragraph, sentences_count, average_length_sentence,
    max_length_sentence, min_length_sentence,standard_deviation_sentence]

    features = {
        "words_count": words_count,
        "log_words_count": log_words_count,
        "unique_words_count": unique_words_count,
        "log_unique_words_count": log_unique_words_count,
        "average_word_length": average_word_length,
        "max_length_word": max_length_word,
        "min_length_word": min_length_word,
        "standard_deviation_words": standard_deviation_words,
        "chars_count": chars_count,
        "hmpz_count": hmpz_count,
        "paragraphs_count": paragraphs_count,
        "is_first_paragraph_less_than_or_equal_to_10": is_first_paragraph_less_than_or_equal_to_10,
        "average_length_paragraph": average_length_paragraph,
        "max_length_paragraph": max_length_paragraph,
        "min_length_paragraph": min_length_paragraph,
        "sentences_count": sentences_count,
        "average_length_sentence": average_length_sentence,
        "max_length_sentence": max_length_sentence,
        "min_length_sentence": min_length_sentence,
        "standard_deviation_sentence": standard_deviation_sentence
    }
    
    return features










        
