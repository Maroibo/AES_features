def calculate_pronoun_features(essay):
    """Extract pronoun features using CAMeL Tools with optimized processing."""
    features = {}
    pronoun_counts = defaultdict(int)
    group_counts = defaultdict(int)
    
    # Get cached sentences
    sentences = split_into_sentences(essay)
    if not sentences:
        return {}
    
    # Track sentence-level statistics
    sentences_with_pronoun = defaultdict(set)
    
    # Process each sentence
    for sent_idx, sentence in enumerate(sentences):
        if not sentence.strip():
            continue
        
        # Check cache first
        if sentence in _SENTENCE_CACHE and 'analyses' in _SENTENCE_CACHE[sentence]:
            analyses = _SENTENCE_CACHE[sentence]['analyses']
        else:
            # Get POS tags using CAMeL Tools
            mle = get_disambiguator()
            words = simple_word_tokenize(sentence)
            analyses = mle.disambiguate(words)
            
            # Cache the results
            if sentence in _SENTENCE_CACHE:
                _SENTENCE_CACHE[sentence]['analyses'] = analyses
            else:
                _SENTENCE_CACHE[sentence] = {'analyses': analyses}
        
        # Process each word analysis
        for word_idx, analysis in enumerate(analyses):
            if not analysis.analyses:
                continue
            
            # Get the top analysis
            top_analysis = analysis.analyses[0].analysis
            pos = top_analysis.get('pos', '')
            
            # Check for pronouns using CAMeL Tools POS tags
            if pos == 'pron' or pos == 'pron_dem':
                # Get more specific features
                pron_type = pos
                gen = top_analysis.get('gen', '')
                num = top_analysis.get('num', '')
                per = top_analysis.get('per', '')
                
                # Create pronoun feature name
                feature_key = f"{pron_type}"
                if per:
                    feature_key += f"_{per}"
                if gen:
                    feature_key += f"_{gen}"
                if num:
                    feature_key += f"_{num}"
                
                # Increment counts
                pronoun_counts[feature_key] += 1
                sentences_with_pronoun[feature_key].add(sent_idx)
                
                # Count pronoun groups efficiently
                if pos == 'pron_dem':
                    group_counts['demonstrative'] += 1
                elif pos == 'pron' and 'rat' in top_analysis and top_analysis['rat'] == 'r':
                    group_counts['relative'] += 1
                elif per == '1':
                    group_counts['first_person'] += 1
                elif per == '2':
                    group_counts['second_person'] += 1
                elif per == '3':
                    group_counts['third_person'] += 1
    
    # Add counts to features (only for pronouns that actually appear)
    for pron_type, count in pronoun_counts.items():
        features[f"pron_{pron_type.lower()}"] = count
    
    for group, count in group_counts.items():
        features[f"group_{group}"] = count
    
    # Add sentence-level statistics
    total_sentences = len(sentences)
    if total_sentences > 0:
        for pron_type, sent_set in sentences_with_pronoun.items():
            features[f"sent_count_{pron_type.lower()}"] = len(sent_set)
            features[f"sent_percent_{pron_type.lower()}"] = (len(sent_set) / total_sentences) * 100
    
    return features

def calculate_possessive_features(essay):
    """Extract possessive features efficiently using CAMeL Tools."""
    features = {}
    poss_counts = defaultdict(int)
    
    # Get cached sentences
    sentences = split_into_sentences(essay)
    if not sentences:
        return {}
    
    # Track sentence-level statistics
    sentences_with_poss = set()
    
    # Get morphological analyzer
    analyzer = get_analyzer()
    
    # Process each sentence
    for sent_idx, sentence in enumerate(sentences):
        if not sentence.strip():
            continue
        
        has_poss_in_sentence = False
        words = simple_word_tokenize(sentence)
        
        for word in words:
            # Analyze the word
            analyses = analyzer.analyze(word)
            
            for analysis in analyses:
                # Check for possessive features in CAMeL Tools analysis
                # Look for enclitic possessive pronouns
                enc0 = analysis.get('enc0', '0')
                
                if enc0 != '0' and 'POSS' in enc0:
                    has_poss_in_sentence = True
                    
                    # Determine possessive person
                    if '1S' in enc0 or '1P' in enc0:
                        poss_counts['first_person_poss'] += 1
                    elif '2' in enc0:  # All 2nd person forms
                        poss_counts['second_person_poss'] += 1
                    elif '3' in enc0:  # All 3rd person forms
                        poss_counts['third_person_poss'] += 1
                    
                    # We found at least one possessive in this analysis
                    break
            
            if has_poss_in_sentence:
                poss_counts['general_possessive'] += 1
                sentences_with_poss.add(sent_idx)
    
    # Add counts to features
    for poss_type, count in poss_counts.items():
        features[f"poss_{poss_type}"] = count
    
    # Add sentence-level statistics
    total_sentences = len(sentences)
    if total_sentences > 0:
        features["sentences_with_poss"] = len(sentences_with_poss)
        features["sent_percent_poss"] = (len(sentences_with_poss) / total_sentences) * 100
    
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
    try:
        # Common Arabic coordinating conjunctions
        conjunctions = ['و', 'أو', 'ثم', 'ف', 'لكن', 'بل', 'أم', 'حتى']
        
        # Split into sentences first
        sentences = list(filter(str.strip, re.split(r'[.،!؛:؟]', essay)))
        
        # Initialize lists to store metrics
        clauses_per_sentence = []
        sentence_depths = []
        leaf_depths = []
        clause_lengths = []
        
        tagger = get_tagger()  # Get tagger once for all sentences
        
        for sentence in sentences:
            try:
                # Create a regex pattern for splitting on conjunctions
                pattern = '|'.join(r'\s+{}\s+'.format(conj) for conj in conjunctions)
                
                # Split sentence into clauses
                clauses = re.split(pattern, sentence)
                clauses = [clause.strip() for clause in clauses if clause.strip()]
                
                # Store number of clauses in this sentence
                clauses_per_sentence.append(len(clauses))
                
                # Calculate clause lengths
                for clause in clauses:
                    words = word_tokenize(clause)
                    clause_lengths.append(len(words))
                
                # Get parse tree information if tagger is available
                if tagger is not None:
                    try:
                        normalized = normalize_unicode(dediac_ar(sentence))
                        tokens = simple_word_tokenize(normalized)
                        analyses = tagger.tag(tokens)
                        
                        # Calculate parse tree depths
                        max_depth = 0
                        leaf_depth_sum = 0
                        leaf_count = 0
                        
                        for analysis in analyses:
                            # The analysis is now a string, not an object
                            if analysis:  # If there's a POS tag
                                depth = len(analysis.split('/'))  # Count levels in POS tag
                                max_depth = max(max_depth, depth)
                                
                                # If it's a leaf node (no further subdivisions)
                                if '/' not in analysis:
                                    leaf_depth_sum += depth
                                    leaf_count += 1
                        
                        sentence_depths.append(max_depth if max_depth > 0 else 1)
                        if leaf_count > 0:
                            leaf_depths.append(leaf_depth_sum / leaf_count)
                        else:
                            leaf_depths.append(1)
                    except Exception:
                        sentence_depths.append(1)
                        leaf_depths.append(1)
                else:
                    sentence_depths.append(1)
                    leaf_depths.append(1)
                    
            except Exception:
                sentence_depths.append(1)
                leaf_depths.append(1)
        
        # Calculate final metrics with error handling
        mean_clause = sum(clause_lengths) / len(clause_lengths) if clause_lengths else 1
        clause_per_s = sum(clauses_per_sentence) / len(sentences) if sentences else 1
        sent_ave_depth = sum(sentence_depths) / len(sentences) if sentences else 1
        ave_leaf_depth = sum(leaf_depths) / len(leaf_depths) if leaf_depths else 1
        max_clause_in_s = max(clauses_per_sentence) if clauses_per_sentence else 1
        
        return {
            "mean_clause": mean_clause,
            "clause_per_s": clause_per_s,
            "sent_ave_depth": sent_ave_depth,
            "ave_leaf_depth": ave_leaf_depth,
            "max_clause_in_s": max_clause_in_s
        }
        
    except Exception:
        # Return default values if calculation fails
        return {
            "mean_clause": 0,
            "clause_per_s": 0,
            "sent_ave_depth": 0,
            "ave_leaf_depth": 0,
            "max_clause_in_s": 0
        }

def calculate_complexity_features(essay):
    """
    Calculates complexity-related features for Arabic text including:
    - sentences: total number of sentences
    - paragraphs: total number of paragraphs
    - long_words: words with 7 or more characters
    """
    # Split into sentences using Arabic punctuation
    sentences = list(filter(str.strip, re.split(r'[.،!؛:؟]', essay)))
    sentences_count = len(sentences)
    
    # Split into paragraphs
    paragraphs = essay.split('\n\n')
    paragraphs_count = len(paragraphs)
    
    # Process words
    words = word_tokenize(re.sub(r'[^\w\s]', '', essay))
    
    # Count long words (7 or more characters)
    long_words_count = sum(1 for word in words if len(word) >= 7)
    
    return {
        "sentences": sentences_count,
        "long_words": long_words_count
    }

# First, create a global variable to store the top N words
_TOP_N_WORDS = None

def get_top_n_words_from_essays(essays, n=100):
    """
    Get the top N most frequent words across all essays.
    
    Args:
        essays (list): List of essay texts
        n (int): Number of top words to consider
    Returns:
        set: Set of top N words
    """
    global _TOP_N_WORDS
    if _TOP_N_WORDS is not None:
        return _TOP_N_WORDS
        
    # Count words across all essays
    total_word_counts = Counter()
    
    for essay in essays:
        # Normalize and tokenize text
        normalized_text = normalize_unicode(essay)
        words = word_tokenize(re.sub(r'[^\w\s]', '', normalized_text))
        
        # Remove stop words and normalize words
        words = [dediac_ar(word) for word in words if word not in ARABIC_STOPWORDS]
        
        # Update counts
        total_word_counts.update(words)
    
    # Get top N words
    _TOP_N_WORDS = set(word for word, _ in total_word_counts.most_common(n))
    return _TOP_N_WORDS

def calculate_top_n_word_features(essay, n=100):
    """
    Calculates features related to top N words in the essay.
    Only includes features for the pre-determined top N words across all essays.
    
    Args:
        essay (str): The essay text
        n (int): Number of top words to consider (default 300)
    """
    # Normalize and tokenize text
    normalized_text = normalize_unicode(essay)
    words = word_tokenize(re.sub(r'[^\w\s]', '', normalized_text))
    
    # Remove stop words and normalize words
    words = [dediac_ar(word) for word in words if word not in ARABIC_STOPWORDS]
    
    # Get sentences for sentence-level features
    sentences = list(filter(str.strip, re.split(r'[.،!؛:؟]', essay)))
    
    # Count word frequencies in this essay
    word_counts = Counter(words)
    
    # Count sentences containing each word
    word_sentence_counts = defaultdict(int)
    word_sentence_percentages = defaultdict(float)
    
    for sentence in sentences:
        # Normalize and tokenize sentence
        sent_words = set(dediac_ar(w) for w in word_tokenize(re.sub(r'[^\w\s]', '', sentence))
                        if w not in ARABIC_STOPWORDS)
        
        # Count sentences containing each word
        for word in sent_words:
            if word in _TOP_N_WORDS:  # Only count if word is in top N
                word_sentence_counts[word] += 1
    
    # Calculate sentence percentages
    total_sentences = len(sentences)
    for word in word_sentence_counts:
        word_sentence_percentages[word] = (word_sentence_counts[word] / total_sentences 
                                         if total_sentences > 0 else 0)
    
    features = {}
    
    # Add features only for the pre-determined top N words
    for word in _TOP_N_WORDS:
        # Word count features
        features[f"top_n_word_count_{word}"] = word_counts[word]
        
        # Sentence count features
        features[f"top_n_num_sent_have_{word}"] = word_sentence_counts[word]
        
        # Sentence percentage features
        features[f"top_n_percentage_sent_have_{word}"] = word_sentence_percentages[word]
    
    return features

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
