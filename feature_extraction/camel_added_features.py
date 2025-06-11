# Standard library imports
import re
import math
import numpy as np
from collections import Counter, defaultdict
from functools import lru_cache
import multiprocessing as mp
from tqdm import tqdm

# Import camel-tools
try:
    from camel_tools.disambig.mle import MLEDisambiguator
    from camel_tools.utils.normalize import normalize_unicode
    from camel_tools.tokenizers.word import simple_word_tokenize
    from camel_tools.morphology.database import MorphologyDB
    from camel_tools.morphology.analyzer import Analyzer
except ImportError as e:
    print(f"Error importing camel-tools: {e}")
    print("Please ensure camel-tools is installed correctly with: pip install --upgrade camel-tools")
    raise

# Initialize global variables
_mle_disambiguator = None
_morph_analyzer = None
_TOP_BIGRAMS = None
_SENTENCE_SPLIT_REGEX = re.compile(r'[.!?;؟،:]\s+')

# Cache for processed sentences to avoid redundant processing
_SENTENCE_CACHE = {}

def get_disambiguator():
    """Get or initialize the MLE disambiguator."""
    global _mle_disambiguator
    if _mle_disambiguator is None:
        _mle_disambiguator = MLEDisambiguator.pretrained()
    return _mle_disambiguator

def get_analyzer():
    """Get or initialize the morphological analyzer."""
    global _morph_analyzer
    if _morph_analyzer is None:
        # Load the database and create an analyzer instance
        db = MorphologyDB.builtin_db()
        _morph_analyzer = Analyzer(db)
    return _morph_analyzer

def initialize_models():
    """Initialize CAMeL Tools models."""
    try:
        # Initialize both models
        get_disambiguator()
        get_analyzer()
        return True
    except Exception as e:
        print(f"Error initializing models: {e}")
        return False

@lru_cache(maxsize=10000)
def get_tag_bigrams(text):
    """Extract POS tag bigrams from text with caching for repeated patterns."""
    if not text or not text.strip():
        return []
    
    # Check cache for this exact text
    if text in _SENTENCE_CACHE:
        return _SENTENCE_CACHE[text]['bigrams']
    
    # Get POS tags using CAMeL Tools
    mle = get_disambiguator()
    words = simple_word_tokenize(text)
    analyses = mle.disambiguate(words)
    
    # Extract POS tags from analyses
    tags = []
    for analysis in analyses:
        if analysis.analyses:
            # Get the most likely analysis
            pos = analysis.analyses[0].analysis.get('pos', 'UNK')
            tags.append(pos)
        else:
            tags.append('UNK')  # Unknown tag
    
    # Create bigrams efficiently
    bigrams = list(zip(tags[:-1], tags[1:]))
    
    # Cache results
    _SENTENCE_CACHE[text] = {'tags': tags, 'bigrams': bigrams, 'analyses': analyses}
    
    return bigrams

def process_batch_for_bigrams(essays_batch):
    """Process a batch of essays to extract bigrams (for parallel processing)."""
    if not initialize_models():
        return []
    
    all_bigrams = []
    for essay in essays_batch:
        all_bigrams.extend(get_tag_bigrams(essay))
    return all_bigrams

def initialize_top_bigrams(essays, n=200, batch_size=100, n_jobs=None):
    """
    Initialize the top N most frequent tag bigrams efficiently using parallel processing.
    
    Args:
        essays (list): List of essay texts
        n (int): Number of top bigrams to extract
        batch_size (int): Number of essays to process in each batch
        n_jobs (int): Number of CPU cores to use, None for auto-detect
    
    Returns:
        dict: Dictionary of {bigram: rank} for the top N bigrams
    """
    global _TOP_BIGRAMS
    
    # Initialize models if not already initialized
    if not initialize_models():
        raise RuntimeError("Failed to initialize CAMeL Tools models")
    
    if not essays:
        print("WARNING: No essays provided for bigram initialization")
        _TOP_BIGRAMS = {}
        return _TOP_BIGRAMS
    
    print(f"Extracting bigrams from {len(essays)} essays...")
    
    # Determine optimal number of processes
    if n_jobs is None:
        n_jobs = max(1, mp.cpu_count() - 1)  # Leave one core free
    
    # Process in parallel
    all_bigrams = []
    
    if n_jobs > 1 and len(essays) > batch_size:
        # Split essays into batches
        batches = [essays[i:i+batch_size] for i in range(0, len(essays), batch_size)]
        
        print(f"Processing {len(batches)} batches with {n_jobs} workers...")
        
        # Create a pool of workers
        with mp.Pool(processes=n_jobs) as pool:
            # Map the function to process each batch
            results = list(tqdm(
                pool.imap(process_batch_for_bigrams, batches),
                total=len(batches),
                desc="Processing batches"
            ))
            
            # Combine all results
            for batch_bigrams in results:
                all_bigrams.extend(batch_bigrams)
    else:
        # Single process mode
        print("Using single process mode for bigram extraction...")
        all_bigrams = process_batch_for_bigrams(essays)
    
    # Count bigrams efficiently
    print(f"Counting {len(all_bigrams)} extracted bigrams...")
    bigram_counts = Counter(all_bigrams)
    
    # Get top N bigrams
    _TOP_BIGRAMS = {}
    
    # Ensure we don't try to get more bigrams than exist
    n = min(n, len(bigram_counts))
    
    if n == 0:
        print("WARNING: No bigrams found in essays!")
        return _TOP_BIGRAMS
    
    for rank, (bigram, count) in enumerate(bigram_counts.most_common(n)):
        _TOP_BIGRAMS[bigram] = rank
    
    print(f"Initialized {len(_TOP_BIGRAMS)} top tag bigrams")
    
    # Debug: show the most common bigrams
    if len(_TOP_BIGRAMS) > 0:
        top_3 = list(bigram_counts.most_common(3))
        print(f"Top 3 bigrams with counts: {top_3}")
    
    return _TOP_BIGRAMS

def calculate_bigram_features(essay):
    """Calculate bigram-related features from text efficiently using CAMeL Tools."""
    global _TOP_BIGRAMS
    
    if _TOP_BIGRAMS is None:
        print("WARNING: _TOP_BIGRAMS is None! Features will not be extracted.")
        return {}
    
    if len(_TOP_BIGRAMS) == 0:
        print("WARNING: _TOP_BIGRAMS is empty! Features will not be extracted.")
        return {}
    
    features = {}
    
    # Get bigrams from essay (cached if already processed)
    essay_bigrams = get_tag_bigrams(essay)
    
    if not essay_bigrams:
        # No bigrams found in essay
        features["total_top_bigrams"] = 0
        features["top_bigram_ratio"] = 0.0
        return features
    
    # Count occurrences of top bigrams efficiently
    bigram_counts = Counter(bigram for bigram in essay_bigrams if bigram in _TOP_BIGRAMS)
    
    # Add counts to features (only for bigrams that actually appear)
    for bigram, count in bigram_counts.items():
        # Clean up feature name to avoid problematic characters
        pos1 = str(bigram[0]).replace(' ', '_').replace('-', '_')
        pos2 = str(bigram[1]).replace(' ', '_').replace('-', '_')
        feature_name = f"bigram_{pos1}_{pos2}"
        features[feature_name] = count
    
    # Add total counts
    features["total_top_bigrams"] = sum(bigram_counts.values())
    features["top_bigram_ratio"] = sum(bigram_counts.values()) / len(essay_bigrams)
    
    return features

@lru_cache(maxsize=10000)
def split_into_sentences(text):
    """Split Arabic text into sentences with caching."""
    sentences = _SENTENCE_SPLIT_REGEX.split(text)
    return tuple([s.strip() for s in sentences if s.strip()])

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

def process_essay(essay, prompt=None):
    """Process an essay efficiently to extract all features using CAMeL Tools."""
    if not essay or not isinstance(essay, str):
        return {}
        
    if not initialize_models():
        raise RuntimeError("Failed to initialize CAMeL Tools models")
    
    features = {}
    try:
        # Calculate features - order matters for efficient caching
        
        # First check if bigram features are initialized
        global _TOP_BIGRAMS
        has_bigrams = _TOP_BIGRAMS is not None and len(_TOP_BIGRAMS) > 0
        
        # Calculate different feature types
        if has_bigrams:
            bigram_features = calculate_bigram_features(essay)
        else:
            bigram_features = {}
        
        pronoun_features = calculate_pronoun_features(essay)
        poss_features = calculate_possessive_features(essay)
        
        # Combine all features
        features.update(bigram_features)
        features.update(pronoun_features)
        features.update(poss_features)
        
        # Add basic counts for essay statistics
        words = simple_word_tokenize(essay)
        features['word_count'] = len(words)
        
        return features
    except Exception as e:
        print(f"Error processing essay: {e}")
        return {} 