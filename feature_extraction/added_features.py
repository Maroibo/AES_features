# Standard library imports
import os
import re
import math
from collections import Counter, defaultdict

# Third-party imports
import numpy as np
import torch
import nltk
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')
# Initialize NLTK properly
import nltk.data
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Initialize Arabic stopwords at module level
ARABIC_STOPWORDS = set(stopwords.words('arabic'))

# Import other dependencies
from transformers import AutoTokenizer, AutoModel

# Import camel-tools with error handling
try:
    from camel_tools.sentiment import SentimentAnalyzer
    from camel_tools.tokenizers.word import simple_word_tokenize
    from camel_tools.utils.normalize import normalize_unicode
    from camel_tools.disambig.mle import MLEDisambiguator
    from camel_tools.tagger.default import DefaultTagger
    from camel_tools.utils.dediac import dediac_ar
except ImportError as e:
    print(f"Error importing camel-tools: {e}")
    print("Please ensure camel-tools is installed correctly with: pip install --upgrade camel-tools")
    raise

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Cache models at module level for reuse
_sentiment_analyzer = None
_bert_tokenizer = None
_bert_model = None
_mle_disambiguator = None
_default_tagger = None

# Initialize models with error handling
def initialize_models():
    global _sentiment_analyzer, _bert_tokenizer, _bert_model, _mle_disambiguator, _default_tagger
    try:
        if _sentiment_analyzer is None:
            try:
                _sentiment_analyzer = SentimentAnalyzer.pretrained()
            except Exception as e:
                print(f"Warning: Could not initialize sentiment analyzer: {e}")
                print("Sentiment features will be set to 0")
                _sentiment_analyzer = None
        
        if _bert_tokenizer is None or _bert_model is None:
            model_name = "CAMeL-Lab/bert-base-arabic-camelbert-mix"
            _bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            _bert_model = AutoModel.from_pretrained(model_name).to(device)
        
        if _mle_disambiguator is None:
            try:
                _mle_disambiguator = MLEDisambiguator.pretrained()
            except Exception as e:
                print(f"Warning: Could not initialize disambiguator: {e}")
                print("Some features depending on disambiguation will be limited")
                _mle_disambiguator = None
        
        if _default_tagger is None and _mle_disambiguator is not None:
            _default_tagger = DefaultTagger(_mle_disambiguator, 'pos')
        
        return True
    except Exception as e:
        print(f"Error initializing models: {e}")
        return False

# Modified getter functions with error handling
def get_sentiment_analyzer():
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        _sentiment_analyzer = SentimentAnalyzer.pretrained()
    return _sentiment_analyzer

def get_bert_model(model_name="CAMeL-Lab/bert-base-arabic-camelbert-mix"):
    global _bert_tokenizer, _bert_model
    if _bert_tokenizer is None:
        _bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
    if _bert_model is None:
        _bert_model = AutoModel.from_pretrained(model_name).to(device)
    return _bert_tokenizer, _bert_model

def get_disambiguator():
    global _mle_disambiguator
    if _mle_disambiguator is None:
        _mle_disambiguator = MLEDisambiguator.pretrained()
    return _mle_disambiguator

def get_tagger():
    global _default_tagger, _mle_disambiguator
    if _default_tagger is None:
        if _mle_disambiguator is None:
            _mle_disambiguator = get_disambiguator()
        if _mle_disambiguator is not None:
            _default_tagger = DefaultTagger(_mle_disambiguator, 'pos')  # Exactly as in docs
    return _default_tagger

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

def calculate_sentiment_scores(essay):
    """
    Calculates sentiment scores and proportions for Arabic text.
    Returns default values if sentiment analyzer is not available.
    """
    if _sentiment_analyzer is None:
        return (0, 0, 0, 0, 1, 0)  # Default to neutral sentiment
    
    try:
        # Normalize and tokenize the text
        normalized_text = normalize_unicode(essay)
        sentences = list(filter(str.strip, re.split(r'[.،!؛:؟]', normalized_text)))
        total_sentences = len(sentences)
        
        if total_sentences == 0:
            return (0, 0, 0, 0, 1, 0)  # Default to neutral sentiment for empty text
        
        # Initialize counters
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        # Calculate sentiment for each sentence in batches
        batch_size = 8  # Process 8 sentences at a time
        positive_scores = []
        negative_scores = []
        neutral_scores = []  # Added neutral scores list
        
        for i in range(0, total_sentences, batch_size):
            batch = sentences[i:i + batch_size]
            try:
                # Get sentiment predictions for the batch
                sentiments = _sentiment_analyzer.predict(batch)
                
                # Process each sentiment prediction
                for sentiment in sentiments:
                    if sentiment == 'positive':
                        positive_scores.append(1)
                        negative_scores.append(0)
                        neutral_scores.append(0)
                        positive_count += 1
                    elif sentiment == 'negative':
                        positive_scores.append(0)
                        negative_scores.append(1)
                        neutral_scores.append(0)
                        negative_count += 1
                    else:  # neutral
                        positive_scores.append(0)
                        negative_scores.append(0)
                        neutral_scores.append(1)
                        neutral_count += 1
            except Exception as e:
                # For failed batches, count as neutral
                for _ in batch:
                    positive_scores.append(0)
                    negative_scores.append(0)
                    neutral_scores.append(1)
                    neutral_count += 1
        
        # Calculate overall scores (proportion of each sentiment type)
        overall_positivity = sum(positive_scores) / total_sentences
        overall_negativity = sum(negative_scores) / total_sentences
        overall_neutrality = sum(neutral_scores) / total_sentences
        
        # Calculate proportions (should sum to 1.0)
        positive_sentence_prop = positive_count / total_sentences
        neutral_sentence_prop = neutral_count / total_sentences
        negative_sentence_prop = negative_count / total_sentences
        
        return (overall_positivity, overall_negativity, overall_neutrality,
                positive_sentence_prop, neutral_sentence_prop, negative_sentence_prop)
    except Exception as e:
        return (0, 0, 0, 0, 1, 0)  # Default to neutral sentiment

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
    words = word_tokenize(re.sub(r'[^\w\s]', '', essay))
    words_count = len(words)
    
    # Calculate word length variance
    word_lengths = [len(word) for word in words]
    word_var = np.var(word_lengths)
    
    # Calculate sentence length variance
    sentences = list(filter(str.strip, re.split(r'[.،!؛:؟]', essay)))
    sentence_lengths = [len(word_tokenize(sentence)) for sentence in sentences]
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

def calculate_wordtype_features(essay):
    """
    Calculates wordtype-related features for Arabic text including:
    - wordtypes: total number of unique words in the essay
    """
    # Process words (removing punctuation and normalizing)
    words = word_tokenize(re.sub(r'[^\w\s]', '', essay))
    
    # Get unique words using set
    unique_words = set(dediac_ar(word) for word in words)  # Using dediac_ar to normalize words
    
    # Count unique words
    wordtypes_count = len(unique_words)
    
    return {
        "wordtypes": wordtypes_count
    }

def calculate_prompt_adherence_features(essay, prompt, model_name="CAMeL-Lab/bert-base-arabic-camelbert-mix"):
    """
    Calculates prompt adherence features using sentence embeddings with GPU acceleration.
    """
    # Get cached tokenizer and model
    tokenizer, model = get_bert_model(model_name)
    
    def get_embedding(text):
        # Tokenize and get model output
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        # Move inputs to GPU if available
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        # Use mean pooling to get text embedding
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
        return embedding
    
    # Process texts in batches for better GPU utilization
    def process_texts_in_batches(texts, batch_size=8):
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            # Tokenize batch
            batch_inputs = tokenizer(batch, return_tensors="pt", truncation=True, 
                                  max_length=512, padding=True)
            # Move to GPU
            batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
            
            with torch.no_grad():
                outputs = model(**batch_inputs)
            # Get embeddings for batch
            batch_embeddings = outputs.last_hidden_state.mean(dim=1)
            embeddings.extend(batch_embeddings.cpu())
        return embeddings
    
    # Get prompt embedding
    prompt_embedding = get_embedding(prompt)
    
    # Split essay into sentences and get embeddings
    sentences = list(filter(str.strip, re.split(r'[.،!؛:؟]', essay)))
    sentence_embeddings = process_texts_in_batches(sentences)
    
    # Move prompt embedding to CPU for calculations
    prompt_embedding = prompt_embedding.cpu()
    
    # Calculate dot scores using vectorized operations
    dot_scores = torch.stack([torch.dot(emb, prompt_embedding) for emb in sentence_embeddings])
    
    # Calculate features
    features = {
        "max_sentence_dot_score": dot_scores.max().item() if len(dot_scores) else 0,
        "mean_sentence_dot_score": dot_scores.mean().item() if len(dot_scores) else 0,
        "min_sentence_dot_score": dot_scores.min().item() if len(dot_scores) else 0,
        "dot_score": torch.dot(get_embedding(essay).cpu(), prompt_embedding).item()
    }
    
    return features

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

def extract_additional_features(essay, prompt=None, existing_features=None):
    """
    Extract basic text features without duplicating calculations from other functions.
    If existing_features is provided, will skip calculations that are already present.
    """
    # Initialize features dictionary with existing features if provided
    features = {} if existing_features is None else existing_features.copy()
    
    # Skip all calculations if these basic features are already computed
    if all(key in features for key in ["mean_word", "ess_char_len", "mean_sent"]):
        return features
    
    # Prepare the text
    # For words analysis
    words = word_tokenize(re.sub(r'[^\w\s]', '', essay))
    words_count = len(words)
    
    # For character analysis
    chars_count = len(essay.replace(" ", ""))  # not counting spaces
    
    # For sentence analysis (using Arabic punctuation)
    sentences = list(filter(str.strip, re.split(r'[.،!؛:؟]', essay)))
    sentences_count = len(sentences)
    
    # For paragraph analysis
    paragraphs = essay.split('\n\n')
    paragraphs_count = len(paragraphs)
    
    # Calculate mean_word (average characters per word)
    total_word_length = sum(len(word) for word in words)
    mean_word = total_word_length / words_count if words_count > 0 else 0
    
    # Calculate ess_char_len (total characters in essay)
    ess_char_len = chars_count
    
    # Calculate mean_sent (average words per sentence)
    sentence_lengths = [len(word_tokenize(sentence)) for sentence in sentences]
    mean_sent = sum(sentence_lengths) / sentences_count if sentences_count > 0 else 0
    
    # characters_per_word (same as mean_word but added for completeness)
    characters_per_word = mean_word
    
    # avg_word_len (same as mean_word but added for completeness)
    avg_word_len = mean_word
    
    # avg_words_per_sentence (same as mean_sent but added for completeness)
    avg_words_per_sentence = mean_sent
    
    # Calculate sentences_per_paragraph
    sentences_per_paragraph_count = []
    for paragraph in paragraphs:
        paragraph_sentences = list(filter(str.strip, re.split(r'[.،!؛:؟]', paragraph)))
        sentences_per_paragraph_count.append(len(paragraph_sentences))
    
    sentences_per_paragraph = sum(sentences_per_paragraph_count) / paragraphs_count if paragraphs_count > 0 else 0
    
    # Count periods and commas
    period_count = essay.count('.')
    comma_count = essay.count('،')  # Using Arabic comma
    
    # Add basic features to the dictionary
    basic_features = {
        "ess_char_len": ess_char_len,
        "mean_sent": mean_sent,
        "avg_word_len": avg_word_len,
        "avg_words_per_sentence": avg_words_per_sentence,
        "sentences_per_paragraph": sentences_per_paragraph,
        "period_count": period_count,
        "comma_count": comma_count,
    }
    
    # Update features dictionary
    features.update(basic_features)
    
    return features

def add_features(essay, existing_features=None, prompt=None):
    """
    Updated add_features to include prompt parameter for prompt adherence features
    """
    additional_features = extract_additional_features(essay, prompt)
    
    if existing_features is None:
        return additional_features
    
    combined_features = existing_features.copy()
    combined_features.update(additional_features)
    
    return combined_features

def compute_additional_features(essay):
    """
    Compute only the additional features from an essay.
    """
    return extract_additional_features(essay)

def process_essay(essay, prompt=None):
    """
    Process an Arabic essay to compute all available features with GPU acceleration where possible.
    """
    # Pre-initialize models
    if not initialize_models():
        raise RuntimeError("Model initialization failed")
    
    # Initialize results dictionary
    features = {}
    
    try:
        # Use torch.cuda.synchronize() to ensure GPU operations are complete
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Calculate readability measures (MADAD and OSMAN)
        madad_score, osman_score = calculate_madad_osman(essay)
        features["madad_score"] = madad_score
        features["osman_score"] = osman_score
        
        # Calculate sentiment scores
        sentiment_results = calculate_sentiment_scores(essay)
        features["overall_positivity"] = sentiment_results[0]
        features["overall_negativity"] = sentiment_results[1]
        features["positive_sentence_prop"] = sentiment_results[2]
        features["neutral_sentence_prop"] = sentiment_results[3]
        features["negative_sentence_prop"] = sentiment_results[4]
        
        # Calculate remaining features
        features.update(calculate_clause_features(essay))
        features.update(calculate_complexity_features(essay))
        features.update(calculate_variance_features(essay))
        features.update(calculate_wordtype_features(essay))
        
        if prompt is not None:
            features.update(calculate_prompt_adherence_features(essay, prompt))
        
        features.update(calculate_top_n_word_features(essay))
        
        # Calculate all remaining additional features
        additional_features = extract_additional_features(essay, prompt)
        
        # Update features dictionary with any remaining features not already added
        for key, value in additional_features.items():
            if key not in features:
                features[key] = value
        
        # Ensure all GPU operations are complete before returning
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        return features
        
    except Exception as e:
        print(f"Error in essay processing: {e}")
        raise

