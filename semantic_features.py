from nltk.corpus import stopwords
from camel_tools_init import get_disambiguator,_mle_disambiguator
from camel_tools.utils.normalize import normalize_unicode
from essay_proccessing import split_into_words, count_chars, split_into_sentences
import math
import re
import pandas as pd

_mle_disambiguator = get_disambiguator()
# Initialize Arabic stopwords
ARABIC_STOPWORDS = set(stopwords.words('arabic'))

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
    complex_words_dc_count = 0
    
    for i, word in enumerate(words):
        word_normalized = normalize_unicode(word.strip())
        word_syllables = syllabify_arabic_word(word)
        syllable_count_per_word = len(word_syllables)
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
        "complex_words_dc": complex_words_dc_count
    }
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


# df=pd.read_csv('../../../../shared/Arabic_Dataset/cleaned_cqc.csv')