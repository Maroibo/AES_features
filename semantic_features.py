from nltk.corpus import stopwords
from essay_proccessing import split_into_sentences
import torch
from transformers import pipeline


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Initialize Arabic stopwords
ARABIC_STOPWORDS = set(stopwords.words('arabic'))


def calculate_sentiment_scores(essay):
    """
    Calculates sentiment scores and proportions for Arabic text.
    Returns default values if sentiment analyzer is not available.
    """    

    sentiment_analyzer = pipeline('sentiment-analysis', model='CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment')


    # Normalize and tokenize the text
    sentences = split_into_sentences(essay)
    total_sentences = len(sentences)
    # Initialize counters
    # positive_count = 0
    # negative_count = 0
    # neutral_count = 0

    positive_confidence = 0
    negative_confidence = 0
    neutral_confidence = 0
    
    # Calculate sentiment for each sentence in batches
    batch_size = 8  # Process 4 sentences at a time
    positive_scores = []
    negative_scores = []
    neutral_scores = []  # Added neutral scores list
    
    for i in range(0, total_sentences, batch_size):
        batch = sentences[i:i + batch_size]
        # Get sentiment predictions for the batch
        sentiments = sentiment_analyzer(batch)
        
        # Process each sentiment prediction
        for sentiment in sentiments:
            if sentiment['label'] == 'positive':
                positive_scores.append(1)
                negative_scores.append(0)
                neutral_scores.append(0)
                # positive_count += 1
                positive_confidence += sentiment['score']
            elif sentiment['label'] == 'negative':
                positive_scores.append(0)
                negative_scores.append(1)
                neutral_scores.append(0)
                negative_confidence += sentiment['score']
                # negative_count += 1
            else:  # neutral
                positive_scores.append(0)
                negative_scores.append(0)
                neutral_scores.append(1)
                neutral_confidence += sentiment['score']
                # neutral_count += 1   
    # Calculate overall scores (proportion of each sentiment type)
    positive_sentence_prop = sum(positive_scores) / total_sentences
    neutral_sentence_prop = sum(negative_scores) / total_sentences
    negative_sentence_prop = sum(neutral_scores) / total_sentences
    
    # Calculate proportions (should sum to 1.0)
    # positive_sentence_prop = positive_count / total_sentences
    # neutral_sentence_prop = neutral_count / total_sentences
    # negative_sentence_prop = negative_count / total_sentences

    overall_positivity = positive_confidence / total_sentences
    overall_negativity = negative_confidence / total_sentences
    overall_neutrality = neutral_confidence / total_sentences
    
    return {
        "overall_positivity": overall_positivity,
        "overall_negativity": overall_negativity,
        "overall_neutrality": overall_neutrality,
        "positive_sentence_prop": positive_sentence_prop,
        "neutral_sentence_prop": neutral_sentence_prop,
        "negative_sentence_prop": negative_sentence_prop
    }
    
def calculate_sent_match_words(essay):
    """
    Calculates the number of max and the average number words matched between the sentences in the essay
    """
    sentences = split_into_sentences(essay)
    max_matched_words = 0
    avg_matched_words = 0
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            matched_words = len(set(sentences[i].split()) & set(sentences[j].split()))
            max_matched_words = max(max_matched_words, matched_words)
            avg_matched_words += matched_words
    avg_matched_words /= len(sentences)
    return {
        "max_matched_words": max_matched_words,
        "avg_matched_words": avg_matched_words
    }

def calculate_prompt_adherence_features(essay, prompt, _bert_tokenizer, _bert_model):
    """
    Calculates prompt adherence features using sentence embeddings with GPU acceleration.
    """
    
    # Process texts in batches for better GPU utilization
    def get_embedding(texts, batch_size=8):
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            # Tokenize batch
            batch_inputs = _bert_tokenizer(batch, return_tensors="pt", truncation=True, 
                                  max_length=512, padding=True)
            # Move to GPU
            batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
            
            with torch.no_grad():
                outputs = _bert_model(**batch_inputs)
            # Get embeddings for batch
            batch_embeddings = outputs.last_hidden_state.mean(dim=1)
            embeddings.extend(batch_embeddings.cpu())
        return embeddings
    
    # Get prompt embedding
    prompt_embedding = get_embedding([prompt])
    

    sentences = split_into_sentences(essay)
    sentence_embeddings = get_embedding(sentences)
    
    # Move prompt embedding to CPU for calculations
    prompt_embedding = prompt_embedding[0].cpu()
    print(len(prompt_embedding))
    
    # Calculate dot scores using vectorized operations
    dot_scores = torch.stack([torch.dot(emb, prompt_embedding) for emb in sentence_embeddings])
    
    # Calculate features
    features = {
        "max_sentence_dot_score": dot_scores.max().item() ,
        "mean_sentence_dot_score": dot_scores.mean().item() ,
        "min_sentence_dot_score": dot_scores.min().item(),
        "dot_score": torch.dot(get_embedding([essay])[0].cpu(), prompt_embedding).item()
    }
    
    return features

def calculate_sim(text1, text2, _bert_tokenizer, _bert_model):
    """
    Calculates semantic similarity between two texts (paragraphs or sentences) using embeddings.
    
    Args:
        text1 (str): First text to compare
        text2 (str): Second text to compare
        model_name (str): Name of the BERT model to use
        
    Returns:
        float: Maximum similarity score between the texts
    """
    def get_embedding(text):
        # Tokenize and get embedding for a single text
        inputs = _bert_tokenizer(text, return_tensors="pt", truncation=True, 
                               max_length=512, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = _bert_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)[0].cpu()
    
    # Get embeddings for both texts
    embedding1 = get_embedding(text1)
    embedding2 = get_embedding(text2)
    
    # Calculate cosine similarity
    similarity = torch.dot(embedding1, embedding2) / (torch.norm(embedding1) * torch.norm(embedding2))
    
    return similarity.item()


def calculate_semantic_similarities(intro, body, conclusion,_bert_tokenizer, _bert_model):
    """
    Calculates semantic similarities between essay parts (intro, body, conclusion) at both paragraph and sentence levels.
    
    Args:
        intro (str): Introduction paragraph
        body (str): Body paragraph
        conclusion (str): Conclusion paragraph
        
    Returns:
        dict: Dictionary containing:
            - max_paragraph_sim: Maximum similarity between any two paragraphs (intro, body, conclusion)
            - avg_paragraph_sim: Average similarity between all paragraph pairs
            - max_sent_sim: Maximum similarity between any two sentences across all paragraphs
            - avg_sent_sim: Average similarity between all sentence pairs
    """
    # Get all sentences from each part
    intro_sentences = split_into_sentences(intro)
    body_sentences = split_into_sentences(body)
    conclusion_sentences = split_into_sentences(conclusion)
    
    # Calculate paragraph similarity
    paragraphs = [intro, body, conclusion]
    paragraph_similarities = []
    for i in range(len(paragraphs)):
        for j in range(i + 1, len(paragraphs)):
            similarity = calculate_sim(paragraphs[i], paragraphs[j], _bert_tokenizer, _bert_model)
            paragraph_similarities.append(similarity)
    
    max_paragraph_sim = max(paragraph_similarities) if paragraph_similarities else 0.0
    avg_paragraph_sim = sum(paragraph_similarities) / len(paragraph_similarities) if paragraph_similarities else 0.0
    
    # Calculate sentence similarity across all sentences
    all_sentences = intro_sentences + body_sentences + conclusion_sentences
    sentence_similarities = []
    if len(all_sentences) > 1:
        for i in range(len(all_sentences)):
            for j in range(i + 1, len(all_sentences)):
                similarity = calculate_sim(all_sentences[i], all_sentences[j], _bert_tokenizer, _bert_model )
                sentence_similarities.append(similarity)
    
    max_sent_sim = max(sentence_similarities) if sentence_similarities else 0.0
    avg_sent_sim = sum(sentence_similarities) / len(sentence_similarities) if sentence_similarities else 0.0
    
    return {
        "max_paragraph_sim": max_paragraph_sim,
        "avg_paragraph_sim": avg_paragraph_sim,
        "max_sent_sim": max_sent_sim,
        "avg_sent_sim": avg_sent_sim
    }





